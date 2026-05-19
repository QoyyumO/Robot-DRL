"""
PPO agent for vision-based discrete control (Schulman et al., 2017).
Actor-critic with shared CNN backbone; clipped surrogate + GAE advantages.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from typing import Dict, Optional, Tuple

from utils import PREPROCESS_WIDTH, PREPROCESS_HEIGHT
from config_loader import cfg

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

OBS_H = PREPROCESS_HEIGHT
OBS_W = PREPROCESS_WIDTH

_ppo = cfg["ppo"]
_DEFAULT_LR: float = _ppo["learning_rate"]
_DEFAULT_GAMMA: float = _ppo["gamma"]
_DEFAULT_GAE_LAMBDA: float = _ppo["gae_lambda"]
_DEFAULT_CLIP: float = _ppo["clip_range"]
_DEFAULT_ENTROPY: float = _ppo["entropy_coef"]
_DEFAULT_VALUE_COEF: float = _ppo["value_coef"]
_DEFAULT_MAX_GRAD_NORM: float = _ppo["max_grad_norm"]
_DEFAULT_BATCH_SIZE: int = _ppo["batch_size"]
_DEFAULT_N_EPOCHS: int = _ppo["n_epochs"]


def _cnn_backbone(inputs: "tf.Tensor") -> "tf.Tensor":
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.layers.Dense(512, activation="relu")(x)


def build_actor_critic(
    n_actions: int,
    input_h: int = OBS_H,
    input_w: int = OBS_W,
) -> "tf.keras.Model":
    """Shared backbone; policy logits + scalar value outputs."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
    inputs = tf.keras.layers.Input(shape=(input_h, input_w, 1), dtype=tf.float32)
    features = _cnn_backbone(inputs)
    logits = tf.keras.layers.Dense(n_actions, name="policy_logits")(features)
    value = tf.keras.layers.Dense(1, name="value")(features)
    return tf.keras.Model(inputs=inputs, outputs=[logits, value], name="actor_critic")


class PPOAgent:
    """Proximal Policy Optimization with discrete categorical policy."""

    def __init__(
        self,
        n_actions: int,
        obs_shape: Tuple[int, int] = (OBS_H, OBS_W),
        learning_rate: float = _DEFAULT_LR,
        gamma: float = _DEFAULT_GAMMA,
        gae_lambda: float = _DEFAULT_GAE_LAMBDA,
        clip_range: float = _DEFAULT_CLIP,
        entropy_coef: float = _DEFAULT_ENTROPY,
        value_coef: float = _DEFAULT_VALUE_COEF,
        max_grad_norm: float = _DEFAULT_MAX_GRAD_NORM,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        n_epochs: int = _DEFAULT_N_EPOCHS,
        mixed_precision: bool = False,
    ):
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self._batch_size = batch_size
        self._n_epochs = n_epochs

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.model = build_actor_critic(n_actions, obs_shape[0], obs_shape[1])
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._compiled_update = tf.function(self._ppo_update_graph)

    def _obs_batch_4d(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 2:
            obs = obs[np.newaxis]
        if obs.ndim == 3 and obs.shape[-1] != 1:
            obs = obs[:, :, :, np.newaxis]
        return obs.astype(np.float32)

    def _forward(self, obs_4d: "tf.Tensor") -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        logits, value = self.model(obs_4d, training=False)
        logits = tf.cast(logits, tf.float32)
        value = tf.cast(tf.squeeze(value, axis=-1), tf.float32)
        dist = tf.keras.distributions.Categorical(logits=logits)
        return dist, value, logits

    def act_batch(
        self, obs_batch: np.ndarray, training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample actions for N envs.
        Returns: actions (N,), log_probs (N,), values (N,)
        """
        obs_4d = tf.constant(self._obs_batch_4d(obs_batch))
        dist, values, _ = self._forward(obs_4d)
        if training:
            actions = dist.sample()
        else:
            actions = tf.argmax(dist.logits, axis=-1, output_type=tf.int32)
        log_probs = dist.log_prob(actions)
        return (
            actions.numpy().astype(np.int32),
            log_probs.numpy().astype(np.float32),
            values.numpy().astype(np.float32),
        )

    def evaluate_actions(
        self, obs: np.ndarray, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Log-probs, values, entropy for stored (obs, action) pairs."""
        obs_4d = tf.constant(self._obs_batch_4d(obs))
        actions_t = tf.constant(actions.astype(np.int32))
        dist, values, _ = self._forward(obs_4d)
        log_probs = dist.log_prob(actions_t)
        entropy = dist.entropy()
        return (
            log_probs.numpy().astype(np.float32),
            values.numpy().astype(np.float32),
            entropy.numpy().astype(np.float32),
        )

    @staticmethod
    def compute_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: float,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generalized Advantage Estimation.
        rewards, values, dones: shape (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - float(dones[t])
            next_value = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def _ppo_update_graph(
        self,
        obs: "tf.Tensor",
        actions: "tf.Tensor",
        old_log_probs: "tf.Tensor",
        advantages: "tf.Tensor",
        returns: "tf.Tensor",
    ) -> Dict[str, "tf.Tensor"]:
        with tf.GradientTape() as tape:
            logits, values = self.model(obs, training=True)
            logits = tf.cast(logits, tf.float32)
            values = tf.cast(tf.squeeze(values, axis=-1), tf.float32)
            dist = tf.keras.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            adv = tf.cast(advantages, tf.float32)
            adv = (adv - tf.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-8)

            ratio = tf.exp(log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            policy_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))

            ret = tf.cast(returns, tf.float32)
            value_loss = tf.reduce_mean(tf.square(ret - values))

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * tf.reduce_mean(entropy)
            )

        grads = tape.gradient(loss, self.model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": tf.reduce_mean(entropy),
        }

    def train_on_rollout(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """Run n_epochs of minibatch PPO updates on flattened rollout data."""
        n = len(obs)
        metrics_sum: Dict[str, float] = {}
        n_updates = 0

        for _ in range(self._n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self._batch_size):
                batch_idx = idx[start : start + self._batch_size]
                if len(batch_idx) < 2:
                    continue
                obs_b = tf.constant(self._obs_batch_4d(obs[batch_idx]))
                act_b = tf.constant(actions[batch_idx].astype(np.int32))
                old_lp = tf.constant(old_log_probs[batch_idx].astype(np.float32))
                adv_b = tf.constant(advantages[batch_idx].astype(np.float32))
                ret_b = tf.constant(returns[batch_idx].astype(np.float32))

                out = self._compiled_update(obs_b, act_b, old_lp, adv_b, ret_b)
                for k, v in out.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + float(v.numpy())
                n_updates += 1

        if n_updates == 0:
            return {}
        return {k: v / n_updates for k, v in metrics_sum.items()}

    def save(self, path: str) -> None:
        self.model.save(path)

    def load(self, path: str) -> None:
        self.model = build_actor_critic(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        if path.endswith(".keras"):
            self.model.load_weights(path)
        else:
            self.model.load_weights(path, by_name=False, skip_mismatch=False)
