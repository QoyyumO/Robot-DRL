"""
Discrete Soft Actor-Critic for vision-based control (Haarnoja et al., 2018).
Twin Q-networks + categorical policy with entropy regularisation.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from typing import Optional, Tuple

from utils import PREPROCESS_WIDTH, PREPROCESS_HEIGHT
from config_loader import cfg

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

OBS_H = PREPROCESS_HEIGHT
OBS_W = PREPROCESS_WIDTH

_sac = cfg["sac"]
_DEFAULT_LR: float = _sac["learning_rate"]
_DEFAULT_GAMMA: float = _sac["gamma"]
_DEFAULT_TAU: float = _sac["tau"]
_DEFAULT_ALPHA: float = _sac["alpha"]
_DEFAULT_BUFFER_SIZE: int = _sac["buffer_size"]
_DEFAULT_BATCH_SIZE: int = _sac["batch_size"]


def _cnn_backbone(inputs: "tf.Tensor") -> "tf.Tensor":
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.layers.Dense(512, activation="relu")(x)


def build_q_network(
    n_actions: int,
    input_h: int = OBS_H,
    input_w: int = OBS_W,
) -> "tf.keras.Model":
    """Outputs Q(s, a) for all discrete actions."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
    inputs = tf.keras.layers.Input(shape=(input_h, input_w, 1), dtype=tf.float32)
    features = _cnn_backbone(inputs)
    q_values = tf.keras.layers.Dense(n_actions)(features)
    return tf.keras.Model(inputs=inputs, outputs=q_values)


def build_policy_network(
    n_actions: int,
    input_h: int = OBS_H,
    input_w: int = OBS_W,
) -> "tf.keras.Model":
    """Categorical policy logits."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
    inputs = tf.keras.layers.Input(shape=(input_h, input_w, 1), dtype=tf.float32)
    features = _cnn_backbone(inputs)
    logits = tf.keras.layers.Dense(n_actions)(features)
    return tf.keras.Model(inputs=inputs, outputs=logits)


class SACAgent:
    """Discrete SAC: twin critics, softmax policy, fixed temperature alpha."""

    def __init__(
        self,
        n_actions: int,
        obs_shape: Tuple[int, int] = (OBS_H, OBS_W),
        learning_rate: float = _DEFAULT_LR,
        gamma: float = _DEFAULT_GAMMA,
        tau: float = _DEFAULT_TAU,
        alpha: float = _DEFAULT_ALPHA,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        mixed_precision: bool = False,
    ):
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self._batch_size = batch_size

        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.q1 = build_q_network(n_actions, obs_shape[0], obs_shape[1])
        self.q2 = build_q_network(n_actions, obs_shape[0], obs_shape[1])
        self.q1_target = build_q_network(n_actions, obs_shape[0], obs_shape[1])
        self.q2_target = build_q_network(n_actions, obs_shape[0], obs_shape[1])
        self.policy = build_policy_network(n_actions, obs_shape[0], obs_shape[1])

        self._sync_targets()
        self._q_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._pi_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self._compiled_train = tf.function(self._train_graph)

        self._buffer_s = np.zeros((buffer_size, obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        self._buffer_a = np.zeros(buffer_size, dtype=np.int32)
        self._buffer_r = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_s2 = np.zeros((buffer_size, obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        self._buffer_done = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_size = buffer_size
        self._buffer_pos = 0
        self._buffer_len = 0

    def _soft_update(self, source: "tf.keras.Model", target: "tf.keras.Model") -> None:
        for s_w, t_w in zip(source.trainable_variables, target.trainable_variables):
            t_w.assign(self.tau * s_w + (1.0 - self.tau) * t_w)

    def _sync_targets(self) -> None:
        self.q1_target.set_weights(self.q1.get_weights())
        self.q2_target.set_weights(self.q2.get_weights())

    def _obs_to_batch(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 2:
            obs = obs[np.newaxis, :, :, np.newaxis]
        elif obs.ndim == 3:
            obs = obs[np.newaxis]
        return obs.astype(np.float32)

    def select_actions_batch(self, obs_batch: np.ndarray, training: bool = True) -> np.ndarray:
        obs_4d = obs_batch[:, :, :, np.newaxis].astype(np.float32)
        logits = tf.cast(self.policy(tf.constant(obs_4d), training=False), tf.float32)
        if training:
            actions = tf.random.categorical(logits, 1, dtype=tf.int32)[:, 0]
        else:
            actions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        return actions.numpy().astype(np.int32)

    def store_batch(
        self,
        s_batch: np.ndarray,
        a_batch: np.ndarray,
        r_batch: np.ndarray,
        s2_batch: np.ndarray,
        done_batch: np.ndarray,
    ) -> None:
        N = len(a_batch)
        pos = np.arange(self._buffer_pos, self._buffer_pos + N) % self._buffer_size
        self._buffer_s[pos] = s_batch[:, :, :, np.newaxis].astype(np.float32)
        self._buffer_a[pos] = a_batch.astype(np.int32)
        self._buffer_r[pos] = r_batch.astype(np.float32)
        self._buffer_s2[pos] = s2_batch[:, :, :, np.newaxis].astype(np.float32)
        self._buffer_done[pos] = done_batch.astype(np.float32)
        self._buffer_pos += N
        self._buffer_len = min(self._buffer_len + N, self._buffer_size)

    def _train_graph(
        self,
        s: "tf.Tensor",
        a: "tf.Tensor",
        r: "tf.Tensor",
        s2: "tf.Tensor",
        done: "tf.Tensor",
    ) -> "tf.Tensor":
        """One SAC gradient step; returns mean critic loss."""
        a = tf.cast(a, tf.int32)
        r = tf.cast(r, tf.float32)
        done = tf.cast(done, tf.float32)

        # ── Critic update ───────────────────────────────────────────────────
        with tf.GradientTape() as q_tape:
            q1 = tf.cast(self.q1(s, training=True), tf.float32)
            q2 = tf.cast(self.q2(s, training=True), tf.float32)
            batch_idx = tf.stack([tf.range(tf.shape(s)[0]), a], axis=1)
            q1_a = tf.gather_nd(q1, batch_idx)
            q2_a = tf.gather_nd(q2, batch_idx)

            next_logits = tf.cast(self.policy(s2, training=False), tf.float32)
            next_log_pi = tf.nn.log_softmax(next_logits)
            next_probs = tf.nn.softmax(next_logits)

            q1_next = tf.cast(self.q1_target(s2, training=False), tf.float32)
            q2_next = tf.cast(self.q2_target(s2, training=False), tf.float32)
            min_q_next = tf.minimum(q1_next, q2_next)
            soft_v = tf.reduce_sum(
                next_probs * (min_q_next - self.alpha * next_log_pi), axis=1
            )
            target = r + self.gamma * (1.0 - done) * soft_v
            critic_loss = tf.reduce_mean(
                tf.square(target - q1_a) + tf.square(target - q2_a)
            )

        q_vars = self.q1.trainable_variables + self.q2.trainable_variables
        q_grads = q_tape.gradient(critic_loss, q_vars)
        self._q_optimizer.apply_gradients(zip(q_grads, q_vars))

        # ── Policy update ───────────────────────────────────────────────────
        with tf.GradientTape() as pi_tape:
            logits = tf.cast(self.policy(s, training=True), tf.float32)
            log_pi = tf.nn.log_softmax(logits)
            probs = tf.nn.softmax(logits)
            q1_pi = tf.cast(self.q1(s, training=False), tf.float32)
            q2_pi = tf.cast(self.q2(s, training=False), tf.float32)
            min_q = tf.minimum(q1_pi, q2_pi)
            policy_loss = tf.reduce_mean(
                tf.reduce_sum(probs * (self.alpha * log_pi - min_q), axis=1)
            )

        pi_grads = pi_tape.gradient(policy_loss, self.policy.trainable_variables)
        self._pi_optimizer.apply_gradients(zip(pi_grads, self.policy.trainable_variables))

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)
        return critic_loss

    def train_step(self) -> Optional[float]:
        if self._buffer_len < self._batch_size:
            return None
        idx = np.random.choice(self._buffer_len, self._batch_size, replace=False)
        s = tf.constant(self._buffer_s[idx])
        a = tf.constant(self._buffer_a[idx])
        r = tf.constant(self._buffer_r[idx])
        s2 = tf.constant(self._buffer_s2[idx])
        done = tf.constant(self._buffer_done[idx])
        loss = self._compiled_train(s, a, r, s2, done)
        return float(loss.numpy())

    @staticmethod
    def _component_path(base: str, suffix: str) -> str:
        stem = f"{base}_{suffix}"
        if os.path.isfile(stem):
            return stem
        keras_path = f"{stem}.keras"
        if os.path.isfile(keras_path):
            return keras_path
        return keras_path

    def save(self, path: str) -> None:
        base = path[:-6] if path.endswith(".keras") else path
        self.q1.save(f"{base}_q1.keras")
        self.q2.save(f"{base}_q2.keras")
        self.policy.save(f"{base}_policy.keras")

    def load(self, path: str) -> None:
        base = path[:-6] if path.endswith(".keras") else path
        self.q1 = build_q_network(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        self.q2 = build_q_network(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        self.policy = build_policy_network(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        self.q1.load_weights(self._component_path(base, "q1"))
        self.q2.load_weights(self._component_path(base, "q2"))
        self.policy.load_weights(self._component_path(base, "policy"))
        self.q1_target = build_q_network(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        self.q2_target = build_q_network(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        self._sync_targets()
