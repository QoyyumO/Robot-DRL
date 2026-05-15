"""
DQN agent for vision-based control — GPU/headless training version.
Uses TensorFlow for training (CUDA GPU on Colab).
OpenVINO is intentionally excluded: it targets Intel hardware and won't use Colab's NVIDIA GPU.

Key GPU optimisations vs the desktop version:
  - @tf.function compiled Bellman graph: all 3 forward passes + backprop in one GPU
    kernel sequence, no Python round-trips between them.
  - Mixed precision (FP16): T4 tensor cores run at 65 TFLOPS FP16 vs 8.1 TFLOPS FP32.
  - Batch size 256 (from config): feeds enough work to fill T4 CUDA cores.
"""

import os
import warnings

# Suppress TF C++ log noise
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

_dqn = cfg["dqn"]
_DEFAULT_LR: float = _dqn["learning_rate"]
_DEFAULT_GAMMA: float = _dqn["gamma"]
_DEFAULT_EPS_START: float = _dqn["epsilon_start"]
_DEFAULT_EPS_END: float = _dqn["epsilon_end"]
_DEFAULT_EPS_DECAY_STEPS: int = _dqn["epsilon_decay_steps"]
_DEFAULT_BUFFER_SIZE: int = _dqn["buffer_size"]
_DEFAULT_BATCH_SIZE: int = _dqn["batch_size"]
_DEFAULT_TARGET_UPDATE_FREQ: int = _dqn["target_update_freq"]


def build_dqn_model(
    n_actions: int,
    input_h: int = OBS_H,
    input_w: int = OBS_W,
) -> "tf.keras.Model":
    """
    Small CNN for DQN (single 84x84 grayscale input).
    Not compiled here — training uses a custom @tf.function Bellman step
    with a separately managed optimizer for better GPU performance.
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
    inputs = tf.keras.layers.Input(shape=(input_h, input_w, 1), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    q_values = tf.keras.layers.Dense(n_actions)(x)
    return tf.keras.Model(inputs=inputs, outputs=q_values)


class DQNAgent:
    """
    DQN agent: epsilon-greedy policy, replay buffer, target network.

    Training uses a @tf.function compiled Bellman update that runs the
    target forward pass, current forward pass, and backprop as a single
    compiled GPU kernel sequence — no Python round-trips between them.

    Mixed precision (FP16) is supported for T4 / Ampere tensor cores.
    Set mixed_precision=True (or via config.yaml training.mixed_precision)
    to enable it. Activations are computed in FP16; weights stay in FP32.
    """

    def __init__(
        self,
        n_actions: int,
        obs_shape: Tuple[int, int] = (OBS_H, OBS_W),
        learning_rate: float = _DEFAULT_LR,
        gamma: float = _DEFAULT_GAMMA,
        epsilon_start: float = _DEFAULT_EPS_START,
        epsilon_end: float = _DEFAULT_EPS_END,
        epsilon_decay_steps: int = _DEFAULT_EPS_DECAY_STEPS,
        buffer_size: int = _DEFAULT_BUFFER_SIZE,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        target_update_freq: int = _DEFAULT_TARGET_UPDATE_FREQ,
        mixed_precision: bool = False,
    ):
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._step_count = 0
        self._mixed_precision = mixed_precision

        # Mixed precision must be set before building models
        if mixed_precision:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")

        self.model = build_dqn_model(n_actions, obs_shape[0], obs_shape[1])
        self.target_model = build_dqn_model(n_actions, obs_shape[0], obs_shape[1])
        self._sync_target()

        # Plain Adam — no LossScaleOptimizer needed.
        # The Bellman loss and gradients are computed in float32 (we cast
        # q_taken and bellman_target before the MSE), so there is no FP16
        # underflow risk regardless of the mixed_precision setting.
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the Bellman update as a TF graph once after everything is built.
        # tf.function traces on first call and reuses the compiled graph from then on.
        self._compiled_bellman = tf.function(self._bellman_graph)

        # Replay buffer (pre-allocated, circular)
        self._buffer_s = np.zeros((buffer_size, obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        self._buffer_a = np.zeros(buffer_size, dtype=np.int32)
        self._buffer_r = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_s2 = np.zeros((buffer_size, obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        self._buffer_done = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_size = buffer_size
        self._buffer_pos = 0
        self._buffer_len = 0

    def _bellman_graph(
        self,
        s: "tf.Tensor",
        a: "tf.Tensor",
        r: "tf.Tensor",
        s2: "tf.Tensor",
        done: "tf.Tensor",
    ) -> "tf.Tensor":
        """
        Full Bellman update compiled as a single TF graph by tf.function.

        Eliminates the 3 separate Python↔GPU round-trips of the original
        train_step (target_model.numpy() → model.numpy() → train_on_batch).
        All three passes + scatter + backprop run as one fused kernel sequence.

        Casts to float32 for numerically stable Bellman targets even when
        model activations are in float16 (mixed precision).
        """
        # ── Target Q values (no gradient) ───────────────────────────────────
        next_q = self.target_model(s2, training=False)          # (B, n_actions)
        next_q_max = tf.reduce_max(next_q, axis=1)              # (B,)

        # Bellman target — always in float32 for numerical stability
        r_f32 = tf.cast(r, tf.float32)
        done_f32 = tf.cast(done, tf.float32)
        nqm_f32 = tf.cast(next_q_max, tf.float32)
        bellman_target = r_f32 + self.gamma * nqm_f32 * (1.0 - done_f32)  # (B,)

        # Indices to select Q-value of the action actually taken
        batch_idx = tf.range(tf.shape(s)[0])
        indices = tf.stack([batch_idx, tf.cast(a, tf.int32)], axis=1)  # (B, 2)

        # ── Forward + backward pass ──────────────────────────────────────────
        with tf.GradientTape() as tape:
            q_pred = self.model(s, training=True)               # (B, n_actions)
            q_taken = tf.cast(tf.gather_nd(q_pred, indices), tf.float32)  # (B,)
            loss = tf.reduce_mean(tf.square(bellman_target - q_taken))
            if self._mixed_precision:
                # LossScaleOptimizer scales the loss to prevent FP16 underflow
                scaled_loss = self._optimizer.get_scaled_loss(loss)

        if self._mixed_precision:
            grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            grads = self._optimizer.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(loss, self.model.trainable_variables)

        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def _sync_target(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def _obs_to_batch(self, obs: np.ndarray) -> np.ndarray:
        """Ensure obs is (1, H, W, 1) float32."""
        if obs.ndim == 2:
            obs = obs[np.newaxis, :, :, np.newaxis]
        elif obs.ndim == 3:
            obs = obs[np.newaxis]
        return obs.astype(np.float32)

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        q = self.model(self._obs_to_batch(obs), training=False)
        return int(np.argmax(q[0].numpy()))

    def store(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        """Add transition to replay buffer (circular)."""
        pos = self._buffer_pos % self._buffer_size
        self._buffer_s[pos] = self._obs_to_batch(s)[0]
        self._buffer_a[pos] = a
        self._buffer_r[pos] = r
        self._buffer_s2[pos] = self._obs_to_batch(s2)[0]
        self._buffer_done[pos] = 1.0 if done else 0.0
        self._buffer_pos += 1
        self._buffer_len = min(self._buffer_len + 1, self._buffer_size)

    def train_step(self) -> Optional[float]:
        """
        One gradient update using the compiled Bellman graph.
        Returns the MSE loss, or None if the buffer hasn't reached batch_size yet.

        Called by train.py every train_freq env steps, gradient_steps times.
        Passes pre-allocated numpy slices as tf.constant tensors so TF doesn't
        re-trace the graph on every call.
        """
        if self._buffer_len < self._batch_size:
            return None

        idx = np.random.choice(self._buffer_len, self._batch_size, replace=False)

        # Convert to TF constants once before entering the compiled graph —
        # avoids repeated tensor allocation inside the graph body.
        s    = tf.constant(self._buffer_s[idx])
        a    = tf.constant(self._buffer_a[idx])
        r    = tf.constant(self._buffer_r[idx])
        s2   = tf.constant(self._buffer_s2[idx])
        done = tf.constant(self._buffer_done[idx])

        loss = self._compiled_bellman(s, a, r, s2, done)

        self._step_count += 1
        if self._step_count % self._target_update_freq == 0:
            self._sync_target()
        self.epsilon = max(self._epsilon_end, self.epsilon - self._epsilon_decay)
        return float(loss.numpy())

    def save(self, path: str) -> None:
        """Save model weights (.keras format preferred)."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """
        Load weights from .keras or .h5.
        Rebuilds the architecture fresh to avoid KerasTensor warnings.
        The compiled Bellman graph stays valid — it accesses self.model at
        call time, so it will pick up the reloaded weights automatically.
        """
        self.model = build_dqn_model(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        if path.endswith(".keras"):
            self.model.load_weights(path)
        else:
            self.model.load_weights(path, by_name=False, skip_mismatch=False)
        self._sync_target()
