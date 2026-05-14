"""
DQN agent for vision-based control — GPU/headless training version.
Uses TensorFlow for training (CUDA GPU on Colab).
OpenVINO is intentionally excluded: it targets Intel hardware and won't use Colab's NVIDIA GPU.
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


def _get_custom_objects_for_h5_load():
    """Compat shim for .h5 models saved with older Keras (handles 'reduction' in MSE config)."""
    if not HAS_TF:
        return {}

    class _MSECompat(tf.keras.losses.MeanSquaredError):
        def __init__(self, name="mean_squared_error", reduction="sum_over_batch_size", **kwargs):
            kwargs.pop("reduction", None)
            super().__init__(name=name, **kwargs)

    return {
        "mse": _MSECompat,
        "mean_squared_error": _MSECompat,
        "MeanSquaredError": _MSECompat,
    }


def build_dqn_model(
    n_actions: int,
    input_h: int = OBS_H,
    input_w: int = OBS_W,
    learning_rate: float = _DEFAULT_LR,
) -> "tf.keras.Model":
    """Small CNN for DQN (single 84x84 grayscale input)."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
    inputs = tf.keras.layers.Input(shape=(input_h, input_w, 1), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    q_values = tf.keras.layers.Dense(n_actions)(x)
    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return model


class DQNAgent:
    """
    DQN agent: epsilon-greedy policy, replay buffer, target network.
    Trains with TensorFlow — runs on Colab's CUDA GPU automatically when
    a GPU is available (TF selects it without any extra configuration).
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

        self.model = build_dqn_model(n_actions, obs_shape[0], obs_shape[1], learning_rate)
        self.target_model = build_dqn_model(n_actions, obs_shape[0], obs_shape[1], learning_rate)
        self._sync_target()

        self._buffer_s = np.zeros((buffer_size, obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        self._buffer_a = np.zeros(buffer_size, dtype=np.int32)
        self._buffer_r = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_s2 = np.zeros((buffer_size, obs_shape[0], obs_shape[1], 1), dtype=np.float32)
        self._buffer_done = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_size = buffer_size
        self._buffer_pos = 0
        self._buffer_len = 0

    def _sync_target(self) -> None:
        self.target_model.set_weights(self.model.get_weights())

    def _obs_batch(self, obs: np.ndarray) -> np.ndarray:
        """Ensure obs is (1, H, W, 1) float32."""
        if obs.ndim == 2:
            obs = np.expand_dims(np.expand_dims(obs, 0), -1)
        elif obs.ndim == 3:
            obs = np.expand_dims(obs, -1)
        return obs.astype(np.float32)

    def select_action(self, obs: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        obs_batch = self._obs_batch(obs)
        if training and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        q = self.model(obs_batch, training=False)
        return int(np.argmax(q[0].numpy()))

    def store(self, s: np.ndarray, a: int, r: float, s2: np.ndarray, done: bool) -> None:
        """Add transition to replay buffer (circular)."""
        pos = self._buffer_pos % self._buffer_size
        self._buffer_s[pos] = self._obs_batch(s)[0]
        self._buffer_a[pos] = a
        self._buffer_r[pos] = r
        self._buffer_s2[pos] = self._obs_batch(s2)[0]
        self._buffer_done[pos] = 1.0 if done else 0.0
        self._buffer_pos += 1
        self._buffer_len = min(self._buffer_len + 1, self._buffer_size)

    def train_step(self) -> Optional[float]:
        """One gradient update. Returns loss or None if buffer not yet full enough."""
        if self._buffer_len < self._batch_size:
            return None
        idx = np.random.choice(self._buffer_len, self._batch_size, replace=False)
        s = self._buffer_s[idx]
        a = self._buffer_a[idx]
        r = self._buffer_r[idx]
        s2 = self._buffer_s2[idx]
        done = self._buffer_done[idx]
        next_q = self.target_model(s2, training=False)
        next_q_max = np.max(next_q.numpy(), axis=1)
        target = r + self.gamma * next_q_max * (1 - done)
        target_full = self.model(s, training=False).numpy()
        target_full[np.arange(self._batch_size), a] = target
        loss = self.model.train_on_batch(s, target_full)
        self._step_count += 1
        if self._step_count % self._target_update_freq == 0:
            self._sync_target()
        self.epsilon = max(self._epsilon_end, self.epsilon - self._epsilon_decay)
        return float(loss)

    def save(self, path: str) -> None:
        """Save model weights (.keras format preferred)."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load weights from .keras or .h5 — builds fresh architecture to avoid KerasTensor warnings."""
        self.model = build_dqn_model(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        if path.endswith(".keras"):
            self.model.load_weights(path)
        else:
            self.model.load_weights(path, by_name=False, skip_mismatch=False)
        self._sync_target()
