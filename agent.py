"""
DQN agent for vision-based control (extend/curl/left/right): TensorFlow for training,
OpenVINO for deployment on Iris Xe (GPU). Includes utility to convert .keras/.h5 to IR.
"""

import os
import warnings

# Reduce TensorFlow C++ log noise (e.g. "rebuild with SSE4/AVX" and oneDNN messages)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from typing import Optional, Tuple

from utils import PREPROCESS_WIDTH, PREPROCESS_HEIGHT

# --- TensorFlow DQN (training) ---
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# --- OpenVINO (inference) ---
try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False


# Observation: single 84x84 grayscale -> (1, 84, 84, 1) for TF/OpenVINO
OBS_H = PREPROCESS_HEIGHT
OBS_W = PREPROCESS_WIDTH


def _get_custom_objects_for_h5_load():
    """
    Custom objects when loading .h5 saved with older Keras (handles 'reduction' in MeanSquaredError config).
    """
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
    learning_rate: float = 2.5e-4,
) -> "tf.keras.Model":
    """Build a small CNN for DQN (single 84x84 grayscale input)."""
    if not HAS_TF:
        raise RuntimeError("TensorFlow is required for training. Install with: pip install tensorflow")
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
    DQN agent: train with TensorFlow, optional save to .h5.
    Uses epsilon-greedy and a simple replay buffer.
    """

    def __init__(
        self,
        n_actions: int,
        obs_shape: Tuple[int, int] = (OBS_H, OBS_W),
        learning_rate: float = 2.5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 50000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
    ):
        if not HAS_TF:
            raise RuntimeError("TensorFlow is required. Install with: pip install tensorflow")
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.epsilon = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = (epsilon_start - epsilon_end) / max(1, epsilon_decay_steps)
        self._epsilon_decay_steps = epsilon_decay_steps
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._step_count = 0

        self.model = build_dqn_model(n_actions, obs_shape[0], obs_shape[1], learning_rate)
        self.target_model = build_dqn_model(n_actions, obs_shape[0], obs_shape[1], learning_rate)
        self._sync_target()

        # Simple replay buffer: (s, a, r, s', done)
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
        """Ensure obs is (1, H, W, 1)."""
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
        """Add transition to replay buffer."""
        pos = self._buffer_pos % self._buffer_size
        self._buffer_s[pos] = self._obs_batch(s)[0]
        self._buffer_a[pos] = a
        self._buffer_r[pos] = r
        self._buffer_s2[pos] = self._obs_batch(s2)[0]
        self._buffer_done[pos] = 1.0 if done else 0.0
        self._buffer_pos += 1
        self._buffer_len = min(self._buffer_len + 1, self._buffer_size)

    def train_step(self) -> Optional[float]:
        """One gradient step; returns loss or None if not enough samples."""
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
        """Save model (.keras or .h5; .keras avoids legacy HDF5 warning)."""
        self.model.save(path)

    def load(self, path: str) -> None:
        """Load model from .keras or .h5: build fresh architecture and load weights only (avoids KerasTensor warning)."""
        self.model = build_dqn_model(self.n_actions, self.obs_shape[0], self.obs_shape[1])
        if path.endswith(".keras"):
            self.model.load_weights(path)
        else:
            self.model.load_weights(path, by_name=False, skip_mismatch=False)
        self._sync_target()


def convert_h5_to_openvino_ir(
    h5_path: str,
    output_dir: str,
    output_name: str = "dqn_ir",
    n_actions: int = 5,
) -> Tuple[str, str]:
    """
    Convert a TensorFlow .keras or .h5 model to OpenVINO IR (.xml + .bin).
    Builds architecture, loads weights only, then passes the Keras model directly to OpenVINO
    (no SavedModel step) to avoid the _DictWrapper / tf.saved_model.save() bug.
    If you still see _DictWrapper: try `pip install wrapt==1.14.1` and rerun.
    """
    if not HAS_TF:
        raise RuntimeError("TensorFlow required for conversion. pip install tensorflow")
    if not HAS_OPENVINO:
        raise RuntimeError("OpenVINO required. pip install openvino")
    os.makedirs(output_dir, exist_ok=True)
    # Weights-only load; then convert Keras model object directly (no SavedModel)
    model = build_dqn_model(n_actions)
    if h5_path.endswith(".keras"):
        model.load_weights(h5_path)
    else:
        model.load_weights(h5_path, by_name=False, skip_mismatch=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Keras "Expected: keras_tensor" when OpenVINO traces
        ov_model = ov.convert_model(model)
    xml_path = os.path.join(output_dir, f"{output_name}.xml")
    ov.save_model(ov_model, xml_path)
    bin_path = xml_path.replace(".xml", ".bin")
    if not os.path.isfile(bin_path):
        bin_path = os.path.join(output_dir, f"{output_name}.bin")
    return (xml_path, bin_path)


class OpenVINOInference:
    """Run DQN inference using OpenVINO on GPU (Iris Xe)."""

    def __init__(self, ir_xml_path: str, device: str = "GPU"):
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO required. pip install openvino")
        self._core = ov.Core()
        self._model = self._core.read_model(ir_xml_path)
        self._compiled = self._core.compile_model(self._model, device)
        self._input_name = self._compiled.input(0)
        self._output_name = self._compiled.output(0)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Forward pass: obs (H,W) or (1,H,W,1) -> Q values (n_actions,)."""
        if obs.ndim == 2:
            obs = np.expand_dims(np.expand_dims(obs.astype(np.float32), 0), -1)
        elif obs.ndim == 3:
            obs = np.expand_dims(obs.astype(np.float32), -1)
        result = self._compiled([obs])
        return np.array(result[self._output_name]).flatten()

    def select_action(self, obs: np.ndarray) -> int:
        """Greedy action from Q values."""
        q = self(obs)
        return int(np.argmax(q))
