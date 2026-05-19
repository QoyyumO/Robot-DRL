"""
Image preprocessing pipeline for vision-based robotic arm control.
GPU/headless version: Tkinter display helpers removed.
"""
from __future__ import annotations

import cv2
import numpy as np

from config_loader import cfg

_prep = cfg["preprocessing"]

# Preprocessing dimensions loaded from config.yaml
PREPROCESS_WIDTH: int = _prep["width"]
PREPROCESS_HEIGHT: int = _prep["height"]
_BILATERAL_D: int = _prep["bilateral_d"]
_BILATERAL_SIGMA_COLOR: float = _prep["bilateral_sigma_color"]
_BILATERAL_SIGMA_SPACE: float = _prep["bilateral_sigma_space"]


def preprocess_frame(
    frame: np.ndarray,
    width: int = PREPROCESS_WIDTH,
    height: int = PREPROCESS_HEIGHT,
    bilateral_d: int = _BILATERAL_D,
    bilateral_sigma_color: float = _BILATERAL_SIGMA_COLOR,
    bilateral_sigma_space: float = _BILATERAL_SIGMA_SPACE,
) -> np.ndarray:
    """
    Preprocess a raw RGB frame for the DQN pipeline.

    Steps:
    1. Resize to width x height (default 84x84).
    2. Convert to grayscale.
    3. Apply Bilateral Filter for denoising while preserving edges.
    4. Normalize pixel values to [0, 1].

    Returns:
        Preprocessed frame of shape (height, width), float32 in [0, 1].
    """
    if frame is None or frame.size == 0:
        raise ValueError("preprocess_frame: frame must be a non-empty numpy array")

    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    filtered = cv2.bilateralFilter(
        gray,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    normalized = filtered.astype(np.float32) / 255.0
    np.clip(normalized, 0.0, 1.0, out=normalized)
    return normalized


def preprocess_frame_uint8(
    frame: np.ndarray,
    width: int = PREPROCESS_WIDTH,
    height: int = PREPROCESS_HEIGHT,
    **kwargs
) -> np.ndarray:
    """Same pipeline as preprocess_frame but returns uint8 [0, 255]."""
    normalized = preprocess_frame(frame, width=width, height=height, **kwargs)
    return (normalized * 255.0).astype(np.uint8)
