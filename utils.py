"""
Image preprocessing pipeline for vision-based robotic arm control (up/down/left/right).
Uses OpenCV for resize, grayscale, bilateral filtering, and normalization.
"""
from __future__ import annotations

import cv2
import numpy as np
from typing import Any


# Default preprocessing dimensions (DQN-style observation space)
PREPROCESS_WIDTH = 84
PREPROCESS_HEIGHT = 84


def preprocess_frame(
    frame: np.ndarray,
    width: int = PREPROCESS_WIDTH,
    height: int = PREPROCESS_HEIGHT,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
) -> np.ndarray:
    """
    Preprocess a raw RGB frame for the DQN / OpenVINO pipeline.

    Steps:
    1. Resize to width x height (default 84x84).
    2. Convert to grayscale to focus on geometric patterns.
    3. Apply Bilateral Filter for denoising while preserving edges.
    4. Normalize pixel values to the [0, 1] range.

    Args:
        frame: Raw image as numpy array (H, W, 3) BGR or RGB, uint8.
        width: Target width (default 84).
        height: Target height (default 84).
        bilateral_d: Diameter of pixel neighborhood for bilateral filter.
        bilateral_sigma_color: Filter sigma in the color space.
        bilateral_sigma_space: Filter sigma in the coordinate space.

    Returns:
        Preprocessed frame of shape (height, width), float32 in [0, 1].
    """
    if frame is None or frame.size == 0:
        raise ValueError("preprocess_frame: frame must be a non-empty numpy array")

    # 1. Resize to target dimensions
    resized = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    # 2. Convert to grayscale (handles both BGR and RGB by using standard conversion)
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    # 3. Bilateral filter: denoise while preserving edges
    filtered = cv2.bilateralFilter(
        gray,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    # 4. Normalize to [0, 1] as float32 (expected by many ML frameworks)
    normalized = filtered.astype(np.float32) / 255.0
    np.clip(normalized, 0.0, 1.0, out=normalized)

    return normalized


def preprocess_frame_uint8(
    frame: np.ndarray,
    width: int = PREPROCESS_WIDTH,
    height: int = PREPROCESS_HEIGHT,
    **kwargs,
) -> np.ndarray:
    """
    Same pipeline as preprocess_frame but returns uint8 [0, 255] for display
    (e.g., in the Visual Feedback panel).
    """
    normalized = preprocess_frame(frame, width=width, height=height, **kwargs)
    return (normalized * 255.0).astype(np.uint8)


def numpy_to_tk_photo(
    arr: np.ndarray,
    is_grayscale: bool = True,
) -> Any:
    """
    Convert a numpy image (H, W) or (H, W, 3) to Tkinter PhotoImage (PIL.ImageTk.PhotoImage).
    Used to display preprocessed frames in the GUI.
    """
    from PIL import Image
    from PIL import ImageTk

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    if is_grayscale and len(arr.shape) == 2:
        pil_img = Image.fromarray(arr, mode="L")
    else:
        if len(arr.shape) == 2:
            pil_img = Image.fromarray(arr, mode="L")
        else:
            # BGR from OpenCV -> RGB for PIL
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(arr_rgb, mode="RGB")
    return ImageTk.PhotoImage(pil_img)
