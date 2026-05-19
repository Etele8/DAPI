"""Shared image-array helpers for the segmentation pipeline."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


def validate_odd_kernel_size(value: int, *, minimum: int = 1, name: str) -> int:
    if value < minimum or value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}")
    return value


def ensure_grayscale_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != shape:
        raise ValueError("artifact_mask must have the same height/width as the image")
    return (mask > 0).astype(np.uint8)


def normalize_float_map(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_v = float(image.min())
    max_v = float(image.max())
    if max_v <= min_v:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_v) / (max_v - min_v)


def to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(normalize_float_map(image) * 255.0), 0, 255).astype(np.uint8)


def ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected grayscale or BGR image, got shape {image.shape}")
    if image.dtype == np.uint8:
        return image.copy()
    return to_u8(image)


def select_source_channel(image_bgr: np.ndarray, source: Literal["value", "blue", "max_blue_value"]) -> np.ndarray:
    blue = image_bgr[:, :, 0].astype(np.float32)
    green = image_bgr[:, :, 1].astype(np.float32)
    red = image_bgr[:, :, 2].astype(np.float32)
    value = np.maximum.reduce([blue, green, red])

    if source == "blue":
        return blue
    if source == "value":
        return value
    if source == "max_blue_value":
        return np.maximum(blue, value)
    raise ValueError(f"Unsupported source: {source}")
