from __future__ import annotations

import numpy as np


def create_border_mask(image: np.ndarray, margin: int = 1) -> np.ndarray:
    """
    Mask borders so touching objects can be excluded.
    """
    h, w = image.shape
    mask = np.zeros_like(image, dtype=np.uint8)

    mask[:margin, :] = 1
    mask[-margin:, :] = 1
    mask[:, :margin] = 1
    mask[:, -margin:] = 1

    return mask


def create_scale_bar_mask(image: np.ndarray) -> np.ndarray:
    """
    Very simple v1:
    mask bottom-right region where scale bar is located.
    """
    h, w = image.shape

    mask = np.zeros_like(image, dtype=np.uint8)

    h0 = int(h * 0.96)
    w0 = int(w * 0.91)

    mask[h0:h, w0:w] = 1

    return mask


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out = np.maximum(out, m)
    return out