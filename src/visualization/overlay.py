from __future__ import annotations

import cv2
import numpy as np

from src.segmentation.segment import RegionDetection


def draw_filter_overlay(
    image_gray_u8: np.ndarray,
    labels: np.ndarray,
    objects: list[RegionDetection],
) -> np.ndarray:
    """
    Render accepted objects in green and rejected objects in red.
    """
    if image_gray_u8.ndim == 2:
        base = cv2.cvtColor(image_gray_u8, cv2.COLOR_GRAY2BGR)
    else:
        base = image_gray_u8.copy()
    overlay = base.copy()

    for obj in objects:
        mask = labels == obj.label_id
        if not np.any(mask):
            continue

        color = (0, 255, 0) if obj.accepted else (0, 0, 255)
        overlay[mask] = color

    return cv2.addWeighted(base, 0.55, overlay, 0.45, 0.0)
