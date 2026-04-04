from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class ConnectedObject:
    object_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    touches_border: bool
    raw_contour: np.ndarray
    area_px2_raw: float


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def contour_to_xy(contour: np.ndarray) -> np.ndarray:
    xy = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(xy) >= 2 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    return xy


def polygon_area(points_xy: np.ndarray) -> float:
    if len(points_xy) < 3:
        return 0.0
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_counter_clockwise(points_xy: np.ndarray) -> np.ndarray:
    if polygon_area(points_xy) < 0:
        return points_xy[::-1].copy()
    return points_xy.copy()


def extract_connected_objects(binary_mask: np.ndarray) -> list[ConnectedObject]:
    binary_mask = ensure_binary_mask(binary_mask)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    height, width = binary_mask.shape
    objects: list[ConnectedObject] = []

    for label_id in range(1, num_labels):
        component_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        x, y, w, h, area_px = stats[label_id]
        centroid = (float(centroids[label_id][0]), float(centroids[label_id][1]))
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height

        objects.append(
            ConnectedObject(
                object_id=len(objects) + 1,
                mask=component_mask,
                bbox=(int(x), int(y), int(w), int(h)),
                centroid_xy=centroid,
                touches_border=touches_border,
                raw_contour=ensure_counter_clockwise(contour_to_xy(contour)),
                area_px2_raw=float(area_px),
            )
        )

    return objects
