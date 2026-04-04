from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class QCResult:
    flags: list[str]
    keep: bool


def _segments_intersect(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> bool:
    def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return (
            min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9
            and min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9
        )

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) <= 1e-9 and on_segment(p1, p2, q1):
        return True
    if abs(o2) <= 1e-9 and on_segment(p1, p2, q2):
        return True
    if abs(o3) <= 1e-9 and on_segment(q1, q2, p1):
        return True
    if abs(o4) <= 1e-9 and on_segment(q1, q2, p2):
        return True
    return False


def polygon_self_intersects(points_xy: np.ndarray) -> bool:
    n_points = len(points_xy)
    if n_points < 4:
        return False

    for i in range(n_points):
        p1 = points_xy[i]
        p2 = points_xy[(i + 1) % n_points]
        for j in range(i + 1, n_points):
            if abs(i - j) <= 1:
                continue
            if i == 0 and j == n_points - 1:
                continue
            q1 = points_xy[j]
            q2 = points_xy[(j + 1) % n_points]
            if _segments_intersect(p1, p2, q1, q2):
                return True
    return False


def suspicious_merge(mask: np.ndarray, contour_xy: np.ndarray) -> bool:
    contour = contour_xy.reshape(-1, 1, 2).astype(np.float32)
    hull = cv2.convexHull(contour)
    area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull))
    x, y, w, h = cv2.boundingRect(contour.astype(np.int32))
    aspect_ratio = float(max(w, h) / max(min(w, h), 1))
    solidity = area / hull_area if hull_area > 0 else 0.0
    return solidity < 0.82 and aspect_ratio > 2.8 and int(np.count_nonzero(mask)) > 40


def build_qc_flags(
    *,
    touches_border: bool,
    area_px2_raw: float,
    smoothing_failed: bool,
    smoothing_reason: str | None,
    self_intersects: bool,
    merged_object: bool,
    volume_failed: bool,
    calibration_available: bool,
    min_area_px2: float,
) -> QCResult:
    flags: list[str] = []
    if touches_border:
        flags.append("touches_border")
    if area_px2_raw < min_area_px2:
        flags.append("tiny_contour")
    if smoothing_failed:
        flags.append("failed_contour_smoothing")
        if smoothing_reason:
            flags.append(f"smoothing:{smoothing_reason}")
    if self_intersects:
        flags.append("self_intersecting_contour")
    if merged_object:
        flags.append("suspicious_merged_object")
    if volume_failed:
        flags.append("failed_volume_computation")
    if not calibration_available:
        flags.append("missing_calibration")

    hard_exclusions = {
        "touches_border",
        "tiny_contour",
        "failed_contour_smoothing",
        "self_intersecting_contour",
        "failed_volume_computation",
    }
    keep = not any(flag in hard_exclusions for flag in flags)
    return QCResult(flags=flags, keep=keep)
