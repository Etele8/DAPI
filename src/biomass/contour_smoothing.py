from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.biomass.contours import ensure_counter_clockwise, polygon_area


@dataclass(slots=True)
class SmoothedContour:
    points_xy: np.ndarray
    failed: bool
    reason: str | None = None


def _resample_closed_polygon(points_xy: np.ndarray, n_points: int) -> np.ndarray:
    closed = np.vstack([points_xy, points_xy[0]])
    segment_vectors = np.diff(closed, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    perimeter = float(segment_lengths.sum())
    if perimeter <= 0:
        raise ValueError("contour perimeter is zero")

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    targets = np.linspace(0.0, perimeter, n_points + 1)[:-1]
    samples = np.zeros((n_points, 2), dtype=np.float64)

    edge_idx = 0
    for i, target in enumerate(targets):
        while edge_idx < len(segment_lengths) - 1 and cumulative[edge_idx + 1] < target:
            edge_idx += 1
        edge_length = segment_lengths[edge_idx]
        start = closed[edge_idx]
        if edge_length <= 0:
            samples[i] = start
            continue
        alpha = (target - cumulative[edge_idx]) / edge_length
        samples[i] = start + alpha * segment_vectors[edge_idx]

    return samples


def _smooth_wraparound(points_xy: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return points_xy.copy()
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")

    radius = window // 2
    padded = np.vstack([points_xy[-radius:], points_xy, points_xy[:radius]])
    smoothed = np.zeros_like(points_xy)
    for i in range(len(points_xy)):
        smoothed[i] = padded[i : i + window].mean(axis=0)
    return smoothed


def smooth_contour(points_xy: np.ndarray, n_points: int = 64, smoothing_window: int = 5) -> SmoothedContour:
    if len(points_xy) < 5:
        return SmoothedContour(points_xy=points_xy.copy(), failed=True, reason="too_few_raw_points")

    unique_points = np.unique(np.round(points_xy, 6), axis=0)
    if len(unique_points) < 5:
        return SmoothedContour(points_xy=points_xy.copy(), failed=True, reason="too_few_unique_points")

    try:
        resampled = _resample_closed_polygon(points_xy, n_points=max(n_points, 16))
        smoothed = _smooth_wraparound(resampled, window=smoothing_window)
        smoothed = ensure_counter_clockwise(smoothed)
        if abs(polygon_area(smoothed)) <= 1e-6:
            return SmoothedContour(points_xy=smoothed, failed=True, reason="degenerate_smoothed_polygon")
        return SmoothedContour(points_xy=smoothed, failed=False, reason=None)
    except ValueError as exc:
        return SmoothedContour(points_xy=points_xy.copy(), failed=True, reason=str(exc))
