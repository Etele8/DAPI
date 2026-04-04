from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(slots=True)
class SliceElement:
    start_s: float
    end_s: float
    width_start: float
    width_end: float
    volume_px3: float
    triangle_count: int


@dataclass(slots=True)
class ZederVolumeResult:
    volume_px3: float
    axis_start_xy: tuple[float, float]
    axis_end_xy: tuple[float, float]
    slice_elements: list[SliceElement]
    approximation_notes: list[str]


def _pairwise_longest_chord(points_xy: np.ndarray) -> tuple[float, tuple[int, int]]:
    deltas = points_xy[:, None, :] - points_xy[None, :, :]
    squared = np.sum(deltas * deltas, axis=2)
    i, j = np.unravel_index(np.argmax(squared), squared.shape)
    return float(np.sqrt(squared[i, j])), (int(i), int(j))


def longest_chord(points_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    length, (i, j) = _pairwise_longest_chord(points_xy)
    return length, points_xy[i], points_xy[j]


def _half_plane_clip(points_xy: np.ndarray, normal_xy: np.ndarray, threshold: float, keep_greater: bool) -> np.ndarray:
    if len(points_xy) == 0:
        return points_xy
    clipped: list[np.ndarray] = []
    eps = 1e-9

    def inside(point: np.ndarray) -> bool:
        value = float(np.dot(point, normal_xy))
        return value >= threshold - eps if keep_greater else value <= threshold + eps

    prev = points_xy[-1]
    prev_inside = inside(prev)
    for current in points_xy:
        curr_inside = inside(current)
        if curr_inside != prev_inside:
            direction = current - prev
            denom = float(np.dot(direction, normal_xy))
            if abs(denom) > eps:
                alpha = (threshold - float(np.dot(prev, normal_xy))) / denom
                clipped.append(prev + alpha * direction)
        if curr_inside:
            clipped.append(current.copy())
        prev = current
        prev_inside = curr_inside

    if not clipped:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def _clip_polygon_to_slab(points_xy: np.ndarray, axis_unit: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    clipped = _half_plane_clip(points_xy, axis_unit, start_s, keep_greater=True)
    clipped = _half_plane_clip(clipped, axis_unit, end_s, keep_greater=False)
    return clipped


def _line_polygon_intersections(points_xy: np.ndarray, axis_unit: np.ndarray, normal_unit: np.ndarray, target_s: float) -> np.ndarray:
    intersections: list[np.ndarray] = []
    eps = 1e-7
    for i in range(len(points_xy)):
        p0 = points_xy[i]
        p1 = points_xy[(i + 1) % len(points_xy)]
        s0 = float(np.dot(p0, axis_unit))
        s1 = float(np.dot(p1, axis_unit))

        if abs(s0 - target_s) <= eps:
            intersections.append(p0)
        if abs(s1 - target_s) <= eps:
            intersections.append(p1)
        if (s0 - target_s) * (s1 - target_s) < 0:
            alpha = (target_s - s0) / (s1 - s0)
            intersections.append(p0 + alpha * (p1 - p0))

    if not intersections:
        return np.zeros((0,), dtype=np.float64)

    points = np.asarray(intersections, dtype=np.float64)
    unique = np.unique(np.round(points, 6), axis=0)
    return unique @ normal_unit


def _cross_section_width(points_xy: np.ndarray, axis_unit: np.ndarray, normal_unit: np.ndarray, target_s: float) -> float:
    coords = _line_polygon_intersections(points_xy, axis_unit, normal_unit, target_s)
    if len(coords) < 2:
        return 0.0
    return float(coords.max() - coords.min())


def _triangle_count_for_strip(strip_polygon_xy: np.ndarray) -> int:
    return max(0, len(strip_polygon_xy) - 2)


def estimate_zeder_volume(
    points_xy: np.ndarray,
    *,
    min_slice_length_px: float = 0.75,
    max_depth: int = 8,
    width_linearity_tol_px: float = 0.35,
) -> ZederVolumeResult:
    chord_length, axis_start, axis_end = longest_chord(points_xy)
    if chord_length <= 0:
        raise ValueError("longest chord is zero")

    axis_unit = (axis_end - axis_start) / chord_length
    normal_unit = np.array([-axis_unit[1], axis_unit[0]], dtype=np.float64)
    projections = points_xy @ axis_unit
    s_min = float(projections.min())
    s_max = float(projections.max())
    elements: list[SliceElement] = []

    def recurse(start_s: float, end_s: float, depth: int) -> None:
        strip = _clip_polygon_to_slab(points_xy, axis_unit, start_s, end_s)
        if len(strip) < 3:
            return

        slice_length = end_s - start_s
        if slice_length <= 0:
            return

        w0 = _cross_section_width(points_xy, axis_unit, normal_unit, start_s)
        w1 = _cross_section_width(points_xy, axis_unit, normal_unit, end_s)
        mid_s = 0.5 * (start_s + end_s)
        w_mid = _cross_section_width(points_xy, axis_unit, normal_unit, mid_s)
        linear_mid = 0.5 * (w0 + w1)

        should_split = (
            depth < max_depth
            and slice_length > min_slice_length_px
            and abs(w_mid - linear_mid) > width_linearity_tol_px
        )
        if should_split:
            recurse(start_s, mid_s, depth + 1)
            recurse(mid_s, end_s, depth + 1)
            return

        radius0 = max(0.0, 0.5 * w0)
        radius1 = max(0.0, 0.5 * w1)
        volume = math.pi * slice_length * (radius0 * radius0 + radius0 * radius1 + radius1 * radius1) / 3.0
        elements.append(
            SliceElement(
                start_s=start_s,
                end_s=end_s,
                width_start=w0,
                width_end=w1,
                volume_px3=float(volume),
                triangle_count=_triangle_count_for_strip(strip),
            )
        )

    recurse(s_min, s_max, depth=0)
    if not elements:
        raise ValueError("no slice elements generated")

    notes = [
        "Contour is smoothed and represented as a closed polygon.",
        "The practical Zeder-style implementation uses the longest chord as a global cell axis.",
        "The contour polygon is recursively subdivided into perpendicular slab polygons until cross-section widths are locally linear.",
        "Each slab polygon is triangle-counted for bookkeeping, and its biovolume is estimated as a rotational frustum from the slab boundary widths.",
    ]
    return ZederVolumeResult(
        volume_px3=float(sum(element.volume_px3 for element in elements)),
        axis_start_xy=(float(axis_start[0]), float(axis_start[1])),
        axis_end_xy=(float(axis_end[0]), float(axis_end[1])),
        slice_elements=elements,
        approximation_notes=notes,
    )
