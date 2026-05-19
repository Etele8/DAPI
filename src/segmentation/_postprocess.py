"""Morphological cleanup, optional watershed split, and per-region measurement."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import MorphologyConfig, RegionFilterConfig, SplitConfig
from src.segmentation._utils import ensure_bgr_u8, validate_odd_kernel_size


@dataclass(slots=True)
class RegionDetection:
    label_id: int
    area_px: int
    bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    width_px: int
    height_px: int
    aspect_ratio: float
    solidity: float
    convexity: float
    circularity: float
    eccentricity: float
    touches_border: bool
    touches_artifact_mask: bool
    accepted: bool
    rejection_reasons: list[str]


def _fill_small_holes(binary_mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    if max_hole_area_px <= 0:
        return binary_mask

    inverse = cv2.bitwise_not(binary_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    filled = binary_mask.copy()
    height, width = binary_mask.shape

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]
        touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
        if not touches_border and area <= max_hole_area_px:
            filled[labels == label_id] = 255

    return filled


def morphological_cleanup(binary_mask: np.ndarray, config: MorphologyConfig) -> np.ndarray:
    cleaned = binary_mask.copy()

    if config.opening_kernel_size > 1:
        open_kernel_size = validate_odd_kernel_size(
            config.opening_kernel_size,
            minimum=3,
            name="opening_kernel_size",
        )
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    if config.closing_kernel_size > 1:
        close_kernel_size = validate_odd_kernel_size(
            config.closing_kernel_size,
            minimum=3,
            name="closing_kernel_size",
        )
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    if config.fill_holes:
        cleaned = _fill_small_holes(cleaned, config.min_hole_area_px)

    return cleaned


def split_touching_cells(binary_mask: np.ndarray, image_bgr: np.ndarray, config: SplitConfig) -> np.ndarray:
    if not config.enabled:
        return binary_mask

    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_distance = float(distance.max())
    if max_distance <= 0:
        return binary_mask

    threshold_value = max(max_distance * config.distance_threshold_ratio, float(config.min_distance_px))
    _, sure_foreground = cv2.threshold(distance, threshold_value, 255, 0)
    sure_foreground = sure_foreground.astype(np.uint8)

    num_markers, markers = cv2.connectedComponents(sure_foreground)
    if num_markers <= 1:
        return binary_mask

    sure_background = cv2.dilate(binary_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    unknown = cv2.subtract(sure_background, sure_foreground)
    markers = markers + 1
    markers[unknown > 0] = 0

    watershed_input = ensure_bgr_u8(image_bgr)
    markers = cv2.watershed(watershed_input.copy(), markers)
    return np.where(markers > 1, 255, 0).astype(np.uint8)


def _compute_eccentricity(contour: np.ndarray) -> float:
    if len(contour) < 5:
        return 0.0
    (_, _), (major_axis, minor_axis), _ = cv2.fitEllipse(contour)
    major = max(float(major_axis), float(minor_axis))
    minor = min(float(major_axis), float(minor_axis))
    if major <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, 1.0 - (minor * minor) / (major * major))))


def connected_components_and_regions(
    binary_mask: np.ndarray,
    artifact_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[tuple[int, np.ndarray, np.ndarray]]]:
    del artifact_mask
    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    components: list[tuple[int, np.ndarray, np.ndarray]] = []

    for label_id in range(1, num_labels):
        component_mask = np.where(labels == label_id, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        components.append((label_id, component_mask, contour))

    return labels, components


def measure_region(
    label_id: int,
    component_mask: np.ndarray,
    contour: np.ndarray,
    image_shape: tuple[int, int],
    *,
    artifact_mask: np.ndarray | None = None,
    config: RegionFilterConfig | None = None,
) -> RegionDetection:
    height, width = image_shape
    artifact_mask_bool = artifact_mask.astype(bool) if artifact_mask is not None else None

    x, y, w, h = cv2.boundingRect(contour)
    area_px = int(cv2.countNonZero(component_mask))
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        centroid_xy = (x + w / 2.0, y + h / 2.0)
    else:
        centroid_xy = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    hull = cv2.convexHull(contour)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(hull))
    perimeter = float(cv2.arcLength(contour, True))
    hull_perimeter = float(cv2.arcLength(hull, True))

    aspect_ratio = float(max(w, h) / max(min(w, h), 1))
    solidity = float(contour_area / hull_area) if hull_area > 0 else 0.0
    convexity = float(hull_perimeter / perimeter) if perimeter > 0 else 0.0
    circularity = float((4.0 * np.pi * contour_area) / (perimeter * perimeter)) if perimeter > 0 else 0.0
    eccentricity = _compute_eccentricity(contour)
    touches_border = x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
    touches_artifact_mask = bool(
        artifact_mask_bool is not None and np.any(component_mask.astype(bool) & artifact_mask_bool)
    )

    rejection_reasons: list[str] = []
    if config is not None:
        if area_px < config.min_area_px:
            rejection_reasons.append("area_too_small")
        if config.max_area_px is not None and area_px > config.max_area_px:
            rejection_reasons.append("area_too_large")
        if w < config.min_width_px:
            rejection_reasons.append("width_too_small")
        if h < config.min_height_px:
            rejection_reasons.append("height_too_small")
        if aspect_ratio > config.max_aspect_ratio:
            rejection_reasons.append("aspect_ratio_too_high")
        if solidity < config.min_solidity:
            rejection_reasons.append("solidity_too_low")
        if convexity < config.min_convexity:
            rejection_reasons.append("convexity_too_low")
        if circularity < config.min_circularity:
            rejection_reasons.append("circularity_too_low")
        if eccentricity > config.max_eccentricity:
            rejection_reasons.append("eccentricity_too_high")
        if config.exclude_border_touching and touches_border:
            rejection_reasons.append("touches_border")
    if touches_artifact_mask:
        rejection_reasons.append("touches_artifact_mask")

    return RegionDetection(
        label_id=label_id,
        area_px=area_px,
        bbox=(x, y, w, h),
        centroid_xy=centroid_xy,
        width_px=w,
        height_px=h,
        aspect_ratio=aspect_ratio,
        solidity=solidity,
        convexity=convexity,
        circularity=circularity,
        eccentricity=eccentricity,
        touches_border=touches_border,
        touches_artifact_mask=touches_artifact_mask,
        accepted=not rejection_reasons,
        rejection_reasons=rejection_reasons,
    )


def filter_regions(
    labels: np.ndarray,
    components: list[tuple[int, np.ndarray, np.ndarray]],
    config: RegionFilterConfig,
    artifact_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[RegionDetection]]:
    filtered_mask = np.zeros(labels.shape, dtype=np.uint8)
    regions: list[RegionDetection] = []

    for label_id, component_mask, contour in components:
        region = measure_region(
            label_id,
            component_mask,
            contour,
            labels.shape,
            artifact_mask=artifact_mask,
            config=config,
        )
        if region.accepted:
            filtered_mask[labels == label_id] = 255

        regions.append(region)

    _, relabeled, _, _ = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)
    return relabeled.astype(np.int32), regions
