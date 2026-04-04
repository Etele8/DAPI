from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


ThresholdMethod = Literal["otsu", "otsu_positive", "percentile", "percentile_positive", "adaptive", "hysteresis"]


@dataclass(slots=True)
class PreprocessingConfig:
    use_median_blur: bool = True
    median_kernel_size: int = 3


@dataclass(slots=True)
class BlueDominanceConfig:
    blue_green_weight: float = 0.6
    blue_red_weight: float = 0.5
    blue_channel_weight: float = 0.25
    local_suppression_kernel_size: int = 35
    local_suppression_strength: float = 0.95


@dataclass(slots=True)
class BlobEnhancementConfig:
    source: Literal["value", "blue", "max_blue_value"] = "max_blue_value"
    top_hat_kernel_size: int = 11
    dog_sigma_small: float = 0.7
    dog_sigma_large: float = 3.0
    top_hat_weight: float = 0.6
    dog_weight: float = 0.8


@dataclass(slots=True)
class LocalSuppressionConfig:
    blur_kernel_size: int = 31
    source: Literal["value", "blue", "max_blue_value"] = "max_blue_value"


@dataclass(slots=True)
class EvidenceFusionConfig:
    blue_weight: float = 0.50
    blob_weight: float = 0.70
    suppression_weight: float = 0.30
    gamma: float = 0.5


@dataclass(slots=True)
class ThresholdConfig:
    method: ThresholdMethod = "hysteresis"
    otsu_scale: float = 1.0
    percentile: float = 95.0
    adaptive_block_size: int = 31
    adaptive_c: float = -4.0
    positive_floor_percentile: float = 85.0
    positive_floor_value: int = 8
    hysteresis_low_scale: float = 0.87
    hysteresis_high_scale: float = 1.05
    pre_blur_kernel_size: int = 1


@dataclass(slots=True)
class MorphologyConfig:
    opening_kernel_size: int = 1
    closing_kernel_size: int = 0
    fill_holes: bool = True
    min_hole_area_px: int = 12


@dataclass(slots=True)
class RegionFilterConfig:
    min_area_px: int = 8
    max_area_px: int | None = 2000
    min_width_px: int = 2
    min_height_px: int = 2
    max_aspect_ratio: float = 3.5
    min_solidity: float = 0.57
    min_convexity: float = 0.85
    min_circularity: float = 0.32
    max_eccentricity: float = 0.95
    exclude_border_touching: bool = False


@dataclass(slots=True)
class SplitConfig:
    enabled: bool = False
    distance_threshold_ratio: float = 0.45
    min_distance_px: int = 2


@dataclass(slots=True)
class SegmentationConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    blue_dominance: BlueDominanceConfig = field(default_factory=BlueDominanceConfig)
    blob_enhancement: BlobEnhancementConfig = field(default_factory=BlobEnhancementConfig)
    local_suppression: LocalSuppressionConfig = field(default_factory=LocalSuppressionConfig)
    fusion: EvidenceFusionConfig = field(default_factory=EvidenceFusionConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    region_filter: RegionFilterConfig = field(default_factory=RegionFilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)


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


@dataclass(slots=True)
class SegmentationResult:
    preprocessed_bgr: np.ndarray
    blue_dominance_map: np.ndarray
    blob_enhanced_map: np.ndarray
    local_suppression_map: np.ndarray
    evidence_map: np.ndarray
    binary_mask_raw: np.ndarray
    binary_mask_clean: np.ndarray
    filtered_mask: np.ndarray
    candidate_labels: np.ndarray
    labels: np.ndarray
    regions: list[RegionDetection]


def _validate_odd_kernel_size(value: int, *, minimum: int = 1, name: str) -> int:
    if value < minimum or value % 2 == 0:
        raise ValueError(f"{name} must be an odd integer >= {minimum}")
    return value


def _ensure_grayscale_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if mask.shape != shape:
        raise ValueError("artifact_mask must have the same height/width as the image")
    return (mask > 0).astype(np.uint8)


def _normalize_float_map(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_v = float(image.min())
    max_v = float(image.max())
    if max_v <= min_v:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_v) / (max_v - min_v)


def _to_u8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(_normalize_float_map(image) * 255.0), 0, 255).astype(np.uint8)


def _ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected grayscale or BGR image, got shape {image.shape}")
    if image.dtype == np.uint8:
        return image.copy()
    return _to_u8(image)


def _select_source_channel(image_bgr: np.ndarray, source: Literal["value", "blue", "max_blue_value"]) -> np.ndarray:
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


def preprocess_image(image_bgr: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    image_bgr = _ensure_bgr_u8(image_bgr)
    if not config.use_median_blur:
        return image_bgr
    kernel = _validate_odd_kernel_size(config.median_kernel_size, minimum=1, name="median_kernel_size")
    if kernel == 1:
        return image_bgr
    return cv2.medianBlur(image_bgr, kernel)


def compute_blue_dominance_map(image_bgr: np.ndarray, config: BlueDominanceConfig) -> np.ndarray:
    image_f = image_bgr.astype(np.float32) / 255.0
    blue = image_f[:, :, 0]
    green = image_f[:, :, 1]
    red = image_f[:, :, 2]

    blue_advantage = (
        config.blue_channel_weight * blue
        + config.blue_green_weight * np.maximum(blue - green, 0.0)
        + config.blue_red_weight * np.maximum(blue - red, 0.0)
    )

    kernel_size = _validate_odd_kernel_size(
        config.local_suppression_kernel_size,
        minimum=3,
        name="local_suppression_kernel_size",
    )
    local_background = cv2.GaussianBlur(blue_advantage, (kernel_size, kernel_size), 0)
    locally_suppressed = np.maximum(
        blue_advantage - (config.local_suppression_strength * local_background),
        0.0,
    )
    return _normalize_float_map(locally_suppressed)


def compute_blob_enhancement_map(image_bgr: np.ndarray, config: BlobEnhancementConfig) -> np.ndarray:
    source = _select_source_channel(image_bgr, config.source)
    source_u8 = _to_u8(source)

    kernel_size = _validate_odd_kernel_size(
        config.top_hat_kernel_size,
        minimum=3,
        name="top_hat_kernel_size",
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    top_hat = cv2.morphologyEx(source_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    dog_small = cv2.GaussianBlur(source, (0, 0), config.dog_sigma_small)
    dog_large = cv2.GaussianBlur(source, (0, 0), config.dog_sigma_large)
    dog = np.maximum(dog_small - dog_large, 0.0)

    combined = config.top_hat_weight * _normalize_float_map(top_hat) + config.dog_weight * _normalize_float_map(dog)
    return _normalize_float_map(combined)


def compute_local_suppression_map(image_bgr: np.ndarray, config: LocalSuppressionConfig) -> np.ndarray:
    source = _select_source_channel(image_bgr, config.source)
    blur_kernel = _validate_odd_kernel_size(
        config.blur_kernel_size,
        minimum=3,
        name="blur_kernel_size",
    )
    background = cv2.GaussianBlur(source, (blur_kernel, blur_kernel), 0)
    suppressed = np.maximum(source - background, 0.0)
    return _normalize_float_map(suppressed)


def fuse_evidence_maps(
    blue_dominance_map: np.ndarray,
    blob_enhanced_map: np.ndarray,
    local_suppression_map: np.ndarray,
    config: EvidenceFusionConfig,
) -> np.ndarray:
    fused = (
        config.blue_weight * blue_dominance_map
        + config.blob_weight * blob_enhanced_map
        + config.suppression_weight * local_suppression_map
    )
    fused = _normalize_float_map(fused)
    if config.gamma <= 0:
        raise ValueError("fusion.gamma must be > 0")
    return np.power(fused, config.gamma, dtype=np.float32)


def _compute_otsu_threshold(values_u8: np.ndarray) -> int:
    flat = np.asarray(values_u8, dtype=np.uint8).reshape(-1)
    if flat.size == 0:
        return 255
    if np.all(flat == flat[0]):
        return int(flat[0])

    histogram = np.bincount(flat, minlength=256).astype(np.float64)
    probabilities = histogram / flat.size
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * np.arange(256, dtype=np.float64))
    mu_total = mu[-1]

    denominator = omega * (1.0 - omega)
    numerator = (mu_total * omega - mu) ** 2
    sigma_b2 = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return int(np.argmax(sigma_b2))


def _prepare_threshold_image(evidence_map: np.ndarray, config: ThresholdConfig) -> np.ndarray:
    evidence_u8 = _to_u8(evidence_map)
    if config.pre_blur_kernel_size <= 1:
        return evidence_u8

    kernel_size = _validate_odd_kernel_size(
        config.pre_blur_kernel_size,
        minimum=3,
        name="pre_blur_kernel_size",
    )
    return cv2.GaussianBlur(evidence_u8, (kernel_size, kernel_size), 0)


def _positive_tail_mask(
    evidence_u8: np.ndarray,
    config: ThresholdConfig,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    valid_values = evidence_u8 if valid_mask is None else evidence_u8[valid_mask]
    if valid_values.size == 0:
        return np.zeros_like(evidence_u8, dtype=bool)
    floor_percentile = float(np.clip(config.positive_floor_percentile, 0.0, 100.0))
    percentile_floor = int(np.percentile(valid_values, floor_percentile))
    floor_value = max(int(config.positive_floor_value), percentile_floor)
    out = evidence_u8 >= floor_value
    if valid_mask is not None:
        out &= valid_mask
    return out


def _positive_floor_threshold(
    evidence_u8: np.ndarray,
    config: ThresholdConfig,
    valid_mask: np.ndarray | None = None,
) -> int:
    valid_values = evidence_u8 if valid_mask is None else evidence_u8[valid_mask]
    if valid_values.size == 0:
        return int(config.positive_floor_value)
    floor_percentile = float(np.clip(config.positive_floor_percentile, 0.0, 100.0))
    percentile_floor = int(np.percentile(valid_values, floor_percentile))
    return max(int(config.positive_floor_value), percentile_floor)


def _threshold_from_method(
    evidence_u8: np.ndarray,
    positive_mask: np.ndarray,
    *,
    method: Literal["otsu", "otsu_positive", "percentile", "percentile_positive"],
    scale: float,
    percentile: float,
    valid_mask: np.ndarray | None = None,
) -> int:
    percentile = float(np.clip(percentile, 0.0, 100.0))
    valid_values = evidence_u8 if valid_mask is None else evidence_u8[valid_mask]

    if method == "otsu":
        base_threshold = _compute_otsu_threshold(valid_values)
    elif method == "otsu_positive":
        base_threshold = _compute_otsu_threshold(evidence_u8[positive_mask])
    elif method == "percentile":
        if valid_values.size == 0:
            base_threshold = 255
        else:
            base_threshold = int(np.percentile(valid_values, percentile))
    elif method == "percentile_positive":
        base_threshold = int(np.percentile(evidence_u8[positive_mask], percentile))
    else:
        raise ValueError(f"Unsupported threshold method: {method}")

    return int(np.clip(round(base_threshold * scale), 0, 255))


def _apply_hysteresis_threshold(evidence_u8: np.ndarray, low_threshold: int, high_threshold: int) -> np.ndarray:
    if low_threshold > high_threshold:
        low_threshold = high_threshold

    low_mask = evidence_u8 >= low_threshold
    high_mask = evidence_u8 >= high_threshold
    if not np.any(high_mask):
        return np.zeros_like(evidence_u8, dtype=np.uint8)

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(low_mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(num_labels, dtype=bool)
    keep[np.unique(labels[high_mask])] = True
    keep[0] = False
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def threshold_evidence_map(
    evidence_map: np.ndarray,
    config: ThresholdConfig,
    artifact_mask: np.ndarray | None = None,
) -> np.ndarray:
    evidence_u8 = _prepare_threshold_image(evidence_map, config)
    valid_mask = None if artifact_mask is None else ~artifact_mask.astype(bool)
    positive_floor = _positive_floor_threshold(evidence_u8, config, valid_mask=valid_mask)
    positive_mask = _positive_tail_mask(evidence_u8, config, valid_mask=valid_mask)
    if not np.any(positive_mask):
        return np.zeros_like(evidence_u8, dtype=np.uint8)

    if config.method == "otsu":
        threshold_value = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method="otsu",
            scale=config.otsu_scale,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        binary = np.where(evidence_u8 >= threshold_value, 255, 0).astype(np.uint8)
        if valid_mask is not None:
            binary[~valid_mask] = 0
        return binary

    if config.method == "otsu_positive":
        threshold_value = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method="otsu_positive",
            scale=config.otsu_scale,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        binary = np.where(evidence_u8 >= threshold_value, 255, 0).astype(np.uint8)
        if valid_mask is not None:
            binary[~valid_mask] = 0
        return binary

    if config.method == "percentile":
        threshold_value = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method="percentile",
            scale=1.0,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        binary = np.where(evidence_u8 >= threshold_value, 255, 0).astype(np.uint8)
        if valid_mask is not None:
            binary[~valid_mask] = 0
        return binary

    if config.method == "percentile_positive":
        threshold_value = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method="percentile_positive",
            scale=1.0,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        binary = np.where(evidence_u8 >= threshold_value, 255, 0).astype(np.uint8)
        if valid_mask is not None:
            binary[~valid_mask] = 0
        return binary

    if config.method == "adaptive":
        block_size = _validate_odd_kernel_size(
            config.adaptive_block_size,
            minimum=3,
            name="adaptive_block_size",
        )
        return cv2.adaptiveThreshold(
            evidence_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            config.adaptive_c,
        )
        if valid_mask is not None:
            binary[~valid_mask] = 0
        return binary

    if config.method == "hysteresis":
        high_threshold = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method="otsu_positive",
            scale=config.hysteresis_high_scale,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        low_threshold = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method="percentile_positive",
            scale=config.hysteresis_low_scale,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        low_threshold = max(low_threshold, positive_floor)
        low_threshold = min(low_threshold, high_threshold)
        binary = _apply_hysteresis_threshold(evidence_u8, low_threshold, high_threshold)
        if valid_mask is not None:
            binary[~valid_mask] = 0
        return binary

    raise ValueError(f"Unsupported threshold method: {config.method}")


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
        open_kernel_size = _validate_odd_kernel_size(
            config.opening_kernel_size,
            minimum=3,
            name="opening_kernel_size",
        )
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, open_kernel)

    if config.closing_kernel_size > 1:
        close_kernel_size = _validate_odd_kernel_size(
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

    watershed_input = _ensure_bgr_u8(image_bgr)
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


def filter_regions(
    labels: np.ndarray,
    components: list[tuple[int, np.ndarray, np.ndarray]],
    config: RegionFilterConfig,
    artifact_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[RegionDetection]]:
    filtered_mask = np.zeros(labels.shape, dtype=np.uint8)
    regions: list[RegionDetection] = []
    height, width = labels.shape
    artifact_mask_bool = artifact_mask.astype(bool) if artifact_mask is not None else None

    for label_id, component_mask, contour in components:
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

        accepted = not rejection_reasons
        if accepted:
            filtered_mask[labels == label_id] = 255

        regions.append(
            RegionDetection(
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
                accepted=accepted,
                rejection_reasons=rejection_reasons,
            )
        )

    _, relabeled, _, _ = cv2.connectedComponentsWithStats(filtered_mask, connectivity=8)
    return relabeled.astype(np.int32), regions


def segment_cells(
    image_bgr: np.ndarray,
    config: SegmentationConfig | None = None,
    artifact_mask: np.ndarray | None = None,
) -> SegmentationResult:
    config = config or SegmentationConfig()
    preprocessed_bgr = preprocess_image(image_bgr, config.preprocessing)
    artifact_mask_u8 = _ensure_grayscale_mask(artifact_mask, preprocessed_bgr.shape[:2])

    blue_dominance_map = compute_blue_dominance_map(preprocessed_bgr, config.blue_dominance)
    blob_enhanced_map = compute_blob_enhancement_map(preprocessed_bgr, config.blob_enhancement)
    local_suppression_map = compute_local_suppression_map(preprocessed_bgr, config.local_suppression)
    evidence_map = fuse_evidence_maps(
        blue_dominance_map,
        blob_enhanced_map,
        local_suppression_map,
        config.fusion,
    )

    binary_mask_raw = threshold_evidence_map(evidence_map, config.threshold, artifact_mask=artifact_mask_u8)
    if artifact_mask_u8 is not None:
        binary_mask_raw = np.where(artifact_mask_u8 > 0, 0, binary_mask_raw).astype(np.uint8)
    binary_mask_clean = morphological_cleanup(binary_mask_raw, config.morphology)
    if artifact_mask_u8 is not None:
        binary_mask_clean = np.where(artifact_mask_u8 > 0, 0, binary_mask_clean).astype(np.uint8)
    binary_mask_clean = split_touching_cells(binary_mask_clean, preprocessed_bgr, config.split)
    if artifact_mask_u8 is not None:
        binary_mask_clean = np.where(artifact_mask_u8 > 0, 0, binary_mask_clean).astype(np.uint8)

    labels_pre_filter, components = connected_components_and_regions(binary_mask_clean, artifact_mask_u8)
    labels, regions = filter_regions(labels_pre_filter, components, config.region_filter, artifact_mask_u8)
    filtered_mask = np.where(labels > 0, 255, 0).astype(np.uint8)

    return SegmentationResult(
        preprocessed_bgr=preprocessed_bgr,
        blue_dominance_map=blue_dominance_map,
        blob_enhanced_map=blob_enhanced_map,
        local_suppression_map=local_suppression_map,
        evidence_map=evidence_map,
        binary_mask_raw=binary_mask_raw,
        binary_mask_clean=binary_mask_clean,
        filtered_mask=filtered_mask,
        candidate_labels=labels_pre_filter,
        labels=labels,
        regions=regions,
    )


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image for writing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def render_label_map(labels: np.ndarray) -> np.ndarray:
    labels_u16 = np.clip(labels, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    labels_u8 = np.where(labels_u16 > 0, (labels_u16 * 37) % 255, 0).astype(np.uint8)
    return cv2.applyColorMap(labels_u8, cv2.COLORMAP_TURBO)


def render_detection_overlay(
    image_bgr: np.ndarray,
    labels: np.ndarray,
    regions: list[RegionDetection],
) -> np.ndarray:
    base = _ensure_bgr_u8(image_bgr)
    overlay = base.copy()

    for region in regions:
        color = (0, 220, 0) if region.accepted else (0, 0, 255)
        overlay[labels == region.label_id] = color
        x, y, w, h = region.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)

    return cv2.addWeighted(base, 0.65, overlay, 0.35, 0.0)


def save_debug_outputs(
    output_dir: str | Path,
    original_image_bgr: np.ndarray,
    result: SegmentationResult,
    artifact_mask: np.ndarray | None = None,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    files = {
        "original_image": output_dir / "original.png",
        "blue_dominance_map": output_dir / "blue_dominance.png",
        "blob_enhanced_map": output_dir / "blob_enhanced.png",
        "local_suppression_map": output_dir / "local_suppression.png",
        "evidence_map": output_dir / "evidence_map.png",
        "binary_mask_raw": output_dir / "binary_mask_raw.png",
        "binary_mask_clean": output_dir / "binary_mask_clean.png",
        "filtered_mask": output_dir / "filtered_mask.png",
        "labels": output_dir / "labels.png",
        "overlay": output_dir / "overlay.png",
    }

    _write_image(files["original_image"], _ensure_bgr_u8(original_image_bgr))
    _write_image(files["blue_dominance_map"], _to_u8(result.blue_dominance_map))
    _write_image(files["blob_enhanced_map"], _to_u8(result.blob_enhanced_map))
    _write_image(files["local_suppression_map"], _to_u8(result.local_suppression_map))
    _write_image(files["evidence_map"], _to_u8(result.evidence_map))
    _write_image(files["binary_mask_raw"], result.binary_mask_raw)
    _write_image(files["binary_mask_clean"], result.binary_mask_clean)
    _write_image(files["filtered_mask"], result.filtered_mask)
    _write_image(files["labels"], render_label_map(result.labels))
    _write_image(files["overlay"], render_detection_overlay(original_image_bgr, result.candidate_labels, result.regions))

    if artifact_mask is not None:
        files["artifact_mask"] = output_dir / "artifact_mask.png"
        artifact_mask_u8 = _ensure_grayscale_mask(artifact_mask, result.filtered_mask.shape)
        _write_image(files["artifact_mask"], artifact_mask_u8 * 255)

    return files


def segment_objects(
    image: np.ndarray,
    THRESH_FACTOR: float = 0.8,
    artifact_mask: np.ndarray | None = None,
    min_area_px: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Backward-compatible wrapper around the new evidence-driven cell segmentation pipeline.
    """
    if THRESH_FACTOR < 0 or THRESH_FACTOR > 2:
        raise ValueError("THRESH_FACTOR must be between 0 and 2")
    if min_area_px < 1:
        raise ValueError("min_area_px must be >= 1")

    config = SegmentationConfig(
        threshold=ThresholdConfig(method="otsu", otsu_scale=THRESH_FACTOR),
        region_filter=RegionFilterConfig(min_area_px=min_area_px),
    )
    result = segment_cells(_ensure_bgr_u8(image), config=config, artifact_mask=artifact_mask)
    return result.labels, result.filtered_mask
