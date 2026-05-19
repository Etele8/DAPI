"""Convert the fused evidence map to a binary mask via configured thresholding."""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from src.config import ThresholdConfig
from src.segmentation._utils import to_u8, validate_odd_kernel_size


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
    evidence_u8 = to_u8(evidence_map)
    if config.pre_blur_kernel_size <= 1:
        return evidence_u8

    kernel_size = validate_odd_kernel_size(
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


def _apply_valid_mask(binary: np.ndarray, valid_mask: np.ndarray | None) -> np.ndarray:
    if valid_mask is not None:
        binary[~valid_mask] = 0
    return binary


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

    method = config.method
    if method in {"otsu", "otsu_positive", "percentile", "percentile_positive"}:
        scale = config.otsu_scale if method.startswith("otsu") else 1.0
        threshold_value = _threshold_from_method(
            evidence_u8,
            positive_mask,
            method=method,
            scale=scale,
            percentile=config.percentile,
            valid_mask=valid_mask,
        )
        binary = np.where(evidence_u8 >= threshold_value, 255, 0).astype(np.uint8)
        return _apply_valid_mask(binary, valid_mask)

    if method == "adaptive":
        block_size = validate_odd_kernel_size(
            config.adaptive_block_size,
            minimum=3,
            name="adaptive_block_size",
        )
        binary = cv2.adaptiveThreshold(
            evidence_u8,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            config.adaptive_c,
        )
        return _apply_valid_mask(binary, valid_mask)

    if method == "hysteresis":
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
        return _apply_valid_mask(binary, valid_mask)

    raise ValueError(f"Unsupported threshold method: {method}")
