"""Build the per-pixel evidence map from a BGR microscopy image.

The evidence map is a normalized float32 stack that fuses three views:
blue-channel dominance, blob-shape enhancement, and local background
suppression. The thresholding stage operates on this fused map.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import (
    BlobEnhancementConfig,
    BlueDominanceConfig,
    EvidenceFusionConfig,
    LocalSuppressionConfig,
    PreprocessingConfig,
)
from src.segmentation._utils import (
    ensure_bgr_u8,
    normalize_float_map,
    select_source_channel,
    to_u8,
    validate_odd_kernel_size,
)


def preprocess_image(image_bgr: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    image_bgr = ensure_bgr_u8(image_bgr)
    if not config.use_median_blur:
        return image_bgr
    kernel = validate_odd_kernel_size(config.median_kernel_size, minimum=1, name="median_kernel_size")
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

    kernel_size = validate_odd_kernel_size(
        config.local_suppression_kernel_size,
        minimum=3,
        name="local_suppression_kernel_size",
    )
    local_background = cv2.GaussianBlur(blue_advantage, (kernel_size, kernel_size), 0)
    locally_suppressed = np.maximum(
        blue_advantage - (config.local_suppression_strength * local_background),
        0.0,
    )
    return normalize_float_map(locally_suppressed)


def compute_blob_enhancement_map(image_bgr: np.ndarray, config: BlobEnhancementConfig) -> np.ndarray:
    source = select_source_channel(image_bgr, config.source)
    source_u8 = to_u8(source)

    kernel_size = validate_odd_kernel_size(
        config.top_hat_kernel_size,
        minimum=3,
        name="top_hat_kernel_size",
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    top_hat = cv2.morphologyEx(source_u8, cv2.MORPH_TOPHAT, kernel).astype(np.float32)

    dog_small = cv2.GaussianBlur(source, (0, 0), config.dog_sigma_small)
    dog_large = cv2.GaussianBlur(source, (0, 0), config.dog_sigma_large)
    dog = np.maximum(dog_small - dog_large, 0.0)

    combined = config.top_hat_weight * normalize_float_map(top_hat) + config.dog_weight * normalize_float_map(dog)
    return normalize_float_map(combined)


def compute_local_suppression_map(image_bgr: np.ndarray, config: LocalSuppressionConfig) -> np.ndarray:
    source = select_source_channel(image_bgr, config.source)
    blur_kernel = validate_odd_kernel_size(
        config.blur_kernel_size,
        minimum=3,
        name="blur_kernel_size",
    )
    background = cv2.GaussianBlur(source, (blur_kernel, blur_kernel), 0)
    suppressed = np.maximum(source - background, 0.0)
    return normalize_float_map(suppressed)


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
    fused = normalize_float_map(fused)
    if config.gamma <= 0:
        raise ValueError("fusion.gamma must be > 0")
    return np.power(fused, config.gamma, dtype=np.float32)
