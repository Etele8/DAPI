"""Top-level evidence-driven cell segmentation pipeline.

Orchestrates the per-stage modules:

- `_evidence`: preprocess + per-channel evidence maps + fusion
- `_threshold`: convert the fused map to a binary mask
- `_postprocess`: morphological cleanup, optional watershed split, region measurement
- `_debug`: write per-stage debug artifacts

Public symbols are also re-exported through `src.segmentation` for callers
that import from the package root.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import RegionFilterConfig, SegmentationConfig, ThresholdConfig
from src.segmentation._debug import (
    render_detection_overlay,
    render_label_map,
    save_debug_outputs,
)
from src.segmentation._evidence import (
    compute_blob_enhancement_map,
    compute_blue_dominance_map,
    compute_local_suppression_map,
    fuse_evidence_maps,
    preprocess_image,
)
from src.segmentation._postprocess import (
    RegionDetection,
    connected_components_and_regions,
    filter_regions,
    measure_region,
    morphological_cleanup,
    split_touching_cells,
)
from src.segmentation._threshold import threshold_evidence_map
from src.segmentation._utils import ensure_bgr_u8, ensure_grayscale_mask


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


def _zero_artifact_pixels(mask: np.ndarray, artifact_mask_u8: np.ndarray | None) -> np.ndarray:
    if artifact_mask_u8 is None:
        return mask
    return np.where(artifact_mask_u8 > 0, 0, mask).astype(np.uint8)


def segment_cells(
    image_bgr: np.ndarray,
    config: SegmentationConfig | None = None,
    artifact_mask: np.ndarray | None = None,
) -> SegmentationResult:
    config = config or SegmentationConfig()
    preprocessed_bgr = preprocess_image(image_bgr, config.preprocessing)
    artifact_mask_u8 = ensure_grayscale_mask(artifact_mask, preprocessed_bgr.shape[:2])

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
    binary_mask_raw = _zero_artifact_pixels(binary_mask_raw, artifact_mask_u8)

    binary_mask_clean = morphological_cleanup(binary_mask_raw, config.morphology)
    binary_mask_clean = _zero_artifact_pixels(binary_mask_clean, artifact_mask_u8)
    binary_mask_clean = split_touching_cells(binary_mask_clean, preprocessed_bgr, config.split)
    binary_mask_clean = _zero_artifact_pixels(binary_mask_clean, artifact_mask_u8)

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


def segment_objects(
    image: np.ndarray,
    THRESH_FACTOR: float = 0.8,
    artifact_mask: np.ndarray | None = None,
    min_area_px: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible wrapper that returns (labels, filtered_mask)."""
    if THRESH_FACTOR < 0 or THRESH_FACTOR > 2:
        raise ValueError("THRESH_FACTOR must be between 0 and 2")
    if min_area_px < 1:
        raise ValueError("min_area_px must be >= 1")

    config = SegmentationConfig(
        threshold=ThresholdConfig(method="otsu", otsu_scale=THRESH_FACTOR),
        region_filter=RegionFilterConfig(min_area_px=min_area_px),
    )
    result = segment_cells(ensure_bgr_u8(image), config=config, artifact_mask=artifact_mask)
    return result.labels, result.filtered_mask


__all__ = [
    "RegionDetection",
    "SegmentationResult",
    "connected_components_and_regions",
    "filter_regions",
    "measure_region",
    "morphological_cleanup",
    "preprocess_image",
    "render_detection_overlay",
    "render_label_map",
    "save_debug_outputs",
    "segment_cells",
    "segment_objects",
    "split_touching_cells",
    "threshold_evidence_map",
]
