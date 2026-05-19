"""Render and write per-stage debug artifacts for a SegmentationResult."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.segmentation._postprocess import RegionDetection
from src.segmentation._utils import ensure_bgr_u8, ensure_grayscale_mask, to_u8

if TYPE_CHECKING:
    from src.segmentation.segment import SegmentationResult


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
    base = ensure_bgr_u8(image_bgr)
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
    result: "SegmentationResult",
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

    _write_image(files["original_image"], ensure_bgr_u8(original_image_bgr))
    _write_image(files["blue_dominance_map"], to_u8(result.blue_dominance_map))
    _write_image(files["blob_enhanced_map"], to_u8(result.blob_enhanced_map))
    _write_image(files["local_suppression_map"], to_u8(result.local_suppression_map))
    _write_image(files["evidence_map"], to_u8(result.evidence_map))
    _write_image(files["binary_mask_raw"], result.binary_mask_raw)
    _write_image(files["binary_mask_clean"], result.binary_mask_clean)
    _write_image(files["filtered_mask"], result.filtered_mask)
    _write_image(files["labels"], render_label_map(result.labels))
    _write_image(
        files["overlay"],
        render_detection_overlay(original_image_bgr, result.candidate_labels, result.regions),
    )

    if artifact_mask is not None:
        files["artifact_mask"] = output_dir / "artifact_mask.png"
        artifact_mask_u8 = ensure_grayscale_mask(artifact_mask, result.filtered_mask.shape)
        _write_image(files["artifact_mask"], artifact_mask_u8 * 255)

    return files
