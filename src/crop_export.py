from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import cv2
import numpy as np

from src.config import CropExportConfig
from src.proposal_generation import ProposalCandidate, ProposalGenerationResult


@dataclass(slots=True)
class CropRecord:
    crop_id: str
    image_id: str
    candidate_id: int
    source_bbox: tuple[int, int, int, int]
    crop_bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_px: int
    touches_border: bool
    qc_flag: str
    profile_name: str
    crop_path: str
    overlay_path: str
    mask_path: str


@dataclass(slots=True)
class CropExportResult:
    records: list[CropRecord]
    files: dict[str, Path]


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image for writing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _relative_to(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _compute_crop_bbox(
    image_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    *,
    padding_px: int,
    min_crop_size_px: int,
    force_square: bool,
    clamp_to_image: bool,
) -> tuple[int, int, int, int]:
    image_h, image_w = image_shape
    x, y, w, h = bbox
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    crop_w = max(w + 2 * padding_px, min_crop_size_px)
    crop_h = max(h + 2 * padding_px, min_crop_size_px)
    if force_square:
        crop_w = crop_h = max(crop_w, crop_h)

    left = int(round(cx - (crop_w / 2.0)))
    top = int(round(cy - (crop_h / 2.0)))
    crop_w = int(crop_w)
    crop_h = int(crop_h)

    if clamp_to_image:
        left = min(max(left, 0), max(image_w - crop_w, 0))
        top = min(max(top, 0), max(image_h - crop_h, 0))
        crop_w = min(crop_w, image_w)
        crop_h = min(crop_h, image_h)

    return left, top, crop_w, crop_h


def _overlay_crop(crop_image: np.ndarray, crop_mask: np.ndarray) -> np.ndarray:
    overlay = crop_image.copy()
    overlay[crop_mask > 0] = (0, 220, 0)
    return cv2.addWeighted(crop_image, 0.65, overlay, 0.35, 0.0)


def export_candidate_crops(
    *,
    image_bgr: np.ndarray,
    proposal_result: ProposalGenerationResult,
    output_dir: str | Path,
    config: CropExportConfig,
) -> CropExportResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    records: list[CropRecord] = []

    images_dir = output_dir / config.images_subdir
    overlays_dir = output_dir / config.overlays_subdir
    masks_dir = output_dir / config.masks_subdir

    for candidate in proposal_result.candidates:
        if not candidate.exportable:
            continue

        crop_bbox = _compute_crop_bbox(
            image_bgr.shape[:2],
            candidate.bbox,
            padding_px=config.padding_px,
            min_crop_size_px=config.min_crop_size_px,
            force_square=config.force_square,
            clamp_to_image=config.clamp_to_image,
        )
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        crop_image = image_bgr[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        crop_mask = np.where(
            proposal_result.candidate_labels[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] == candidate.candidate_id,
            255,
            0,
        ).astype(np.uint8)

        crop_id = f"{candidate.image_id}_c{candidate.candidate_id:04d}"
        image_path = images_dir / f"{crop_id}.png"
        overlay_path = overlays_dir / f"{crop_id}.png"
        mask_path = masks_dir / f"{crop_id}.png"

        if config.export_images:
            _write_image(image_path, crop_image)
        if config.export_overlays:
            _write_image(overlay_path, _overlay_crop(crop_image, crop_mask))
        if config.export_masks:
            _write_image(mask_path, crop_mask)

        record = CropRecord(
            crop_id=crop_id,
            image_id=candidate.image_id,
            candidate_id=candidate.candidate_id,
            source_bbox=candidate.bbox,
            crop_bbox=crop_bbox,
            centroid_xy=candidate.centroid_xy,
            area_px=candidate.area_px,
            touches_border=candidate.touches_border,
            qc_flag=";".join(candidate.qc_flags),
            profile_name=candidate.profile_name,
            crop_path=_relative_to(image_path, output_dir) if config.export_images else "",
            overlay_path=_relative_to(overlay_path, output_dir) if config.export_overlays else "",
            mask_path=_relative_to(mask_path, output_dir) if config.export_masks else "",
        )
        records.append(record)

    csv_path = output_dir / config.metadata_csv_name
    jsonl_path = output_dir / config.metadata_jsonl_name
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "crop_id",
                "image_id",
                "candidate_id",
                "source_bbox",
                "crop_bbox",
                "centroid",
                "area_px",
                "touches_border",
                "qc_flag",
                "profile_name",
                "crop_path",
                "overlay_path",
                "mask_path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "crop_id": record.crop_id,
                    "image_id": record.image_id,
                    "candidate_id": record.candidate_id,
                    "source_bbox": ",".join(str(value) for value in record.source_bbox),
                    "crop_bbox": ",".join(str(value) for value in record.crop_bbox),
                    "centroid": f"{record.centroid_xy[0]:.3f},{record.centroid_xy[1]:.3f}",
                    "area_px": record.area_px,
                    "touches_border": record.touches_border,
                    "qc_flag": record.qc_flag,
                    "profile_name": record.profile_name,
                    "crop_path": record.crop_path,
                    "overlay_path": record.overlay_path,
                    "mask_path": record.mask_path,
                }
            )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            payload["source_bbox"] = list(record.source_bbox)
            payload["crop_bbox"] = list(record.crop_bbox)
            payload["centroid_xy"] = list(record.centroid_xy)
            handle.write(json.dumps(payload) + "\n")

    files["crop_metadata_csv"] = csv_path
    files["crop_metadata_jsonl"] = jsonl_path
    return CropExportResult(records=records, files=files)
