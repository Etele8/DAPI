from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv

import cv2
import numpy as np

from src.biomass import BiomassPipelineResult, run_biomass_stage
from src.classifier_dataset import ingest_annotation_manifest
from src.config import BiomassConfig, ClassifierDatasetConfig, LocalRefinementConfig
from src.segmentation.segment import RegionDetection, segment_cells


@dataclass(slots=True)
class RefinedCropRecord:
    crop_id: str
    image_id: str
    candidate_id: str
    label: str
    refinement_status: str
    qc_flag: str
    refined_area_px: float
    refined_mask_path: str
    refined_overlay_path: str
    biomass_volume_um3_zeder: float | None


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image for writing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _load_positive_rows(manifest_path: str | Path, dataset_config: ClassifierDatasetConfig) -> list[dict[str, str]]:
    rows = ingest_annotation_manifest(manifest_path)
    return [row for row in rows if row.get("label", "").strip() in dataset_config.positive_labels]


def _choose_region(regions: list[RegionDetection], shape: tuple[int, int], center_distance_weight: float) -> RegionDetection | None:
    if not regions:
        return None
    center_x = shape[1] / 2.0
    center_y = shape[0] / 2.0

    def score(region: RegionDetection) -> float:
        dx = region.centroid_xy[0] - center_x
        dy = region.centroid_xy[1] - center_y
        return float(region.area_px) - (center_distance_weight * (dx * dx + dy * dy))

    accepted = [region for region in regions if region.accepted]
    pool = accepted if accepted else regions
    return max(pool, key=score)


def _render_refinement_overlay(crop_image: np.ndarray, mask: np.ndarray, region: RegionDetection | None) -> np.ndarray:
    overlay = crop_image.copy()
    overlay[mask > 0] = (0, 220, 0)
    if region is not None:
        x, y, w, h = region.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 200, 255), 1)
    return cv2.addWeighted(crop_image, 0.65, overlay, 0.35, 0.0)


def refine_positive_crops(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    refinement_config: LocalRefinementConfig,
    dataset_config: ClassifierDatasetConfig,
    biomass_config: BiomassConfig | None = None,
    run_biomass: bool = False,
) -> tuple[list[RefinedCropRecord], dict[str, Path]]:
    manifest_path = Path(manifest_path)
    manifest_root = manifest_path.parent.parent if manifest_path.parent.name == "annotation" else manifest_path.parent
    positive_rows = _load_positive_rows(manifest_path, dataset_config)
    output_dir = Path(output_dir)
    files: dict[str, Path] = {}
    refined_rows: list[RefinedCropRecord] = []

    masks_dir = output_dir / refinement_config.refined_masks_subdir
    overlays_dir = output_dir / refinement_config.refined_overlays_subdir
    biomass_dir = output_dir / refinement_config.biomass_subdir

    for row in positive_rows:
        crop_id = row["crop_id"]
        crop_path = manifest_root / row["crop_path"]
        crop_image = _read_image(crop_path)
        segmentation = segment_cells(crop_image, config=refinement_config.segmentation)
        chosen = _choose_region(segmentation.regions, crop_image.shape[:2], refinement_config.center_distance_weight)

        refined_mask = np.zeros(crop_image.shape[:2], dtype=np.uint8)
        qc_flags: list[str] = []
        area_px = 0.0
        biomass_result: BiomassPipelineResult | None = None

        if chosen is None:
            status = "failed"
            qc_flags.append("refinement_failed_no_region")
        else:
            refined_mask = np.where(segmentation.candidate_labels == chosen.label_id, 255, 0).astype(np.uint8)
            area_px = float(np.count_nonzero(refined_mask))
            status = "refined"
            if chosen.touches_border:
                qc_flags.append("refined_touches_crop_border")
            if run_biomass and biomass_config is not None:
                biomass_result = run_biomass_stage(
                    image_id=crop_id,
                    original_image=crop_image,
                    binary_mask=refined_mask,
                    output_dir=biomass_dir / crop_id,
                    config=biomass_config,
                )

        mask_path = masks_dir / f"{crop_id}.png"
        overlay_path = overlays_dir / f"{crop_id}.png"
        _write_image(mask_path, refined_mask)
        _write_image(overlay_path, _render_refinement_overlay(crop_image, refined_mask, chosen))

        refined_rows.append(
            RefinedCropRecord(
                crop_id=crop_id,
                image_id=row["image_id"],
                candidate_id=row["candidate_id"],
                label=row["label"],
                refinement_status=status,
                qc_flag=";".join(qc_flags),
                refined_area_px=area_px,
                refined_mask_path=mask_path.relative_to(output_dir).as_posix(),
                refined_overlay_path=overlay_path.relative_to(output_dir).as_posix(),
                biomass_volume_um3_zeder=(
                    None
                    if biomass_result is None
                    else biomass_result.summary.total_volume_um3_zeder
                ),
            )
        )

    metadata_path = output_dir / refinement_config.refined_metadata_name
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "crop_id",
                "image_id",
                "candidate_id",
                "label",
                "refinement_status",
                "qc_flag",
                "refined_area_px",
                "refined_mask_path",
                "refined_overlay_path",
                "biomass_volume_um3_zeder",
            ],
        )
        writer.writeheader()
        for row in refined_rows:
            writer.writerow(asdict(row))

    files["refined_metadata_csv"] = metadata_path
    return refined_rows, files
