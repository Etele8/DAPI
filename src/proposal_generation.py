from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import cv2
import numpy as np

from src.config import ProposalFilterConfig, SegmentationConfig
from src.segmentation.segment import (
    RegionDetection,
    SegmentationResult,
    connected_components_and_regions,
    measure_region,
    render_label_map,
    segment_cells,
)


@dataclass(slots=True)
class ProposalCandidate:
    image_id: str
    candidate_id: int
    bbox: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_px: int
    width_px: int
    height_px: int
    aspect_ratio: float
    solidity: float
    convexity: float
    circularity: float
    eccentricity: float
    touches_border: bool
    touches_artifact_mask: bool
    exportable: bool
    profile_name: str
    qc_flags: list[str]


@dataclass(slots=True)
class ProposalGenerationResult:
    segmentation: SegmentationResult
    candidate_labels: np.ndarray
    proposal_mask: np.ndarray
    candidates: list[ProposalCandidate]


def _write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image for writing: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def _format_bbox(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(value) for value in bbox)


def _format_centroid(centroid_xy: tuple[float, float]) -> str:
    return f"{centroid_xy[0]:.3f},{centroid_xy[1]:.3f}"


def _candidate_qc_flags(region: RegionDetection, config: ProposalFilterConfig) -> tuple[bool, list[str]]:
    flags: list[str] = []
    exportable = True

    if region.area_px < config.min_area_px:
        flags.append("below_min_area")
        exportable = False
    if config.max_area_px is not None and region.area_px > config.max_area_px:
        flags.append("above_max_area")
        exportable = False
    if region.touches_border:
        flags.append("touches_border")
        if not config.keep_border_touching:
            exportable = False
    if region.area_px <= config.small_area_warn_px:
        flags.append("very_small_candidate")
    if region.area_px >= config.large_area_warn_px:
        flags.append("very_large_candidate")
    if region.aspect_ratio >= config.merged_aspect_ratio_warn:
        flags.append("possible_elongated_or_merged")
    if region.eccentricity >= config.merged_eccentricity_warn:
        flags.append("high_eccentricity")
    if region.solidity <= config.merged_solidity_warn:
        flags.append("low_solidity_possible_merge")
    if region.touches_artifact_mask:
        flags.append("touches_artifact_mask")
    return exportable, flags


def _to_candidate(image_id: str, profile_name: str, region: RegionDetection, config: ProposalFilterConfig) -> ProposalCandidate:
    exportable, qc_flags = _candidate_qc_flags(region, config)
    return ProposalCandidate(
        image_id=image_id,
        candidate_id=region.label_id,
        bbox=region.bbox,
        centroid_xy=region.centroid_xy,
        area_px=region.area_px,
        width_px=region.width_px,
        height_px=region.height_px,
        aspect_ratio=region.aspect_ratio,
        solidity=region.solidity,
        convexity=region.convexity,
        circularity=region.circularity,
        eccentricity=region.eccentricity,
        touches_border=region.touches_border,
        touches_artifact_mask=region.touches_artifact_mask,
        exportable=exportable,
        profile_name=profile_name,
        qc_flags=qc_flags,
    )


def generate_proposals(
    *,
    image_id: str,
    image_bgr: np.ndarray,
    segmentation_config: SegmentationConfig,
    filter_config: ProposalFilterConfig,
    profile_name: str,
    artifact_mask: np.ndarray | None = None,
) -> ProposalGenerationResult:
    segmentation = segment_cells(image_bgr, config=segmentation_config, artifact_mask=artifact_mask)
    candidate_labels, components = connected_components_and_regions(segmentation.binary_mask_clean, artifact_mask)
    candidates: list[ProposalCandidate] = []

    for label_id, component_mask, contour in components:
        region = measure_region(
            label_id,
            component_mask,
            contour,
            candidate_labels.shape,
            artifact_mask=artifact_mask,
            config=None,
        )
        candidates.append(_to_candidate(image_id, profile_name, region, filter_config))

    return ProposalGenerationResult(
        segmentation=segmentation,
        candidate_labels=candidate_labels,
        proposal_mask=np.where(candidate_labels > 0, 255, 0).astype(np.uint8),
        candidates=sorted(candidates, key=lambda item: item.candidate_id),
    )


def render_candidate_id_overlay(image_bgr: np.ndarray, candidate_labels: np.ndarray, candidates: list[ProposalCandidate]) -> np.ndarray:
    overlay = image_bgr.copy()
    for candidate in candidates:
        mask = candidate_labels == candidate.candidate_id
        color = (0, 220, 0) if candidate.exportable else (0, 0, 255)
        overlay[mask] = color
        x, y, w, h = candidate.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 1)
        cv2.putText(
            overlay,
            str(candidate.candidate_id),
            (x, max(12, y + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return cv2.addWeighted(image_bgr, 0.6, overlay, 0.4, 0.0)


def render_crop_box_overlay(image_bgr: np.ndarray, candidates: list[ProposalCandidate]) -> np.ndarray:
    overlay = image_bgr.copy()
    for candidate in candidates:
        if not candidate.exportable:
            continue
        x, y, w, h = candidate.bbox
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (40, 200, 255), 1)
        cv2.putText(
            overlay,
            f"C{candidate.candidate_id:04d}",
            (x, max(12, y + 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def write_candidate_records(
    output_dir: str | Path,
    candidates: list[ProposalCandidate],
    *,
    csv_name: str,
    jsonl_name: str,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    csv_path = output_dir / csv_name
    jsonl_path = output_dir / jsonl_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "candidate_id",
                "bbox",
                "centroid",
                "area_px",
                "width_px",
                "height_px",
                "aspect_ratio",
                "solidity",
                "convexity",
                "circularity",
                "eccentricity",
                "touches_border",
                "touches_artifact_mask",
                "exportable",
                "profile_name",
                "qc_flags",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "image_id": candidate.image_id,
                    "candidate_id": candidate.candidate_id,
                    "bbox": _format_bbox(candidate.bbox),
                    "centroid": _format_centroid(candidate.centroid_xy),
                    "area_px": candidate.area_px,
                    "width_px": candidate.width_px,
                    "height_px": candidate.height_px,
                    "aspect_ratio": candidate.aspect_ratio,
                    "solidity": candidate.solidity,
                    "convexity": candidate.convexity,
                    "circularity": candidate.circularity,
                    "eccentricity": candidate.eccentricity,
                    "touches_border": candidate.touches_border,
                    "touches_artifact_mask": candidate.touches_artifact_mask,
                    "exportable": candidate.exportable,
                    "profile_name": candidate.profile_name,
                    "qc_flags": ";".join(candidate.qc_flags),
                }
            )

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = asdict(candidate)
            payload["bbox"] = list(candidate.bbox)
            payload["centroid_xy"] = list(candidate.centroid_xy)
            handle.write(json.dumps(payload) + "\n")

    return {"candidates_csv": csv_path, "candidates_jsonl": jsonl_path}


def save_proposal_debug_outputs(
    output_dir: str | Path,
    *,
    original_image_bgr: np.ndarray,
    proposal_result: ProposalGenerationResult,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    files = {
        "original_image": output_dir / "original.png",
        "proposal_mask": output_dir / "proposal_mask.png",
        "proposal_labels": output_dir / "proposal_labels.png",
        "candidate_overlay": output_dir / "candidate_overlay.png",
        "candidate_ids": output_dir / "candidate_ids.png",
        "crop_boxes": output_dir / "crop_boxes.png",
    }
    _write_image(files["original_image"], original_image_bgr)
    _write_image(files["proposal_mask"], proposal_result.proposal_mask)
    _write_image(files["proposal_labels"], render_label_map(proposal_result.candidate_labels))
    _write_image(
        files["candidate_overlay"],
        render_candidate_id_overlay(original_image_bgr, proposal_result.candidate_labels, proposal_result.candidates),
    )
    _write_image(
        files["candidate_ids"],
        render_candidate_id_overlay(original_image_bgr, proposal_result.candidate_labels, proposal_result.candidates),
    )
    _write_image(files["crop_boxes"], render_crop_box_overlay(original_image_bgr, proposal_result.candidates))
    return files
