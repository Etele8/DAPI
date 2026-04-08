from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import statistics

import numpy as np

from src.biomass.calibration import Calibration
from src.biomass.contour_smoothing import smooth_contour
from src.biomass.contours import ConnectedObject, extract_connected_objects, polygon_area
from src.biomass.qc import build_qc_flags, polygon_self_intersects, suspicious_merge
from src.biomass.volume_baselines import equivalent_sphere_volume_from_area, rod_volume_from_area_and_length
from src.biomass.volume_zeder import estimate_zeder_volume, longest_chord
from src.config import BiomassConfig


@dataclass(slots=True)
class BiomassObjectMeasurement:
    image_id: str
    object_id: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    touches_border: bool
    area_px2_raw: float
    area_px2_smooth: float
    perimeter_px: float
    longest_chord_px: float
    longest_chord_endpoints: tuple[tuple[float, float], tuple[float, float]]
    volume_px3_zeder: float
    volume_px3_rod: float
    volume_px3_baseline: float
    area_um2: float
    perimeter_um: float
    longest_chord_um: float
    volume_um3_zeder: float
    volume_um3_rod: float
    volume_um3_baseline: float
    qc_flags: list[str]
    keep: bool
    raw_contour_xy: np.ndarray = field(repr=False)
    smoothed_contour_xy: np.ndarray = field(repr=False)


@dataclass(slots=True)
class BiomassSummary:
    image_id: str
    n_objects_total: int
    n_objects_kept: int
    total_volume_px3_zeder: float
    mean_volume_px3_zeder: float
    median_volume_px3_zeder: float
    total_volume_px3_rod: float
    total_volume_px3_baseline: float
    total_volume_um3_zeder: float
    mean_volume_um3_zeder: float
    median_volume_um3_zeder: float
    total_volume_um3_rod: float
    total_volume_um3_baseline: float


@dataclass(slots=True)
class BiomassDebugFiles:
    files: dict[str, Path]


@dataclass(slots=True)
class BiomassPipelineResult:
    calibration: Calibration
    objects: list[BiomassObjectMeasurement]
    summary: BiomassSummary
    debug_files: BiomassDebugFiles
    approximation_notes: list[str]


def _perimeter(points_xy: np.ndarray) -> float:
    closed = np.vstack([points_xy, points_xy[0]])
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def _format_bbox(bbox: tuple[int, int, int, int]) -> str:
    return ",".join(str(v) for v in bbox)


def _format_centroid(centroid_xy: tuple[float, float]) -> str:
    return f"{centroid_xy[0]:.3f},{centroid_xy[1]:.3f}"


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_method_notes(path: Path, notes: list[str], config: BiomassConfig) -> None:
    payload = {
        "zeder_approximation_notes": notes,
        "config": {
            "microns_per_pixel": config.microns_per_pixel,
            "min_area_px2": config.min_area_px2,
            "smoothing_points": config.smoothing_points,
            "smoothing_window": config.smoothing_window,
            "min_slice_length_px": config.min_slice_length_px,
            "zeder_max_depth": config.zeder_max_depth,
            "zeder_width_linearity_tol_px": config.zeder_width_linearity_tol_px,
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _object_measurement(
    image_id: str,
    obj: ConnectedObject,
    calibration: Calibration,
    config: BiomassConfig,
) -> tuple[BiomassObjectMeasurement, list[str]]:
    smoothing = smooth_contour(
        obj.raw_contour,
        n_points=config.smoothing_points,
        smoothing_window=config.smoothing_window,
    )
    smooth_contour_xy = smoothing.points_xy
    area_px2_smooth = abs(polygon_area(smooth_contour_xy))
    perimeter_px = _perimeter(smooth_contour_xy)
    chord_px, chord_start, chord_end = longest_chord(smooth_contour_xy)

    approximation_notes: list[str] = []
    volume_failed = False
    try:
        zeder = estimate_zeder_volume(
            smooth_contour_xy,
            min_slice_length_px=config.min_slice_length_px,
            max_depth=config.zeder_max_depth,
            width_linearity_tol_px=config.zeder_width_linearity_tol_px,
        )
        volume_px3_zeder = zeder.volume_px3
        approximation_notes = zeder.approximation_notes
    except ValueError:
        volume_px3_zeder = float("nan")
        volume_failed = True

    volume_px3_rod = rod_volume_from_area_and_length(area_px2_smooth, chord_px) if not volume_failed else float("nan")
    volume_px3_baseline = equivalent_sphere_volume_from_area(area_px2_smooth) if not volume_failed else float("nan")

    self_intersects = polygon_self_intersects(smooth_contour_xy) if not smoothing.failed else False
    merged_flag = suspicious_merge(obj.mask, smooth_contour_xy) if not smoothing.failed else False
    qc = build_qc_flags(
        touches_border=obj.touches_border,
        area_px2_raw=obj.area_px2_raw,
        smoothing_failed=smoothing.failed,
        smoothing_reason=smoothing.reason,
        self_intersects=self_intersects,
        merged_object=merged_flag,
        volume_failed=volume_failed,
        calibration_available=calibration.is_available,
        min_area_px2=config.min_area_px2,
    )

    measurement = BiomassObjectMeasurement(
        image_id=image_id,
        object_id=obj.object_id,
        bbox=obj.bbox,
        centroid=obj.centroid_xy,
        touches_border=obj.touches_border,
        area_px2_raw=float(obj.area_px2_raw),
        area_px2_smooth=float(area_px2_smooth),
        perimeter_px=float(perimeter_px),
        longest_chord_px=float(chord_px),
        longest_chord_endpoints=(
            (float(chord_start[0]), float(chord_start[1])),
            (float(chord_end[0]), float(chord_end[1])),
        ),
        volume_px3_zeder=float(volume_px3_zeder),
        volume_px3_rod=float(volume_px3_rod),
        volume_px3_baseline=float(volume_px3_baseline),
        area_um2=calibration.area_um2(area_px2_smooth),
        perimeter_um=calibration.length_um(perimeter_px),
        longest_chord_um=calibration.length_um(chord_px),
        volume_um3_zeder=calibration.volume_um3(volume_px3_zeder),
        volume_um3_rod=calibration.volume_um3(volume_px3_rod),
        volume_um3_baseline=calibration.volume_um3(volume_px3_baseline),
        qc_flags=qc.flags,
        keep=qc.keep,
        raw_contour_xy=obj.raw_contour,
        smoothed_contour_xy=smooth_contour_xy,
    )
    return measurement, approximation_notes


def _summary_from_objects(image_id: str, objects: list[BiomassObjectMeasurement]) -> BiomassSummary:
    kept = [obj for obj in objects if obj.keep]
    zeder_px = [obj.volume_px3_zeder for obj in kept if np.isfinite(obj.volume_px3_zeder)]
    rod_px = [obj.volume_px3_rod for obj in kept if np.isfinite(obj.volume_px3_rod)]
    baseline_px = [obj.volume_px3_baseline for obj in kept if np.isfinite(obj.volume_px3_baseline)]
    zeder_values = [obj.volume_um3_zeder for obj in kept if np.isfinite(obj.volume_um3_zeder)]
    rod_values = [obj.volume_um3_rod for obj in kept if np.isfinite(obj.volume_um3_rod)]
    baseline_values = [obj.volume_um3_baseline for obj in kept if np.isfinite(obj.volume_um3_baseline)]

    return BiomassSummary(
        image_id=image_id,
        n_objects_total=len(objects),
        n_objects_kept=len(kept),
        total_volume_px3_zeder=float(sum(zeder_px)) if zeder_px else float("nan"),
        mean_volume_px3_zeder=float(statistics.fmean(zeder_px)) if zeder_px else float("nan"),
        median_volume_px3_zeder=float(statistics.median(zeder_px)) if zeder_px else float("nan"),
        total_volume_px3_rod=float(sum(rod_px)) if rod_px else float("nan"),
        total_volume_px3_baseline=float(sum(baseline_px)) if baseline_px else float("nan"),
        total_volume_um3_zeder=float(sum(zeder_values)) if zeder_values else float("nan"),
        mean_volume_um3_zeder=float(statistics.fmean(zeder_values)) if zeder_values else float("nan"),
        median_volume_um3_zeder=float(statistics.median(zeder_values)) if zeder_values else float("nan"),
        total_volume_um3_rod=float(sum(rod_values)) if rod_values else float("nan"),
        total_volume_um3_baseline=float(sum(baseline_values)) if baseline_values else float("nan"),
    )


def _write_outputs(
    output_dir: Path,
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    labels: np.ndarray,
    objects: list[BiomassObjectMeasurement],
    summary: BiomassSummary,
    approximation_notes: list[str],
    config: BiomassConfig,
) -> BiomassDebugFiles:
    from src.biomass.visualization import (
        draw_contour_overlay,
        draw_labeled_overlay,
        draw_object_ids,
        render_object_panel,
        write_image,
    )

    files = {
        "original_image": output_dir / "original.png",
        "binary_mask": output_dir / "binary_mask.png",
        "labeled_object_overlay": output_dir / "labeled_overlay.png",
        "contour_overlay": output_dir / "contour_overlay.png",
        "object_ids": output_dir / "object_ids.png",
        "per_object_csv": output_dir / "per_object.csv",
        "per_image_csv": output_dir / "per_image.csv",
        "method_notes": output_dir / "method_notes.json",
    }

    write_image(files["original_image"], original_image)
    write_image(files["binary_mask"], binary_mask)
    write_image(files["labeled_object_overlay"], draw_labeled_overlay(original_image, labels))
    write_image(files["contour_overlay"], draw_contour_overlay(original_image, objects))
    write_image(files["object_ids"], draw_object_ids(original_image, objects))

    object_rows = [
        {
            "image_id": obj.image_id,
            "object_id": obj.object_id,
            "bbox": _format_bbox(obj.bbox),
            "centroid": _format_centroid(obj.centroid),
            "touches_border": obj.touches_border,
            "area_px2_raw": obj.area_px2_raw,
            "area_px2_smooth": obj.area_px2_smooth,
            "perimeter_px": obj.perimeter_px,
            "longest_chord_px": obj.longest_chord_px,
            "volume_px3_zeder": obj.volume_px3_zeder,
            "volume_px3_rod": obj.volume_px3_rod,
            "volume_px3_baseline": obj.volume_px3_baseline,
            "area_um2": obj.area_um2,
            "perimeter_um": obj.perimeter_um,
            "longest_chord_um": obj.longest_chord_um,
            "volume_um3_zeder": obj.volume_um3_zeder,
            "volume_um3_rod": obj.volume_um3_rod,
            "volume_um3_baseline": obj.volume_um3_baseline,
            "qc_flags": ";".join(obj.qc_flags),
        }
        for obj in objects
    ]
    _write_csv(
        files["per_object_csv"],
        object_rows,
        [
            "image_id",
            "object_id",
            "bbox",
            "centroid",
            "touches_border",
            "area_px2_raw",
            "area_px2_smooth",
            "perimeter_px",
            "longest_chord_px",
            "volume_px3_zeder",
            "volume_px3_rod",
            "volume_px3_baseline",
            "area_um2",
            "perimeter_um",
            "longest_chord_um",
            "volume_um3_zeder",
            "volume_um3_rod",
            "volume_um3_baseline",
            "qc_flags",
        ],
    )

    _write_csv(
        files["per_image_csv"],
        [
            {
                "image_id": summary.image_id,
                "n_objects_total": summary.n_objects_total,
                "n_objects_kept": summary.n_objects_kept,
                "total_volume_px3_zeder": summary.total_volume_px3_zeder,
                "mean_volume_px3_zeder": summary.mean_volume_px3_zeder,
                "median_volume_px3_zeder": summary.median_volume_px3_zeder,
                "total_volume_px3_rod": summary.total_volume_px3_rod,
                "total_volume_px3_baseline": summary.total_volume_px3_baseline,
                "total_volume_um3_zeder": summary.total_volume_um3_zeder,
                "mean_volume_um3_zeder": summary.mean_volume_um3_zeder,
                "median_volume_um3_zeder": summary.median_volume_um3_zeder,
                "total_volume_um3_rod": summary.total_volume_um3_rod,
                "total_volume_um3_baseline": summary.total_volume_um3_baseline,
            }
        ],
        [
            "image_id",
            "n_objects_total",
            "n_objects_kept",
            "total_volume_px3_zeder",
            "mean_volume_px3_zeder",
            "median_volume_px3_zeder",
            "total_volume_px3_rod",
            "total_volume_px3_baseline",
            "total_volume_um3_zeder",
            "mean_volume_um3_zeder",
            "median_volume_um3_zeder",
            "total_volume_um3_rod",
            "total_volume_um3_baseline",
        ],
    )

    panels_dir = output_dir / "object_panels"
    for obj in sorted(objects, key=lambda item: item.area_px2_smooth, reverse=True)[: config.max_debug_panels]:
        panel = render_object_panel(original_image, obj)
        path = panels_dir / f"object_{obj.object_id:04d}.png"
        write_image(path, panel)
        files[f"panel_{obj.object_id:04d}"] = path

    _save_method_notes(files["method_notes"], approximation_notes, config)
    return BiomassDebugFiles(files=files)


def run_biomass_stage(
    *,
    image_id: str,
    original_image: np.ndarray,
    binary_mask: np.ndarray,
    output_dir: str | Path,
    config: BiomassConfig | None = None,
) -> BiomassPipelineResult:
    config = config or BiomassConfig()
    calibration = Calibration(microns_per_pixel=config.microns_per_pixel)
    connected_objects = extract_connected_objects(binary_mask)

    labels = np.zeros(binary_mask.shape[:2], dtype=np.int32)
    for obj in connected_objects:
        labels[obj.mask > 0] = obj.object_id

    measurements: list[BiomassObjectMeasurement] = []
    notes: list[str] = []
    for obj in connected_objects:
        measurement, approximation_notes = _object_measurement(image_id, obj, calibration, config)
        measurements.append(measurement)
        if approximation_notes and not notes:
            notes = approximation_notes

    summary = _summary_from_objects(image_id, measurements)
    debug_files = _write_outputs(
        Path(output_dir),
        original_image=original_image,
        binary_mask=np.where(binary_mask > 0, 255, 0).astype(np.uint8),
        labels=labels,
        objects=measurements,
        summary=summary,
        approximation_notes=notes,
        config=config,
    )
    return BiomassPipelineResult(
        calibration=calibration,
        objects=measurements,
        summary=summary,
        debug_files=debug_files,
        approximation_notes=notes,
    )
