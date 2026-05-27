"""Compute biomass from Cellpose instance-label masks.

For each <stem>.png paired with <stem><mask-suffix>.{png,tif} in a directory,
feeds the instance labels straight into the biovolume computation (one
measurement per Cellpose instance, so touching cells the model separated are
not re-merged). Writes the standard per-image biomass outputs under
<output-dir>/<stem>/ and an aggregate dataset-level CSV across all images.

This is pure CPU geometry — fast, no GPU needed. Run it locally on masks you
pulled back from the GPU pod.

Calibration: pass --microns-per-pixel to get volumes in um^3. Without it,
volumes are reported in px^3 only (um^3 columns will be NaN).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.biomass import run_biomass_from_labels
from src.config import BiomassConfig


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_MASK_EXTS = (".png", ".tif", ".tiff")
_ARTEFACT_MARKERS = ("_cp_outlines", "_cp_flows", "_flows", "_overlay")


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def _find_mask(masks_dir: Path, stem: str, mask_suffix: str) -> Path | None:
    for ext in _MASK_EXTS:
        candidate = masks_dir / f"{stem}{mask_suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def _list_pairs(images_dir: Path, masks_dir: Path, mask_suffix: str) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        stem = image_path.stem
        if stem.endswith(mask_suffix) or any(m in stem for m in _ARTEFACT_MARKERS):
            continue
        mask_path = _find_mask(masks_dir, stem, mask_suffix)
        if mask_path is not None:
            pairs.append((stem, image_path, mask_path))
    return pairs


def _safe(value: float) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.6g}"


def run(
    *,
    images_dir: Path,
    masks_dir: Path,
    mask_suffix: str,
    output_dir: Path,
    microns_per_pixel: float | None,
) -> int:
    pairs = _list_pairs(images_dir, masks_dir, mask_suffix)
    if not pairs:
        raise SystemExit(
            f"No (image, mask) pairs found. Looked for <stem>{mask_suffix}.<ext> in {masks_dir}."
        )

    config = BiomassConfig(microns_per_pixel=microns_per_pixel)
    output_dir.mkdir(parents=True, exist_ok=True)
    if microns_per_pixel is None:
        print("WARNING: no --microns-per-pixel given; um^3 columns will be empty (px^3 only).")
    print(f"Computing biomass for {len(pairs)} images -> {output_dir.resolve()}")

    aggregate_rows: list[dict[str, object]] = []
    for stem, image_path, mask_path in pairs:
        image = _read_image(image_path, flags=cv2.IMREAD_COLOR)
        labels = _read_image(mask_path, flags=cv2.IMREAD_UNCHANGED)
        if image is None or labels is None:
            print(f"  SKIP {stem}: could not read image or mask")
            continue
        if labels.ndim == 3:
            labels = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
        if labels.shape[:2] != image.shape[:2]:
            labels = cv2.resize(
                labels.astype(np.int32),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        result = run_biomass_from_labels(
            image_id=stem,
            original_image=image,
            label_image=labels,
            output_dir=output_dir / stem,
            config=config,
        )
        s = result.summary
        aggregate_rows.append(
            {
                "image_id": s.image_id,
                "n_objects_total": s.n_objects_total,
                "n_objects_kept": s.n_objects_kept,
                "total_volume_px3_zeder": s.total_volume_px3_zeder,
                "mean_volume_px3_zeder": s.mean_volume_px3_zeder,
                "median_volume_px3_zeder": s.median_volume_px3_zeder,
                "total_volume_um3_zeder": s.total_volume_um3_zeder,
                "mean_volume_um3_zeder": s.mean_volume_um3_zeder,
                "median_volume_um3_zeder": s.median_volume_um3_zeder,
            }
        )
        print(
            f"  {stem}: objects={s.n_objects_total} kept={s.n_objects_kept} "
            f"total_vol_px3={_safe(s.total_volume_px3_zeder)} "
            f"total_vol_um3={_safe(s.total_volume_um3_zeder)}"
        )

    summary_csv = output_dir / "biomass_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "n_objects_total",
                "n_objects_kept",
                "total_volume_px3_zeder",
                "mean_volume_px3_zeder",
                "median_volume_px3_zeder",
                "total_volume_um3_zeder",
                "mean_volume_um3_zeder",
                "median_volume_um3_zeder",
            ],
        )
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow(row)

    total_cells = sum(int(r["n_objects_kept"]) for r in aggregate_rows)
    total_px3 = sum(
        float(r["total_volume_px3_zeder"])
        for r in aggregate_rows
        if isinstance(r["total_volume_px3_zeder"], (int, float)) and not math.isnan(float(r["total_volume_px3_zeder"]))
    )
    print()
    print(f"Aggregate: {len(aggregate_rows)} images, {total_cells} kept cells")
    print(f"Total biovolume (Zeder): {total_px3:.6g} px^3"
          + (f"" if microns_per_pixel is None else f"  ({total_px3 * (microns_per_pixel ** 3):.6g} um^3)"))
    print(f"Per-image summary: {summary_csv.resolve()}")
    print(f"Per-image details + overlays under: {output_dir.resolve()}/<image>/")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellpose-biomass",
        description="Compute biovolume from Cellpose instance-label masks.",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory with source images and their instance-label masks.",
    )
    parser.add_argument(
        "--masks-dir",
        default=None,
        help="Directory containing the mask label images. Defaults to --dir.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_cp_masks",
        help="Suffix identifying mask files. Default _cp_masks (Cellpose CLI output).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cellpose_biomass",
        help="Where to write per-image biomass outputs and the aggregate summary.",
    )
    parser.add_argument(
        "--microns-per-pixel",
        type=float,
        default=None,
        help="Calibration for um^3 volumes. Omit to report px^3 only.",
    )

    args = parser.parse_args(argv)
    images_dir = Path(args.dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else images_dir
    if not images_dir.exists():
        raise SystemExit(f"--dir does not exist: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"--masks-dir does not exist: {masks_dir}")

    return run(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mask_suffix=args.mask_suffix,
        output_dir=Path(args.output_dir),
        microns_per_pixel=args.microns_per_pixel,
    )


if __name__ == "__main__":
    raise SystemExit(main())
