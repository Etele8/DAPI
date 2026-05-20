"""Rebuild full-image instance-label masks from per-crop annotations.

For each sample, walks the crop metadata + annotation manifest under
`outputs/annot/<sample>/` and pastes every accepted crop's mask back into the
full source-image coordinate frame, assigning each crop a unique instance ID.
The result is a Cellpose-ready (image, _masks.tif) pair you can open directly
in Cellpose GUI as a pre-populated starting point for full-image annotation.

Per-crop mask precedence:
  1. The human-edited mask (annotation/edited_masks/<crop_id>.png) if present
  2. Otherwise the proposal mask (crops/masks/<crop_id>.png)

Per-crop inclusion rule (controllable via --include):
  - "all_non_invalid" (default): include positives, unlabeled, exclude
    explicitly invalid. Best for bootstrapping when proposals are accurate.
  - "labeled_positive_only": include only label == single_valid. Safest but
    drops most unlabeled crops.
  - "edited_only": include only crops with an edited mask. Highest quality
    starting point, lowest coverage.

Overlaps (multiple crops covering the same pixel) are resolved first-write-
wins so each cell gets one stable instance ID.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import cv2
import numpy as np


IncludePolicy = Literal["all_non_invalid", "labeled_positive_only", "edited_only"]


@dataclass(slots=True)
class _CropRow:
    crop_id: str
    image_id: str
    crop_bbox: tuple[int, int, int, int]      # (left, top, width, height) in source-image coords
    proposal_mask_rel: str                    # relative to <annot_root>/<sample>/
    edited_mask_rel: str                      # relative to <annot_root>/<sample>/  ("" if none)
    mask_was_edited: bool
    label: str


@dataclass(slots=True)
class ReconstructionSummary:
    sample: str
    full_image_path: Path
    output_image: Path
    output_mask: Path
    instances_pasted: int
    crops_skipped_missing_mask: int
    crops_skipped_by_policy: int
    crops_with_overlap: int


def _read_image_bytes(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def _write_image_bytes(path: Path, image: np.ndarray) -> bool:
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)
    return True


def _parse_bbox(text: str) -> tuple[int, int, int, int]:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 comma-separated ints for bbox, got: {text!r}")
    return parts[0], parts[1], parts[2], parts[3]


def _load_sample_rows(sample_dir: Path) -> list[_CropRow]:
    """Join crop_metadata.csv + annotation_manifest.csv for one sample."""
    crop_meta_path = sample_dir / "crop_metadata.csv"
    manifest_path = sample_dir / "annotation" / "annotation_manifest.csv"
    if not crop_meta_path.exists():
        raise FileNotFoundError(f"Missing crop_metadata.csv at {crop_meta_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing annotation_manifest.csv at {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest_by_id = {row["crop_id"]: row for row in csv.DictReader(handle)}

    rows: list[_CropRow] = []
    with crop_meta_path.open(newline="", encoding="utf-8") as handle:
        for crop in csv.DictReader(handle):
            crop_id = crop["crop_id"]
            manifest = manifest_by_id.get(crop_id, {})
            rows.append(
                _CropRow(
                    crop_id=crop_id,
                    image_id=crop["image_id"],
                    crop_bbox=_parse_bbox(crop["crop_bbox"]),
                    proposal_mask_rel=crop.get("mask_path", "") or "",
                    edited_mask_rel=(manifest.get("edited_mask_path") or "").strip(),
                    mask_was_edited=(manifest.get("mask_was_edited") or "").strip().lower() == "true",
                    label=(manifest.get("label") or "").strip().lower(),
                )
            )
    return rows


def _row_passes_policy(row: _CropRow, policy: IncludePolicy) -> bool:
    if policy == "edited_only":
        return row.mask_was_edited and bool(row.edited_mask_rel)
    if policy == "labeled_positive_only":
        return row.label == "single_valid"
    # default: all_non_invalid
    return row.label != "invalid"


def _pick_mask_path(sample_dir: Path, row: _CropRow) -> Path | None:
    if row.mask_was_edited and row.edited_mask_rel:
        candidate = sample_dir / row.edited_mask_rel
        if candidate.exists():
            return candidate
    if row.proposal_mask_rel:
        candidate = sample_dir / row.proposal_mask_rel
        if candidate.exists():
            return candidate
    return None


def reconstruct_sample(
    *,
    sample: str,
    annot_root: Path,
    source_image_path: Path,
    output_dir: Path,
    policy: IncludePolicy,
) -> ReconstructionSummary:
    sample_dir = annot_root / sample
    rows = _load_sample_rows(sample_dir)

    source_image = _read_image_bytes(source_image_path)
    if source_image is None:
        raise FileNotFoundError(f"Could not read source image: {source_image_path}")
    image_h, image_w = source_image.shape[:2]

    # uint16 supports up to 65535 instances per image; plenty.
    label_canvas = np.zeros((image_h, image_w), dtype=np.uint16)

    next_id = 1
    skipped_by_policy = 0
    skipped_missing_mask = 0
    overlapping = 0

    for row in rows:
        if not _row_passes_policy(row, policy):
            skipped_by_policy += 1
            continue

        mask_path = _pick_mask_path(sample_dir, row)
        if mask_path is None:
            skipped_missing_mask += 1
            continue

        mask = _read_image_bytes(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped_missing_mask += 1
            continue

        left, top, crop_w, crop_h = row.crop_bbox
        # Resize mask to match the crop_bbox size in source coords if it drifted.
        if mask.shape[:2] != (crop_h, crop_w):
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

        right = min(left + crop_w, image_w)
        bottom = min(top + crop_h, image_h)
        if right <= left or bottom <= top:
            skipped_missing_mask += 1
            continue
        mask_slice = mask[: bottom - top, : right - left]

        canvas_region = label_canvas[top:bottom, left:right]
        binary = mask_slice > 0
        if not np.any(binary):
            skipped_missing_mask += 1
            continue

        # First-write-wins: only paint pixels that are currently background.
        paintable = binary & (canvas_region == 0)
        if not np.any(paintable):
            overlapping += 1
            continue
        if np.any(binary & (canvas_region != 0)):
            overlapping += 1

        canvas_region[paintable] = next_id
        label_canvas[top:bottom, left:right] = canvas_region
        next_id += 1

    output_image_path = output_dir / f"{sample}.png"
    output_mask_path = output_dir / f"{sample}_masks.tif"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_image_path, output_image_path)
    _write_image_bytes(output_mask_path, label_canvas)

    return ReconstructionSummary(
        sample=sample,
        full_image_path=source_image_path,
        output_image=output_image_path,
        output_mask=output_mask_path,
        instances_pasted=next_id - 1,
        crops_skipped_missing_mask=skipped_missing_mask,
        crops_skipped_by_policy=skipped_by_policy,
        crops_with_overlap=overlapping,
    )


def _resolve_source_image(samples_root: Path, sample: str) -> Path:
    primary = samples_root / f"{sample}.png"
    if primary.exists():
        return primary
    for ext in (".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = samples_root / f"{sample}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No source image found for {sample!r} under {samples_root} "
        "(looked for .png / .jpg / .jpeg / .tif / .tiff)"
    )


def reconstruct_all(
    *,
    annot_root: Path,
    samples_root: Path,
    output_dir: Path,
    policy: IncludePolicy,
    samples: Sequence[str] | None = None,
) -> list[ReconstructionSummary]:
    if samples is None:
        candidates = sorted(p.name for p in annot_root.iterdir() if (p / "crop_metadata.csv").exists())
    else:
        candidates = list(samples)

    if not candidates:
        raise SystemExit(f"No samples with crop_metadata.csv found under {annot_root}")

    summaries: list[ReconstructionSummary] = []
    for sample in candidates:
        try:
            source_image = _resolve_source_image(samples_root, sample)
        except FileNotFoundError as exc:
            print(f"  SKIP {sample}: {exc}")
            continue
        summary = reconstruct_sample(
            sample=sample,
            annot_root=annot_root,
            source_image_path=source_image,
            output_dir=output_dir,
            policy=policy,
        )
        summaries.append(summary)
        print(
            f"  {sample}: instances={summary.instances_pasted:4d} "
            f"by_policy_skip={summary.crops_skipped_by_policy:4d} "
            f"missing={summary.crops_skipped_missing_mask:3d} "
            f"overlap_skip={summary.crops_with_overlap:3d}"
        )
    return summaries


def _write_summary_csv(output_dir: Path, summaries: Iterable[ReconstructionSummary]) -> Path:
    summary_path = output_dir / "reconstruction_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample",
                "instances_pasted",
                "crops_skipped_by_policy",
                "crops_skipped_missing_mask",
                "crops_with_overlap",
                "output_image",
                "output_mask",
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "sample": s.sample,
                    "instances_pasted": s.instances_pasted,
                    "crops_skipped_by_policy": s.crops_skipped_by_policy,
                    "crops_skipped_missing_mask": s.crops_skipped_missing_mask,
                    "crops_with_overlap": s.crops_with_overlap,
                    "output_image": str(s.output_image),
                    "output_mask": str(s.output_mask),
                }
            )
    return summary_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellpose-reconstruct",
        description="Rebuild full-image instance-label masks from per-crop annotations.",
    )
    parser.add_argument(
        "--annot-root",
        default="outputs/annot",
        help="Annotation root with per-sample crop_metadata.csv + annotation_manifest.csv.",
    )
    parser.add_argument(
        "--samples-root",
        default="data/samples",
        help="Where the source images live (e.g. data/samples/<sample>.png).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/cellpose_fullimage_seed",
        help="Output directory: receives <sample>.png + <sample>_masks.tif per sample.",
    )
    parser.add_argument(
        "--policy",
        choices=("all_non_invalid", "labeled_positive_only", "edited_only"),
        default="all_non_invalid",
        help="Which crops to include in the reconstruction.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        default=None,
        help="Reconstruct only the named sample(s). Repeat for multiple. Default: all.",
    )

    args = parser.parse_args(argv)

    annot_root = Path(args.annot_root)
    samples_root = Path(args.samples_root)
    output_dir = Path(args.out_dir)
    if not annot_root.exists():
        raise SystemExit(f"--annot-root does not exist: {annot_root}")
    if not samples_root.exists():
        raise SystemExit(f"--samples-root does not exist: {samples_root}")

    print(f"Reconstructing into {output_dir.resolve()} (policy={args.policy})")
    summaries = reconstruct_all(
        annot_root=annot_root,
        samples_root=samples_root,
        output_dir=output_dir,
        policy=args.policy,
        samples=args.samples,
    )
    summary_csv = _write_summary_csv(output_dir, summaries)

    total_instances = sum(s.instances_pasted for s in summaries)
    print()
    print(f"Wrote {len(summaries)} full-image seeds to {output_dir.resolve()}")
    print(f"Total instances pasted: {total_instances}")
    print(f"Summary: {summary_csv}")
    print()
    print("Open the seeds in Cellpose GUI:")
    print(f"  python -m cellpose --gui")
    print("  File -> Load image -> pick any <sample>.png in the output dir.")
    print("  The matching <sample>_masks.tif loads automatically as a starting point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
