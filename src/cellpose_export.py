"""Pack human-edited annotation crops into a Cellpose-trainable directory.

Cellpose's CLI training step (`python -m cellpose --train`) expects a folder
of paired files:

    <crop_id>.png         # the input image
    <crop_id>_masks.tif   # uint16 instance-label image (0 = background)

For our single-cell crops, every mask is a single instance, so the label
image is just `mask_binary > 0` cast to uint16. We pull each pair from the
annotation manifest where `mask_was_edited=true`, optionally filtering to
positive labels, and write a train/val split.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


POSITIVE_LABELS_DEFAULT = ("single_valid",)


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


def _manifest_root(manifest_path: Path) -> Path:
    return manifest_path.parent.parent


def _row_is_eligible(row: dict[str, str], positive_labels: Sequence[str], require_positive: bool) -> bool:
    if (row.get("mask_was_edited") or "").strip().lower() != "true":
        return False
    edited_rel = (row.get("edited_mask_path") or "").strip()
    if not edited_rel:
        return False
    if not require_positive:
        return True
    label = (row.get("label") or "").strip().lower()
    if not label:
        return False
    return label in {lbl.lower() for lbl in positive_labels}


def collect_training_rows(
    annot_root: Path,
    *,
    positive_labels: Sequence[str] = POSITIVE_LABELS_DEFAULT,
    require_positive: bool = True,
) -> list[tuple[Path, dict[str, str]]]:
    """Walk every manifest under annot_root and yield rows ready for export."""
    out: list[tuple[Path, dict[str, str]]] = []
    for manifest_path in sorted(annot_root.glob("*/annotation/annotation_manifest.csv")):
        with manifest_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if _row_is_eligible(row, positive_labels, require_positive):
                    out.append((manifest_path, row))
    return out


def _binary_mask_to_label_image(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 1, 0).astype(np.uint16)


def _export_pair(
    *,
    manifest_path: Path,
    row: dict[str, str],
    split_dir: Path,
) -> bool:
    crop_id = (row.get("crop_id") or "").strip()
    if not crop_id:
        return False
    manifest_root = _manifest_root(manifest_path)
    crop_src = manifest_root / row["crop_path"]
    mask_src = manifest_root / row["edited_mask_path"]

    crop_image = _read_image_bytes(crop_src, flags=cv2.IMREAD_UNCHANGED)
    mask_image = _read_image_bytes(mask_src, flags=cv2.IMREAD_GRAYSCALE)
    if crop_image is None or mask_image is None:
        return False

    if mask_image.shape[:2] != crop_image.shape[:2]:
        mask_image = cv2.resize(
            mask_image,
            (crop_image.shape[1], crop_image.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    label_image = _binary_mask_to_label_image(mask_image)
    crop_out = split_dir / f"{crop_id}.png"
    mask_out = split_dir / f"{crop_id}_masks.tif"
    if not _write_image_bytes(crop_out, crop_image):
        return False
    if not _write_image_bytes(mask_out, label_image):
        return False
    return True


def export_for_cellpose(
    *,
    annot_root: Path,
    out_dir: Path,
    val_fraction: float,
    seed: int,
    positive_labels: Sequence[str] = POSITIVE_LABELS_DEFAULT,
    require_positive: bool = True,
) -> dict[str, object]:
    rows = collect_training_rows(
        annot_root,
        positive_labels=positive_labels,
        require_positive=require_positive,
    )
    if not rows:
        raise SystemExit(
            f"No eligible rows under {annot_root}. "
            "Need rows with mask_was_edited=true"
            + (f" and label in {tuple(positive_labels)}" if require_positive else "")
        )

    if not 0.0 <= val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0.0, 1.0).")

    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    val_rows = shuffled[:n_val]
    train_rows = shuffled[n_val:]

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"

    exported = {"train": 0, "val": 0, "skipped": 0}
    summary_records: list[dict[str, str]] = []

    def _run(split_dir: Path, items: Iterable[tuple[Path, dict[str, str]]], split_name: str) -> None:
        for manifest_path, row in items:
            if _export_pair(manifest_path=manifest_path, row=row, split_dir=split_dir):
                exported[split_name] += 1
                summary_records.append(
                    {
                        "split": split_name,
                        "crop_id": row.get("crop_id", ""),
                        "image_id": row.get("image_id", ""),
                        "label": row.get("label", ""),
                        "source_manifest": str(manifest_path),
                    }
                )
            else:
                exported["skipped"] += 1

    _run(train_dir, train_rows, "train")
    _run(val_dir, val_rows, "val")

    summary_csv = out_dir / "export_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "crop_id", "image_id", "label", "source_manifest"],
        )
        writer.writeheader()
        writer.writerows(summary_records)

    return {
        "exported": exported,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "summary_csv": summary_csv,
        "total_rows_considered": len(rows),
    }


def _print_runpod_instructions(out_dir: Path, exported: dict[str, int]) -> None:
    train_count = exported["train"]
    val_count = exported["val"]
    base_model = "cyto3"
    print()
    print("=" * 72)
    print("Cellpose training bundle ready.")
    print("=" * 72)
    print(f"  Train images : {train_count}")
    print(f"  Val images   : {val_count}")
    print(f"  Output       : {out_dir.resolve()}")
    print()
    print("Next steps")
    print("-" * 72)
    print(f"1. Zip the export and upload to RunPod (or scp):")
    print(f"   powershell> Compress-Archive -Path {out_dir}\\train,{out_dir}\\val -DestinationPath cellpose_data.zip")
    print()
    print("2. On the RunPod GPU pod (recommended template: PyTorch 2.x + CUDA 12.x),")
    print("   open a terminal and run:")
    print()
    print("       pip install 'cellpose[gui]'")
    print("       unzip cellpose_data.zip -d cellpose_data")
    print("       python -m cellpose --train \\")
    print("           --dir cellpose_data/train \\")
    print("           --test_dir cellpose_data/val \\")
    print(f"           --pretrained_model {base_model} \\")
    print("           --mask_filter _masks \\")
    print("           --chan 0 --chan2 0 \\")
    print("           --learning_rate 0.1 \\")
    print("           --weight_decay 0.0001 \\")
    print("           --n_epochs 200 \\")
    print("           --batch_size 8 \\")
    print("           --use_gpu \\")
    print("           --verbose")
    print()
    print("   The trained model lands in cellpose_data/train/models/")
    print()
    print("3. Download the model file back to this machine, e.g.:")
    print("       outputs/cellpose_training/trained_model")
    print()
    print("4. Sanity-check on a held-out crop or full image (CPU is fine for inference):")
    print("       python -m cellpose \\")
    print("           --dir <some_images_dir> \\")
    print("           --pretrained_model outputs/cellpose_training/trained_model \\")
    print("           --chan 0 --chan2 0 --save_png --no_npy --verbose")
    print()
    print("Tips")
    print("-" * 72)
    print(" - If training loss plateaus, try --learning_rate 0.05.")
    print(" - If val IoU is low, double the epochs first; small datasets need more passes.")
    print(" - To watch GPU live on the pod: `watch -n 1 nvidia-smi`.")
    print()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellpose-export",
        description="Export human-edited crops + masks as Cellpose training data.",
    )
    parser.add_argument(
        "--annot-root",
        default="outputs/annot",
        help="Annotation root containing per-sample manifests.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/cellpose_training",
        help="Destination directory for the training bundle.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Fraction of edited crops held out for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="RNG seed used for the train/val split.",
    )
    parser.add_argument(
        "--positive-label",
        action="append",
        dest="positive_labels",
        default=None,
        help="Label value(s) treated as positive. Repeat for multiple. Default: single_valid.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Also include edited rows with an empty label. Off by default.",
    )

    args = parser.parse_args(argv)

    annot_root = Path(args.annot_root)
    out_dir = Path(args.out_dir)
    if not annot_root.exists():
        raise SystemExit(f"--annot-root does not exist: {annot_root}")

    positive_labels = tuple(args.positive_labels) if args.positive_labels else POSITIVE_LABELS_DEFAULT

    result = export_for_cellpose(
        annot_root=annot_root,
        out_dir=out_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
        positive_labels=positive_labels,
        require_positive=not args.include_unlabeled,
    )

    exported = result["exported"]
    print(f"Considered {result['total_rows_considered']} edited rows.")
    print(f"Wrote train={exported['train']} val={exported['val']} skipped={exported['skipped']}")
    print(f"Summary: {result['summary_csv']}")
    _print_runpod_instructions(out_dir, exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
