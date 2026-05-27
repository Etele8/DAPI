"""Package Cellpose-GUI-annotated full images into a train/val bundle.

Cellpose GUI saves an `<image>_seg.npy` next to each image you annotate. This
reads those, extracts the final instance masks, and writes clean
`<image>.png` + `<image>_masks.tif` pairs into train/ and val/ subdirectories,
ready for `cellpose --train --mask_filter _masks`.

Only images with a `_seg.npy` are included (i.e. ones you actually annotated);
seed-only images are ignored. The split is stratified by instance count so
that both dense and sparse images appear in train and val.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class AnnotatedImage:
    stem: str               # e.g. "Image_24969"
    image_path: Path        # source .png
    seg_path: Path          # <stem>_seg.npy
    n_instances: int


def _load_masks_from_seg(seg_path: Path) -> np.ndarray:
    data = np.load(seg_path, allow_pickle=True).item()
    masks = data["masks"]
    return np.asarray(masks)


def _find_annotated_images(seed_dir: Path) -> list[AnnotatedImage]:
    out: list[AnnotatedImage] = []
    for seg_path in sorted(seed_dir.glob("*_seg.npy")):
        stem = seg_path.name[: -len("_seg.npy")]
        image_path = seed_dir / f"{stem}.png"
        if not image_path.exists():
            print(f"  SKIP {stem}: no matching {image_path.name}")
            continue
        masks = _load_masks_from_seg(seg_path)
        out.append(
            AnnotatedImage(
                stem=stem,
                image_path=image_path,
                seg_path=seg_path,
                n_instances=int(masks.max()),
            )
        )
    return out


def _stratified_split(
    images: list[AnnotatedImage],
    *,
    val_fraction: float,
    dense_threshold: int,
    seed: int,
) -> tuple[list[AnnotatedImage], list[AnnotatedImage]]:
    rng = random.Random(seed)
    dense = [im for im in images if im.n_instances >= dense_threshold]
    sparse = [im for im in images if im.n_instances < dense_threshold]

    def _split(group: list[AnnotatedImage]) -> tuple[list[AnnotatedImage], list[AnnotatedImage]]:
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_val = max(int(round(len(shuffled) * val_fraction)), 1) if shuffled else 0
        return shuffled[n_val:], shuffled[:n_val]

    train_dense, val_dense = _split(dense)
    train_sparse, val_sparse = _split(sparse)
    return train_dense + train_sparse, val_dense + val_sparse


def _write_pair(dest_dir: Path, image: AnnotatedImage) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image.image_path, dest_dir / f"{image.stem}.png")
    masks = _load_masks_from_seg(image.seg_path).astype(np.uint16)
    mask_path = dest_dir / f"{image.stem}_masks.tif"
    ok, encoded = cv2.imencode(".tif", masks)
    if not ok:
        raise ValueError(f"Failed to encode mask for {image.stem}")
    encoded.tofile(mask_path)


def pack(
    *,
    seed_dir: Path,
    out_dir: Path,
    val_fraction: float,
    dense_threshold: int,
    seed: int,
) -> dict:
    images = _find_annotated_images(seed_dir)
    if not images:
        raise SystemExit(f"No *_seg.npy annotations found under {seed_dir}")

    train_images, val_images = _stratified_split(
        images,
        val_fraction=val_fraction,
        dense_threshold=dense_threshold,
        seed=seed,
    )

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    for image in train_images:
        _write_pair(train_dir, image)
    for image in val_images:
        _write_pair(val_dir, image)

    summary_path = out_dir / "pack_summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "stem", "n_instances"])
        writer.writeheader()
        for split_name, group in (("train", train_images), ("val", val_images)):
            for image in sorted(group, key=lambda im: im.stem):
                writer.writerow({"split": split_name, "stem": image.stem, "n_instances": image.n_instances})

    return {
        "train": train_images,
        "val": val_images,
        "summary_csv": summary_path,
        "train_dir": train_dir,
        "val_dir": val_dir,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellpose-pack",
        description="Package Cellpose-GUI-annotated full images into a train/val bundle.",
    )
    parser.add_argument(
        "--seed-dir",
        default="outputs/cellpose_fullimage_seed",
        help="Directory containing <image>.png + <image>_seg.npy from Cellpose GUI.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/cellpose_fullimage_train",
        help="Destination for the train/ and val/ bundle.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of images held out for validation (within each density stratum).",
    )
    parser.add_argument(
        "--dense-threshold",
        type=int,
        default=150,
        help="Images with >= this many instances are stratified as 'dense'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="RNG seed for the split.",
    )

    args = parser.parse_args(argv)

    seed_dir = Path(args.seed_dir)
    if not seed_dir.exists():
        raise SystemExit(f"--seed-dir does not exist: {seed_dir}")

    result = pack(
        seed_dir=seed_dir,
        out_dir=Path(args.out_dir),
        val_fraction=args.val_fraction,
        dense_threshold=args.dense_threshold,
        seed=args.seed,
    )

    train_images = result["train"]
    val_images = result["val"]
    total_train_inst = sum(im.n_instances for im in train_images)
    total_val_inst = sum(im.n_instances for im in val_images)

    print(f"Packed into {Path(args.out_dir).resolve()}")
    print(f"  train: {len(train_images)} images, {total_train_inst} instances")
    print(f"  val  : {len(val_images)} images, {total_val_inst} instances")
    print(f"  summary: {result['summary_csv']}")
    print()
    print("Val images:", ", ".join(sorted(im.stem for im in val_images)))
    print()
    print("Train on a GPU pod (full images -> min_train_masks back to default):")
    print("  bash /opt/runpod_train.sh <data_dir> --n_epochs 100 --min_train_masks 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
