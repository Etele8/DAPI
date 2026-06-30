"""Overlay instance-mask outlines on the original image for visual QC.

Draws a 1-px boundary around every instance in a label image on top of the
source image. This is the right view for judging boundary accuracy: you see
the cell and the predicted edge as a thin line, rather than a filled blob that
hides the boundary. Pass --enhance for a contrast stretch on very dark images.

Pairs files in a directory by stem: for each <stem>.png that is NOT a mask,
it looks for <stem><mask-suffix>.{png,tif,tiff} as the label image. Masks may
live in a separate directory via --masks-dir.

Defaults assume the Cellpose CLI convention (<stem>_cp_masks.png), but
--mask-suffix lets you overlay ground truth (_masks), reconstructed seeds, or
Cellpose-GUI annotations saved as <stem>_seg.npy (use --mask-suffix _seg).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_MASK_EXTS = (".npy", ".png", ".tif", ".tiff")
# Cellpose-generated artefacts that should never be treated as source images.
_ARTEFACT_MARKERS = ("_cp_outlines", "_cp_flows", "_flows", "_overlay")


def _load_npy_masks(path: Path) -> np.ndarray | None:
    """Pull the instance-label array out of a Cellpose GUI _seg.npy file."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception:
        return None
    if data.dtype == object:
        try:
            payload = data.item()
        except Exception:
            return None
        if isinstance(payload, dict):
            for key in ("masks", "Mask", "labels"):
                value = payload.get(key)
                if isinstance(value, np.ndarray):
                    return value
        return None
    return data if data.ndim == 2 else None


def _load_mask_any(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".npy":
        return _load_npy_masks(path)
    return _read_image(path, flags=cv2.IMREAD_UNCHANGED)


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    if not path.exists():
        return None
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Encode failed for {path}")
    encoded.tofile(path)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image.copy()


def _enhance_contrast(image_bgr: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.5) -> np.ndarray:
    """Percentile contrast stretch so dim DAPI cells become visible."""
    out = np.zeros_like(image_bgr)
    for c in range(image_bgr.shape[2]):
        channel = image_bgr[:, :, c].astype(np.float32)
        lo = float(np.percentile(channel, low_pct))
        hi = float(np.percentile(channel, high_pct))
        if hi <= lo:
            out[:, :, c] = image_bgr[:, :, c]
            continue
        stretched = np.clip((channel - lo) / (hi - lo), 0.0, 1.0) * 255.0
        out[:, :, c] = stretched.astype(np.uint8)
    return out


def _instance_boundaries(labels: np.ndarray) -> np.ndarray:
    """Boolean map of 1-px borders: a foreground pixel that differs from a 4-neighbour."""
    lbl = labels.astype(np.int32)
    boundary = np.zeros(lbl.shape, dtype=bool)
    boundary[:-1, :] |= lbl[:-1, :] != lbl[1:, :]
    boundary[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    boundary[:, :-1] |= lbl[:, :-1] != lbl[:, 1:]
    boundary[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    return boundary & (lbl > 0)


def overlay_outlines(
    image_bgr: np.ndarray,
    labels: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    enhance: bool = True,
) -> np.ndarray:
    base = _to_bgr(image_bgr)
    if enhance:
        base = _enhance_contrast(base)
    if labels.shape[:2] != base.shape[:2]:
        labels = cv2.resize(
            labels.astype(np.int32),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    boundary = _instance_boundaries(labels)
    out = base.copy()
    out[boundary] = color
    return out


def _find_mask(masks_dir: Path, stem: str, mask_suffix: str) -> Path | None:
    for ext in _MASK_EXTS:
        candidate = masks_dir / f"{stem}{mask_suffix}{ext}"
        if candidate.exists():
            return candidate
    return None


def process_dir(
    *,
    images_dir: Path,
    masks_dir: Path,
    mask_suffix: str,
    output_dir: Path,
    color: tuple[int, int, int],
    enhance: bool,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        stem = image_path.stem
        if stem.endswith(mask_suffix) or any(marker in stem for marker in _ARTEFACT_MARKERS):
            continue
        mask_path = _find_mask(masks_dir, stem, mask_suffix)
        if mask_path is None:
            continue
        image = _read_image(image_path)
        labels = _load_mask_any(mask_path)
        if image is None or labels is None:
            print(f"  SKIP {stem}: could not read image or mask")
            continue
        overlay = overlay_outlines(image, labels, color=color, enhance=enhance)
        out_path = output_dir / f"{stem}_overlay.png"
        _write_image(out_path, overlay)
        n_instances = int(np.asarray(labels).max())
        print(f"  {stem}: {n_instances} instances -> {out_path.name}")
        n_written += 1
    return n_written


def _parse_color(text: str) -> tuple[int, int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("color must be 'B,G,R', e.g. 0,255,255")
    return parts[0], parts[1], parts[2]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mask-overlay",
        description="Overlay instance-mask outlines on the original image for visual QC.",
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory containing the source images (and masks, unless --masks-dir is given).",
    )
    parser.add_argument(
        "--masks-dir",
        default=None,
        help="Directory containing the mask label images. Defaults to --dir.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_cp_masks",
        help="Suffix identifying mask files. Examples: _cp_masks (Cellpose CLI output), "
        "_masks (ground-truth .tif), _seg (Cellpose GUI .npy annotations).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write <stem>_overlay.png. Defaults to <dir>/overlays.",
    )
    parser.add_argument(
        "--color",
        type=_parse_color,
        default=(0, 0, 255),
        help="Outline color as B,G,R. Default 0,0,255 (red) — high contrast on blue/green fluorescence.",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Apply a percentile contrast stretch to the source image. Off by default; "
        "useful for very dark images, but tends to amplify background noise on fluorescence.",
    )

    args = parser.parse_args(argv)

    images_dir = Path(args.dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else images_dir
    output_dir = Path(args.output_dir) if args.output_dir else images_dir / "overlays"
    if not images_dir.exists():
        raise SystemExit(f"--dir does not exist: {images_dir}")
    if not masks_dir.exists():
        raise SystemExit(f"--masks-dir does not exist: {masks_dir}")

    print(f"Overlaying masks ('{args.mask_suffix}') onto images in {images_dir}")
    n = process_dir(
        images_dir=images_dir,
        masks_dir=masks_dir,
        mask_suffix=args.mask_suffix,
        output_dir=output_dir,
        color=args.color,
        enhance=args.enhance,
    )
    if n == 0:
        raise SystemExit(
            f"No image/mask pairs found. Looked for <stem>{args.mask_suffix}.<ext> in {masks_dir}."
        )
    print(f"\nWrote {n} overlays to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
