"""Sweep cellprob_threshold for a Cellpose model and visualize the effect.

Lower cellprob_threshold = more permissive = picks up fainter cells (at the
risk of eventually adding background noise). For each image in --dir, this runs
the model at several thresholds, counts detected instances, and writes an
outline overlay per (image, threshold). You then pick the lowest threshold that
still only adds real faint cells and not junk.

Run it on the GPU pod for speed (seconds per image); CPU works but each
full-image inference is slow.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.mask_overlay import overlay_outlines


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
_MASK_MARKERS = ("_masks", "_cp_masks", "_overlay")


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Encode failed for {path}")
    encoded.tofile(path)


def _list_source_images(image_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_EXTS:
            continue
        if any(marker in path.stem for marker in _MASK_MARKERS):
            continue
        out.append(path)
    return out


def _parse_thresholds(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def sweep(
    *,
    model_path: str,
    image_dir: Path,
    thresholds: Sequence[float],
    output_dir: Path,
    use_gpu: bool,
    flow_threshold: float | None,
    color: tuple[int, int, int],
) -> dict[str, dict[float, int]]:
    from cellpose import models

    print(f"Loading model: {model_path} (gpu={use_gpu})")
    model = models.CellposeModel(pretrained_model=model_path, gpu=use_gpu)

    image_paths = _list_source_images(image_dir)
    if not image_paths:
        raise SystemExit(f"No source images found in {image_dir}")
    print(f"Sweeping {len(thresholds)} thresholds over {len(image_paths)} images")

    eval_kwargs: dict[str, object] = {}
    if flow_threshold is not None:
        eval_kwargs["flow_threshold"] = flow_threshold

    counts: dict[str, dict[float, int]] = {}
    for path in image_paths:
        stem = path.stem
        image = _read_image(path)
        if image is None:
            print(f"  SKIP {stem}: could not read")
            continue
        counts[stem] = {}
        for t in thresholds:
            output = model.eval(image, cellprob_threshold=t, **eval_kwargs)
            masks = output[0] if isinstance(output, tuple) else output
            n = int(np.asarray(masks).max())
            counts[stem][t] = n

            overlay = overlay_outlines(image, np.asarray(masks), color=color, enhance=False)
            _write_image(output_dir / f"thr_{t:g}" / f"{stem}_overlay.png", overlay)
        trend = "  ".join(f"{t:g}:{counts[stem][t]}" for t in thresholds)
        print(f"  {stem}: {trend}")

    # counts table CSV
    csv_path = output_dir / "sweep_counts.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", *[f"thr_{t:g}" for t in thresholds]])
        for stem, by_t in counts.items():
            writer.writerow([stem, *[by_t[t] for t in thresholds]])

    print()
    print("Count table (rows=images, cols=thresholds):")
    header = "image".ljust(22) + "".join(f"{t:>8g}" for t in thresholds)
    print(header)
    for stem, by_t in counts.items():
        print(stem.ljust(22) + "".join(f"{by_t[t]:>8d}" for t in thresholds))
    print()
    print(f"Overlays per threshold: {output_dir.resolve()}/thr_<t>/")
    print(f"Counts CSV: {csv_path.resolve()}")
    print()
    print("Pick the lowest threshold where counts stop climbing steeply and the")
    print("overlays don't show new specks in empty background. That's your value")
    print("for `--cellprob_threshold` at inference time.")
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellpose-sweep",
        description="Sweep cellprob_threshold for a Cellpose model; count cells and write overlays.",
    )
    parser.add_argument("--model", required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--dir", required=True, help="Directory of source images to run on.")
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=[0.0, -1.0, -2.0, -3.0],
        help="Comma-separated cellprob_threshold values. Default: 0,-1,-2,-3.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cellpose_sweep",
        help="Where to write per-threshold overlays + sweep_counts.csv.",
    )
    parser.add_argument(
        "--flow-threshold",
        type=float,
        default=None,
        help="Optional flow_threshold override. Default: Cellpose default.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU. Default uses GPU if available.",
    )
    parser.add_argument(
        "--color",
        default="0,0,255",
        help="Outline color B,G,R for overlays. Default red.",
    )

    args = parser.parse_args(argv)
    color_parts = [int(x) for x in args.color.split(",")]
    if len(color_parts) != 3:
        raise SystemExit("--color must be B,G,R, e.g. 0,0,255")

    image_dir = Path(args.dir)
    if not image_dir.exists():
        raise SystemExit(f"--dir does not exist: {image_dir}")

    sweep(
        model_path=args.model,
        image_dir=image_dir,
        thresholds=args.thresholds,
        output_dir=Path(args.output_dir),
        use_gpu=not args.cpu,
        flow_threshold=args.flow_threshold,
        color=(color_parts[0], color_parts[1], color_parts[2]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
