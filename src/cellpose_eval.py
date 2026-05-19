"""Run a Cellpose model on annotated crops and inspect the results.

For each <crop_id>.png / <crop_id>_masks.tif pair under --data-dir, loads the
image, runs the model, compares the prediction to the ground-truth mask, and
writes a side-by-side visualization (input | ground truth | prediction).

Pass --compare-baseline to also run base `cpsam` on the same data and add a
fourth panel to each visualization, so fine-tuned vs out-of-the-box quality
is visible at a glance. A summary CSV and aggregate IoU stats are printed at
the end.

Designed to run on a CPU laptop; cpsam is a 4 GB transformer, so the first
load takes ~30 s and per-crop inference takes a few seconds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class EvalRecord:
    crop_id: str
    iou_finetuned: float
    iou_baseline: float | None
    n_instances_finetuned: int
    n_instances_baseline: int | None


def _read_image(path: Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Could not decode {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Encode failed for {path}")
    encoded.tofile(path)


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(intersection / union) if union > 0 else 0.0


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image.copy()


def _overlay_mask(base_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    over = base_bgr.copy()
    over[mask > 0] = color
    return cv2.addWeighted(base_bgr, 0.6, over, 0.4, 0.0)


def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(panel, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _stitch_horizontal(panels: list[np.ndarray]) -> np.ndarray:
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    out = []
    for p in panels:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[: p.shape[0], : p.shape[1]] = p
        out.append(canvas)
    return np.hstack(out)


def _list_pairs(data_dir: Path, mask_suffix: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(data_dir.iterdir()):
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        stem = img_path.stem
        if stem.endswith(mask_suffix):
            continue
        for ext in (".tif", ".tiff", ".png"):
            candidate = data_dir / f"{stem}{mask_suffix}{ext}"
            if candidate.exists():
                pairs.append((img_path, candidate))
                break
    return pairs


def _run_model(model, image: np.ndarray) -> np.ndarray:
    """Cellpose v4 returns (masks, flows, styles); we only need masks."""
    output = model.eval(image)
    masks = output[0] if isinstance(output, tuple) else output
    return masks


def evaluate(
    *,
    model_path: str,
    data_dir: Path,
    output_dir: Path,
    compare_baseline: bool,
    mask_suffix: str,
) -> dict:
    from cellpose import models

    print(f"Loading fine-tuned model: {model_path}")
    main_model = models.CellposeModel(pretrained_model=model_path, gpu=False)

    baseline_model = None
    if compare_baseline:
        print("Loading baseline: cpsam")
        baseline_model = models.CellposeModel(pretrained_model="cpsam", gpu=False)

    pairs = _list_pairs(data_dir, mask_suffix=mask_suffix)
    if not pairs:
        raise SystemExit(f"No image/mask pairs found under {data_dir}")
    print(f"Found {len(pairs)} pairs in {data_dir}")

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    records: list[EvalRecord] = []
    for idx, (img_path, mask_path) in enumerate(pairs):
        image = _read_image(img_path)
        gt_mask = _read_image(mask_path, flags=cv2.IMREAD_UNCHANGED)
        gt_binary = gt_mask > 0

        pred_mask = _run_model(main_model, image)
        pred_binary = pred_mask > 0
        iou_ft = _binary_iou(pred_binary, gt_binary)

        iou_bl: float | None = None
        pred_bl: np.ndarray | None = None
        if baseline_model is not None:
            pred_bl = _run_model(baseline_model, image)
            iou_bl = _binary_iou(pred_bl > 0, gt_binary)

        records.append(
            EvalRecord(
                crop_id=img_path.stem,
                iou_finetuned=iou_ft,
                iou_baseline=iou_bl,
                n_instances_finetuned=int(pred_mask.max()),
                n_instances_baseline=int(pred_bl.max()) if pred_bl is not None else None,
            )
        )

        base_bgr = _to_bgr(image)
        panels = [
            _label_panel(base_bgr, "input"),
            _label_panel(_overlay_mask(base_bgr, gt_binary, (0, 220, 0)), "ground truth"),
            _label_panel(_overlay_mask(base_bgr, pred_binary, (40, 200, 255)), f"finetuned  IoU={iou_ft:.2f}"),
        ]
        if pred_bl is not None and iou_bl is not None:
            panels.append(
                _label_panel(
                    _overlay_mask(base_bgr, pred_bl > 0, (200, 80, 255)),
                    f"baseline  IoU={iou_bl:.2f}",
                )
            )
        _write_image(viz_dir / f"{img_path.stem}.png", _stitch_horizontal(panels))

        if iou_bl is None:
            print(f"  [{idx+1:3d}/{len(pairs)}] {img_path.stem}: IoU={iou_ft:.3f}")
        else:
            marker = "+" if iou_ft > iou_bl + 0.02 else ("-" if iou_ft < iou_bl - 0.02 else "=")
            print(f"  [{idx+1:3d}/{len(pairs)}] {img_path.stem}: ft={iou_ft:.3f}  baseline={iou_bl:.3f}  {marker}")

    csv_path = output_dir / "eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "crop_id",
                "iou_finetuned",
                "iou_baseline",
                "n_instances_finetuned",
                "n_instances_baseline",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    "crop_id": r.crop_id,
                    "iou_finetuned": f"{r.iou_finetuned:.4f}",
                    "iou_baseline": "" if r.iou_baseline is None else f"{r.iou_baseline:.4f}",
                    "n_instances_finetuned": r.n_instances_finetuned,
                    "n_instances_baseline": "" if r.n_instances_baseline is None else r.n_instances_baseline,
                }
            )

    ious_ft = np.array([r.iou_finetuned for r in records])
    print()
    print("=" * 60)
    print(f"Fine-tuned model across {len(records)} crops:")
    print(f"  mean IoU       : {ious_ft.mean():.3f}")
    print(f"  median IoU     : {float(np.median(ious_ft)):.3f}")
    print(f"  fraction > 0.7 : {(ious_ft > 0.7).mean() * 100:.0f}%")
    print(f"  fraction > 0.8 : {(ious_ft > 0.8).mean() * 100:.0f}%")
    print(f"  fraction > 0.9 : {(ious_ft > 0.9).mean() * 100:.0f}%")

    if compare_baseline:
        ious_bl = np.array([r.iou_baseline for r in records if r.iou_baseline is not None])
        delta = ious_ft - ious_bl
        wins_ft = int((delta > 0.05).sum())
        wins_bl = int((delta < -0.05).sum())
        ties = len(delta) - wins_ft - wins_bl
        print()
        print(f"Baseline cpsam across {len(records)} crops:")
        print(f"  mean IoU       : {ious_bl.mean():.3f}")
        print(f"  median IoU     : {float(np.median(ious_bl)):.3f}")
        print()
        print("Per-crop comparison (>5% IoU change = a win):")
        print(f"  fine-tuned wins : {wins_ft}")
        print(f"  baseline wins   : {wins_bl}")
        print(f"  ties            : {ties}")
        print(f"  mean delta      : {delta.mean():+.3f}")

    print()
    print(f"Visualizations: {viz_dir.resolve()}")
    print(f"CSV:            {csv_path.resolve()}")
    return {"records": records, "csv": csv_path, "viz_dir": viz_dir}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cellpose-eval",
        description="Score a Cellpose model on annotated crops and write side-by-side visualizations.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to a fine-tuned checkpoint, or a built-in name like 'cpsam' or 'cyto3'.",
    )
    parser.add_argument(
        "--data-dir",
        default="outputs/cellpose_training/val",
        help="Directory containing <crop>.png + <crop>_masks.tif pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cellpose_eval",
        help="Where to write per-crop visualizations and eval_results.csv.",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run the base cpsam model and add a fourth comparison panel per crop.",
    )
    parser.add_argument(
        "--mask-suffix",
        default="_masks",
        help="Filename suffix that identifies the mask file paired with each image.",
    )

    args = parser.parse_args(argv)

    evaluate(
        model_path=args.model,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        compare_baseline=args.compare_baseline,
        mask_suffix=args.mask_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
