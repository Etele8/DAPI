"""Small task runner that wraps the most common workflows.

Examples:

    python tasks.py test
    python tasks.py list-profiles
    python tasks.py segment Tv143
    python tasks.py proposals Tv143
    python tasks.py annotate Tv143
    python tasks.py prepare-dataset Tv143
    python tasks.py refine Tv143 --run-biomass

Run with no args to see the full subcommand list. All paths default to the
conventional layout: data in `data/samples`, outputs in `outputs/<stage>`.
Override anything by passing flags after the subcommand — they are forwarded
to the underlying CLI verbatim.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

DATA_ROOT = "data/samples"
OUT_SEGMENT = "outputs/segment"   # param-tuning playground for full-image segmentation
OUT_ANNOT = "outputs/annot"       # human-in-the-loop pipeline root: proposals, manifests, classifier splits, refined masks
DEFAULT_PROFILE_SEGMENT = "default"
DEFAULT_PROFILE_PROPOSAL = "proposal_high_recall"


def _run(cmd: Sequence[str]) -> int:
    printable = " ".join(cmd)
    print(f"$ {printable}")
    return subprocess.call([sys.executable, *cmd])


def _segment_cmd(sample: str, extra: list[str]) -> list[str]:
    return [
        "-m", "src.cli",
        "--mode", "segment",
        "--profile", DEFAULT_PROFILE_SEGMENT,
        "--data-root", DATA_ROOT,
        "--sample", sample,
        "--output-dir", OUT_SEGMENT,
        *extra,
    ]


def _proposals_cmd(sample: str, extra: list[str]) -> list[str]:
    return [
        "-m", "src.cli",
        "--mode", "proposals",
        "--profile", DEFAULT_PROFILE_PROPOSAL,
        "--data-root", DATA_ROOT,
        "--sample", sample,
        "--output-dir", OUT_ANNOT,
        *extra,
    ]


def _annotate_cmd(sample: str | None, extra: list[str]) -> list[str]:
    if sample is None:
        return ["-m", "src.annotation_cli", "--root", OUT_ANNOT, *extra]
    manifest = Path(OUT_ANNOT) / sample / "annotation" / "annotation_manifest.csv"
    return ["-m", "src.annotation_cli", "--manifest", str(manifest), *extra]


def _prepare_dataset_cmd(sample: str, extra: list[str]) -> list[str]:
    manifest = Path(OUT_ANNOT) / sample / "annotation" / "annotation_manifest.csv"
    return [
        "-m", "src.cli",
        "--mode", "prepare-dataset",
        "--profile", DEFAULT_PROFILE_PROPOSAL,
        "--manifest", str(manifest),
        "--output-dir", OUT_ANNOT,
        *extra,
    ]


def _refine_cmd(sample: str, extra: list[str]) -> list[str]:
    manifest = Path(OUT_ANNOT) / sample / "annotation" / "annotation_manifest.csv"
    return [
        "-m", "src.cli",
        "--mode", "refine",
        "--profile", DEFAULT_PROFILE_PROPOSAL,
        "--manifest", str(manifest),
        "--output-dir", OUT_ANNOT,
        *extra,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tasks.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="task", required=True)

    sub.add_parser("test", help="Run the test suite via pytest.")
    sub.add_parser("list-profiles", help="List bundled segmentation profiles.")
    sub.add_parser(
        "cellpose-export",
        help="Pack human-edited crops + masks into a Cellpose training bundle.",
    )
    sub.add_parser(
        "cellpose-eval",
        help="Score a trained Cellpose model on annotated crops, write visualizations.",
    )
    sub.add_parser(
        "cellpose-reconstruct",
        help="Rebuild full-image instance masks from per-crop annotations (seeds for Cellpose GUI).",
    )
    sub.add_parser(
        "cellpose-pack",
        help="Package Cellpose-GUI-annotated full images into a train/val bundle.",
    )
    sub.add_parser(
        "overlay",
        help="Overlay instance-mask outlines on the original image for visual QC.",
    )
    sub.add_parser(
        "cellpose-sweep",
        help="Sweep cellprob_threshold for a model; count cells + write overlays per threshold.",
    )

    def _add_sample_cmd(name: str, help_: str, required: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_)
        if required:
            p.add_argument("sample", help="Sample stem, e.g. Tv143.")
        else:
            p.add_argument("sample", nargs="?", default=None, help="Sample stem; omit to scan the whole proposals root.")
        return p

    _add_sample_cmd("segment", "Run full-image segmentation on a sample.")
    _add_sample_cmd("proposals", "Run proposal generation + crop export on a sample.")
    _add_sample_cmd("annotate", "Open the annotation tool on a sample's manifest.", required=False)
    _add_sample_cmd("prepare-dataset", "Prepare the classifier dataset from a sample's manifest.")
    _add_sample_cmd("refine", "Refine positively-labeled crops for a sample.")

    args, extra = parser.parse_known_args(argv)

    if args.task == "test":
        return subprocess.call([sys.executable, "-m", "pytest", *extra])
    if args.task == "list-profiles":
        return _run(["-m", "src.cli", "--list-profiles", *extra])
    if args.task == "cellpose-export":
        return _run([
            "-m", "src.cellpose_export",
            "--annot-root", OUT_ANNOT,
            "--out-dir", "outputs/cellpose_training",
            *extra,
        ])
    if args.task == "cellpose-eval":
        return _run([
            "-m", "src.cellpose_eval",
            *extra,
        ])
    if args.task == "cellpose-reconstruct":
        return _run([
            "-m", "src.cellpose_reconstruct",
            "--annot-root", OUT_ANNOT,
            "--samples-root", DATA_ROOT,
            "--out-dir", "outputs/cellpose_fullimage_seed",
            *extra,
        ])
    if args.task == "cellpose-pack":
        return _run([
            "-m", "src.cellpose_pack",
            "--seed-dir", "outputs/cellpose_fullimage_seed",
            "--out-dir", "outputs/cellpose_fullimage_train",
            *extra,
        ])
    if args.task == "overlay":
        return _run([
            "-m", "src.mask_overlay",
            *extra,
        ])
    if args.task == "cellpose-sweep":
        return _run([
            "-m", "src.cellpose_sweep",
            *extra,
        ])
    if args.task == "segment":
        return _run(_segment_cmd(args.sample, extra))
    if args.task == "proposals":
        return _run(_proposals_cmd(args.sample, extra))
    if args.task == "annotate":
        return _run(_annotate_cmd(args.sample, extra))
    if args.task == "prepare-dataset":
        return _run(_prepare_dataset_cmd(args.sample, extra))
    if args.task == "refine":
        return _run(_refine_cmd(args.sample, extra))

    parser.error(f"Unknown task: {args.task}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
