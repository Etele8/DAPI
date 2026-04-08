from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.config import apply_cli_overrides, available_profiles, default_config
from src.io.discover import discover_samples
from src.io.loader import load_sample
from src.pipeline import run_segmentation_for_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dapi-segment",
        description="Run the evidence-based DAPI cell segmentation pipeline.",
    )
    parser.add_argument(
        "--data-root",
        action="append",
        dest="data_roots",
        default=None,
        help="Input root to scan for samples. Repeat to use multiple roots. Defaults to data/samples.",
    )
    parser.add_argument(
        "--sample",
        help="Process only one discovered sample stem, for example Tv143.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where per-sample debug folders are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N discovered samples after filtering.",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Bundled parameter profile name to load from src/profiles, for example default.",
    )
    parser.add_argument(
        "--profile-file",
        default=None,
        help="Path to a custom TOML profile file. Overrides --profile when provided.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print bundled profile names and exit.",
    )
    parser.add_argument(
        "--microns-per-pixel",
        type=float,
        default=None,
        help="Microscope calibration for biomass conversion. If omitted, physical-unit columns are written as NaN and flagged.",
    )
    return parser


def _select_samples(sample_stem: str | None, limit: int | None, stems: Sequence[str]) -> list[str]:
    selected = list(stems)
    if sample_stem is not None:
        matches = [stem for stem in selected if stem == sample_stem]
        if not matches:
            available = ", ".join(selected[:10])
            raise SystemExit(
                f"Sample '{sample_stem}' was not found. First discovered samples: {available}"
            )
        selected = matches

    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be >= 1")
        selected = selected[:limit]

    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_profiles:
        for name in available_profiles():
            print(name)
        return 0

    data_roots = args.data_roots or ["data/samples"]
    cfg = apply_cli_overrides(
        default_config(data_roots, profile_name=args.profile, profile_path=args.profile_file),
        microns_per_pixel=args.microns_per_pixel,
    )
    discovered = discover_samples(cfg.discovery)
    discovered_by_stem = {sample.stem: sample for sample in discovered}
    selected_stems = _select_samples(args.sample, args.limit, [sample.stem for sample in discovered])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stem in selected_stems:
        sample = discovered_by_stem[stem]
        loaded = load_sample(sample)
        result = run_segmentation_for_sample(
            loaded,
            output_dir,
            config=cfg.segmentation,
            biomass_config=cfg.biomass,
        )
        accepted_count = sum(region.accepted for region in result.segmentation.regions)
        kept_biomass = result.biomass.summary.n_objects_kept if result.biomass is not None else 0
        print(
            f"{stem}: accepted_regions={accepted_count} biomass_kept={kept_biomass} "
            f"seg_dir={output_dir / f'{stem}_seg'} biomass_dir={output_dir / f'{stem}_biomass'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
