from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.annotation_review import annotate_manifest, annotate_root
from src.classifier_dataset import prepare_classifier_dataset
from src.config import apply_cli_overrides, available_profiles, default_config
from src.io.discover import discover_samples
from src.io.loader import load_sample
from src.pipeline import (
    run_proposal_pipeline_for_sample,
    run_refinement_from_manifest,
    run_segmentation_for_sample,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dapi-segment",
        description="Run the evidence-based DAPI segmentation, proposal, refinement, and annotation workflows.",
    )
    parser.add_argument(
        "--mode",
        choices=("segment", "proposals", "prepare-dataset", "refine", "annotate"),
        default="segment",
        help="Pipeline mode to run.",
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
        help="Directory where pipeline outputs are written.",
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
        help="Bundled parameter profile name to load from src/profiles.",
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
        help="Microscope calibration for biomass conversion.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Annotation manifest path used by prepare-dataset and refine modes.",
    )
    parser.add_argument(
        "--run-biomass",
        action="store_true",
        help="When refining positive crops, also run biomass on the refined masks.",
    )
    parser.add_argument(
        "--annotation-root",
        default=None,
        help="Root directory for recursive annotation manifest or crop-image discovery in annotate mode.",
    )
    parser.add_argument(
        "--include-labeled",
        action="store_true",
        help="In annotate mode, review already labeled rows as well instead of only unlabeled rows.",
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


def _resolve_profile(mode: str, profile: str, profile_file: str | None) -> str:
    if profile_file is not None:
        return profile
    if mode in {"proposals", "refine"} and profile == "default":
        return "proposal_high_recall"
    return profile


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_profiles:
        for name in available_profiles():
            print(name)
        return 0

    profile_name = _resolve_profile(args.mode, args.profile, args.profile_file)
    data_roots = args.data_roots or ["data/samples"]
    cfg = apply_cli_overrides(
        default_config(data_roots, profile_name=profile_name, profile_path=args.profile_file),
        microns_per_pixel=args.microns_per_pixel,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "prepare-dataset":
        if args.manifest is None:
            raise SystemExit("--manifest is required for --mode prepare-dataset")
        files = prepare_classifier_dataset(args.manifest, output_dir=output_dir, config=cfg.proposal.dataset)
        print(f"prepared_dataset manifest={args.manifest} output={files['summary_json']}")
        return 0

    if args.mode == "refine":
        if args.manifest is None:
            raise SystemExit("--manifest is required for --mode refine")
        result = run_refinement_from_manifest(args.manifest, output_dir, app_config=cfg, run_biomass=args.run_biomass)
        print(
            f"refined_crops={len(result.refined_records)} metadata={result.files['refined_metadata_csv']}"
        )
        return 0

    if args.mode == "annotate":
        if args.manifest is not None:
            session = annotate_manifest(args.manifest, config=cfg.proposal.review, include_labeled=args.include_labeled)
            print(
                f"manifest={session.manifest_path} rows_in_scope={session.rows_in_scope} "
                f"label_updates={session.rows_labeled} remaining_unlabeled={session.remaining_unlabeled} "
                f"stopped_early={session.stopped_early}"
            )
            return 0

        annotation_root = args.annotation_root or args.output_dir
        result = annotate_root(annotation_root, config=cfg.proposal.review, include_labeled=args.include_labeled)
        print(
            f"annotation_root={Path(annotation_root)} manifests={result.manifests_processed} "
            f"label_updates={result.rows_labeled} remaining_unlabeled={result.remaining_unlabeled} "
            f"stopped_early={result.stopped_early}"
        )
        return 0

    discovered = discover_samples(cfg.discovery)
    discovered_by_stem = {sample.stem: sample for sample in discovered}
    selected_stems = _select_samples(args.sample, args.limit, [sample.stem for sample in discovered])

    for stem in selected_stems:
        sample = discovered_by_stem[stem]
        loaded = load_sample(sample)

        if args.mode == "proposals":
            result = run_proposal_pipeline_for_sample(loaded, output_dir, app_config=cfg)
            export_count = 0 if result.crop_export is None else len(result.crop_export.records)
            print(
                f"{stem}: proposals={len(result.proposals.candidates)} exported_crops={export_count} "
                f"annotation_manifest={result.annotation_files.get('annotation_manifest', 'n/a')}"
            )
            continue

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
