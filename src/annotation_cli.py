from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.annotation_review import annotate_manifest, annotate_root
from src.config import available_profiles, default_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dapi-annotate",
        description="Interactively annotate exported crop images or recursively discovered image folders.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Root directory to scan recursively for annotation manifests or images.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Single annotation manifest to review.",
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
        "--include-labeled",
        action="store_true",
        help="Review already labeled rows as well instead of only unlabeled rows.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print bundled profile names and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_profiles:
        for name in available_profiles():
            print(name)
        return 0

    if args.manifest is None and args.root is None:
        raise SystemExit("Provide either --manifest or --root")

    cfg = default_config(["data/samples"], profile_name=args.profile, profile_path=args.profile_file)
    if args.manifest is not None:
        result = annotate_manifest(args.manifest, config=cfg.proposal.review, include_labeled=args.include_labeled)
        print(
            f"manifest={result.manifest_path} rows_in_scope={result.rows_in_scope} "
            f"label_updates={result.rows_labeled} mask_updates={result.masks_saved} "
            f"remaining_unlabeled={result.remaining_unlabeled} "
            f"stopped_early={result.stopped_early}"
        )
        return 0

    run_result = annotate_root(args.root, config=cfg.proposal.review, include_labeled=args.include_labeled)
    print(
        f"annotation_root={Path(args.root)} manifests={run_result.manifests_processed} "
        f"label_updates={run_result.rows_labeled} mask_updates={run_result.masks_saved} "
        f"remaining_unlabeled={run_result.remaining_unlabeled} "
        f"stopped_early={run_result.stopped_early}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
