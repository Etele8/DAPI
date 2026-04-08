from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import random

from src.config import ClassifierDatasetConfig


@dataclass(slots=True)
class ClassifierExample:
    split: str
    crop_id: str
    image_id: str
    candidate_id: str
    crop_path: str
    overlay_path: str
    label: str
    class_label: str
    notes: str


def ingest_annotation_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _normalize_label(label: str) -> str:
    return label.strip()


def _class_label(label: str, config: ClassifierDatasetConfig) -> str | None:
    normalized = _normalize_label(label)
    if not normalized:
        return None
    if normalized in config.excluded_labels:
        return None
    if normalized in config.positive_labels:
        return "positive"
    if normalized in config.negative_labels:
        return "negative"
    return normalized


def prepare_classifier_dataset(
    manifest_path: str | Path,
    *,
    output_dir: str | Path,
    config: ClassifierDatasetConfig,
) -> dict[str, Path]:
    rows = ingest_annotation_manifest(manifest_path)
    included = []
    for row in rows:
        mapped = _class_label(row.get("label", ""), config)
        if mapped is None:
            continue
        row_copy = dict(row)
        row_copy["class_label"] = mapped
        included.append(row_copy)

    grouped: dict[str, list[dict[str, str]]] = {}
    group_field = "crop_id" if config.allow_image_leakage else "image_id"
    for row in included:
        grouped.setdefault(row[group_field], []).append(row)

    group_keys = list(grouped)
    random.Random(config.split_seed).shuffle(group_keys)
    total = len(group_keys)
    n_train = int(round(total * config.train_fraction))
    n_val = int(round(total * config.val_fraction))
    n_train = min(n_train, total)
    n_val = min(n_val, max(total - n_train, 0))
    train_keys = set(group_keys[:n_train])
    val_keys = set(group_keys[n_train : n_train + n_val])
    test_keys = set(group_keys[n_train + n_val :])

    split_rows: dict[str, list[ClassifierExample]] = {"train": [], "val": [], "test": []}
    for key, group_rows in grouped.items():
        if key in train_keys:
            split_name = "train"
        elif key in val_keys:
            split_name = "val"
        else:
            split_name = "test"

        for row in group_rows:
            split_rows[split_name].append(
                ClassifierExample(
                    split=split_name,
                    crop_id=row["crop_id"],
                    image_id=row["image_id"],
                    candidate_id=row["candidate_id"],
                    crop_path=row["crop_path"],
                    overlay_path=row.get("overlay_path", ""),
                    label=row["label"],
                    class_label=row["class_label"],
                    notes=row.get("notes", ""),
                )
            )

    output_dir = Path(output_dir) / config.split_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    for split_name, examples in split_rows.items():
        path = output_dir / f"{split_name}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "split",
                    "crop_id",
                    "image_id",
                    "candidate_id",
                    "crop_path",
                    "overlay_path",
                    "label",
                    "class_label",
                    "notes",
                ],
            )
            writer.writeheader()
            for example in examples:
                writer.writerow(asdict(example))
        files[f"{split_name}_csv"] = path

    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "allow_image_leakage": config.allow_image_leakage,
                "counts": {split: len(items) for split, items in split_rows.items()},
                "groups": {split: sorted({item.image_id for item in items}) for split, items in split_rows.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    files["summary_json"] = summary_path
    return files
