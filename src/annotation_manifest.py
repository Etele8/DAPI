from __future__ import annotations

from pathlib import Path
import csv
import json

from src.config import AnnotationExportConfig
from src.crop_export import CropRecord


def write_annotation_manifest(
    records: list[CropRecord],
    *,
    output_dir: str | Path,
    config: AnnotationExportConfig,
) -> dict[str, Path]:
    output_dir = Path(output_dir) / config.manifest_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / config.manifest_name
    label_schema_path = output_dir / "label_schema.json"

    fieldnames = [
        "crop_id",
        "image_id",
        "candidate_id",
        "crop_path",
        "overlay_path",
        "mask_path",
        "edited_mask_path",
        "mask_was_edited",
        "mask_edit_mode",
        "label",
        "notes",
    ]
    if config.include_helper_columns:
        fieldnames.extend(["touches_border", "area_px", "qc_flag", "profile_name"])

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "crop_id": record.crop_id,
                "image_id": record.image_id,
                "candidate_id": record.candidate_id,
                "crop_path": record.crop_path,
                "overlay_path": record.overlay_path,
                "mask_path": record.mask_path,
                "edited_mask_path": "",
                "mask_was_edited": "false",
                "mask_edit_mode": "",
                "label": "",
                "notes": "",
            }
            if config.include_helper_columns:
                row.update(
                    {
                        "touches_border": record.touches_border,
                        "area_px": record.area_px,
                        "qc_flag": record.qc_flag,
                        "profile_name": record.profile_name,
                    }
                )
            writer.writerow(row)

    label_schema_path.write_text(
        json.dumps(
            {
                "label_options": list(config.label_options),
                "notes": "Blank label means not yet annotated. Add additional labels if the workflow grows.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"annotation_manifest": manifest_path, "label_schema": label_schema_path}
