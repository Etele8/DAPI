from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.annotation_review import (
    MANIFEST_FIELDNAMES,
    build_annotation_manifest_from_images,
    collect_annotation_manifests,
    ensure_annotation_manifests,
    load_annotation_rows,
    save_annotation_rows,
)
from src.annotation_manifest import write_annotation_manifest
from src.config import AnnotationExportConfig
from src.config import AnnotationReviewConfig
from src.crop_export import CropRecord


WORKSPACE_TMP = Path("test_tmp")


def _case_dir(name: str) -> Path:
    path = WORKSPACE_TMP / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_image(path: Path) -> None:
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Could not encode test image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


class AnnotationReviewTests(unittest.TestCase):
    def test_collects_manifests_recursively(self) -> None:
        case_dir = _case_dir("annotation_collect")
        manifest_a = case_dir / "sample_a" / "annotation" / "annotation_manifest.csv"
        manifest_b = case_dir / "sample_b" / "annotation" / "annotation_manifest.csv"
        manifest_a.parent.mkdir(parents=True, exist_ok=True)
        manifest_b.parent.mkdir(parents=True, exist_ok=True)
        manifest_a.write_text("crop_id,image_id,candidate_id,crop_path,overlay_path,label,notes\n", encoding="utf-8")
        manifest_b.write_text("crop_id,image_id,candidate_id,crop_path,overlay_path,label,notes\n", encoding="utf-8")

        manifests = collect_annotation_manifests(case_dir, manifest_name="annotation_manifest.csv", recursive=True)
        self.assertEqual(manifests, [manifest_a, manifest_b])

    def test_builds_ad_hoc_manifest_from_images(self) -> None:
        case_dir = _case_dir("annotation_ad_hoc")
        image_path = case_dir / "nested" / "crop_001.png"
        _write_image(image_path)

        config = AnnotationReviewConfig()
        manifest_path = case_dir / config.manifest_name
        built = build_annotation_manifest_from_images(case_dir, manifest_path=manifest_path, config=config)
        rows, fieldnames = load_annotation_rows(built)

        self.assertEqual(built, manifest_path)
        self.assertEqual(fieldnames, MANIFEST_FIELDNAMES)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["crop_path"], "nested/crop_001.png")
        self.assertEqual(rows[0]["overlay_path"], "nested/crop_001.png")
        self.assertEqual(rows[0]["mask_path"], "")
        self.assertEqual(rows[0]["edited_mask_path"], "")
        self.assertEqual(rows[0]["mask_was_edited"], "false")
        self.assertEqual(rows[0]["label"], "")

    def test_ensure_annotation_manifests_falls_back_to_images(self) -> None:
        case_dir = _case_dir("annotation_ensure")
        _write_image(case_dir / "crop_a.png")

        manifests = ensure_annotation_manifests(case_dir, config=AnnotationReviewConfig())
        self.assertEqual(len(manifests), 1)
        self.assertTrue(manifests[0].exists())

    def test_save_annotation_rows_persists_label_updates(self) -> None:
        case_dir = _case_dir("annotation_save")
        manifest_path = case_dir / "annotation_manifest.csv"
        rows = [
            {
                "crop_id": "img_c0001",
                "image_id": "img",
                "candidate_id": "1",
                "crop_path": "crops/images/img_c0001.png",
                "overlay_path": "crops/overlays/img_c0001.png",
                "label": "",
                "notes": "",
            }
        ]
        save_annotation_rows(manifest_path, rows, MANIFEST_FIELDNAMES)

        stored_rows, fieldnames = load_annotation_rows(manifest_path)
        stored_rows[0]["label"] = "single_valid"
        save_annotation_rows(manifest_path, stored_rows, fieldnames)
        reloaded_rows, _ = load_annotation_rows(manifest_path)

        self.assertEqual(reloaded_rows[0]["label"], "single_valid")

    def test_load_annotation_rows_backfills_new_mask_fields(self) -> None:
        case_dir = _case_dir("annotation_backfill")
        manifest_path = case_dir / "annotation_manifest.csv"
        manifest_path.write_text(
            "crop_id,image_id,candidate_id,crop_path,overlay_path,label,notes\n"
            "img_c0001,img,1,crops/images/img_c0001.png,crops/overlays/img_c0001.png,,\n",
            encoding="utf-8",
        )
        rows, fieldnames = load_annotation_rows(manifest_path)

        self.assertIn("mask_path", fieldnames)
        self.assertIn("edited_mask_path", fieldnames)
        self.assertIn("mask_was_edited", fieldnames)
        self.assertIn("mask_edit_mode", fieldnames)
        self.assertEqual(rows[0]["mask_path"], "")
        self.assertEqual(rows[0]["edited_mask_path"], "")
        self.assertEqual(rows[0]["mask_was_edited"], "")
        self.assertEqual(rows[0]["mask_edit_mode"], "")

    def test_annotation_manifest_includes_mask_columns(self) -> None:
        case_dir = _case_dir("annotation_manifest_schema")
        record = CropRecord(
            crop_id="img_c0001",
            image_id="img",
            candidate_id=1,
            source_bbox=(1, 2, 3, 4),
            crop_bbox=(0, 0, 16, 16),
            centroid_xy=(8.0, 8.0),
            area_px=42,
            touches_border=False,
            qc_flag="",
            profile_name="proposal_high_recall",
            crop_path="crops/images/img_c0001.png",
            overlay_path="crops/overlays/img_c0001.png",
            mask_path="crops/masks/img_c0001.png",
        )

        files = write_annotation_manifest([record], output_dir=case_dir, config=AnnotationExportConfig())
        rows, _ = load_annotation_rows(files["annotation_manifest"])
        self.assertEqual(rows[0]["mask_path"], "crops/masks/img_c0001.png")
        self.assertEqual(rows[0]["edited_mask_path"], "")
        self.assertEqual(rows[0]["mask_was_edited"], "false")
        self.assertEqual(rows[0]["mask_edit_mode"], "")


if __name__ == "__main__":
    unittest.main()
