from __future__ import annotations

import csv
import shutil
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.annotation_manifest import write_annotation_manifest
from src.classifier_dataset import prepare_classifier_dataset
from src.config import ProposalWorkflowConfig, SegmentationConfig
from src.crop_export import export_candidate_crops
from src.local_refinement import refine_positive_crops
from src.proposal_generation import generate_proposals, write_candidate_records
from src.segmentation.segment import save_debug_outputs, segment_cells


WORKSPACE_TMP = Path("test_tmp")


def _case_dir(name: str) -> Path:
    path = WORKSPACE_TMP / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _make_synthetic_cells(background_bgr: tuple[int, int, int]) -> np.ndarray:
    image = np.full((128, 128, 3), background_bgr, dtype=np.uint8)
    centers = [(32, 36), (70, 48), (94, 88), (42, 94)]

    for center in centers:
        cv2.circle(image, center, 7, (220, 180, 140), -1)
        cv2.circle(image, center, 3, (255, 245, 235), -1)

    return image


class SegmentationPipelineTests(unittest.TestCase):
    def test_detects_cells_on_black_background(self) -> None:
        image = _make_synthetic_cells((8, 8, 8))
        result = segment_cells(image, config=SegmentationConfig())

        accepted = [region for region in result.regions if region.accepted]
        self.assertGreaterEqual(len(accepted), 4)
        self.assertLess(np.count_nonzero(result.filtered_mask), 900)

    def test_detects_cells_on_dark_green_background(self) -> None:
        image = _make_synthetic_cells((35, 70, 25))
        result = segment_cells(image, config=SegmentationConfig())

        accepted = [region for region in result.regions if region.accepted]
        self.assertGreaterEqual(len(accepted), 4)
        self.assertLess(np.count_nonzero(result.filtered_mask), 900)

    def test_writes_expected_debug_outputs(self) -> None:
        image = _make_synthetic_cells((15, 20, 12))
        result = segment_cells(image, config=SegmentationConfig())

        case_dir = _case_dir("debug_outputs")
        files = save_debug_outputs(case_dir, image, result)
        expected = {
            "original_image",
            "blue_dominance_map",
            "blob_enhanced_map",
            "local_suppression_map",
            "evidence_map",
            "binary_mask_raw",
            "binary_mask_clean",
            "filtered_mask",
            "labels",
            "overlay",
        }
        self.assertEqual(set(files.keys()), expected)
        for path in files.values():
            self.assertTrue(path.exists(), path)

    def test_proposal_exports_candidates_and_annotation_manifest(self) -> None:
        image = _make_synthetic_cells((10, 12, 10))
        proposal_cfg = ProposalWorkflowConfig()

        case_dir = _case_dir("proposal_exports")
        result = generate_proposals(
            image_id="sample_a",
            image_bgr=image,
            segmentation_config=SegmentationConfig(),
            filter_config=proposal_cfg.filter,
            profile_name="proposal_high_recall",
        )
        candidate_files = write_candidate_records(
            case_dir,
            result.candidates,
            csv_name=proposal_cfg.candidates_csv_name,
            jsonl_name=proposal_cfg.candidates_jsonl_name,
        )
        crops = export_candidate_crops(
            image_bgr=image,
            proposal_result=result,
            output_dir=case_dir,
            config=proposal_cfg.crop_export,
        )
        annotation_files = write_annotation_manifest(
            crops.records,
            output_dir=case_dir,
            config=proposal_cfg.annotation,
        )

        self.assertGreaterEqual(len(result.candidates), 4)
        self.assertGreaterEqual(len(crops.records), 4)
        self.assertTrue(candidate_files["candidates_csv"].exists())
        self.assertTrue(annotation_files["annotation_manifest"].exists())

    def test_dataset_split_is_image_grouped(self) -> None:
        proposal_cfg = ProposalWorkflowConfig()
        case_dir = _case_dir("dataset_split")
        manifest_path = case_dir / "annotation_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["crop_id", "image_id", "candidate_id", "crop_path", "overlay_path", "label", "notes"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "crop_id": "img1_c0001",
                    "image_id": "img1",
                    "candidate_id": "1",
                    "crop_path": "crops/images/img1_c0001.png",
                    "overlay_path": "crops/overlays/img1_c0001.png",
                    "label": "single_valid",
                    "notes": "",
                }
            )
            writer.writerow(
                {
                    "crop_id": "img1_c0002",
                    "image_id": "img1",
                    "candidate_id": "2",
                    "crop_path": "crops/images/img1_c0002.png",
                    "overlay_path": "crops/overlays/img1_c0002.png",
                    "label": "merged",
                    "notes": "",
                }
            )
            writer.writerow(
                {
                    "crop_id": "img2_c0001",
                    "image_id": "img2",
                    "candidate_id": "1",
                    "crop_path": "crops/images/img2_c0001.png",
                    "overlay_path": "crops/overlays/img2_c0001.png",
                    "label": "single_valid",
                    "notes": "",
                }
            )

        files = prepare_classifier_dataset(manifest_path, output_dir=case_dir, config=proposal_cfg.dataset)
        train_images: set[str] = set()
        val_images: set[str] = set()
        test_images: set[str] = set()
        split_sets = {"train_csv": train_images, "val_csv": val_images, "test_csv": test_images}
        for key, image_set in split_sets.items():
            with files[key].open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    image_set.add(row["image_id"])

        self.assertTrue(train_images.isdisjoint(val_images))
        self.assertTrue(train_images.isdisjoint(test_images))
        self.assertTrue(val_images.isdisjoint(test_images))

    def test_positive_crop_refinement_runs(self) -> None:
        image = _make_synthetic_cells((10, 12, 10))
        proposal_cfg = ProposalWorkflowConfig()

        case_dir = _case_dir("refinement")
        result = generate_proposals(
            image_id="sample_refine",
            image_bgr=image,
            segmentation_config=SegmentationConfig(),
            filter_config=proposal_cfg.filter,
            profile_name="proposal_high_recall",
        )
        crops = export_candidate_crops(
            image_bgr=image,
            proposal_result=result,
            output_dir=case_dir,
            config=proposal_cfg.crop_export,
        )
        annotation_files = write_annotation_manifest(
            crops.records,
            output_dir=case_dir,
            config=proposal_cfg.annotation,
        )

        manifest_path = annotation_files["annotation_manifest"]
        with manifest_path.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["label"] = "single_valid"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        refined_rows, files = refine_positive_crops(
            manifest_path,
            output_dir=case_dir,
            refinement_config=proposal_cfg.refinement,
            dataset_config=proposal_cfg.dataset,
        )
        self.assertEqual(len(refined_rows), 1)
        self.assertEqual(refined_rows[0].refinement_status, "refined")
        self.assertTrue(files["refined_metadata_csv"].exists())


if __name__ == "__main__":
    unittest.main()
