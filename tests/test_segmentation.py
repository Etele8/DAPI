from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.segmentation.segment import SegmentationConfig, save_debug_outputs, segment_cells


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

        with tempfile.TemporaryDirectory() as tmpdir:
            files = save_debug_outputs(Path(tmpdir), image, result)
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


if __name__ == "__main__":
    unittest.main()
