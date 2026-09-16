"""
Unit and integration tests for Pluto Adaptive 3D Image Segmentation.
"""

import os
import shutil
import unittest
import numpy as np
from PIL import Image

from Utils.utils import (
    load_image_stack,
    natural_sort_key,
    characterize_mask,
    propagate_slice,
    save_masks,
    fill_mask_holes,
)


class TestPlutoCore(unittest.TestCase):
    """Test suite for Pluto segmentation and stack utility functions."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join("data", "cropped_3_3_1")
        cls.test_masks_dir = os.path.join("masks", "test_suite_run")

    def tearDown(self):
        if os.path.isdir(self.test_masks_dir):
            shutil.rmtree(self.test_masks_dir)

    def test_01_natural_sort(self):
        """Test natural numerical sorting key."""
        raw = ["file10.tif", "file2.tif", "file1.tif", "file100.tif", "file20.tif"]
        sorted_files = sorted(raw, key=natural_sort_key)
        expected = ["file1.tif", "file2.tif", "file10.tif", "file20.tif", "file100.tif"]
        self.assertEqual(sorted_files, expected)

    def test_02_load_image_stack(self):
        """Test loading actual dataset data/cropped_3_3_1."""
        self.assertTrue(os.path.isdir(self.data_dir), "Sample dataset data/cropped_3_3_1 must exist.")
        stack, filenames, folder_name = load_image_stack(self.data_dir)

        self.assertEqual(folder_name, "cropped_3_3_1")
        self.assertEqual(len(filenames), 201)
        self.assertEqual(stack.shape, (201, 363, 438))
        self.assertEqual(stack.dtype, np.uint8)

        # Check numerical ordering
        self.assertEqual(filenames[0], "filtered03640000.tif")
        self.assertEqual(filenames[1], "filtered03640001.tif")
        self.assertEqual(filenames[200], "filtered03640200.tif")

    def test_03_characterize_mask(self):
        """Test mask statistical characterization."""
        img = np.array([
            [10, 10, 10, 10],
            [10, 50, 60, 10],
            [10, 70, 80, 10],
            [10, 10, 10, 10]
        ], dtype=np.uint8)

        mask = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]
        ], dtype=np.uint8)

        stats = characterize_mask(img, mask)
        self.assertTrue(stats["valid"])
        self.assertEqual(stats["yc"], 2)
        self.assertEqual(stats["xc"], 2)
        self.assertAlmostEqual(stats["mean"], 65.0, places=2)
        self.assertAlmostEqual(stats["std"], float(np.std([50, 60, 70, 80])), places=2)
        self.assertEqual(stats["area"], 4)

        # Test empty mask
        empty_mask = np.zeros_like(mask)
        empty_stats = characterize_mask(img, empty_mask)
        self.assertFalse(empty_stats["valid"])
        self.assertEqual(empty_stats["area"], 0)

    def test_04_propagate_slice_success(self):
        """Test slice propagation under normal homogeneous conditions."""
        stack, filenames, _ = load_image_stack(self.data_dir)

        # Slice 0 canalicular initial mask (around centroid yc=97, xc=117)
        mask0 = np.zeros(stack[0].shape, dtype=np.uint8)
        mask0[94:102, 114:122] = 1

        stats0 = characterize_mask(stack[0], mask0)
        self.assertTrue(stats0["valid"])

        # Propagate through slices 1 to 5
        curr_yc, curr_xc = stats0["yc"], stats0["xc"]
        curr_mean, curr_std = stats0["mean"], stats0["std"]

        for slice_idx in range(1, 6):
            res = propagate_slice(
                current_slice=stack[slice_idx],
                prev_yc=curr_yc,
                prev_xc=curr_xc,
                mean=curr_mean,
                std=curr_std,
                alpha=0.8,
                n_iterations_max=40,
                k_std=2.0
            )
            self.assertFalse(res["is_empty"], f"Slice {slice_idx} should not be empty")
            self.assertGreater(res["area"], 10)
            self.assertIsNotNone(res["yc"])
            self.assertIsNotNone(res["xc"])

            curr_yc, curr_xc = res["yc"], res["xc"]
            curr_mean, curr_std = res["mean"], res["std"]

    def test_05_propagate_empty_mask_handling(self):
        """Test empty mask detection when seed is invalid or area fails to grow."""
        slice_img = np.full((100, 100), 50, dtype=np.uint8)

        # 1. Out of bounds seed
        res_oob = propagate_slice(slice_img, prev_yc=999, prev_xc=999, mean=50.0, std=5.0)
        self.assertTrue(res_oob["is_empty"])
        self.assertIn("out of bounds", res_oob["reason"])

        # 2. Seed in pixel where intensity differs drastically from mean
        res_mismatch = propagate_slice(slice_img, prev_yc=50, prev_xc=50, mean=200.0, std=2.0)
        self.assertTrue(res_mismatch["is_empty"])
        self.assertIn("failed to grow", res_mismatch["reason"])

    def test_06_save_masks_strictly_binary_png(self):
        """Test saving masks strictly as binary PNGs (0 and 255)."""
        stack, filenames, folder_name = load_image_stack(self.data_dir)

        test_masks = {
            0: np.zeros((363, 438), dtype=np.uint8),
            1: np.ones((363, 438), dtype=np.uint8),
        }
        test_masks[0][50:70, 50:70] = 1

        saved_paths = save_masks(
            masks_dict=test_masks,
            folder_name="test_suite_run",
            filenames=filenames,
            base_dir="masks"
        )

        self.assertEqual(len(saved_paths), 2)
        for p in saved_paths:
            self.assertTrue(os.path.isfile(p))
            self.assertTrue(p.endswith(".png"))
            with Image.open(p) as im:
                self.assertEqual(im.format, "PNG")
                arr = np.array(im)
                unique_vals = set(np.unique(arr))
                # Must contain strictly binary values: subset of {0, 255}
                self.assertTrue(unique_vals.issubset({0, 255}))

    def test_07_recovery_go_back_and_resume(self):
        """Test the workflow of going back n slices, re-characterizing, and resuming."""
        stack, filenames, _ = load_image_stack(self.data_dir)

        # Slice 0 initial mask
        mask0 = np.zeros(stack[0].shape, dtype=np.uint8)
        mask0[95:102, 115:122] = 1
        stats0 = characterize_mask(stack[0], mask0)

        masks = {0: mask0}
        curr_yc, curr_xc = stats0["yc"], stats0["xc"]
        curr_mean, curr_std = stats0["mean"], stats0["std"]

        # Propagate through slices 1 to 3
        for s in range(1, 4):
            res = propagate_slice(stack[s], curr_yc, curr_xc, curr_mean, curr_std)
            masks[s] = res["mask"]
            curr_yc, curr_xc = res["yc"], res["xc"]
            curr_mean, curr_std = res["mean"], res["std"]

        self.assertIn(3, masks)

        # Simulate empty mask on slice 4, user chooses go back n=2 (to slice 2)
        failed_slice = 4
        n_back = 2
        target_slice = failed_slice - n_back  # Slice 2
        self.assertEqual(target_slice, 2)

        # Discard masks after target_slice
        for s in list(masks.keys()):
            if s > target_slice:
                del masks[s]

        self.assertNotIn(3, masks)
        self.assertIn(2, masks)

        # Draw / touch up mask on target_slice
        new_stats = characterize_mask(stack[target_slice], masks[target_slice])
        self.assertTrue(new_stats["valid"])

        # Resume propagation to slice 3 and 4
        res_resumed = propagate_slice(
            stack[target_slice + 1],
            new_stats["yc"],
            new_stats["xc"],
            new_stats["mean"],
            new_stats["std"]
        )
        self.assertFalse(res_resumed["is_empty"])
        masks[target_slice + 1] = res_resumed["mask"]
        self.assertIn(3, masks)


if __name__ == "__main__":
    unittest.main()

