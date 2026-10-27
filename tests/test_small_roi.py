import unittest

import numpy as np

from evaluate_small_roi import masked_metrics, summarize


class SmallRoiMetricTests(unittest.TestCase):
    def test_missing_tiny_region_is_not_diluted_by_background(self):
        reference = np.zeros((20, 20, 3), dtype=np.float32)
        reference[5:7, 5:7] = 1
        missing = np.zeros_like(reference)
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:7, 5:7] = True
        roi = masked_metrics(reference, missing, mask)
        full = masked_metrics(reference, missing, np.ones_like(mask))
        self.assertAlmostEqual(roi["psnr"], 0.)
        self.assertAlmostEqual(full["psnr"], 20.)
        self.assertGreater(masked_metrics(reference, reference, mask)["psnr"], roi["psnr"])

    def test_region_equal_weight_and_empty_summary(self):
        rows = [dict(models={"x": dict(mse=.01, l1=.1, psnr=20, boundary_gradient_l1=None)}),
                dict(models={"x": dict(mse=.1, l1=.2, psnr=10, boundary_gradient_l1=.2)})]
        self.assertEqual(summarize(rows, ["x"])["x"]["mean_psnr"], 15.)
        self.assertIsNone(summarize([], ["x"])["x"]["mean_psnr"])

    def test_reject_empty_or_wrong_resolution(self):
        rgb = np.zeros((2, 2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            masked_metrics(rgb, rgb, np.zeros((2, 2), dtype=bool))
        with self.assertRaises(ValueError):
            masked_metrics(rgb, rgb[:1], np.ones((2, 2), dtype=bool))


if __name__ == "__main__":
    unittest.main()
