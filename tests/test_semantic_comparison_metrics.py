import unittest

import numpy as np

from utils.semantic_comparison_metrics import (
    _connected_regions, _maximum_cardinality_matching,
    evaluate_semantics, inner_boundary, semantic_metrics,
)


class SemanticComparisonMetricsTests(unittest.TestCase):
    def scene(self):
        gt = np.full((100, 100), 1, dtype=np.uint8)
        gt[10:20, 10:20] = 4
        gt[40:60, 60:80] = 2
        return gt

    def test_perfect_maps_all_metrics_and_small_vehicle(self):
        gt = self.scene()
        out = evaluate_semantics([gt.copy()], [gt])
        self.assertEqual(out['miou'], 1.0)
        self.assertEqual(out['macc'], 1.0)
        self.assertEqual(out['b_miou'], 1.0)
        self.assertEqual(out['region_recall']['recall'], 1.0)
        self.assertEqual(out['region_recall']['vehicle_gt'], 1)
        self.assertEqual(out['region_recall']['vehicle_recall'], 1.0)
        self.assertEqual(out['region_recall']['small_gt'], 1)
        self.assertEqual(out['region_recall']['small_recall'], 1.0)
        self.assertEqual(out['per_image'][0]['boundary_dilation_px'], 3)

    def test_confusion_is_dataset_aggregated_not_image_mean(self):
        gt_a = np.ones((10, 10), dtype=np.uint8)
        gt_b = np.ones((20, 20), dtype=np.uint8)
        pred_a = np.full_like(gt_a, 2)
        pred_b = gt_b.copy()
        out = evaluate_semantics([pred_a, pred_b], [gt_a, gt_b])
        self.assertEqual(out['confusion'][1][1], 400)
        self.assertEqual(out['confusion'][1][2], 100)
        self.assertAlmostEqual(out['macc'], 0.8)
        # False positive class 2 has no GT but has positive union, so contributes 0 IoU.
        self.assertAlmostEqual(out['miou'], 0.4)
        self.assertIsNone(out['per_class']['2']['pixel_recall'])
        self.assertEqual(out['per_class']['2']['iou'], 0.0)

    def test_predicting_void_is_a_false_negative(self):
        gt = self.scene()
        pred = gt.copy()
        pred[gt == 4] = 0
        out = evaluate_semantics(pred, gt)
        self.assertEqual(out['confusion'][4][0], 100)
        self.assertEqual(out['per_class']['4']['iou'], 0.0)
        self.assertEqual(out['per_class']['4']['pixel_recall'], 0.0)
        self.assertEqual(out['region_recall']['vehicle_recall'], 0.0)

    def test_void_gt_ignored_for_all_three_domains(self):
        gt = np.ones((80, 80), dtype=np.uint8)
        gt[:, :10] = 0
        pred = gt.copy()
        pred[:, :10] = 4
        out = evaluate_semantics(pred, gt)
        self.assertEqual(out['miou'], 1.0)
        self.assertEqual(out['macc'], 1.0)
        self.assertEqual(out['b_miou'], 1.0)
        self.assertEqual(out['confusion'][0], [0] * 7)
        self.assertEqual(out['region_recall']['vehicle_gt'], 0)
        self.assertEqual(out['per_class']['4']['regions']['pred_components'], 0)

    def test_void_boundary_neighborhood_excluded(self):
        gt = np.ones((80, 80), dtype=np.uint8)
        gt[:, :20] = 0
        pred = gt.copy()
        # The spurious class-2 band at the unknown edge is outside the valid domain.
        # The displaced class-1 inner band can still extend beyond d, legitimately.
        pred[:, 20:22] = 2
        out = evaluate_semantics(pred, gt, boundary_dilation_ratio=0.04)
        self.assertGreater(out['b_miou'], 0.8)
        self.assertLess(out['b_miou'], 1.0)
        self.assertLess(out['miou'], 1.0)
        self.assertEqual(out['per_class']['2']['boundary_union'], 0)

    def test_image_border_is_still_a_boundary_without_void(self):
        gt = np.ones((8, 8), dtype=np.uint8)
        out = evaluate_semantics(gt, gt)
        self.assertEqual(out['per_image'][0]['boundary_valid_pixels'], 64)
        self.assertEqual(out['per_class']['1']['boundary_union'], 28)
        self.assertEqual(int(inner_boundary(gt == 1, 1).sum()), 28)

    def test_native_boundary_width_formula(self):
        gt = np.ones((894, 1592), dtype=np.uint8)
        out = evaluate_semantics(gt, gt)
        self.assertEqual(out['per_image'][0]['boundary_dilation_px'], 37)

    def test_shift_degrades_boundary_and_semantic_metrics(self):
        gt = self.scene()
        pred = np.roll(gt, 4, axis=1)
        out = evaluate_semantics(pred, gt)
        self.assertLess(out['miou'], 1.0)
        self.assertLess(out['b_miou'], 1.0)
        self.assertLess(out['region_recall']['vehicle_recall'], 1.0)

    def test_wrong_class_cannot_match_region(self):
        gt = self.scene()
        pred = gt.copy()
        pred[gt == 4] = 5
        out = evaluate_semantics(pred, gt)
        self.assertEqual(out['region_recall']['vehicle_recall'], 0.0)
        self.assertEqual(out['per_class']['5']['iou'], 0.0)

    def test_minimum_gt_region_area_and_8_connectivity(self):
        gt = np.zeros((30, 30), dtype=bool)
        gt[1:5, 1:5] = True
        gt[5:9, 5:9] = True  # diagonal connection joins the two 16 px patches
        gt[20:23, 20:23] = True  # nine pixels are below fixed GT minimum
        out = _connected_regions(gt, gt, 16, 100, 0.5)
        self.assertEqual(out['gt'], 1)
        self.assertEqual(out['tp'], 1)
        self.assertEqual(out['small_gt'], 1)

    def test_split_prediction_is_one_recalled_gt(self):
        gt = np.zeros((50, 50), dtype=bool)
        gt[10:20, 10:30] = True
        pred = gt.copy()
        pred[10:20, 25] = False  # 150px and40px components, 200px GT
        out = _connected_regions(pred, gt, 16, 1000, 0.5)
        self.assertEqual(out['pred_components'], 2)
        self.assertEqual(out['gt'], 1)
        self.assertEqual(out['tp'], 1)

    def test_merged_prediction_cannot_recall_both_regions(self):
        gt = np.zeros((50, 60), dtype=bool)
        gt[10:20, 5:25] = True
        gt[10:20, 30:40] = True
        pred = gt.copy()
        pred[14, 25:30] = True  # same-class bridge merges both prediction regions
        out = _connected_regions(pred, gt, 16, 1000, 0.5)
        self.assertEqual(out['gt'], 2)
        self.assertEqual(out['pred_components'], 1)
        self.assertEqual(out['tp'], 1)

    def test_matching_uses_augmenting_path_not_greedy(self):
        matching = _maximum_cardinality_matching({1: [1, 2], 2: [1]})
        self.assertEqual(len(matching), 2)
        self.assertEqual(matching, {1: 2, 2: 1})

    def test_all_void_is_undefined_not_perfect(self):
        gt = np.zeros((20, 20), dtype=np.uint8)
        out = evaluate_semantics(np.full_like(gt, 2), gt)
        self.assertIsNone(out['miou'])
        self.assertIsNone(out['macc'])
        self.assertIsNone(out['b_miou'])
        self.assertIsNone(out['region_recall']['recall'])

    def test_validation_and_alias(self):
        gt = self.scene()
        self.assertEqual(semantic_metrics(gt, gt)['miou'], 1.0)
        for pred in (gt.astype(float), gt[:, :5], np.full_like(gt, 7)):
            with self.assertRaises(ValueError):
                evaluate_semantics(pred, gt)
        with self.assertRaises(ValueError):
            evaluate_semantics([], [])
        with self.assertRaises(ValueError):
            evaluate_semantics([gt], [gt, gt])


if __name__ == '__main__':
    unittest.main()
