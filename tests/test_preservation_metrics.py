"""Metric behavior on disappearance, shifted outlines, gaps and missing labels."""
import importlib.util
import unittest

import numpy as np

from utils.preservation_metrics import (boundary_counts,boundary_score,cldice_counts,
    cldice_score,evaluate_masks,roi_box,roi_psnr,small_regions)


class PreservationMetricTests(unittest.TestCase):
    def test_boundary_tolerance_and_missing_target(self):
        gt=np.zeros((40,50),bool);gt[10:25,10:25]=True
        shifted=np.zeros_like(gt);shifted[10:25,12:27]=True
        valid=np.ones_like(gt)
        self.assertEqual(boundary_score(boundary_counts(shifted,gt,valid,2))['f1'],1.)
        self.assertLess(boundary_score(boundary_counts(shifted,gt,valid,0))['f1'],1.)
        self.assertEqual(boundary_score(boundary_counts(np.zeros_like(gt),gt,valid))['f1'],0.)
        with self.assertRaises(ValueError):boundary_counts(gt,gt,valid,-1)

    def test_void_contours_and_frame_exterior_are_not_targets(self):
        valid=np.ones((40,40),bool);valid[:,:12]=False
        gt=np.ones_like(valid);gt[:,:12]=False
        pred=np.ones_like(valid)
        c=boundary_counts(pred,gt,valid)
        self.assertEqual(c['target'],0)
        self.assertEqual(c['predicted'],0)
        self.assertIsNone(boundary_score(c)['f1'])

    def test_cldice_gap_penalty_and_empty_support(self):
        gt=np.zeros((11,31),bool);gt[5,2:29]=True
        pred=gt.copy();pred[5,14:17]=False
        # These one-pixel lines already are exact hard skeletons.
        c=cldice_counts(pred,gt,np.ones_like(gt),gt_skeleton=gt,pred_skeleton=pred)
        s=cldice_score(c)
        self.assertEqual(s['topology_precision'],1.)
        self.assertAlmostEqual(s['topology_recall'],24/27)
        self.assertAlmostEqual(s['cldice'],48/51)
        empty=np.zeros_like(gt)
        self.assertEqual(cldice_score(cldice_counts(empty,gt,np.ones_like(gt),gt_skeleton=gt,pred_skeleton=empty))['cldice'],0.)
        self.assertIsNone(cldice_score(cldice_counts(empty,empty,np.ones_like(gt),gt_skeleton=empty,pred_skeleton=empty))['cldice'])

    def test_small_regions_area_thresholds_and_crop_geometry(self):
        gt=np.ones((100,100),np.uint8)
        gt[10:14,10:14]=4 # area 16, eligible
        gt[30:33,30:35]=4 # area 15, excluded
        gt[50:61,50:61]=4 # area 121 >1%, excluded
        rows,mapping=small_regions(gt)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['area'],16)
        self.assertEqual(int((mapping>0).sum()),16)
        self.assertEqual(rows[0]['crop_xyxy'],[0,0,32,32])
        self.assertEqual(roi_box([98,98,100,100],100,100),[68,68,100,100])
        with self.assertRaises(ValueError):roi_box([-1,0,2,2],10,10)

    def test_native_crop_and_masked_psnr_distinguish_context(self):
        gt=np.zeros((40,40,3),np.uint8);pred=gt.copy();pred[10:14,10:14]=255
        mask=np.zeros((40,40),bool);mask[10:14,10:14]=True
        box=[0,0,32,32]
        object_score=roi_psnr(gt,pred,box,mask)
        context_score=roi_psnr(gt,pred,box)
        self.assertEqual(object_score['psnr'],0.)
        self.assertAlmostEqual(context_score['psnr'],10*np.log10(64))
        self.assertEqual(roi_psnr(gt,gt,box)['psnr'],120.)
        with self.assertRaises(ValueError):roi_psnr(gt,pred,box,np.zeros_like(mask))

    @unittest.skipUnless(importlib.util.find_spec('skimage'),'Hard skeleton test runs in the existing VGGT environment')
    def test_perfect_labels_and_disappearance(self):
        gt=np.ones((100,100),np.uint8)
        gt[10:14,10:14]=4;gt[50:55,50:55]=4
        full=evaluate_masks([gt],[gt])
        self.assertEqual(full['small_object_recall'],1.)
        self.assertEqual(full['boundary_f1'],1.)
        self.assertEqual(full['cldice'],1.)
        pred=gt.copy();pred[10:14,10:14]=1
        missing=evaluate_masks([pred],[gt])
        self.assertEqual((missing['small_tp'],missing['small_gt']),(1,2))
        self.assertEqual(missing['small_object_recall'],.5)
        self.assertLess(missing['cldice'],1.)
        with self.assertRaises(ValueError):evaluate_masks([pred],[])


if __name__=='__main__':unittest.main()
