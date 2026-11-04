"""Report guards for missing values, semantic definitions and exact label IDs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT=Path('/home/zhangys/Prune-Wisely-Reconstruct-Sharply')
spec=importlib.util.spec_from_file_location('finalize_city_benchmark',str(ROOT/'finalize_city_benchmark.py'))
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class CityBenchmarkReportTests(unittest.TestCase):
    def test_missing_metrics_stay_missing(self):
        values=report.metric_values(None,None)
        self.assertTrue(all(value is None for value in values.values()))
        self.assertEqual(report.number(None,'mIoU'),'—')

    def test_region_recall_not_pixel_accuracy(self):
        values=report.metric_values({'PSNR':30,'SSIM':.9,'LPIPS':.1},
                {'miou':.7,'macc':.8,'b_miou':.6,'region_recall':{'recall':.25}})
        self.assertEqual(values['Recall'],.25)
        self.assertEqual(values['mAcc'],.8)
        self.assertEqual(report.number(values['Recall'],'Recall'),'25.00')

    def test_percentage_point_delta(self):
        self.assertEqual(report.number(.0123,'mIoU',delta=True),'+1.23')
        self.assertEqual(report.number(-.01,'LPIPS',delta=True),'-0.01000')
        self.assertEqual(report.number(.2,'PSNR',delta=True),'+0.2000')

    def test_palette_retains_native_label_ids(self):
        labels=np.arange(7,dtype=np.uint8)[None].repeat(3,axis=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'labels_palette.png'
            report.palette_image(labels).save(str(path))
            self.assertTrue(np.array_equal(np.asarray(Image.open(path)),labels))
            self.assertTrue(np.array_equal(np.asarray(Image.open(path).convert('RGB')),
                             np.asarray(report.PALETTE,dtype=np.uint8)[labels]))

    def test_pending_semantics_requires_explicit_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(FileNotFoundError):
                report.semantic_bundle(root,False)
            self.assertEqual(report.semantic_bundle(root,True)['results'],{})

    def test_region_breakdown_keeps_independent_denominators(self):
        row=dict(label='example',semantic_metrics=dict(region_recall=dict(tp=175,gt=283,recall=175./283,
                 small_tp=148,small_gt=248,small_recall=148./248,vehicle_tp=45,vehicle_gt=59,vehicle_recall=45./59)))
        detail=report.region_details(row)
        self.assertEqual((detail['overall_tp'],detail['overall_gt']),(175,283))
        self.assertEqual((detail['small_tp'],detail['small_gt']),(148,248))
        self.assertEqual((detail['vehicle_tp'],detail['vehicle_gt']),(45,59))
        rendered='\n'.join(report.region_table([row]))
        for value in ('175/283','148/248','45/59'):
            self.assertIn(value,rendered)


if __name__=='__main__':
    unittest.main()
