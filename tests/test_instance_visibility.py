import tempfile
import unittest
from pathlib import Path

import numpy as np

from audit_instance_visibility import active_counts, opacity_audit, read_ply_vertices
from make_small_roi_comparison import region_bbox, select_cases


class QualitativeSelectionTests(unittest.TestCase):
    def region(self, number, delta):
        return dict(frame="frame_%03d" % number, local_region=1,
                    models={"control": dict(psnr=30.), "protected": dict(psnr=30. + delta)})

    def test_includes_two_worst_and_median_without_duplicates(self):
        regions = [self.region(i, delta) for i, delta in enumerate([-4, -2, -.1, 0, 1, 2, 6])]
        selected, center = select_cases(regions, "control", "protected")
        negative = [r["psnr_delta"] for r in selected if r["group"] == "degradation"]
        self.assertEqual(negative, [-4, -2])
        self.assertEqual(center, 0)
        self.assertEqual(len(selected), len({r["region"]["frame"] for r in selected}))
        self.assertTrue(any(r["psnr_delta"] == 0 for r in selected))

    def test_fewer_than_two_degradations_are_honestly_retained(self):
        selected, _ = select_cases([self.region(0, -1), self.region(1, 1)], "control", "protected")
        self.assertEqual(sum(r["group"] == "degradation" for r in selected), 1)
        with self.assertRaises(ValueError):
            select_cases([], "control", "protected", degraded=1)

    def test_native_bbox_clamps_padding(self):
        labels = np.zeros((10, 20), dtype=int)
        labels[0:2, 17:19] = 1
        self.assertEqual(region_bbox(labels, 1, 3), (14, 0, 20, 5))


class ActiveOpacityTests(unittest.TestCase):
    def test_logit_threshold_counts_and_raw_floor_distinction(self):
        original = dict(original_ids=np.arange(4), instance_ids=np.array([1, 1, 2, 2]))
        compressed = dict(original_ids=np.array([0, 2]), instance_ids=np.array([1, 2]))
        logits = np.array([0., -10., 0., 0.])
        final_logits = np.array([-10., 0.])
        policy = dict(instances=[dict(instance_id=1, min_keep=1, protected=True, small=True),
                                dict(instance_id=2, min_keep=1, protected=True, small=False)])
        report = opacity_audit(original, compressed, logits, final_logits, policy=policy)
        self.assertEqual(report["groups"]["protected"]["floor_violation_count"], 0)
        self.assertEqual(report["opacity_groups"]["0.01"]["protected_small"]["active_extinct_count"], 1)
        self.assertTrue(report["instances"][0]["opacity_counts"]["0.05"]["active_below_raw_count_floor"])

    def test_alpha_and_logit_parameterizations_agree(self):
        alpha = np.array([.001, .02, .9])
        logits = np.log(alpha) - np.log1p(-alpha)
        ids = np.array([0, 1, 1])
        self.assertEqual(active_counts(ids, alpha, .01, "alpha"), active_counts(ids, logits, .01))

    def test_binary_ply_stride_and_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.ply"
            dtype = np.dtype([("x", "<f4"), ("unused", "u1"), ("opacity", "<f4")])
            records = np.array([(1., 4, -2.), (3., 5, 6.)], dtype=dtype)
            header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                      b"property float x\nproperty uchar unused\nproperty float opacity\nend_header\n")
            path.write_bytes(header + records.tobytes())
            mapped = read_ply_vertices(path)
            self.assertEqual(mapped["opacity"].tolist(), [-2., 6.])
            path.write_bytes(header + records.tobytes()[:-1])
            with self.assertRaises(ValueError):
                read_ply_vertices(path)


if __name__ == "__main__":
    unittest.main()
