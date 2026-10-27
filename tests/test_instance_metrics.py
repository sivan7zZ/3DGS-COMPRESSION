import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.instance_metrics import (inspect_standard_ply, load_metadata,
                                     projected_area_statistics, retention_report)


class InstanceRetentionTests(unittest.TestCase):
    def test_exact_lineage_and_floor_violation(self):
        original = dict(original_ids=np.arange(7), instance_ids=np.array([0, 1, 1, 1, 2, 2, 3]))
        final = dict(original_ids=np.array([5, 1, 0]), instance_ids=np.array([2, 1, 0]))
        policy = dict(instances=[dict(instance_id=1, min_keep=2, protected=True, small=True),
                                dict(instance_id=2, min_keep=1, protected=True, small=False)])
        r = retention_report(original, final, policy=policy)
        self.assertEqual(r["groups"]["all_instances"]["extinct_count"], 1)
        self.assertEqual(r["groups"]["protected_small"]["floor_violation_count"], 1)
        self.assertEqual(r["gaussian_counts"]["unresolved_final"], 1)
        self.assertEqual(r["instances"][0]["retention_fraction"], 1 / 3)

    def test_reject_unknown_duplicate_and_changed_identity(self):
        original = dict(original_ids=np.array([2, 4]), instance_ids=np.array([1, 2]))
        bad = [dict(original_ids=np.array([5]), instance_ids=np.array([1])),
               dict(original_ids=np.array([2, 2]), instance_ids=np.array([1, 1])),
               dict(original_ids=np.array([2]), instance_ids=np.array([2]))]
        for compressed in bad:
            with self.assertRaises(ValueError):
                retention_report(original, compressed)

    def test_unknown_points_and_empty_group(self):
        original = dict(original_ids=np.arange(2), instance_ids=np.zeros(2, dtype=int))
        final = dict(original_ids=np.array([], dtype=int), instance_ids=np.array([], dtype=int))
        report = retention_report(original, final)
        self.assertIsNone(report["groups"]["protected_small"]["median_retention_fraction"])
        self.assertEqual(report["groups"]["all_instances"]["instance_count"], 0)

    def test_metadata_requires_explicit_compressed_lineage(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "meta.npz"
            np.savez(p, instance_ids=np.array([-1, 2], dtype=np.int32))
            self.assertEqual(load_metadata(p)["instance_ids"].tolist(), [0, 2])
            with self.assertRaises(ValueError):
                load_metadata(p, require_original_ids=True)

    def test_training_areas_exclude_test_and_sum_same_view_regions(self):
        tracks = dict(nodes=[dict(instance_id=1, frame="train", area=10),
                             dict(instance_id=1, frame="train", area=20),
                             dict(instance_id=1, frame="test", area=999)])
        areas, excluded = projected_area_statistics(tracks, ["train"], {"train": (10, 10)})
        self.assertEqual(areas[1]["median_projected_area_fraction"], .3)
        self.assertEqual(areas[1]["training_view_count"], 1)
        self.assertEqual(excluded, ["test"])

    def test_sh2_header_and_dog_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "model.ply"
            names = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)] +
                     [f"f_rest_{i}" for i in range(24)] + ["opacity"] +
                     [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
            header = "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
            p.write_text(header + "".join(f"property float {n}\n" for n in names) + "end_header\n")
            info = inspect_standard_ply(p, 2)
            self.assertEqual(info["property_count"], 41)
            self.assertTrue(info["expected_sh_matches"])
            self.assertTrue(info["no_dog_properties"])
            self.assertFalse(info["missing_standard_properties"])
            p.write_text(header + "".join(f"property float {n}\n" for n in names + ["negative_opacity"]) + "end_header\n")
            self.assertFalse(inspect_standard_ply(p, 2)["no_dog_properties"])


if __name__ == "__main__":
    unittest.main()
