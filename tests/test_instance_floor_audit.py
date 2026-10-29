import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.instance_pruning import InstancePruningState, xyz_sha256


class InstanceFloorAuditTests(unittest.TestCase):
    def make(self, root):
        xyz = np.arange(60, dtype=np.float32).reshape(20, 3)
        prior = root / "original.npz"
        np.savez(prior, instance_ids=np.r_[np.ones(8), np.full(8, 2), np.zeros(4)].astype(np.int32),
                 instance_confidence=np.ones(20, dtype=np.float32),
                 instance_source=np.ones(20, dtype=np.uint8),
                 instance_views=np.full(20, 3, dtype=np.uint16),
                 source_xyz_sha256=np.asarray(xyz_sha256(xyz)))
        config = dict(min_original_count=1, min_direct_count=1, min_multiview_direct_count=1,
                      min_keep=3, max_keep_floor=3, base_retention=0., small_retention_boost=0.)
        return InstancePruningState.load(prior, xyz, config=config), xyz

    def test_valid_protected_subset_passes_saved_floor_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            state, xyz = self.make(Path(directory))
            keep, _ = state.select_keep(np.arange(20, dtype=float), xyz, 8)
            state.subset_(keep)
            report = state.audit_floors()
            self.assertEqual(report["violation_count"], 0)
            self.assertEqual(report["gaussian_count"], 8)
            self.assertTrue(all(row["count"] >= row["min_keep"] for row in report["instances"]))

    def test_below_floor_global_control_still_loads_but_protected_resume_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state, xyz = self.make(root)
            keep = np.ones(20, dtype=bool)
            keep[:6] = False  # instance 1 now has 2 atoms, below its floor of 3
            state.subset_(keep)
            saved = root / "global_control.npz"
            state.save(saved, xyz[keep])
            loaded = InstancePruningState.load_resume(saved, xyz[keep])
            # Global controls may load this metadata for rendering/evaluation.
            report = loaded.audit_floors(raise_on_violation=False)
            self.assertEqual(report["violation_count"], 1)
            self.assertEqual(report["violations"][0]["instance_id"], 1)
            self.assertEqual(report["violations"][0]["count"], 2)
            with self.assertRaisesRegex(ValueError, "retention floors violated"):
                loaded.audit_floors()  # explicit protected-mode bootstrap gate


if __name__ == "__main__":
    unittest.main()
