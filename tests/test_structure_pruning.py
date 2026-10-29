import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.instance_pruning import InstancePruningState, xyz_sha256
from utils.structure_pruning import greedy_coverage
from prepare_structure_protection import screen_cell_groups, zhang_suen


class StructureCoverageTests(unittest.TestCase):
    def test_occluded_centers_cannot_become_coverage_candidates(self):
        mask = np.ones((3, 3), dtype=bool)
        ids = np.full((3, 3), 4, dtype=np.int32)
        ids[:, 0] = 1
        candidate, strength, offsets, groups, weights = screen_cell_groups(
            mask, np.ones((3, 3), dtype=np.int32), ids, np.ones((3, 3)), 1, 4)
        self.assertEqual(set(candidate), {1, 4})
        self.assertNotIn(2, candidate)  # no raw/center-projected Gaussian 2
        self.assertEqual(offsets, [0, 2])
        self.assertEqual(len(groups), 1)
        self.assertAlmostEqual(float(weights.mean()), 1.)

    def test_adjacent_small_instances_retain_distinct_cell_constraints(self):
        regions = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
        candidate, _, offsets, groups, _ = screen_cell_groups(
            np.ones((2, 4), dtype=bool), regions, regions, np.ones((2, 4)), 1, 8)
        self.assertEqual(len(groups), 2)
        self.assertEqual(set(candidate), {1, 2})
        self.assertEqual(offsets, [0, 1, 2])

    def test_set_cover_reuses_multiview_atom_without_dropping_groups(self):
        candidates = np.array([1, 3, 2, 3, 3, 4, 5])
        offsets = np.array([0, 2, 4, 5, 7])
        selected = greedy_coverage(candidates, offsets, np.ones(7), budget=2)
        np.testing.assert_array_equal(selected, [3, 4])
        with self.assertRaisesRegex(ValueError, "explicit anchor budget"):
            greedy_coverage(candidates, offsets, np.ones(7), budget=1)

    def test_skeleton_keeps_a_thin_diagonal_connected(self):
        mask = np.eye(12, dtype=bool)
        result = zhang_suen(mask)
        np.testing.assert_array_equal(result, mask)

    def test_visible_anchors_and_instance_floors_share_exact_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            xyz = np.arange(300, dtype=np.float32).reshape(100, 3)
            path = Path(directory) / "prior.npz"
            np.savez(path, instance_ids=np.r_[np.ones(50), np.full(50, 2)].astype(np.int32),
                     instance_confidence=np.ones(100, dtype=np.float32),
                     instance_views=np.full(100, 3, dtype=np.uint16),
                     instance_source=np.ones(100, dtype=np.uint8),
                     source_xyz_sha256=np.asarray(xyz_sha256(xyz)))
            state = InstancePruningState.load(path, xyz, config={"min_keep": 8, "max_keep_floor": 8})
            anchors = np.array([0, 1, 2, 3, 51])
            keep, audit = state.select_keep(np.arange(100, dtype=float), xyz, 16,
                                            required_original_ids=anchors)
            self.assertEqual(keep.sum(), 16)
            self.assertTrue(keep[anchors].all())
            self.assertEqual(keep[:50].sum(), 8)
            self.assertEqual(audit["anchors_plus_floors_cost"], 16)
            with self.assertRaisesRegex(ValueError, "anchors.*floors require"):
                state.select_keep(np.ones(100), xyz, 16,
                                  required_original_ids=np.arange(10))
            missing = int(np.flatnonzero(~keep)[0])
            state.subset_(keep)
            with self.assertRaisesRegex(ValueError, "anchors were lost"):
                state.select_keep(np.ones(16), xyz[keep], 16,
                                  required_original_ids=np.array([missing]))


if __name__ == "__main__":
    unittest.main()
