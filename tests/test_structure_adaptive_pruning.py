import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.instance_pruning import InstancePruningState, xyz_sha256
from utils.structure_pruning import StructurePruningState


class AdaptiveCoverageTests(unittest.TestCase):
    def make(self, count, groups, anchors, budget, instance_ids=None, floor=0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        xyz = np.arange(count * 3, dtype=np.float32).reshape(count, 3)
        instance_ids = np.zeros(count, dtype=np.int32) if instance_ids is None else np.asarray(instance_ids, dtype=np.int32)
        prior = Path(directory.name) / "prior.npz"
        np.savez(prior, instance_ids=instance_ids,
                 instance_confidence=np.ones(count, dtype=np.float32),
                 instance_views=np.full(count, 3, dtype=np.uint16),
                 instance_source=np.ones(count, dtype=np.uint8),
                 source_xyz_sha256=np.asarray(xyz_sha256(xyz)))
        config = None if not floor else dict(min_original_count=1, min_direct_count=1,
                     min_multiview_direct_count=1, min_keep=floor, max_keep_floor=floor,
                     base_retention=0., small_retention_boost=0., spatial_cells=1)
        instance = InstancePruningState.load(prior, xyz, config=config)
        structure = StructurePruningState()
        structure.anchors = np.asarray(anchors, dtype=np.int64)
        structure.candidates = np.asarray([item for group in groups for item in group], dtype=np.int64)
        structure.offsets = np.r_[0, np.cumsum([len(group) for group in groups])]
        structure.group_kind = np.asarray([1 + index % 3 for index in range(len(groups))], dtype=np.uint8)
        structure.group_strength = structure.group_total_strength = None
        structure.manifest = dict(final_budget=budget)
        structure.source_xyz_sha256 = instance.source_xyz_sha256
        return structure, instance, xyz

    def test_already_covered_sps_is_unchanged_and_old_anchors_can_be_replaced(self):
        s, instance, xyz = self.make(4, [[0, 1], [2, 3]], [0, 2], 2)
        keep, audit = s.select_keep_adaptive_final(np.array([0., 10., 0., 9.]), instance.original_ids, 2, instance, xyz)
        np.testing.assert_array_equal(np.flatnonzero(keep), [1, 3])
        self.assertEqual(audit["adaptive_final"]["added_to_instance_SPS"], 0)
        self.assertEqual(audit["structure_coverage"]["missing_anchors"], 2)
        self.assertEqual(audit["structure_coverage"]["uncovered_groups"], 0)

    def test_repairs_missing_cell_with_one_swap_and_preserves_sole_representatives(self):
        s, instance, xyz = self.make(6, [[0, 1], [2, 3], [3, 4]], [0, 2, 4], 3)
        keep, audit = s.select_keep_adaptive_final(np.array([0., 1., 0., 3., 10., 9.]), instance.original_ids, 3, instance, xyz)
        np.testing.assert_array_equal(np.flatnonzero(keep), [1, 3, 4])
        self.assertEqual(audit["adaptive_final"]["added_to_instance_SPS"], 1)
        self.assertEqual(audit["adaptive_final"]["removed_from_instance_SPS"], 1)
        self.assertFalse(audit["adaptive_final"]["fixed_certificate_fallback"])

    def test_repair_preserves_instance_floors(self):
        s, instance, xyz = self.make(6, [[0], [3]], [0, 3], 3, [1, 1, 1, 2, 2, 0], floor=1)
        keep, audit = s.select_keep_adaptive_final(np.array([0., 2., 5., 1., 6., 100.]), instance.original_ids, 3, instance, xyz)
        np.testing.assert_array_equal(np.flatnonzero(keep), [0, 3, 5])
        self.assertTrue(all(not row["floor_violation"] for row in audit["instances"]))

    def test_intermediate_round_keeps_feasible_fixed_certificate(self):
        s, instance, xyz = self.make(4, [[0, 1], [2, 3]], [0, 2], 2)
        keep, audit = s.select_keep_adaptive_final(np.array([0., 10., 0., 9.]), instance.original_ids, 3, instance, xyz)
        self.assertTrue(keep[s.anchors].all())
        self.assertFalse(audit["adaptive_final"]["applied"])

    def test_locally_locked_greedy_solution_uses_explicit_feasible_fallback(self):
        # A,B are high-SPS but then F makes a locally irreducible three-atom
        # cover. The declared E,F certificate proves that exact K=2 is feasible.
        s, instance, xyz = self.make(6, [[0, 2, 4], [1, 3, 4], [2, 5], [3, 5]], [4, 5], 2)
        keep, audit = s.select_keep_adaptive_final(np.array([10., 9., .5, .4, .1, .2]), instance.original_ids, 2, instance, xyz)
        np.testing.assert_array_equal(np.flatnonzero(keep), [4, 5])
        self.assertTrue(audit["adaptive_final"]["fixed_certificate_fallback"])
        self.assertIn("local greedy repair", audit["adaptive_final"]["fallback_reason"])
        self.assertEqual(audit["structure_coverage"]["uncovered_groups"], 0)

    def test_an_unavailable_candidate_is_not_selected(self):
        s, instance, xyz = self.make(5, [[0, 1], [2, 3]], [0, 2], 2)
        available = np.array([True, False, True, True, True])
        instance.subset_(available)
        keep, audit = s.select_keep_adaptive_final(np.array([0., 0., 3., 10.]), instance.original_ids, 2, instance, xyz[available])
        self.assertTrue(keep[0])  # source 1 is unavailable, so source 0 is needed
        self.assertEqual(audit["structure_coverage"]["uncovered_groups"], 0)

    def test_strict_cell_audit_allows_anchor_replacement_but_rejects_missing_cells(self):
        s, _, _ = self.make(4, [[0, 1], [2, 3]], [0, 2], 2)
        self.assertEqual(s.audit_coverage(np.array([1, 3]), require_anchors=False)["uncovered_groups"], 0)
        with self.assertRaisesRegex(ValueError, "screen cells lost coverage"):
            s.audit_coverage(np.array([1]), require_anchors=False)


if __name__ == "__main__":
    unittest.main()
