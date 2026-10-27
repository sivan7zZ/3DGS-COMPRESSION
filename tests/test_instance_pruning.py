import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.instance_pruning import InstancePruningState, xyz_sha256, global_sps_keep


class InstancePruningTests(unittest.TestCase):
    def test_global_sps_exact_budget_stable_ties_and_invalid_scores(self):
        np.testing.assert_array_equal(global_sps_keep(np.array([2., 3., 3., 1.]), 2),
                                      [False, True, True, False])
        np.testing.assert_array_equal(global_sps_keep(np.ones(4), 2), [True, True, False, False])
        with self.assertRaises(ValueError):
            global_sps_keep(np.array([1., np.nan]), 1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def make_state(self, ids, confidence=None, source=None, views=None, config=None, area=None):
        ids = np.asarray(ids, dtype=np.int32)
        rng = np.random.RandomState(12)
        xyz = rng.uniform(-1, 1, (len(ids), 3)).astype(np.float32)
        path = self.root / "priors.npz"
        np.savez(path, instance_ids=ids,
                 instance_confidence=np.ones(len(ids), dtype=np.float32) if confidence is None else confidence,
                 instance_views=np.full(len(ids), 4, dtype=np.uint8) if views is None else views,
                 instance_source=np.ones(len(ids), dtype=np.uint8) if source is None else source,
                 source_xyz_sha256=np.asarray(xyz_sha256(xyz)))
        return InstancePruningState.load(path, xyz, config=config, projected_area=area), xyz

    def test_low_sps_small_instance_survives_exact_global_budget(self):
        state, xyz = self.make_state([1] * 2000 + [2] * 40 + [0] * 100)
        scores = np.r_[np.ones(2000), np.zeros(40), np.ones(100)]
        keep, audit = state.select_keep(scores, xyz, 350)
        self.assertEqual(int(keep.sum()), 350)
        self.assertGreaterEqual(int(keep[2000:2040].sum()), 16)
        self.assertEqual(audit["protected_lost"], 0)
        self.assertEqual(audit["instances"][1]["original_count"], 40)

    def test_floors_immutable_across_pruning_rounds_and_original_ids_follow_rows(self):
        state, xyz = self.make_state([1] * 1000 + [2] * 40 + [0] * 30)
        scores = np.arange(len(xyz), dtype=float)
        first, _ = state.select_keep(scores, xyz, 400)
        expected_ids = np.flatnonzero(first)
        floors = [row["min_keep"] for row in state.policy]
        state.subset_(first)
        xyz = xyz[first]
        np.testing.assert_array_equal(state.original_ids, expected_ids)
        second, audit = state.select_keep(scores[first], xyz, 250)
        self.assertEqual(int(second.sum()), 250)
        self.assertEqual([row["min_keep"] for row in state.policy], floors)
        self.assertTrue(all(row["after"] >= row["min_keep"] for row in audit["instances"]))

    def test_infeasible_budget_fails_explicitly_without_modifying_state(self):
        state, xyz = self.make_state([1] * 50 + [2] * 50)
        before = state.original_ids.copy()
        with self.assertRaisesRegex(ValueError, "floors require.*global keep budget"):
            state.select_keep(np.ones(100), xyz, 20)
        np.testing.assert_array_equal(state.original_ids, before)

    def test_equal_scores_are_deterministic_and_exact(self):
        state, xyz = self.make_state([0] * 100)
        first, _ = state.select_keep(np.ones((100, 1)), xyz, 17)
        second, _ = state.select_keep(np.ones((100, 1)), xyz, 17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.flatnonzero(first), np.arange(17))

    def test_unknown_propagation_only_and_tiny_regions_receive_no_hard_quota(self):
        ids = [0] * 100 + [1] * 100 + [2] * 5 + [3] * 100
        source = np.r_[np.ones(100), np.full(100, 2), np.ones(105)].astype(np.uint8)
        confidence = np.r_[np.ones(205), np.zeros(100)].astype(np.float32)
        state, xyz = self.make_state(ids, source=source, confidence=confidence)
        self.assertTrue(all(row["min_keep"] == 0 for row in state.policy))
        keep, audit = state.select_keep(np.arange(len(xyz)), xyz, 1)
        self.assertEqual(int(keep.sum()), 1)
        self.assertEqual(audit["floor_total"], 0)

    def test_single_view_direct_noise_does_not_create_floor(self):
        state, xyz = self.make_state([1] * 80, views=np.ones(80, dtype=np.uint8))
        self.assertFalse(state.policy[0]["protected"])
        keep, _ = state.select_keep(np.ones(80), xyz, 0)
        self.assertFalse(keep.any())

    def test_existing_city_minus_one_unknown_is_normalized_without_creating_quota(self):
        state, xyz = self.make_state([-1] * 40 + [1] * 80)
        np.testing.assert_array_equal(state.instance_ids[:40], np.zeros(40, dtype=np.int32))
        self.assertEqual([row["instance_id"] for row in state.policy], [1])
        keep, audit = state.select_keep(np.ones(120), xyz, 40)
        self.assertEqual(int(keep.sum()), 40)
        self.assertEqual(audit["unknown_before"], 40)

    def test_direct_evidence_wins_equal_sps_over_propagated_fill(self):
        # The hard floor is made one spatial cell to isolate score weighting.
        ids = [1] * 80
        source = np.r_[np.full(40, 2), np.ones(40)].astype(np.uint8)
        state, xyz = self.make_state(ids, source=source, config={"spatial_cells": 1})
        keep, _ = state.select_keep(np.ones(80), xyz, 16)
        self.assertEqual(int(keep[40:].sum()), 16)

    def test_spatial_representatives_protect_extent_against_central_score_bias(self):
        state, xyz = self.make_state([1] * 125,
                                     config={"base_retention": 0, "small_retention_boost": 0,
                                             "min_keep": 27, "spatial_cells": 27})
        grid = np.asarray([(x, y, z) for x in range(5) for y in range(5) for z in range(5)], dtype=np.float32)
        # Source fingerprint is not recomputed during refinement, but geometry
        # moves are valid. Update frozen source bounds to this synthetic grid.
        state.policy[0]["bbox_min"] = [0, 0, 0]
        state.policy[0]["bbox_max"] = [4, 4, 4]
        scores = -((grid - 2) ** 2).sum(axis=1)
        keep, _ = state.select_keep(scores, grid, 27)
        selected = grid[keep]
        np.testing.assert_array_equal(selected.min(axis=0), [1, 1, 1])
        np.testing.assert_array_equal(selected.max(axis=0), [3, 3, 3])
        # All 27 spatial cells retain representatives, including outer cells.
        cells = np.floor(np.clip(selected / 4, 0, 1 - 1e-8) * 3).astype(int)
        self.assertEqual(len(np.unique(cells, axis=0)), 27)

    def test_projected_area_overrides_gaussian_count_proxy(self):
        state, _ = self.make_state([1] * 80 + [2] * 80, area={1: 0.20, 2: 0.001})
        self.assertFalse(state.policy[0]["small"])
        self.assertTrue(state.policy[1]["small"])
        self.assertEqual(state.policy[0]["smallness_source"], "train_projected_area")
        self.assertGreater(state.policy[1]["smallness"], state.policy[0]["smallness"])

    def test_saved_subset_roundtrip_uses_current_xyz_and_preserves_original_policy(self):
        state, xyz = self.make_state([1] * 300 + [2] * 40 + [0] * 30)
        keep, _ = state.select_keep(np.arange(len(xyz)), xyz, 200)
        state.subset_(keep)
        current_xyz = xyz[keep] + np.float32(0.0125)
        path = self.root / "checkpoint.instances.npz"
        state.save(path, current_xyz)
        loaded = InstancePruningState.load_resume(path, current_xyz)
        np.testing.assert_array_equal(loaded.original_ids, np.flatnonzero(keep))
        self.assertEqual(loaded.policy, state.policy)
        self.assertEqual(loaded.original_count, 370)
        with np.load(path, allow_pickle=False) as data:
            self.assertNotEqual(str(data["source_xyz_sha256"].item()), str(data["current_xyz_sha256"].item()))
        with self.assertRaisesRegex(ValueError, "current_xyz_sha256 mismatch"):
            InstancePruningState.load_resume(path, current_xyz[::-1])
        policy_path = self.root / "protection_policy.json"
        loaded.export_policy(policy_path)
        policy = json.loads(policy_path.read_text())
        self.assertTrue(all(key in policy["instances"][0] for key in ("instance_id", "min_keep", "protected", "small", "original_count")))

    def test_label_row_misalignment_and_invalid_subset_rejected(self):
        state, xyz = self.make_state([1] * 50)
        with self.assertRaisesRegex(ValueError, "source_xyz_sha256 mismatch"):
            InstancePruningState.load(self.root / "priors.npz", xyz[::-1])
        with self.assertRaisesRegex(ValueError, "boolean keep mask"):
            state.subset_(np.arange(50))
        with self.assertRaisesRegex(ValueError, "aligned"):
            state.select_keep(np.ones(49), xyz, 20)
        with self.assertRaisesRegex(ValueError, "finite"):
            state.select_keep(np.full(50, np.nan), xyz, 20)

    def test_external_metadata_loss_below_floor_is_reported(self):
        state, xyz = self.make_state([1] * 50 + [2] * 50)
        accidental = np.r_[np.zeros(50, dtype=bool), np.ones(50, dtype=bool)]
        state.subset_(accidental)
        with self.assertRaisesRegex(ValueError, "already has 0 points below immutable floor"):
            state.select_keep(np.ones(50), xyz[accidental], 40)

    def test_torch_scores_return_bool_on_input_device(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable in this Python")
        state, xyz = self.make_state([1] * 80 + [0] * 20)
        keep, _ = state.select_keep(torch.ones(100, 1), torch.as_tensor(xyz), 40)
        self.assertEqual(keep.dtype, torch.bool)
        self.assertEqual(keep.device.type, "cpu")
        self.assertEqual(int(keep.sum()), 40)


if __name__ == "__main__":
    unittest.main()
