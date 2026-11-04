import unittest

import numpy as np

from utils.adaptive_constraints import (DynamicConnectivityGuard, articulation_points,
                                        build_topology, graph_from_edges,
                                        select_feasible_prune)


class AdaptiveConstraintTests(unittest.TestCase):
    def test_global_order_is_not_per_instance_and_scores_unchanged(self):
        scores = np.array([100., 1., 2., 99., 3., 98.])
        original = scores.copy()
        instance = np.array([0, 0, 0, 1, 1, 1])
        mask, audit = select_feasible_prune(scores, instance, {0: 1, 1: 1}, 3)
        np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 4])
        np.testing.assert_array_equal(scores, original)
        self.assertEqual(audit["score_order"], "unchanged_global_ascending_stable")
        self.assertTrue(audit["feasible"])

    def test_quota_is_updated_within_batch(self):
        mask, audit = select_feasible_prune(np.arange(6.), [0, 0, 0, 1, 1, 1], {0: 2, 1: 2}, 4)
        np.testing.assert_array_equal(mask, [True, False, False, True, False, False])
        self.assertFalse(audit["feasible"])
        self.assertEqual(audit["unfilled_prune_count"], 2)
        self.assertEqual(audit["instance_protected"], 4)
        self.assertTrue(all(row["count_after"] >= row["minimum"] for row in audit["instance_audit"]))

    def test_boundary_quota_unknown_instance_is_respected(self):
        mask, audit = select_feasible_prune([0, 1, 2, 3], [-1]*4, {}, 3,
                                          boundary_mask=[1, 1, 1, 0], boundary_min_counts={-1: 2})
        np.testing.assert_array_equal(mask, [True, False, False, True])
        self.assertFalse(audit["feasible"])
        self.assertEqual(audit["boundary_protected"], 2)

    def test_hard_boundary_and_active_mask(self):
        mask, audit = select_feasible_prune([0, 1, 2, 3], [0]*4, {0: 1}, 2,
                                          active_mask=[1, 1, 0, 1], boundary_mask=[1, 0, 0, 0], boundary_mode="hard")
        np.testing.assert_array_equal(mask, [False, True, False, True])
        self.assertEqual(audit["count_after"], 1)

    def test_existing_broken_quota_raises(self):
        with self.assertRaises(ValueError):
            select_feasible_prune([1, 2], [0, 0], {0: 3}, 1)

    def test_infeasible_explicit_raise(self):
        with self.assertRaisesRegex(ValueError, "infeasible"):
            select_feasible_prune([1, 2], [0, 0], {0: 2}, 1, raise_on_infeasible=True)

    def test_tarjan_chain_and_cycle(self):
        chain = graph_from_edges(np.arange(5), [[0, 1], [1, 2], [2, 3], [3, 4]])
        critical, audit = articulation_points(chain)
        np.testing.assert_array_equal(critical, [False, True, True, True, False])
        self.assertEqual(audit["components"], 1)
        cycle = graph_from_edges(np.arange(5), [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]])
        critical, _ = articulation_points(cycle)
        self.assertFalse(critical.any())

    def test_dynamic_cycle_blocks_second_cut(self):
        graph = graph_from_edges(np.arange(4), [[0, 1], [1, 2], [2, 3], [3, 0]])
        guard = DynamicConnectivityGuard(graph, 4)
        self.assertTrue(guard.can_remove(0))
        guard.remove(0)
        # Node 2 was not initially an articulation, but now bridges 1 and 3.
        self.assertFalse(guard.can_remove(2))
        self.assertTrue(guard.can_remove(1))
        guard.remove(1)
        self.assertTrue(guard.can_remove(2))

    def test_global_cycle_selection_does_not_use_stale_mask(self):
        graph = graph_from_edges(np.arange(4), [[0, 1], [1, 2], [2, 3], [3, 0]])
        mask, audit = select_feasible_prune([0, 2, 1, 3], [0]*4, {0: 1}, 2, topology=graph)
        np.testing.assert_array_equal(mask, [True, True, False, False])
        self.assertEqual(audit["topology_protected"], 1)

    def test_chain_global_selection_skips_bridge(self):
        graph = graph_from_edges(np.arange(5), [[0, 1], [1, 2], [2, 3], [3, 4]])
        mask, audit = select_feasible_prune([4, 1, 0, 2, 3], [0]*5, {0: 2}, 3, topology=graph)
        # All low-score internal nodes were infeasible at their visit; endpoints
        # can be removed but a later iteration cannot retroactively change order.
        np.testing.assert_array_equal(np.flatnonzero(mask), [0, 4])
        self.assertFalse(audit["feasible"])

    def test_round_start_articulation_stays_protected_after_leaf_removal(self):
        graph = graph_from_edges(np.arange(3), [[0, 1], [1, 2]])
        # After removing 0, node 1 is a live leaf. Its initial critical flag
        # still prohibits deletion in this compression round.
        topology = dict(graph=graph, critical_mask=np.array([False, True, False]))
        mask, audit = select_feasible_prune([0., 1., 2.], [0]*3, {}, 2, topology=topology)
        np.testing.assert_array_equal(mask, [True, False, True])
        self.assertEqual(audit["topology_initial_critical_count"], 1)
        self.assertEqual(audit["topology_initial_protected"], 1)
        self.assertEqual(audit["topology_dynamic_protected"], 0)

    def test_raw_graph_derives_immutable_initial_articulations(self):
        graph = graph_from_edges(np.arange(3), [[0, 1], [1, 2]])
        mask, audit = select_feasible_prune([0., 1., 2.], [0]*3, {}, 3, topology=graph)
        np.testing.assert_array_equal(mask, [True, False, True])
        self.assertFalse(audit["feasible"])
        self.assertEqual(audit["topology_initial_protected"], 1)

    def test_round_start_critical_mask_alignment_is_validated(self):
        graph = graph_from_edges(np.arange(3), [[0, 1], [1, 2]])
        with self.assertRaisesRegex(ValueError, "critical mask is not aligned"):
            select_feasible_prune([0., 1., 2.], [0]*3, {}, 2,
                                  topology=dict(graph=graph, critical_mask=np.zeros(2, bool)))

    def test_search_limit_rejects_conservatively(self):
        count = 30
        graph = graph_from_edges(np.arange(count), [[i, (i+1) % count] for i in range(count)])
        guard = DynamicConnectivityGuard(graph, count, max_search_nodes=3)
        self.assertFalse(guard.can_remove(0))
        self.assertEqual(guard.search_limited, 1)
        self.assertTrue(guard.active.all())

    def test_sparse_disconnected_graph_tarjan(self):
        graph = graph_from_edges([2, 5, 9, 12, 16], [[0, 1], [1, 2], [3, 4]])
        critical, audit = articulation_points(graph)
        np.testing.assert_array_equal(critical, [False, True, False, False, False])
        self.assertEqual(audit["components"], 2)

    def test_pca_line_and_cross_instance_edges(self):
        xyz = np.column_stack((np.arange(40, dtype=np.float32), np.zeros(40), np.zeros(40))).astype(np.float32)
        instance = np.repeat([0, 1], 20)
        result = build_topology(xyz, instance, config=dict(knn=6, graph_knn=2, distance_threshold=1.5, knn_backend="flann"))
        self.assertTrue(result["thin_candidate_mask"].all())
        graph = result["graph"]
        for local, original in enumerate(graph["node_ids"]):
            neighbors = graph["node_ids"][graph["neighbors"][graph["offsets"][local]:graph["offsets"][local+1]]]
            self.assertTrue((instance[neighbors] == instance[original]).all())
        self.assertEqual(result["audit"]["components"], 2)
        self.assertGreater(result["critical_mask"].sum(), 20)

    def test_pca_boundary_excludes_edges(self):
        xyz = np.column_stack((np.arange(10, dtype=np.float32), np.zeros(10), np.zeros(10))).astype(np.float32)
        boundary = np.zeros(10, bool)
        boundary[5] = True
        result = build_topology(xyz, np.zeros(10, np.int64), boundary_mask=boundary,
                                config=dict(knn=4, graph_knn=2, distance_threshold=1.5))
        graph = result["graph"]
        local = int(np.flatnonzero(graph["node_ids"] == 5)[0])
        self.assertEqual(graph["offsets"][local], graph["offsets"][local+1])

    def test_empty_and_disabled_topology(self):
        result = build_topology(np.empty((0, 3)), np.empty(0, np.int64))
        self.assertEqual(result["audit"]["nodes"], 0)
        result = build_topology(np.ones((4, 3)), np.zeros(4, np.int64), config=dict(enabled=False))
        self.assertFalse(result["thin_candidate_mask"].any())

    def test_nan_score_rejected(self):
        with self.assertRaises(ValueError):
            select_feasible_prune([1, np.nan], [0, 0], {}, 1)


if __name__ == "__main__":
    unittest.main()
