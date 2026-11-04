"""CPU tests exercise real GaussianModel optimizer mutations and merge moments."""
import argparse
import os
import tempfile
import unittest

import numpy as np
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.adaptive_mutation import (CompressionState, covariance_from_parameters,
                                     find_safe_merge_pairs, moment_match_pairs)


def make_model(n=6, warm_optimizer=True):
    xyz = torch.arange(n, dtype=torch.float32)[:, None].repeat(1, 3)*0.05
    rotation = torch.zeros(n, 4)
    rotation[:, 0] = 1
    tensors = [xyz, torch.zeros(n, 1, 3), torch.zeros(n, 15, 3),
               torch.zeros(n, 3), rotation, torch.full((n, 1), -0.3),
               torch.arange(n, dtype=torch.float32), torch.ones(n, 1), torch.ones(n, 1)]
    checkpoint = (3, *tensors, {"state": {}, "param_groups": []}, 1.0)
    parser = argparse.ArgumentParser()
    args = OptimizationParams(parser).extract(parser.parse_args([]))
    model = GaussianModel(3)
    model.load_checkpoint_model(checkpoint, args, device="cpu")
    if warm_optimizer:
        for group in model.optimizer.param_groups:
            group["params"][0].grad = torch.full_like(group["params"][0], 0.25)
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
    return model


def metadata(n):
    return {"instance_ids": np.full(n, 3, dtype=np.int64),
            "confidence": np.full(n, 0.9, dtype=np.float32),
            "original_ids": np.arange(n, dtype=np.int64),
            "boundary_mask": np.zeros(n, dtype=bool),
            "topology_critical": np.zeros(n, dtype=bool),
            "visibility": np.ones(n, dtype=np.float32),
            "structure_risk": np.linspace(0, 0.2, n).astype(np.float32),
            "extra_vector": np.arange(n*3, dtype=np.float32).reshape(n, 3)}


class MomentMatchingTests(unittest.TestCase):
    def test_center_covariance_sh_and_alpha_union(self):
        params = {"xyz": torch.tensor([[0., 0., 0.], [2., 0., 0.]]),
                  "f_dc": torch.tensor([[[0., 1., 2.]], [[2., 3., 4.]]]),
                  "f_rest": torch.ones(2, 15, 3),
                  "scaling": torch.tensor([[0., np.log(2.), np.log(3.)], [0., np.log(2.), np.log(3.)]], dtype=torch.float32),
                  "rotation": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]]),
                  "opacity": torch.logit(torch.tensor([[0.2], [0.6]]))}
        merged, weights = moment_match_pairs(params, [[0, 1]], [3., 1.])
        np.testing.assert_allclose(weights, [[0.5, 0.5]], atol=1e-6)
        torch.testing.assert_close(merged["xyz"], torch.tensor([[1., 0., 0.]]))
        expected = torch.diag(torch.tensor([2., 4., 9.]))[None]
        torch.testing.assert_close(covariance_from_parameters(merged["scaling"], merged["rotation"]), expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(merged["f_dc"], torch.tensor([[[1., 2., 3.]]]))
        torch.testing.assert_close(torch.sigmoid(merged["opacity"]), torch.tensor([[0.68]]))
        self.assertAlmostEqual(float(merged["rotation"].norm()), 1., places=6)

    def test_general_rotated_covariance_and_zero_visibility_fallback(self):
        torch.manual_seed(11)
        n = 30
        p = {"xyz": torch.randn(n, 3), "f_dc": torch.randn(n, 1, 3), "f_rest": torch.randn(n, 15, 3),
             "scaling": torch.randn(n, 3)*0.5, "rotation": torch.randn(n, 4), "opacity": torch.randn(n, 1)}
        pairs = np.arange(n).reshape(-1, 2)
        m, w = moment_match_pairs(p, pairs, np.zeros(n))
        original_covariance = covariance_from_parameters(p["scaling"], p["rotation"])
        expected = []
        for k, (a, b) in enumerate(pairs):
            d1 = p["xyz"][a]-m["xyz"][k]
            d2 = p["xyz"][b]-m["xyz"][k]
            expected.append(w[k, 0]*(original_covariance[a]+d1[:, None]*d1[None, :])+
                            w[k, 1]*(original_covariance[b]+d2[:, None]*d2[None, :]))
        torch.testing.assert_close(covariance_from_parameters(m["scaling"], m["rotation"]), torch.stack(expected), atol=2e-5, rtol=2e-5)
        self.assertTrue(all(torch.isfinite(x).all() for x in m.values()))


class StateMutationTests(unittest.TestCase):
    def test_prune_restore_metadata_exact_native_fields_and_momentum(self):
        model = make_model()
        state = CompressionState(model, metadata(6))
        before = {k: v.detach().clone() for k, v in state._params().items()}
        old_moments = {group["name"]: {key: value.clone() for key, value in model.optimizer.state[group["params"][0]].items()}
                       for group in model.optimizer.param_groups}
        mask = np.array([False, True, False, True, False, False])
        self.assertEqual(state.remove(mask, 1, np.arange(6)), 2)
        np.testing.assert_array_equal(state.metadata["original_ids"], [0, 2, 4, 5])
        for group in model.optimizer.param_groups:
            native_state = model.optimizer.state[group["params"][0]]
            torch.testing.assert_close(native_state["exp_avg"], old_moments[group["name"]]["exp_avg"][~mask])
            torch.testing.assert_close(native_state["step"], old_moments[group["name"]]["step"])
        pool = state.pool_arrays()
        np.testing.assert_array_equal(pool["metadata"]["original_ids"], [1, 3])
        self.assertTrue(all(v.device.type == "cpu" for v in pool["params"].values()))
        self.assertEqual(state.restore([1, 0], 1, target_count=5), 1)
        np.testing.assert_array_equal(state.metadata["original_ids"], [0, 2, 4, 5, 3])
        for key, value in state._params().items():
            torch.testing.assert_close(value[-1], before[key][3], rtol=0, atol=0)
        for group in model.optimizer.param_groups:
            native_state = model.optimizer.state[group["params"][0]]
            self.assertTrue(torch.equal(native_state["exp_avg"][-1], torch.zeros_like(native_state["exp_avg"][-1])))
            torch.testing.assert_close(native_state["exp_avg"][:-1], old_moments[group["name"]]["exp_avg"][~mask])
        self.assertEqual(state.audit()["count"], 5)
        self.assertEqual(state.audit()["pool_rows"], 1)

    def test_merging_new_lineage_zero_moments_and_auxiliary_alignment(self):
        model = make_model()
        state = CompressionState(model, metadata(6))
        expected, _ = moment_match_pairs(state._params(), [[0, 1]], state.metadata["visibility"])
        old_survivor = model._xyz[2:].detach().clone()
        self.assertEqual(state.merge([[0, 1]], 2), 1)
        self.assertEqual(state.count, 5)
        np.testing.assert_array_equal(state.metadata["original_ids"], [2, 3, 4, 5, -1])
        self.assertEqual(state.metadata["row_ids"][-1], 6)
        self.assertEqual(state.metadata["lineage_count"][-1], 2)
        self.assertEqual(state.lineage[0]["parent_original_ids"], [0, 1])
        self.assertEqual(state.audit()["pool_rows"], 0)
        torch.testing.assert_close(model._xyz[:-1], old_survivor, atol=0, rtol=0)
        for key, value in state._params().items():
            torch.testing.assert_close(value[-1:], expected[key])
        for group in model.optimizer.param_groups:
            self.assertEqual(float(model.optimizer.state[group["params"][0]]["exp_avg"][-1].abs().sum()), 0.)
        self.assertEqual(state.metadata["extra_vector"].shape, (5, 3))
        mask = np.zeros(5, dtype=bool)
        mask[-1] = True
        state.remove(mask, 3)
        self.assertEqual(state.restore([0], 3, 5), 0)  # -1 has no teacher sparse support.

    def test_reject_unsafe_or_duplicate_merge_before_mutation(self):
        for key, value in (("instance_ids", -1), ("confidence", 0.2), ("boundary_mask", True), ("topology_critical", True)):
            m = metadata(6)
            m[key][0] = value
            state = CompressionState(make_model(), m)
            with self.assertRaises(ValueError):
                state.merge([[0, 1]], 0)
            self.assertEqual(state.count, 6)
        state = CompressionState(make_model(), metadata(6))
        with self.assertRaises(ValueError):
            state.merge([[0, 1], [1, 2]], 0)
        self.assertEqual(state.count, 6)

    def test_pool_expiry_and_save_load_roundtrip(self):
        model = make_model()
        state = CompressionState(model, metadata(6), pool_rounds=2)
        state.remove(np.array([True, False, False, False, False, False]), 1)
        state.remove(np.array([True, False, False, False, False]), 2)
        state.remove(np.array([True, False, False, False]), 3)
        np.testing.assert_array_equal(state.pool_arrays()["metadata"]["original_ids"], [1, 2])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "compression_state.pt")
            state.save(path, {"round": 3, "controller": {"p": 0.2}})
            restored, extra = CompressionState.load(model, path)
            self.assertEqual(extra["round"], 3)
            self.assertEqual(restored.audit(), state.audit())
            np.testing.assert_array_equal(restored.pool_arrays()["metadata"]["row_ids"], [1, 2])
            self.assertEqual(restored.restore([0], 3, 4), 1)

    def test_uninitialized_optimizer_and_invalid_append(self):
        state = CompressionState(make_model(warm_optimizer=False), metadata(6))
        state.remove(np.array([True, False, False, False, False, False]), 1)
        self.assertEqual(state.restore([0], 1, 6), 1)
        self.assertTrue(all(not x for x in state.model.optimizer.state.values()))
        with self.assertRaises(ValueError):
            state.append({key: value[:1] for key, value in state._params().items()},
                         {key: value[:1] for key, value in state.metadata.items()})

    def test_multiple_rounds_leave_untouched_native_row_and_adam_exact(self):
        model = make_model(12)
        state = CompressionState(model, metadata(12))
        untouched = 11
        raw = {key: value[untouched].detach().clone() for key, value in state._params().items()}
        moments = {group["name"]: {key: value[untouched].clone() if value.ndim else value.clone()
                                   for key, value in model.optimizer.state[group["params"][0]].items()}
                   for group in model.optimizer.param_groups}
        mask = np.isin(state.metadata["original_ids"], [0, 1, 2])
        state.remove(mask, 1)
        state.merge([[0, 1]], 1)
        state.restore([1], 1, 9)
        state.remove(np.isin(state.metadata["original_ids"], [5, 6]), 2)
        pool = state.pool_arrays(2)
        selected = np.flatnonzero(np.isin(pool["metadata"]["original_ids"], [0, 5]))
        state.restore(selected, 2, 9)
        found = int(np.flatnonzero(state.metadata["original_ids"] == untouched)[0])
        for key, value in state._params().items():
            torch.testing.assert_close(value[found], raw[key], atol=0, rtol=0)
        for group in model.optimizer.param_groups:
            native = model.optimizer.state[group["params"][0]]
            for key, value in native.items():
                compared = value[found] if value.ndim else value
                torch.testing.assert_close(compared, moments[group["name"]][key], atol=0, rtol=0)
        self.assertEqual(state.audit()["count"], 9)


class MergeCandidateTests(unittest.TestCase):
    def test_safe_knn_pairs_obey_protection_appearance_distance_and_quota(self):
        model = make_model(10, warm_optimizer=False)
        m = metadata(10)
        m["instance_ids"][8:] = 9
        m["boundary_mask"][0] = True
        m["topology_critical"][1] = True
        m["confidence"][2] = 0.1
        with torch.no_grad():
            model._features_dc[3] = 10
            model._xyz[4] = 100
        pairs = find_safe_merge_pairs(model, m, 5, instance_min_counts={3: 7, 9: 1})
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(np.unique(pairs)), 4)
        self.assertFalse(np.isin(pairs, [0, 1, 2, 3, 4]).any())
        self.assertTrue(np.all(m["instance_ids"][pairs[:, 0]] == m["instance_ids"][pairs[:, 1]]))
        self.assertEqual(sum(m["instance_ids"][pairs[:, 0]] == 3), 1)
        self.assertEqual(sum(m["instance_ids"][pairs[:, 0]] == 9), 1)


if __name__ == "__main__":
    unittest.main()
