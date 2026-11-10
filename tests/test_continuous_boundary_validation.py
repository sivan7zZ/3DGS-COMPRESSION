import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validate_continuous_boundary_results import (
    COMPONENTS, PARAMETERS, MetadataContinuity, components_array, float32_error,
    sha256_file, validate_event, validate_payload, validate_score_receipt,
)


class ContinuousBoundaryValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def arrays(self, count=9):
        value = np.linspace(0., 1., count, dtype=np.float32)
        sam, depth, rgb = value, value ** 2, np.sqrt(value)
        combined = np.float32(.4) * sam + np.float32(.4) * depth + np.float32(.2) * rgb
        boost = np.log1p(np.float32(5.) * combined)
        raw = np.linspace(-10., 3., count, dtype=np.float32)
        raw[0] = -np.inf
        scores = raw.copy()
        scores[1:] += boost[1:]
        return dict(scores=scores, raw_pup_scores=raw, sam_boundary_strength=sam,
                    depth_boundary_strength=depth, rgb_boundary_strength=rgb,
                    combined_boundary_strength=combined, boundary_multiplier=np.float32(1.) + np.float32(5.) * combined,
                    boundary_log_boost=boost, boundary_mask=np.zeros(count, bool),
                    row_ids=np.arange(count, dtype=np.int64), original_ids=np.arange(count, dtype=np.int64))

    def receipt(self, arrays=None, name="continuous"):
        arrays = self.arrays() if arrays is None else arrays
        path = self.folder / (name + ".npz")
        np.savez_compressed(str(path), **arrays)
        return dict(path=str(path), sha256=sha256_file(path), mode="pup_continuous_boundary",
                    continuous_boundary=True, boundary_score=dict(PARAMETERS), count=len(arrays["scores"]),
                    negative_infinity_count=int(np.isneginf(arrays["scores"]).sum()),
                    boundary_count=int(arrays["boundary_mask"].sum()))

    def test_independent_continuous_score_formula_and_singular_fisher(self):
        result = validate_score_receipt(self.receipt(), 9)
        self.assertTrue(result["singular_scores_unchanged"])
        self.assertEqual(result["negative_infinity_count"], 1)
        self.assertEqual(result["combined"]["unique_values"], 9)
        self.assertEqual(result["increased_finite_scores"], 8)
        self.assertEqual(result["legacy_boundary_count"], 0)

    def test_bool_mask_does_not_control_continuous_boost(self):
        arrays = self.arrays()
        before = validate_score_receipt(self.receipt(arrays), 9)
        arrays["boundary_mask"][:] = True
        after = validate_score_receipt(self.receipt(arrays), 9)
        self.assertEqual(before["increased_finite_scores"], after["increased_finite_scores"])
        self.assertEqual(before["combined"], after["combined"])

    def test_numpy_torch_float32_rounding_tolerance_is_bounded(self):
        arrays = self.arrays()
        arrays["boundary_log_boost"][4] = np.nextafter(arrays["boundary_log_boost"][4], np.float32(np.inf))
        arrays["scores"][4] = np.nextafter(arrays["scores"][4], np.float32(np.inf))
        validate_score_receipt(self.receipt(arrays), 9)
        arrays["scores"][4] += .01
        with self.assertRaises(ValueError):
            validate_score_receipt(self.receipt(arrays), 9)
        # Cancellation can magnify the relative error at a near-zero result;
        # rounding is bounded by the operand spacing instead.
        raw, boost = np.array([-1.5], np.float32), np.array([1.5], np.float32)
        self.assertGreater(float32_error(np.array([1e-7], np.float32), raw+boost, operands=(raw, boost)), 0.)

    def test_torch_cpu_log1p_snapshots_pass_independent_numpy_recomputation(self):
        import torch
        from utils.pruning_scores import continuous_boundary_scores
        arrays = self.arrays(1003)
        result = continuous_boundary_scores(torch.from_numpy(arrays["raw_pup_scores"]),
            *(arrays[key] for key in COMPONENTS), config=PARAMETERS)
        for key, value in result.items():
            arrays[key] = value.numpy()
        record = validate_score_receipt(self.receipt(arrays), 1003)
        self.assertGreater(record["increased_finite_scores"], 1000)

    def test_rejects_boolean_binary_nonfinite_or_wrong_formula(self):
        original = self.arrays()
        for mode in ("boolean", "binary", "bad_mixture", "bad_multiplier", "bad_boost", "nan", "plus_inf", "singular_changed"):
            arrays = copy.deepcopy(original)
            if mode == "boolean":
                arrays[COMPONENTS[0]] = arrays[COMPONENTS[0]].astype(bool)
            elif mode == "binary":
                for key in COMPONENTS:
                    arrays[key][:] = 0.
                arrays["combined_boundary_strength"][:] = 0.
                arrays["boundary_multiplier"][:] = 1.
                arrays["boundary_log_boost"][:] = 0.
                arrays["scores"] = arrays["raw_pup_scores"].copy()
            elif mode == "bad_mixture":
                arrays["combined_boundary_strength"][2] += .01
            elif mode == "bad_multiplier":
                arrays["boundary_multiplier"][2] += .01
            elif mode == "bad_boost":
                arrays["boundary_log_boost"][2] += .01
            elif mode == "nan":
                arrays["raw_pup_scores"][1] = np.nan
            elif mode == "plus_inf":
                arrays["scores"][1] = np.inf
            else:
                arrays["scores"][0] = -100.
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    validate_score_receipt(self.receipt(arrays), 9)

    def test_rejects_receipt_and_row_identity_corruption(self):
        receipt = self.receipt()
        for key, bad in (("mode", "pup_boundary_weighted"), ("sha256", "0"*64), ("count", 10)):
            broken = copy.deepcopy(receipt)
            broken[key] = bad
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_score_receipt(broken, 9)
        arrays = self.arrays()
        arrays["row_ids"][3] = arrays["row_ids"][2]
        with self.assertRaises(ValueError):
            validate_score_receipt(self.receipt(arrays), 9)

    def source(self):
        return {COMPONENTS[0]: np.array([.05, .25, .5, .85], np.float32),
                COMPONENTS[1]: np.array([.9, .3, .2, .6], np.float32),
                COMPONENTS[2]: np.array([.1, .7, .4, .8], np.float32)}

    def state_fixture(self):
        source = self.source()
        lineage = [dict(row_id=4, parent_row_ids=[0, 1], merge_weights=[.25, .75])]
        model = {key: np.concatenate((value[2:], np.array([np.float32(.25)*value[0]+np.float32(.75)*value[1]], np.float32)))
                 for key, value in source.items()}
        model.update(row_ids=np.array([2, 3, 4], np.int64), original_ids=np.array([2, 3, -1], np.int64),
                     confidence=np.ones(3, np.float32), extra_vector=np.ones((3, 2), np.float32))
        pool = {key: value[:2].copy() for key, value in source.items()}
        pool.update(row_ids=np.array([0, 1], np.int64), original_ids=np.array([0, 1], np.int64),
                    confidence=np.ones(2, np.float32), extra_vector=np.ones((2, 2), np.float32))
        payload = dict(metadata=copy.deepcopy(model), lineage=lineage,
                       removed_pool=[dict(metadata=pool, active=np.ones(2, bool),
                                          params=dict(xyz=np.zeros((2, 3), np.float32)), original_score=np.zeros(2, np.float32))])
        return source, model, payload

    def test_source_restore_merge_and_save_resume_metadata_alignment(self):
        source, model, payload = self.state_fixture()
        continuity = MetadataContinuity(source, payload["lineage"])
        result = validate_payload(payload, model, continuity)
        lineage = continuity.audit_lineage()
        self.assertTrue(result["export_resume_metadata_exact"])
        self.assertIn("extra_vector", result["floating_metadata_fields"])
        self.assertEqual(lineage["saved_merge_weights_recomputed"], 1)
        self.assertIsNone(lineage["merge_weight_limitation"])

    def test_saturated_metadata_uses_normalized_double_precision_merge_weights(self):
        source = {key: np.ones(2, np.float32) for key in COMPONENTS}
        weights = [.8020124435424805, .1979876458644867]
        self.assertGreater(sum(weights), 1.)
        continuity = MetadataContinuity(source,
            [dict(row_id=2, parent_row_ids=[0, 1], merge_weights=weights)])
        metadata = {key: np.ones(1, np.float32) for key in COMPONENTS}
        metadata.update(row_ids=np.array([2], np.int64), original_ids=np.array([-1], np.int64))
        continuity.observe(metadata)
        self.assertEqual(continuity.audit_lineage()["saved_merge_weights_recomputed"], 1)

    def test_rejects_changed_restore_row_or_recurring_merged_value(self):
        source, model, payload = self.state_fixture()
        for mode in ("source_reordered", "merged_changed", "wrong_parent_weight"):
            continuity = MetadataContinuity(source, payload["lineage"])
            original = copy.deepcopy(model)
            continuity.observe(original)
            if mode == "source_reordered":
                original[COMPONENTS[0]][0] = source[COMPONENTS[0]][0]
            elif mode == "merged_changed":
                original[COMPONENTS[1]][2] += .01
            else:
                continuity.lineage[4]["merge_weights"] = [.5, .5]
                with self.assertRaises(ValueError):
                    continuity.audit_lineage()
                continue
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    continuity.observe(original)

    def test_rejects_saved_dtype_length_or_export_disagreement(self):
        source, model, original = self.state_fixture()
        for mode in ("export_differs", "pool_missing_field", "pool_vector_unaligned", "pool_duplicate_active"):
            payload = copy.deepcopy(original)
            if mode == "export_differs":
                payload["metadata"]["extra_vector"][0, 0] = 0.
            elif mode == "pool_missing_field":
                payload["removed_pool"][0]["metadata"].pop(COMPONENTS[0])
            elif mode == "pool_vector_unaligned":
                payload["removed_pool"][0]["metadata"]["extra_vector"] = np.ones((2, 3), np.float32)
            else:
                payload["removed_pool"][0]["metadata"]["row_ids"][0] = 2
                payload["removed_pool"][0]["metadata"]["original_ids"][0] = 2
                for key in COMPONENTS:
                    payload["removed_pool"][0]["metadata"][key][0] = source[key][2]
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    validate_payload(payload, model, MetadataContinuity(source, payload["lineage"]))

    def selection(self, before, removed):
        return dict(target_reached=True, score_order="unchanged_global_ascending_stable", ranking_scope="global",
                    boundary_mode="none", boundary_protected=0, boundary_audit=[], requested_prune=removed,
                    actual_prune=removed, count_before=before, count_after=before-removed)

    def event(self):
        primary = self.receipt(self.arrays(1000), "primary")
        primary["post_merge"] = self.receipt(self.arrays(593), "post")
        selection = self.selection(1000, 402)
        selection["post_merge_pruning"] = self.selection(593, 5)
        event = dict(round_id=1, iteration=15700, before=1000, target=600, after_pruning=598,
                     after_merging=593, temporary_count=588, after_reallocation=600, pruned=407,
                     merged=5, restored=12, selection=selection, pup_scores=primary,
                     quota_audit=dict(instance_violations=0, boundary_violations=0, instances=[]))
        config = dict(gaussian_merging=dict(max_fraction=.01), budget_reallocation=dict(reserve_ratio=.02))
        return event, config

    def test_continuous_event_exact_budget_and_postmerge_score_audit(self):
        event, config = self.event()
        result = validate_event(event, config)
        self.assertEqual(result["merge_cap"], 10)
        self.assertEqual(result["restored"], 12)
        self.assertEqual(result["post_merge_scores"]["count"], 593)
        event["selection"]["boundary_mode"] = "quota"
        with self.assertRaises(ValueError):
            validate_event(event, config)


if __name__ == "__main__":
    unittest.main()
