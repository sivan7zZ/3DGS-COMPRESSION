import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from validate_boundary_weighted_results import (
    replay_rir_dpra, sha256_file, validate_event, validate_score_receipt, validate_snapshot,
)
from utils.rir_dpra_schedule import RIRDPRA


class BoundaryWeightedValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def receipt(self, count=7, name="scores", raw=None, weighted=None):
        raw = np.linspace(-10., 10., count, dtype=np.float32) if raw is None else np.asarray(raw, np.float32)
        raw[0] = -np.inf
        boundary = np.arange(count) % 2 == 0
        scores = raw.copy()
        scores[boundary & np.isfinite(scores)] += 2.
        if weighted is not None:
            scores = np.asarray(weighted, np.float32)
        path = self.folder / (name + ".npz")
        np.savez_compressed(str(path), raw_pup_scores=raw, scores=scores,
                            boundary_mask=boundary, row_ids=np.arange(count, dtype=np.int64))
        return dict(path=str(path), sha256=sha256_file(path), mode="pup_boundary_weighted",
                    boundary_log_weight=2., count=count, boundary_count=int(boundary.sum()),
                    negative_infinity_count=int(np.isneginf(raw).sum()))

    def selection(self, before, removed):
        return dict(target_reached=True, score_order="unchanged_global_ascending_stable", ranking_scope="global",
                    boundary_mode="none", boundary_protected=0, boundary_audit=[],
                    requested_prune=removed, actual_prune=removed, count_before=before, count_after=before-removed)

    def event(self):
        config = dict(gaussian_merging=dict(enabled=True, max_fraction=.01),
                      budget_reallocation=dict(enabled=True, reserve_ratio=.02))
        # 1000 -> 598 after ordinary pruning -> 593 after five merges -> 588 -> 600.
        primary = self.receipt(1000, "primary")
        primary["post_merge"] = self.receipt(593, "post_merge")
        selection = self.selection(1000, 402)
        selection["post_merge_pruning"] = self.selection(593, 5)
        event = dict(round_id=1, iteration=15700, before=1000, target=600, after_pruning=598,
                     after_merging=593, temporary_count=588, after_reallocation=600,
                     pruned=407, merged=5, restored=12, selection=selection, pup_scores=primary,
                     quota_audit=dict(instance_violations=0, boundary_violations=0,
                                      instances=[dict(current=20, minimum=8, boundary_minimum=0)]))
        return event, config

    def test_recomputes_exact_weight_and_keeps_singular_boundary_scores(self):
        record = validate_score_receipt(self.receipt(), 7)
        self.assertTrue(record["weighted_values_exact"])
        self.assertEqual(record["negative_infinity_count"], 1)
        self.assertEqual(record["finite_boundary_boosted"], 3)

    def test_rejects_wrong_weights_nan_positive_infinity_and_checksum(self):
        invalid_values = [np.array([-np.inf, 0., 4., 0., 4., 0., 4.], np.float32),
                          np.array([-np.inf, np.nan, 4., 0., 4., 0., 4.], np.float32),
                          np.array([-np.inf, np.inf, 4., 0., 4., 0., 4.], np.float32)]
        for values in invalid_values:
            with self.subTest(values=str(values)):
                with self.assertRaises(ValueError):
                    validate_score_receipt(self.receipt(weighted=values), 7)
        receipt = self.receipt()
        receipt["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            validate_score_receipt(receipt, 7)

    def test_rejects_count_mask_and_active_lineage_corruption(self):
        receipt = self.receipt()
        path = Path(receipt["path"])
        with np.load(str(path)) as archive:
            original = {key: archive[key] for key in archive.files}
        for key, bad in (("row_ids", np.zeros(7, np.int64)),
                         ("boundary_mask", np.ones(7, np.int8)),
                         ("scores", original["scores"][:6])):
            arrays = dict(original)
            arrays[key] = bad
            np.savez_compressed(str(path), **arrays)
            receipt["sha256"] = sha256_file(path)
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    validate_score_receipt(receipt, 7)

    def test_exact_merge_refill_accounting_includes_post_merge_scores(self):
        event, config = self.event()
        result = validate_event(event, config)
        self.assertEqual(result["merge_cap"], 10)
        self.assertEqual(result["reserved"], 12)
        self.assertEqual(result["actual_refill_fraction"], .02)
        self.assertEqual(result["post_merge_scores"]["count"], 593)

    def test_rejects_hidden_boundary_constraint_or_budget_drift(self):
        original, config = self.event()
        for path, value in ((["restored"], 11), (["merged"], 11),
                            (["temporary_count"], 589), (["pruned"], 406),
                            (["selection", "boundary_mode"], "quota"),
                            (["selection", "post_merge_pruning", "boundary_protected"], 1)):
            event = copy.deepcopy(original)
            current = event
            for key in path[:-1]:
                current = current[key]
            current[path[-1]] = value
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_event(event, config)

    def stage(self, target=178809, recovered=True):
        settings = dict(virtual_start=15000, virtual_end=25199, pruning_start_iter=15200,
                        pruning_stop_iter=25200, pruning_interval=500,
                        pruning_interval_maximum=1500, beta=.99)
        schedule = RIRDPRA(settings, target)
        current, checks, events = 1788093, [], []
        with contextlib.redirect_stdout(io.StringIO()):
            schedule.initialize(15200, current)
            for iteration in range(15700, 25200, 500):
                if not schedule.is_check(iteration):
                    continue
                decision = schedule.decide(iteration, current, .01 if recovered else 2.)
                checks.append(decision)
                if decision["trigger"]:
                    if current > target:
                        events.append(dict(iteration=iteration, before=current,
                                           after_reallocation=decision["round_target"], rir=decision))
                    current = decision["round_target"]
                    schedule.complete(iteration, current, decision["current_loss"])
        return dict(source_count=1788093, events=events, rir_checks=checks), dict(rir_dpra=settings, pruning_target=target)

    def test_replays_original_schedule_for_10x_20x_and_forced_events(self):
        for target in (178809, 89405):
            for recovered in (True, False):
                with self.subTest(target=target, recovered=recovered):
                    stage, config = self.stage(target, recovered)
                    record = replay_rir_dpra(stage, config)
                    self.assertTrue(record["matches_original"])
                    self.assertEqual(len(record["event_iterations"]), 5)
                    self.assertEqual(record["final_count"], target)

    def test_rejects_missing_rir_check_and_changed_decision(self):
        original, config = self.stage()
        for mode in ("missing_check", "changed_trigger", "changed_budget"):
            stage = copy.deepcopy(original)
            if mode == "missing_check":
                stage["rir_checks"].pop(1)
            elif mode == "changed_trigger":
                stage["rir_checks"][0]["forced"] = not stage["rir_checks"][0]["forced"]
            else:
                stage["events"][0]["after_reallocation"] -= 1
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    replay_rir_dpra(stage, config)

    def test_source_audit_checks_frozen_and_live_bytes(self):
        repository, output = self.folder / "repo", self.folder / "output"
        repository.mkdir()
        frozen = output / "source_snapshot"
        frozen.mkdir(parents=True)
        live, copy_path = repository / "train.py", frozen / "train.py"
        live.write_text("original\n")
        copy_path.write_bytes(live.read_bytes())
        (frozen / "manifest.json").write_text(json.dumps(dict(files={"train.py": sha256_file(live)})))
        self.assertTrue(validate_snapshot(output, repository)["current_sources_exact"])
        for path in (live, copy_path):
            path.write_text("changed\n")
            with self.assertRaises(ValueError):
                validate_snapshot(output, repository)
            path.write_text("original\n")


if __name__ == "__main__":
    unittest.main()
