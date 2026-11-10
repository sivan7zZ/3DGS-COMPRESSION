"""CPU-only independent audit of the boundary-weighted SH3 compression runs.

This reads trusted local checkpoints through ``verify_stage``; it never renders,
trains, or loads a CUDA model. Pure score, budget and RIR audits need only NumPy.
"""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "output/city2_boundary_weighted_20261106"
METHOD = "PUP_boundary_weighted_RIR_DPRA"
LOG_WEIGHT = 2.0


def read(path):
    return json.loads(Path(path).read_text())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".validation-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_snapshot(root, repository=ROOT):
    """Check both the frozen copy and the code used to produce the experiment."""
    folder = Path(root) / "source_snapshot"
    manifest = read(folder / "manifest.json")
    if not manifest.get("files"):
        raise ValueError("Missing frozen source hashes")
    for name, digest in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid frozen source path: " + name)
        for path in (folder / relative, Path(repository) / relative):
            if sha256_file(path) != digest:
                raise ValueError("Frozen experiment source changed: " + str(path))
    return dict(files=len(manifest["files"]), manifest_sha256=sha256_file(folder / "manifest.json"),
                frozen_copies_exact=True, current_sources_exact=True)


def validate_score_receipt(receipt, expected_count, log_weight=LOG_WEIGHT):
    """Recompute every weighted value, including the singular-Fisher convention."""
    if receipt.get("mode") != "pup_boundary_weighted" or receipt.get("boundary_log_weight") != log_weight:
        raise ValueError("Unexpected score mode or boundary weight")
    path = resolve(receipt["path"])
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Score snapshot SHA256 mismatch: " + str(path))
    with np.load(str(path), allow_pickle=False) as archive:
        required = {"scores", "raw_pup_scores", "boundary_mask", "row_ids"}
        if not required.issubset(archive.files):
            raise ValueError("Incomplete raw/weighted score snapshot")
        scores, raw = archive["scores"], archive["raw_pup_scores"]
        boundary, rows = archive["boundary_mask"], archive["row_ids"]
        if any(value.shape != (expected_count,) for value in (scores, raw, boundary, rows)):
            raise ValueError("Score snapshot row count mismatch")
        if scores.dtype.kind != "f" or raw.dtype != scores.dtype or boundary.dtype.kind != "b" or rows.dtype.kind not in "iu":
            raise ValueError("Unexpected score or mask dtype")
        if len(np.unique(rows)) != expected_count or np.any(rows < 0):
            raise ValueError("Duplicate or invalid active score row IDs")
        if any(np.isnan(value).any() or np.isposinf(value).any() for value in (raw, scores)):
            raise ValueError("NaN or +inf in PUP scores")
        expected = raw.copy()
        expected[boundary & np.isfinite(raw)] += log_weight
        if not np.array_equal(scores, expected):
            raise ValueError("Boundary score differs from raw + weight * mask")
        singular = int(np.isneginf(raw).sum())
        if not np.array_equal(np.isneginf(raw), np.isneginf(scores)):
            raise ValueError("Singular Fisher scores were changed")
        if receipt.get("count") != expected_count or receipt.get("negative_infinity_count") != singular:
            raise ValueError("Score receipt count mismatch")
        if receipt.get("boundary_count") != int(boundary.sum()):
            raise ValueError("Score receipt boundary count mismatch")
        return dict(path=str(path), sha256=receipt["sha256"], count=expected_count,
                    boundary_count=int(boundary.sum()), negative_infinity_count=singular,
                    finite_boundary_boosted=int(np.count_nonzero(boundary & np.isfinite(raw))),
                    weighted_values_exact=True, singular_scores_unchanged=True)


def validate_selection(selection):
    if (not selection.get("target_reached") or
            selection.get("score_order") != "unchanged_global_ascending_stable" or
            selection.get("ranking_scope") != "global"):
        raise ValueError("Unexpected constrained score ordering")
    if (selection.get("boundary_mode") != "none" or
            selection.get("boundary_protected") != 0 or selection.get("boundary_audit") != []):
        raise ValueError("Boundary survival constraint was used")
    if selection.get("requested_prune") != selection.get("actual_prune"):
        raise ValueError("Incomplete feasible pruning")


def validate_event(event, config):
    """Account for every removal, safe 2-to-1 merge and restoration slot."""
    before, target = int(event["before"]), int(event["target"])
    merge_config, refill = config["gaussian_merging"], config["budget_reallocation"]
    reserve = int(target * refill["reserve_ratio"]) if refill["enabled"] else 0
    temporary = max(1, target - reserve)
    cap = min(int(before * merge_config["max_fraction"]), before - target) if merge_config["enabled"] else 0
    merged, restored = int(event["merged"]), int(event["restored"])
    if not 0 <= merged <= cap or event["after_pruning"] != temporary + cap:
        raise ValueError("Merge budget or initial pruning count mismatch")
    if event["after_merging"] != event["after_pruning"] - merged:
        raise ValueError("A Gaussian merge did not reduce the count by one")
    if event["temporary_count"] != temporary or event["after_reallocation"] != target:
        raise ValueError("Temporary/final compression budget mismatch")
    if restored != target - temporary or event["after_reallocation"] != temporary + restored:
        raise ValueError("Restoration reserve was not filled exactly")
    expected_pruned = before - event["after_pruning"] + event["after_merging"] - temporary
    if event["pruned"] != expected_pruned or before - event["pruned"] - merged + restored != target:
        raise ValueError("Pruning/merge/refill accounting mismatch")
    audit = event["quota_audit"]
    if audit["instance_violations"] or audit["boundary_violations"]:
        raise ValueError("Historical quota violation")
    if any(row["current"] < row["minimum"] or row["boundary_minimum"] != 0 for row in audit["instances"]):
        raise ValueError("Instance floor violated or boundary floor retained")
    validate_selection(event["selection"])
    if (event["selection"]["count_before"] != before or
            event["selection"]["count_after"] != event["after_pruning"]):
        raise ValueError("Selection audit differs from initial pruning counts")
    primary = validate_score_receipt(event["pup_scores"], before)
    post = None
    if event["after_merging"] > temporary:
        validate_selection(event["selection"]["post_merge_pruning"])
        extra = event["selection"]["post_merge_pruning"]
        if extra["count_before"] != event["after_merging"] or extra["count_after"] != temporary:
            raise ValueError("Post-merge selection audit differs from model counts")
        post = validate_score_receipt(event["pup_scores"]["post_merge"], event["after_merging"])
    elif "post_merge" in event["pup_scores"]:
        raise ValueError("Unexpected score recomputation without post-merge pruning")
    return dict(round_id=event["round_id"], iteration=event["iteration"], before=before, target=target,
                merge_cap=cap, merged=merged, actual_merge_fraction=merged / float(before),
                reserved=reserve, restored=restored, actual_refill_fraction=restored / float(target),
                primary_scores=primary, post_merge_scores=post, exact_budget=True, boundary_constraints_used=False)


def replay_rir_dpra(stage, config):
    """Reimplement the old training-loop decisions using its original scheduler."""
    from utils.schedule_utils import TrainingScheduler
    settings = config["rir_dpra"]
    expected_settings = dict(virtual_start=15000, virtual_end=25199, pruning_start_iter=15200,
                             pruning_stop_iter=25200, pruning_interval=500,
                             pruning_interval_maximum=1500, beta=.99)
    if settings != expected_settings:
        raise ValueError("Original RIR/DPRA settings were changed")
    events, checks = stage["events"], stage["rir_checks"]
    by_iteration = {event["iteration"]: event for event in events}
    if len(events) != 5 or len(by_iteration) != 5:
        raise ValueError("Expected five distinct nonzero compression events")
    target, current = int(config["pruning_target"]), int(stage["source_count"])
    scheduler = TrainingScheduler(SimpleNamespace(pruning_stop_iter=25200))
    scheduler.calculate_pruning_target(current, target)
    with contextlib.redirect_stdout(io.StringIO()):
        rate = scheduler.get_pruning_rate(15200, current, 1)
    stamp, waiting, count, offset, used = 1., 0, 2, 0, []
    for iteration in range(15700, 25200, 500):
        if count > 6:
            break
        if offset >= len(checks):
            raise ValueError("Missing RIR check")
        check = checks[offset]
        offset += 1
        loss = float(check["current_loss"])
        if not math.isfinite(loss) or check["iteration"] != iteration:
            raise ValueError("Invalid reconstruction loss or RIR check interval")
        forced, recovered = waiting + 500 >= 1500, loss < .99 * stamp
        trigger, goal = forced or recovered, max(target, int(round(current * (1. - rate))))
        expected = dict(forced=forced, recovered=recovered, trigger=trigger, pruning_count=count,
                        waiting_iterations=waiting, round_target=goal, loss_stamp=stamp,
                        threshold=.99 * stamp, pruning_ratio=rate)
        for key, value in expected.items():
            same = np.isclose(check.get(key, float("nan")), value, rtol=1e-12, atol=1e-12) if isinstance(value, float) else check.get(key) == value
            if not same:
                raise ValueError("RIR/DPRA replay differs: " + key)
        if trigger:
            if current > target:
                event = by_iteration.get(iteration)
                if event is None or event["before"] != current or event["after_reallocation"] != goal or event["rir"] != check:
                    raise ValueError("Compression event differs from RIR/DPRA decision")
                used.append(iteration)
            current, stamp, waiting = goal, loss, 0
            with contextlib.redirect_stdout(io.StringIO()):
                rate = scheduler.get_pruning_rate(iteration, current, count)
            count += 1
        else:
            waiting += 500
    if offset != len(checks) or set(used) != set(by_iteration) or current != target:
        raise ValueError("Incomplete or extra RIR checks/events")
    return dict(matches_original=True, checks=offset, event_iterations=used, final_count=current)


def validate(root, ratio):
    """Validate one completed SH3 run and save ``full_<ratio>x/validation.json``."""
    from validate_adaptive_results import verify_stage
    from utils.adaptive_schedule import survival_quotas, audit_quotas
    root, ratio = resolve(root), int(ratio)
    if ratio not in (10, 20):
        raise ValueError("Expected ratio 10 or 20")
    frozen = validate_snapshot(root)
    run = root / ("full_%dx" % ratio)
    config, pipeline, stage = read(run / "config.json"), read(run / "pipeline.json"), read(run / "stage3/run_args.json")
    if pipeline["status"] != "complete" or stage["status"] != "complete":
        raise ValueError("Incomplete compression/render/metrics pipeline")
    if pipeline["config"] != config or stage["config"] != config:
        raise ValueError("Pipeline/stage configurations differ")
    if (pipeline.get("distillation") is not False or
            pipeline["completed"] != ["compress", "render_stage3", "metrics"] or
            "teacher_checkpoint" in config or "distill_steps" in config):
        raise ValueError("Expected the explicit compression/render/metrics-only pipeline")
    if (config["pruning_score"] != "pup_boundary_weighted" or config["compression_schedule"] != "rir_dpra" or
            stage["method"] != METHOD or stage["completed_steps"] != 10199 or stage["final_iteration"] != 25199 or
            stage["virtual_start"] != 15000 or stage["virtual_end"] != 25199):
        raise ValueError("Unexpected score mode, method or recovery iteration count")
    if (run / "sh2").exists() or any("distill" in name or "sh2" in name for name in pipeline["completed"]):
        raise ValueError("Unexpected SH distillation output or stage")
    policy = dict(mode="pup_boundary_weighted", boundary_weighted=True, log_weight=LOG_WEIGHT,
                  boundary_constraint_mode="none", preserve_boundary_merge=False)
    protection = read(run / "stage3/protection_policy.json")
    if stage["score_policy"] != policy or protection["score_policy"] != policy or protection["boundary_min"] != {}:
        raise ValueError("Boundary score/quota/merge policy differs")
    if config.get("boundary_score", {}).get("log_weight") != LOG_WEIGHT or config["boundary_protection"]["mode"] != "none":
        raise ValueError("Boundary weighting configuration differs")
    if any(row["boundary_minimum"] != 0 for row in protection["rows"]):
        raise ValueError("Boundary quotas remain in initial protection policy")
    if (config["gaussian_merging"]["max_fraction"] != .01 or config["budget_reallocation"]["reserve_ratio"] != .02 or
            not config["gaussian_merging"]["enabled"] or not config["budget_reallocation"]["enabled"]):
        raise ValueError("Expected 1% merge cap and 2% restoration reserve")
    previous = read(ROOT / ("output/city2_pup_rir_dpra_20261105/full_%dx/config.json" % ratio))
    preserved = ("instance_protection", "topology_protection", "recovery_losses", "preserve_v3_losses",
                 "structure_loss", "shared_metadata", "legacy_structure_roi_dir")
    if any(config[key] != previous[key] for key in preserved):
        raise ValueError("Instance/topology/recovery configuration changed from v3.2")
    shared = resolve(config["shared_metadata"])
    with np.load(str(shared / "adaptive_metadata.npz"), allow_pickle=False) as archive:
        initial = {key: archive[key] for key in ("instance_ids", "instance_fragility", "boundary_mask")}
    quota_config = dict(config, boundary_protection=dict(config["boundary_protection"], enabled=False))
    amin, _, _ = survival_quotas(initial, quota_config)
    recorded = {int(key): int(value) for key, value in protection["instance_min"].items()}
    if amin != recorded or stage["source_count"] != len(initial["instance_ids"]):
        raise ValueError("Initial instance quotas or source count differ")
    if stage["metadata_sha256"] != sha256_file(shared / "adaptive_metadata.npz") or stage["input_ply_sha256"] != sha256_file(resolve(config["input_ply"])):
        raise ValueError("Compression source/metadata changed")
    expected = {10: 178809, 20: 89405}[ratio]
    if config["pruning_target"] != expected or stage["target"] != expected:
        raise ValueError("Unexpected final compression budget")
    checkpoint = run / "stage3/chkpnt25199.pth"
    ply = run / "stage3/point_cloud/iteration_25199/point_cloud.ply"
    if resolve(stage["last_checkpoint"]).resolve() != checkpoint.resolve() or resolve(stage["last_ply"]).resolve() != ply.resolve():
        raise ValueError("Final native models are not the local completed SH3 stage")
    native, metadata = verify_stage(run / "stage3", checkpoint, ply, expected)
    if native["standard_property_count"] != 62:
        raise ValueError("Expected a standard SH3 model")
    quota_audit = audit_quotas(metadata, amin, {})
    replay = replay_rir_dpra(stage, config)
    rounds = [validate_event(event, config) for event in stage["events"]]
    result = dict(status="complete", ratio=ratio, target=expected, SH3=native, method=METHOD,
                  source_snapshot=frozen, score_mode="pup_boundary_weighted", boundary_log_weight=LOG_WEIGHT,
                  boundary_quotas={}, preserve_boundary_merge=False, distillation_updates=0,
                  instance_topology_settings_unchanged=True, rir_dpra_matches_original=True,
                  schedule_replay=replay, recovery_updates=10199, compression_events=len(rounds),
                  event_iterations=replay["event_iterations"], rounds=rounds, quota_violations=0,
                  protected_instances=len(amin), final_quota_audit=quota_audit,
                  metadata_rows_exact=True, merged_total=sum(item["merged"] for item in rounds),
                  restored_total=sum(item["restored"] for item in rounds))
    atomic_json_save(result, run / "validation.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ratio", choices=("10", "20", "all"), default="all")
    args = parser.parse_args()
    ratios = (10, 20) if args.ratio == "all" else (int(args.ratio),)
    results = {str(ratio): validate(args.root, ratio) for ratio in ratios}
    if args.ratio == "all":
        atomic_json_save(dict(status="complete", ratios=results), resolve(args.root) / "final_validation.json")
    print(json.dumps(dict(status="complete", ratios=results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
