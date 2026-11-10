"""Independent CPU validation of continuous SAM/depth/RGB boundary scoring.

The native model and resume payload are read on CPU. This module never trains
or renders. Its score and metadata audits need only NumPy.
"""
import argparse
from pathlib import Path

import numpy as np

from validate_boundary_weighted_results import (
    ROOT, atomic_json_save, read, replay_rir_dpra, resolve, sha256_file,
    validate_selection, validate_snapshot,
)


DEFAULT_OUTPUT = ROOT / "output/city2_continuous_boundary_20261109"
METHOD = "PUP_continuous_boundary_RIR_DPRA"
MODE = "pup_continuous_boundary"
COMPONENTS = ("sam_boundary_strength", "depth_boundary_strength", "rgb_boundary_strength")
PARAMETERS = dict(sam_weight=.4, depth_weight=.4, rgb_edge_weight=.2, gain=5., fusion="log1p")
FLOAT32_EPS = float(np.finfo(np.float32).eps)


def components_array(arrays, count):
    columns = []
    for key in COMPONENTS:
        value = np.asarray(arrays[key])
        if value.shape != (count,) or value.dtype != np.dtype("float32"):
            raise ValueError("Continuous component must be aligned float32: " + key)
        if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1):
            raise ValueError("Continuous component outside finite [0,1]: " + key)
        columns.append(value)
    return np.stack(columns, axis=1)


def float32_error(actual, expected, operands=(), ulps=4):
    """Bound rounding by float32 spacing, including cancellation in raw+boost."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != np.dtype("float32") or expected.dtype != np.dtype("float32") or actual.shape != expected.shape:
        raise ValueError("Expected aligned float32 numerical audit arrays")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite numerical audit arrays")
    spacing = np.abs(np.spacing(expected)).astype(np.float64)
    for value in operands:
        value = np.asarray(value, dtype=np.float32)
        spacing += np.abs(np.spacing(value)).astype(np.float64)
    bound = ulps * np.maximum(spacing, float(np.nextafter(np.float32(0), np.float32(1))))
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    if np.any(error > bound):
        raise ValueError("Continuous score numerical mismatch exceeds float32 ULP bound")
    return float(error.max()) if error.size else 0.


def continuous_summary(value):
    value = np.asarray(value)
    return dict(minimum=float(value.min()), maximum=float(value.max()),
                unique_values=int(len(np.unique(value))),
                interior_values=int(np.count_nonzero((value > 0) & (value < 1))),
                quantiles={str(q): float(np.quantile(value, q)) for q in (0., .01, .1, .25, .5, .75, .9, .99, 1.)})


class MetadataContinuity:
    """Track immutable continuous priors by source or newly merged lineage row."""
    def __init__(self, source, lineage=()):
        self.count = len(source[COMPONENTS[0]])
        self.source = components_array(source, self.count)
        self.lineage = {int(row["row_id"]): row for row in lineage}
        if len(self.lineage) != len(lineage):
            raise ValueError("Duplicate merged lineage row IDs")
        self.merged = {}
        self.observations = 0

    def observe(self, arrays):
        rows, original = np.asarray(arrays["row_ids"]), np.asarray(arrays["original_ids"])
        count = len(rows)
        values = components_array(arrays, count)
        if rows.shape != (count,) or original.shape != (count,) or rows.dtype.kind not in "iu" or original.dtype.kind != "i":
            raise ValueError("Continuous lineage IDs must be aligned integers")
        if len(np.unique(rows)) != count or np.any(rows < 0) or np.any(original < -1):
            raise ValueError("Invalid or duplicate continuous lineage IDs")
        native = original >= 0
        if np.any(original[native] >= self.count) or not np.array_equal(rows[native], original[native]):
            raise ValueError("Restored/native continuous lineage differs from source")
        if not np.array_equal(values[native], self.source[original[native]]):
            raise ValueError("Source/restore continuous components changed or lost row alignment")
        if np.any(rows[~native] < self.count):
            raise ValueError("Merged row reused a source row ID")
        for row, value in zip(rows[~native], values[~native]):
            row = int(row)
            if row in self.merged and not np.array_equal(self.merged[row], value):
                raise ValueError("Continuous merged-row prior changed across score/state snapshots")
            if row not in self.lineage:
                raise ValueError("Missing continuous merged-row lineage")
            self.merged[row] = value.copy()
        self.observations += count
        return dict(rows=count, source_rows=int(native.sum()), merged_rows=int((~native).sum()))

    def audit_lineage(self):
        if set(self.merged) != set(self.lineage):
            raise ValueError("Some merged continuous metadata was never observed")
        exact_weights = 0
        for row, record in self.lineage.items():
            parents = list(map(int, record["parent_row_ids"]))
            if len(parents) != 2 or len(set(parents)) != 2 or any(parent >= row for parent in parents):
                raise ValueError("Invalid continuous merge ancestry")
            try:
                parent_values = np.stack([self.source[p] if p < self.count else self.merged[p] for p in parents])
            except (KeyError, IndexError):
                raise ValueError("Unobserved continuous merge parent")
            current = self.merged[row]
            weights = record.get("merge_weights")
            if weights is not None:
                weights = np.asarray(weights, np.float32)
                if weights.shape != (2,) or not np.isfinite(weights).all() or np.any(weights < 0) or abs(float(weights.sum()) - 1.) > 4 * FLOAT32_EPS:
                    raise ValueError("Invalid saved moment-matching weights")
                # The core normalizes the same moment-matching masses in double
                # precision before returning bounded float32 metadata; two raw
                # float32 masses can sum to slightly more than one.
                stable_weights = weights.astype(np.float64)
                expected = ((stable_weights[0] * parent_values[0].astype(np.float64) +
                             stable_weights[1] * parent_values[1].astype(np.float64)) /
                            stable_weights.sum()).astype(np.float32)
                float32_error(current, expected, operands=(parent_values[0], parent_values[1]))
                exact_weights += 1
            else:
                # No new weight is invented. With legacy lineage records, test
                # whether all channels admit one common convex-combination weight.
                delta = parent_values[0] - parent_values[1]
                channel = int(np.argmax(np.abs(delta)))
                weight = float((current[channel] - parent_values[1, channel]) / delta[channel]) if abs(float(delta[channel])) > 8 * FLOAT32_EPS else .5
                if weight < -8 * FLOAT32_EPS or weight > 1. + 8 * FLOAT32_EPS:
                    raise ValueError("Merged continuous value lies outside parent convex hull")
                expected = weight * parent_values[0].astype(np.float64) + (1. - weight) * parent_values[1].astype(np.float64)
                if np.max(np.abs(current.astype(np.float64) - expected)) > 8 * FLOAT32_EPS:
                    raise ValueError("Merged continuous channels do not share one parent weighting")
        return dict(merged_rows=len(self.merged), observations=self.observations,
                    source_and_restore_values_exact=True, recurring_row_values_exact=True,
                    merge_parent_common_convex_weight_verified=True,
                    saved_merge_weights_recomputed=exact_weights,
                    merge_weight_limitation=None if exact_weights == len(self.merged) else
                    "Legacy lineage omits some moment weights; those rows pass a common-parent convex-combination check, not exact opacity-derived weight recomputation.")


def validate_score_receipt(receipt, expected_count, continuity=None):
    if (receipt.get("mode") != MODE or receipt.get("continuous_boundary") is not True or
            receipt.get("boundary_score") != PARAMETERS):
        raise ValueError("Unexpected continuous score mode or parameters")
    path = resolve(receipt["path"])
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Continuous score snapshot SHA256 mismatch")
    with np.load(str(path), allow_pickle=False) as archive:
        required = {"scores", "raw_pup_scores", "row_ids", "original_ids", "boundary_mask",
                    "combined_boundary_strength", "boundary_multiplier", "boundary_log_boost"} | set(COMPONENTS)
        if not required <= set(archive.files):
            raise ValueError("Incomplete continuous score snapshot")
        arrays = {key: archive[key] for key in required}
    if any(value.shape != (expected_count,) for value in arrays.values()):
        raise ValueError("Continuous score row count mismatch")
    rows, original, mask = arrays["row_ids"], arrays["original_ids"], arrays["boundary_mask"]
    if (rows.dtype.kind not in "iu" or original.dtype.kind != "i" or mask.dtype.kind != "b" or
            len(np.unique(rows)) != expected_count or np.any(rows < 0) or np.any(original < -1) or
            not np.array_equal(rows[original >= 0], original[original >= 0])):
        raise ValueError("Invalid continuous score lineage or legacy boundary mask")
    components = components_array(arrays, expected_count)
    raw, scores = arrays["raw_pup_scores"], arrays["scores"]
    if raw.dtype != np.dtype("float32") or scores.dtype != np.dtype("float32"):
        raise ValueError("Expected native float32 PUP scores")
    if any(np.isnan(value).any() or np.isposinf(value).any() for value in (raw, scores)):
        raise ValueError("NaN or +inf in continuous PUP scores")
    combined = np.float32(.4) * components[:, 0] + np.float32(.4) * components[:, 1] + np.float32(.2) * components[:, 2]
    gain_strength = np.float32(5.) * combined
    multiplier = np.float32(1.) + gain_strength
    boost = np.log1p(gain_strength)
    errors = {}
    for key, expected in (("combined_boundary_strength", combined), ("boundary_multiplier", multiplier), ("boundary_log_boost", boost)):
        errors[key] = float32_error(arrays[key], expected)
    finite = np.isfinite(raw)
    expected = raw[finite] + boost[finite]
    errors["weighted_scores"] = float32_error(scores[finite], expected, operands=(raw[finite], boost[finite]))
    if not np.array_equal(np.isneginf(raw), np.isneginf(scores)):
        raise ValueError("Continuous weighting changed singular Fisher scores")
    if np.any(scores[finite] < raw[finite]):
        raise ValueError("Continuous boundary weighting decreased importance")
    if receipt.get("count") != expected_count or receipt.get("negative_infinity_count") != int((~finite).sum()) or receipt.get("boundary_count") != int(mask.sum()):
        raise ValueError("Continuous score receipt counts differ")
    if not np.any((combined > 0) & (combined < 1)) or len(np.unique(combined)) <= 2:
        raise ValueError("Continuous boundary scoring collapsed to a binary/constant map")
    if continuity is not None:
        continuity.observe(arrays)
    return dict(path=str(path), sha256=receipt["sha256"], count=expected_count,
                negative_infinity_count=int((~finite).sum()), legacy_boundary_count=int(mask.sum()),
                continuous_components={key: continuous_summary(arrays[key]) for key in COMPONENTS},
                combined=continuous_summary(combined), multiplier_range=[float(multiplier.min()), float(multiplier.max())],
                increased_finite_scores=int(np.count_nonzero(scores[finite] > raw[finite])),
                positive_continuous_boost=int(np.count_nonzero(boost > 0)),
                maximum_float32_absolute_errors=errors, float32_ulp_tolerance=4,
                singular_scores_unchanged=True, used_legacy_mask_for_weighting=False)


def validate_event(event, config, continuity=None):
    before, target = int(event["before"]), int(event["target"])
    reserve = int(target * config["budget_reallocation"]["reserve_ratio"])
    temporary = max(1, target - reserve)
    cap = min(int(before * config["gaussian_merging"]["max_fraction"]), before - target)
    merged, restored = int(event["merged"]), int(event["restored"])
    if not 0 <= merged <= cap or event["after_pruning"] != temporary + cap:
        raise ValueError("Continuous run merge cap or initial pruning count differs")
    if event["after_merging"] != event["after_pruning"] - merged or event["temporary_count"] != temporary:
        raise ValueError("Continuous run merge/temporary count differs")
    if restored != target - temporary or event["after_reallocation"] != temporary + restored:
        raise ValueError("Continuous run restoration did not exactly fill its reserve")
    expected_pruned = before - event["after_pruning"] + event["after_merging"] - temporary
    if event["pruned"] != expected_pruned or before - expected_pruned - merged + restored != target:
        raise ValueError("Continuous run exact budget accounting differs")
    audit = event["quota_audit"]
    if audit["instance_violations"] or audit["boundary_violations"] or any(row["current"] < row["minimum"] or row["boundary_minimum"] != 0 for row in audit["instances"]):
        raise ValueError("Instance quota violated or boundary quota retained")
    selection = event["selection"]
    validate_selection(selection)
    if selection["count_before"] != before or selection["count_after"] != event["after_pruning"]:
        raise ValueError("Initial score selection count differs")
    primary = validate_score_receipt(event["pup_scores"], before, continuity)
    post = None
    if event["after_merging"] > temporary:
        extra = selection["post_merge_pruning"]
        validate_selection(extra)
        if extra["count_before"] != event["after_merging"] or extra["count_after"] != temporary:
            raise ValueError("Post-merge selection count differs")
        post = validate_score_receipt(event["pup_scores"]["post_merge"], event["after_merging"], continuity)
    elif "post_merge" in event["pup_scores"]:
        raise ValueError("Unexpected continuous score recomputation")
    return dict(round_id=event["round_id"], iteration=event["iteration"], before=before, target=target,
                merge_cap=cap, merged=merged, actual_merge_fraction=merged / float(before),
                reserved=reserve, restored=restored, actual_refill_fraction=restored / float(target),
                primary_scores=primary, post_merge_scores=post, exact_budget=True, boundary_constraints_used=False)


def validate_payload(payload, metadata, continuity):
    """Check exported, resumable and pooled arrays without creating a model."""
    saved = payload["metadata"]
    if set(saved) != set(metadata) or any(np.asarray(saved[key]).dtype != np.asarray(metadata[key]).dtype or
            not np.array_equal(saved[key], metadata[key]) for key in metadata):
        raise ValueError("Exported continuous metadata differs from saved resume state")
    count = len(metadata["row_ids"])
    all_float_fields = []
    for key, value in saved.items():
        value = np.asarray(value)
        if value.ndim == 0 or len(value) != count:
            raise ValueError("Saved metadata field lost row alignment: " + key)
        if value.dtype.kind == "f":
            if not np.isfinite(value).all():
                raise ValueError("Non-finite saved metadata field: " + key)
            all_float_fields.append(key)
    continuity.observe(saved)
    pool_rows, active_rows = 0, 0
    active_source_ids = set(map(int, saved["row_ids"]))
    for chunk in payload["removed_pool"]:
        size = len(chunk["active"])
        if set(chunk["metadata"]) != set(saved) or chunk["active"].shape != (size,) or chunk["active"].dtype.kind != "b":
            raise ValueError("Removed pool metadata schema differs")
        for key, value in chunk["metadata"].items():
            if np.asarray(value).shape != (size,) + np.asarray(saved[key]).shape[1:]:
                raise ValueError("Removed pool field lost row alignment: " + key)
            if np.asarray(value).dtype != np.asarray(saved[key]).dtype or (np.asarray(value).dtype.kind == "f" and not np.isfinite(value).all()):
                raise ValueError("Removed pool field dtype/value differs: " + key)
        if any(len(value) != size for value in chunk["params"].values()) or len(chunk["original_score"]) != size:
            raise ValueError("Removed pool native parameters are not aligned")
        continuity.observe(chunk["metadata"])
        live = chunk["metadata"]["row_ids"][chunk["active"]]
        if any(int(row) in active_source_ids for row in live):
            raise ValueError("Active pool row also exists in the model or another pool chunk")
        active_source_ids.update(map(int, live))
        pool_rows += size
        active_rows += int(chunk["active"].sum())
    return dict(export_resume_metadata_exact=True, all_metadata_fields_aligned=True,
                floating_metadata_fields=sorted(all_float_fields), pool_rows=pool_rows,
                active_pool_rows=active_rows, source_and_restore_continuous_values_exact=True)


def validate(root, ratio):
    """Audit one completed 10x/20x SH3 run; save its independent receipt."""
    from validate_adaptive_results import verify_stage
    from utils.adaptive_schedule import survival_quotas, audit_quotas
    from utils.distill_utils import load_torch_file
    root, ratio = resolve(root), int(ratio)
    if ratio not in (10, 20):
        raise ValueError("Expected ratio 10 or 20")
    frozen = validate_snapshot(root)
    run = root / ("full_%dx" % ratio)
    config, pipeline, stage = read(run / "config.json"), read(run / "pipeline.json"), read(run / "stage3/run_args.json")
    if pipeline["status"] != "complete" or stage["status"] != "complete" or pipeline["config"] != config or stage["config"] != config:
        raise ValueError("Incomplete or inconsistent continuous SH3 pipeline")
    if pipeline.get("distillation") is not False or pipeline["completed"] != ["compress", "render_stage3", "metrics"] or any(key in config for key in ("teacher_checkpoint", "distill_steps")) or (run / "sh2").exists():
        raise ValueError("Unexpected distillation settings or outputs")
    if (config["pruning_score"] != MODE or config["compression_schedule"] != "rir_dpra" or stage["method"] != METHOD or
            stage["completed_steps"] != 10199 or stage["final_iteration"] != 25199 or stage["virtual_start"] != 15000 or stage["virtual_end"] != 25199):
        raise ValueError("Unexpected continuous method or RIR/DPRA iterations")
    policy = dict(mode=MODE, boundary_weighted=True, log_weight=0., boundary_constraint_mode="none",
                  preserve_boundary_merge=False, continuous_boundary=True, boundary_score=PARAMETERS)
    protection = read(run / "stage3/protection_policy.json")
    if stage["score_policy"] != policy or protection["score_policy"] != policy or protection["boundary_min"] != {} or any(row["boundary_minimum"] for row in protection["rows"]):
        raise ValueError("Continuous boundary score/quota/merge policy differs")
    if config["boundary_score"] != PARAMETERS or config["boundary_protection"]["mode"] != "none":
        raise ValueError("Continuous score configuration differs")
    if not config["gaussian_merging"]["enabled"] or config["gaussian_merging"]["max_fraction"] != .01 or not config["budget_reallocation"]["enabled"] or config["budget_reallocation"]["reserve_ratio"] != .02:
        raise ValueError("Expected 1% merge cap and 2% refill reserve")
    previous = read(ROOT / ("output/city2_boundary_weighted_20261106/full_%dx/config.json" % ratio))
    preserved = ("instance_protection", "topology_protection", "recovery_losses", "preserve_v3_losses", "structure_loss",
                 "shared_metadata", "legacy_structure_roi_dir", "source_path", "input_ply", "gaussian_merging", "budget_reallocation")
    if any(config[key] != previous[key] for key in preserved):
        raise ValueError("Instance/topology/loss/mutation/source settings changed from Nov6")
    shared = resolve(config["shared_metadata"])
    shared_manifest = read(shared / "metadata_manifest.json")
    with np.load(str(shared / "adaptive_metadata.npz"), allow_pickle=False) as archive:
        initial = {key: archive[key] for key in ("instance_ids", "instance_fragility", "boundary_mask", "visibility")}
    source_count = len(initial["instance_ids"])
    # The sidecar path/receipt are also checked against the training manifest below.
    sidecar = resolve(config["continuous_boundary_metadata"])
    continuous_manifest = read(sidecar.parent / "metadata_manifest.json")
    if continuous_manifest.get("status") != "complete":
        raise ValueError("Continuous source metadata is incomplete")
    source_sha, shared_sha = sha256_file(resolve(config["input_ply"])), sha256_file(shared / "adaptive_metadata.npz")
    for key, expected in (("source_ply_sha256", source_sha), ("source_xyz_sha256", shared_manifest["source_xyz_sha256"]),
                          ("shared_metadata_sha256", shared_sha), ("source_count", source_count),
                          ("train_frames", shared_manifest["train_frames"]), ("test_frames_excluded", shared_manifest["test_frames_excluded"]),
                          ("sidecar_sha256", sha256_file(sidecar))):
        if continuous_manifest[key] != expected:
            raise ValueError("Continuous sidecar manifest differs: " + key)
    with np.load(str(sidecar), allow_pickle=False) as archive:
        priors = {key: archive[key] for key in archive.files}
    if any(priors[key].shape != () or priors[key].dtype.kind != "U" for key in ("source_ply_sha256", "source_xyz_sha256")):
        raise ValueError("Continuous sidecar embedded source fingerprints must be scalar strings")
    components_array(priors, source_count)
    if (priors["original_ids"].dtype != np.dtype("int64") or
            not np.array_equal(priors["original_ids"], np.arange(source_count, dtype=np.int64)) or
            priors["visibility"].dtype != initial["visibility"].dtype or
            not np.array_equal(priors["visibility"], initial["visibility"])):
        raise ValueError("Continuous sidecar IDs/visibility differ from original source")
    if str(priors["source_ply_sha256"]) != source_sha or str(priors["source_xyz_sha256"]) != shared_manifest["source_xyz_sha256"]:
        raise ValueError("Continuous sidecar embedded source fingerprints differ")
    if stage["input_ply_sha256"] != source_sha or stage["metadata_sha256"] != shared_sha or stage["source_count"] != source_count:
        raise ValueError("Native source/shared metadata changed")
    # Explicit continuous source hashes are mandatory, not inferred from filenames.
    continuous_receipt = dict(path=str(sidecar), sha256=sha256_file(sidecar),
        manifest_path=str(sidecar.parent / "metadata_manifest.json"),
        manifest_sha256=sha256_file(sidecar.parent / "metadata_manifest.json"),
        shared_metadata_sha256=shared_sha, source_ply_sha256=source_sha,
        source_xyz_sha256=shared_manifest["source_xyz_sha256"], source_count=source_count,
        row_identity_exact=True, visibility_exact=True, fields=list(COMPONENTS))
    if stage["continuous_boundary_metadata"] != continuous_receipt:
        raise ValueError("Training continuous sidecar receipt differs")
    expected = {10: 178809, 20: 89405}[ratio]
    if config["pruning_target"] != expected or stage["target"] != expected:
        raise ValueError("Unexpected exact final count")
    checkpoint, ply = run / "stage3/chkpnt25199.pth", run / "stage3/point_cloud/iteration_25199/point_cloud.ply"
    if resolve(stage["last_checkpoint"]).resolve() != checkpoint.resolve() or resolve(stage["last_ply"]).resolve() != ply.resolve():
        raise ValueError("Final native models are not the completed local SH3 stage")
    native, metadata = verify_stage(run / "stage3", checkpoint, ply, expected)
    if native["standard_property_count"] != 62:
        raise ValueError("Expected standard native SH3 PLY fields")
    quota_config = dict(config, boundary_protection=dict(config["boundary_protection"], enabled=False))
    amin, _, _ = survival_quotas(initial, quota_config)
    if {int(key): int(value) for key, value in protection["instance_min"].items()} != amin:
        raise ValueError("Frozen instance quotas differ")
    quotas = audit_quotas(metadata, amin, {})
    payload = load_torch_file(run / "stage3/compression_state.pt")
    if payload["extra"]["virtual_iteration"] != 25199 or payload["extra"]["config"] != config:
        raise ValueError("Resume state iteration/config differs")
    if payload["extra"].get("continuous_boundary_metadata") != continuous_receipt:
        raise ValueError("Resume state continuous sidecar receipt differs")
    continuity = MetadataContinuity(priors, payload["lineage"])
    rounds = [validate_event(event, config, continuity) for event in stage["events"]]
    metadata_audit = validate_payload(payload, metadata, continuity)
    metadata_audit.update(continuity.audit_lineage())
    if metadata_audit["saved_merge_weights_recomputed"] != metadata_audit["merged_rows"]:
        raise ValueError("Continuous merge weights were not all recorded/recomputed")
    replay = replay_rir_dpra(stage, config)
    result = dict(status="complete", ratio=ratio, target=expected, SH3=native, method=METHOD,
                  source_snapshot=frozen, score_mode=MODE, boundary_score=PARAMETERS, boundary_quotas={},
                  preserve_boundary_merge=False, distillation_updates=0, instance_topology_loss_settings_unchanged=True,
                  continuous_metadata_sha256=sha256_file(sidecar), continuous_manifest_sha256=sha256_file(sidecar.parent / "metadata_manifest.json"),
                  continuous_source_statistics={key: continuous_summary(priors[key]) for key in COMPONENTS},
                  metadata_continuity=metadata_audit, rir_dpra_matches_original=True, schedule_replay=replay,
                  recovery_updates=10199, compression_events=len(rounds), event_iterations=replay["event_iterations"],
                  rounds=rounds, quota_violations=0, protected_instances=len(amin), final_quota_audit=quotas,
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
    print({key: value["status"] for key, value in results.items()})


if __name__ == "__main__":
    main()
