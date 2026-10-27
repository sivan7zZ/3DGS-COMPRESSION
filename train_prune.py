# Copyright (C) 2023, Inria. All rights reserved.
# Derived from PWSR train_prune_DoG.py. See LICENSE.md and THIRD_PARTY_NOTICES.md.
"""Pretrained standard 3DGS -> RIR/DPRA/SPS pruning and recovery."""
import argparse
import os
import random
import json
import numpy as np
import time

import torch
from tqdm import tqdm
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.schedule_utils import TrainingScheduler
from utils.checkpoint_utils import load_checkpoint
from utils.distill_utils import (atomic_json_save, atomic_torch_save, atomic_write,
    capture_rng_state, restore_rng_state, load_torch_file, sha256_file)
from utils.general_utils import safe_state
from utils.instance_pruning import global_sps_keep

def reconstruction_detection(scene, gaussians, pipe, background, dataset, rois=None, roi_weight=0.0):
    with torch.no_grad():
        cameras = scene.getTrainCameras()
        total_loss = 0.0
        for view in cameras:
            target = view.original_image.cuda()
            image = render(view, gaussians, pipe, background,
                           use_trained_exp=dataset.train_test_exp)["render"]
            value = l1_loss(target, image)
            if rois is not None and roi_weight:
                region, _ = rois.losses(view.image_name, image, target)
                value = value + roi_weight * region
            total_loss += value.item()
        return total_loss / len(cameras)


def prune(scene, gaussians, pipe, background, prune_ratio, dataset,
          instance_state=None, rois=None, roi_weight=0.0, score_roi_weight=0.0,
          protect_instances=True, audit_path=None):
    torch.cuda.reset_peak_memory_stats()

    with torch.enable_grad():
        cameras = scene.getTrainCameras()
        scores = torch.zeros_like(gaussians.get_opacity)
        total_loss = 0.0
        for view in tqdm(cameras, desc="Computing Pruning Scores"):
            img_scores = torch.zeros_like(scores, requires_grad=True)
            image = render(
                view, gaussians, pipe, background,
                use_trained_exp=dataset.train_test_exp,
                scores=img_scores,
            )["render"]
            gt = view.original_image.cuda()

            # Build a detached band-pass detail map for the rasterizer's
            # squared per-Gaussian importance gradient.
            with torch.no_grad():
                luma = (
                    image[:3].detach()
                    * image.new_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
                ).sum(0)
                luma -= luma.mean()
                height, width = luma.shape
                spectrum = torch.fft.rfft2(luma, norm="ortho")
                fy = torch.fft.fftfreq(height, device=image.device)
                fx = torch.fft.rfftfreq(width, device=image.device)
                wy, wx = torch.meshgrid(fy, fx, indexing="ij")
                radius = torch.sqrt(wx.square() + wy.square())
                radius /= radius.max().clamp_min(1e-8)
                bandpass = (
                    ((radius - 0.04) / 0.08).clamp(0.0, 1.0)
                    * ((0.45 - radius) / 0.15).clamp(0.0, 1.0)
                )
                detail = torch.fft.irfft2(
                    spectrum * bandpass, s=(height, width), norm="ortho"
                ).abs()
                weight = (detail / detail.mean().clamp_min(1e-8)).clamp(0.0, 3.0)

            objective = image.sum() + 0.10 * (image * weight.unsqueeze(0)).sum()
            if rois is not None and score_roi_weight:
                roi_weight_map = rois.records[view.image_name]["weights"].to(device=image.device, dtype=image.dtype)
                objective = objective + score_roi_weight * (image * roi_weight_map.unsqueeze(0)).sum()
            scores += torch.autograd.grad(objective, img_scores)[0]
            reconstruction = l1_loss(gt, image)
            if rois is not None and roi_weight:
                region, _ = rois.losses(view.image_name, image, gt)
                reconstruction = reconstruction + roi_weight * region
            total_loss += reconstruction.item()

        avg_loss = total_loss / len(cameras)

    before = gaussians.get_xyz.shape[0]
    keep_count = max(1, int(round(before * (1.0 - prune_ratio))))
    if instance_state is not None and protect_instances:
        keep, audit = instance_state.select_keep(scores, gaussians.get_xyz, keep_count)
        audit["selection_mode"] = "instance_constrained_SPS"
        # Counterfactual global SPS, same current model/scores/budget; not an
        # independently retrained baseline. Useful for per-event diagnosis.
        global_keep = global_sps_keep(scores, keep_count)
        global_ids = instance_state.instance_ids[global_keep.cpu().numpy()]
        global_counts = dict(zip(*np.unique(global_ids, return_counts=True)))
        audit["counterfactual_global_SPS"] = [dict(instance_id=row["instance_id"],
            after=int(global_counts.get(row["instance_id"], 0)), min_keep=row["min_keep"],
            small=row["small"], protected=row["protected"]) for row in instance_state.policy]
    else:
        keep = global_sps_keep(scores, keep_count)
        audit = {"selection_mode": "global_SPS", "before": before, "kept": keep_count}
    gaussians.prune_points(~keep)
    if instance_state is not None:
        instance_state.subset_(keep)
    if audit_path:
        atomic_json_save(audit, audit_path)
    return avg_loss



@torch.no_grad()
def evaluate(scene, model, pipeline, background, dataset, iteration):
    cameras = scene.getTestCameras()
    if not cameras:
        raise ValueError("Evaluation requires test cameras")
    metrics = {"iteration": iteration, "representation": "float_tensor", "PSNR": 0., "SSIM": 0.}
    for camera in cameras:
        image = render(camera, model, pipeline, background)["render"]
        target = camera.original_image.to(image.device)
        metrics["PSNR"] += psnr(image, target).mean().item() / len(cameras)
        metrics["SSIM"] += ssim(image, target).mean().item() / len(cameras)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    parser.set_defaults(iterations=25199)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--start_pointcloud", help="Pretrained standard PLY; start a new pruning stage")
    inputs.add_argument("--init_checkpoint", help="Pretrained standard checkpoint; start a new pruning stage")
    inputs.add_argument("--start_checkpoint", help="Resume this pruning stage, including .state.pth sidecar")
    parser.add_argument("--beta", type=float, default=0.99)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", help="Accepted for pipeline compatibility; no viewer started")
    parser.add_argument("--instance_metadata", help="Original same-order label-free instance NPZ")
    parser.add_argument("--instance_policy_config", help="Explicit hard-floor/ranking policy JSON")
    parser.add_argument("--instance_projected_area", help="TRAIN-view median projected-area fractions JSON")
    parser.add_argument("--instance_roi_dir", help="TRAIN-only per-view weighted SAM ROI NPZ directory")
    parser.add_argument("--instance_roi_weight", type=float, default=0.1)
    parser.add_argument("--instance_edge_weight", type=float, default=0.05)
    parser.add_argument("--instance_score_roi_weight", type=float, default=0.25)
    parser.add_argument("--disable_instance_protection", action="store_true", help="Matched global-SPS control; retain only lineage sidecars")
    parser.add_argument("--require_pruning_target", action="store_true", help="Fail full experiments that did not reach the requested Gaussian target")
    args = parser.parse_args()
    if not args.model_path or not args.source_path:
        parser.error("Specify source_path and model_path")
    if args.train_test_exp or args.depths:
        parser.error("This checkpoint pipeline currently supports RGB standard 3DGS without exposure/depth supervision")
    if args.optimizer_type != "default":
        parser.error("This checkpoint pipeline uses the standard Adam optimizer")
    if not 0 < args.beta <= 1:
        parser.error("beta must be in (0, 1]")
    if not args.densify_until_iter < args.iterations < args.pruning_stop_iter:
        parser.error("iterations must be after pretrained initialization and before pruning_stop_iter (default25200)")
    if not args.densify_until_iter < args.pruning_start_iter < args.pruning_stop_iter:
        parser.error("Invalid pruning schedule")
    if args.pruning_interval <= 0 or args.pruning_interval_maximum < args.pruning_interval:
        parser.error("Invalid recovery intervals")
    if args.pruning_target is not None and args.pruning_target <= 0:
        parser.error("pruning_target must be a positive Gaussian count")
    args.source_path = os.path.abspath(args.source_path)
    args.model_path = os.path.abspath(args.model_path)
    for name in ("start_pointcloud", "init_checkpoint", "start_checkpoint"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
            if not os.path.isfile(getattr(args, name)):
                parser.error("Missing input: " + getattr(args, name))
    for name in ("instance_metadata", "instance_policy_config", "instance_projected_area", "instance_roi_dir"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))
    if any(not np.isfinite(getattr(args, name)) or getattr(args, name) < 0
           for name in ("instance_roi_weight", "instance_edge_weight", "instance_score_roi_weight")):
        parser.error("Instance loss weights must be finite and nonnegative")
    if args.disable_instance_protection and args.instance_roi_dir:
        parser.error("The global-SPS control cannot consume instance ROI supervision")
    resume = None
    if args.start_checkpoint:
        resume = load_torch_file(args.start_checkpoint + ".state.pth")
        if resume.get("format") != "pwsr-prune-v1" or resume["checkpoint_sha256"] != sha256_file(args.start_checkpoint):
            parser.error("Invalid checkpoint/sidecar pair")
        for key, value in resume["configuration"].items():
            if key not in {"start_checkpoint", "start_pointcloud", "init_checkpoint", "model_path",
                           "quiet", "disable_viewer", "test_iterations", "save_iterations", "checkpoint_iterations"}:
                if getattr(args, key) != value:
                    parser.error("Resume settings differ: " + key)
    elif os.path.isdir(args.model_path) and any(
        name not in {"train.log"} for name in os.listdir(args.model_path)
    ):
        parser.error("Output directory already contains results; select another directory")
    os.makedirs(args.model_path, exist_ok=True)
    safe_state(args.quiet)
    dataset, opt, pipeline = lp.extract(args), op.extract(args), pp.extract(args)
    model = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, model, shuffle=False, initialize_gaussians=False)
    first_iter = opt.densify_until_iter
    source_iteration = None
    if args.start_checkpoint or args.init_checkpoint:
        state, source_iteration = load_checkpoint(args.start_checkpoint or args.init_checkpoint)
        model.load_checkpoint_model(state, opt, restore_optimizer=bool(resume))
        del state
        if resume:
            first_iter = source_iteration
    else:
        model.load_ply(args.start_pointcloud)
        model.spatial_lr_scale = scene.cameras_extent
        model.training_setup(opt)
    dataset.sh_degree = model.max_sh_degree
    if model.active_sh_degree != model.max_sh_degree:
        parser.error("Input must have its stored SH degree fully activated")
    if opt.pruning_target is not None and opt.pruning_target >= model.get_xyz.shape[0] and not resume:
        parser.error("Pruning target must be smaller than input Gaussian count")
    with open(os.path.join(args.model_path, "cfg_args"), "w") as stream:
        stream.write(str(argparse.Namespace(**vars(dataset))))
    scheduler = TrainingScheduler(opt)
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError("No training cameras")
    view_names = [c.image_name for c in cameras]
    instance_state = None
    policy_config = None
    if args.instance_policy_config:
        with open(args.instance_policy_config) as stream:
            policy_config = json.load(stream)
    if args.instance_metadata or (resume and resume.get("instance_metadata")):
        from utils.instance_pruning import InstancePruningState
        if resume:
            record = resume["instance_metadata"]
            instance_path = args.start_checkpoint + ".instances.npz"
            if sha256_file(instance_path) != record["sha256"]:
                raise ValueError("Instance checkpoint sidecar changed")
            instance_state = InstancePruningState.load_resume(instance_path, model.get_xyz, config=policy_config)
        else:
            area = None
            if args.instance_projected_area:
                with open(args.instance_projected_area) as stream:
                    area = json.load(stream)
            instance_state = InstancePruningState.load(args.instance_metadata, model.get_xyz,
                                                       config=policy_config, projected_area=area)
        instance_state.export_policy(os.path.join(args.model_path, "protection_policy.json"))
        # Check the FINAL budget before any expensive optimization begins.
        target = int(opt.pruning_target if opt.pruning_target is not None else 0.1 * instance_state.original_count)
        if not args.disable_instance_protection and instance_state.policy_dict()["floor_total"] > target:
            raise ValueError("Instance hard floors exceed the final Gaussian budget")
    rois = None
    if args.instance_roi_dir:
        from utils.instance_roi import TrainingInstanceROIs
        rois = TrainingInstanceROIs(args.instance_roi_dir, cameras)
    roi_provenance = rois.provenance if rois is not None else None
    if resume and roi_provenance != resume.get("instance_roi_provenance"):
        raise ValueError("Training ROI content changed since the checkpoint")
    remaining = []
    schedule = {"loss_stamp": 1.0, "iteration_no_prune": 0, "pruning_count": 0, "pruning_ratio": 0.0}
    events = []
    if resume:
        if resume["view_names"] != view_names:
            raise ValueError("Camera ordering changed")
        scheduler.__dict__.update(resume["scheduler"])
        schedule = resume["schedule"]
        remaining = resume["remaining_views"]
        events = resume["events"]
        restore_rng_state(resume["rng"])
    background = torch.tensor([1.,1.,1.] if dataset.white_background else [0.,0.,0.], device="cuda")
    manifest = {"stage": "RIR_DPRA_SPS", "configuration": vars(args), "source_iteration": source_iteration,
                "virtual_start": opt.densify_until_iter, "virtual_end": opt.iterations,
                "initial_gaussians": model.get_xyz.shape[0], "status": "running", "events": events}
    atomic_json_save(manifest, os.path.join(args.model_path, "run_args.json"))
    saves = set(args.save_iterations + args.checkpoint_iterations + args.test_iterations + [opt.iterations])
    metrics = {}
    started = time.monotonic()
    print("Pretrained model loaded; virtual {} -> {}".format(first_iter, opt.iterations), flush=True)
    for iteration in range(first_iter + 1, opt.iterations + 1):
        model.update_learning_rate(iteration)
        if not remaining:
            remaining = list(range(len(cameras)))
        camera = cameras[remaining.pop(random.randint(0, len(remaining)-1))]
        image = render(camera, model, pipeline, background)["render"]
        target = camera.original_image.to(image.device)
        loss = (1-opt.lambda_dssim)*l1_loss(image, target) + opt.lambda_dssim*(1-ssim(image, target))
        if rois is not None:
            region_loss, edge_loss = rois.losses(camera.image_name, image, target)
            loss = loss + args.instance_roi_weight * region_loss + args.instance_edge_weight * edge_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite pruning loss")
        loss.backward()
        # Every labeled iteration updates before topology changes or serialization.
        model.optimizer.step()
        model.optimizer.zero_grad(set_to_none=True)
        if iteration == opt.pruning_start_iter:
            scheduler.calculate_pruning_target(model.get_xyz.shape[0], opt.pruning_target)
            schedule["pruning_count"] = 2
            schedule["pruning_ratio"] = scheduler.get_pruning_rate(iteration, model.get_xyz.shape[0], 1)
        elif (iteration > opt.pruning_start_iter and iteration < opt.pruning_stop_iter
              and (iteration-opt.pruning_start_iter) % opt.pruning_interval == 0
              and 0 < schedule["pruning_count"] <= 6):
            current_loss = reconstruction_detection(scene, model, pipeline, background, dataset, rois, args.instance_roi_weight)
            forced = schedule["iteration_no_prune"] + opt.pruning_interval >= opt.pruning_interval_maximum
            if current_loss < args.beta * schedule["loss_stamp"] or forced:
                before = model.get_xyz.shape[0]
                audit_path = os.path.join(args.model_path, "instance_audits", "iteration_%d.json" % iteration)
                schedule["loss_stamp"] = prune(scene, model, pipeline, background, schedule["pruning_ratio"], dataset,
                    instance_state, rois, args.instance_roi_weight, args.instance_score_roi_weight,
                    not args.disable_instance_protection, audit_path)
                after = model.get_xyz.shape[0]
                schedule["iteration_no_prune"] = 0
                schedule["pruning_ratio"] = scheduler.get_pruning_rate(iteration, after, schedule["pruning_count"])
                schedule["pruning_count"] += 1
                event = {"iteration": iteration, "before": before, "after": after, "forced": forced}
                events.append(event)
                print("Pruning: {}".format(event), flush=True)
            else:
                schedule["iteration_no_prune"] += opt.pruning_interval
        if iteration in saves:
            checkpoint = os.path.join(args.model_path, "chkpnt{}.pth".format(iteration))
            atomic_torch_save((model.capture(), iteration), checkpoint)
            instance_record = None
            if instance_state is not None:
                instance_path = checkpoint + ".instances.npz"
                instance_state.save(instance_path, model.get_xyz)
                instance_record = {"path": instance_path, "sha256": sha256_file(instance_path)}
                instance_state.save(os.path.join(args.model_path, "instance_priors.npz"), model.get_xyz)
            atomic_torch_save({"format": "pwsr-prune-v1", "checkpoint_sha256": sha256_file(checkpoint),
                "configuration": vars(args), "schedule": dict(schedule), "scheduler": dict(scheduler.__dict__),
                "remaining_views": list(remaining), "view_names": view_names, "events": list(events),
                "rng": capture_rng_state(), "instance_metadata": instance_record,
                "instance_roi_provenance": roi_provenance}, checkpoint + ".state.pth")
            ply = os.path.join(args.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
            atomic_write(ply, model.save_ply)
            if instance_state is not None:
                instance_state.save(os.path.join(os.path.dirname(ply), "instance_priors.npz"), model.get_xyz)
            manifest.update(last_checkpoint=checkpoint, last_ply=ply, completed_steps=iteration-opt.densify_until_iter)
            atomic_json_save(manifest, os.path.join(args.model_path, "run_args.json"))
        if iteration in args.test_iterations:
            metrics[str(iteration)] = evaluate(scene, model, pipeline, background, dataset, iteration)
            atomic_json_save(metrics, os.path.join(args.model_path, "metrics_float.json"))
        if not args.quiet and iteration % 100 == 0:
            print("[ITER {}] N={} loss={:.7f}".format(iteration, model.get_xyz.shape[0], loss.item()), flush=True)
    if args.require_pruning_target:
        target = int(opt.pruning_target if opt.pruning_target is not None else scheduler.pruning_target)
        if model.get_xyz.shape[0] != target:
            raise RuntimeError("Final Gaussian count %d differs from target %d" % (model.get_xyz.shape[0], target))
    manifest.update(status="complete", final_gaussians=model.get_xyz.shape[0], elapsed_seconds=time.monotonic()-started)
    atomic_json_save(manifest, os.path.join(args.model_path, "run_args.json"))
    print("Pruning and recovery complete.", flush=True)

if __name__ == "__main__":
    main()
