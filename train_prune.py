# Copyright (C) 2023, Inria. All rights reserved.
# Derived from PWSR train_prune_DoG.py. See LICENSE.md and THIRD_PARTY_NOTICES.md.
"""Pretrained standard 3DGS -> RIR/DPRA/SPS pruning and recovery."""
import argparse
import os
import random
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

def reconstruction_detection(scene, gaussians, pipe, background, dataset):
    with torch.no_grad():
        cameras = scene.getTrainCameras()
        total_loss = sum(
            l1_loss(
                view.original_image.cuda(),
                render(
                    view, gaussians, pipe, background,
                    use_trained_exp=dataset.train_test_exp
                )["render"],
            ).item()
            for view in cameras
        )
        return total_loss / len(cameras)


def prune(scene, gaussians, pipe, background, prune_ratio, dataset):
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
            scores += torch.autograd.grad(objective, img_scores)[0]
            total_loss += l1_loss(gt, image).item()

        avg_loss = total_loss / len(cameras)

    gaussians.prune_gaussians(prune_ratio, scores)
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
            current_loss = reconstruction_detection(scene, model, pipeline, background, dataset)
            forced = schedule["iteration_no_prune"] + opt.pruning_interval >= opt.pruning_interval_maximum
            if current_loss < args.beta * schedule["loss_stamp"] or forced:
                before = model.get_xyz.shape[0]
                schedule["loss_stamp"] = prune(scene, model, pipeline, background, schedule["pruning_ratio"], dataset)
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
            atomic_torch_save({"format": "pwsr-prune-v1", "checkpoint_sha256": sha256_file(checkpoint),
                "configuration": vars(args), "schedule": dict(schedule), "scheduler": dict(scheduler.__dict__),
                "remaining_views": list(remaining), "view_names": view_names, "events": list(events),
                "rng": capture_rng_state()}, checkpoint + ".state.pth")
            ply = os.path.join(args.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
            atomic_write(ply, model.save_ply)
            manifest.update(last_checkpoint=checkpoint, last_ply=ply, completed_steps=iteration-opt.densify_until_iter)
            atomic_json_save(manifest, os.path.join(args.model_path, "run_args.json"))
        if iteration in args.test_iterations:
            metrics[str(iteration)] = evaluate(scene, model, pipeline, background, dataset, iteration)
            atomic_json_save(metrics, os.path.join(args.model_path, "metrics_float.json"))
        if not args.quiet and iteration % 100 == 0:
            print("[ITER {}] N={} loss={:.7f}".format(iteration, model.get_xyz.shape[0], loss.item()), flush=True)
    manifest.update(status="complete", final_gaussians=model.get_xyz.shape[0], elapsed_seconds=time.monotonic()-started)
    atomic_json_save(manifest, os.path.join(args.model_path, "run_args.json"))
    print("Pruning and recovery complete.", flush=True)

if __name__ == "__main__":
    main()
