# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact george.drettakis@inria.fr
#
# SH distillation adapted from VITA-Group/LightGaussian (distill_train.py,
# utils/pose_utils.py), local upstream revision 6676b983e77baadd909effc56a6aaadafa964dcc.
# Uses this repository's standard Gaussian model/rasterizer and checkpoint input.

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.distill_utils import (
    atomic_json_save, atomic_torch_save, atomic_write, augmented_camera,
    capture_rng_state, checkpoint_provenance, distillation_schedule,
    load_torch_file, model_fingerprint, read_standard_checkpoint,
    restore_rng_state, resume_configuration, sha256_file, verify_resume_pair,
)


PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")
INSTANCE_CONFIGURATION_FIELDS = ("instance_metadata", "instance_roi_dir", "instance_roi_weight",
                                 "instance_edge_weight", "instance_balanced_sampling", "structure_priors",
                                 "structure_boundary_weight", "structure_thin_weight", "structure_anchor_weight", "structure_selection_mode")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Checkpoint-based LightGaussian SH distillation")
    model = ModelParams(parser)
    optimization = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.set_defaults(iterations=None, position_lr_max_steps=None, opacity_lr=0.05)
    parser.add_argument("--start_checkpoint")
    parser.add_argument("--teacher_model", help="Frozen, unpruned SH3 checkpoint (required for a fresh run)")
    parser.add_argument("--resume_checkpoint", help="Distillation .pth with matching .state.pth sidecar")
    parser.add_argument("--distill_start_iteration", type=int, default=35000)
    parser.add_argument("--distill_steps", type=int, default=5000)
    parser.add_argument("--new_max_sh", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--test_iterations", nargs="*", type=int, default=[])
    parser.add_argument("--skip_final_eval", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augmented_view", dest="augmented_view", action="store_true")
    parser.add_argument("--no_augmented_view", dest="augmented_view", action="store_false")
    parser.add_argument("--enable_covariance", dest="enable_covariance", action="store_true")
    parser.add_argument("--freeze_covariance", dest="enable_covariance", action="store_false")
    parser.add_argument("--enable_opacity", action="store_true")
    parser.add_argument("--instance_metadata", help="Student same-order .instances.npz (auto-detected beside start checkpoint)")
    parser.add_argument("--instance_roi_dir", help="SAM-derived training-view <image_name>.npz ROI maps")
    parser.add_argument("--instance_roi_weight", type=float, default=0.1)
    parser.add_argument("--instance_edge_weight", type=float, default=0.05)
    parser.add_argument("--instance_balanced_sampling", action="store_true",
                        help="Sample training views with replacement using saved ROI sampling weights")
    parser.add_argument("--structure_selection_mode", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--structure_priors")
    parser.add_argument("--structure_boundary_weight", type=float, default=0.02)
    parser.add_argument("--structure_thin_weight", type=float, default=0.03)
    parser.add_argument("--structure_anchor_weight", type=float, default=0.02)
    parser.add_argument("--quiet", action="store_true")
    parser.set_defaults(augmented_view=True, enable_covariance=True)
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    explicit = set()
    for argument in raw:
        action = parser._option_string_actions.get(argument.split("=")[0])
        if action:
            explicit.add(action.dest)

    resume = None
    if args.resume_checkpoint:
        if args.start_checkpoint or args.teacher_model:
            parser.error("--resume_checkpoint cannot be combined with initial checkpoint inputs")
        sidecar_path = args.resume_checkpoint + ".state.pth"
        if not os.path.isfile(sidecar_path):
            parser.error("Full resume requires sidecar: " + sidecar_path)
        resume = load_torch_file(sidecar_path)
        verify_resume_pair(args.resume_checkpoint, resume)
        resumed_path = os.path.abspath(args.resume_checkpoint)
        try:
            saved_configuration = dict(resume["configuration"])
            for name in INSTANCE_CONFIGURATION_FIELDS:
                saved_configuration.setdefault(name, parser.get_default(name))
            saved = resume_configuration(saved_configuration, vars(args), explicit,
                                         ("model_path", "quiet", "data_device", "skip_final_eval", "resume_checkpoint"))
        except ValueError as error:
            parser.error(str(error))
        args = argparse.Namespace(**saved)
        args.resume_checkpoint = resumed_path
    else:
        if not args.start_checkpoint:
            parser.error("Fresh distillation requires --start_checkpoint (PLY input is not accepted)")
        if not args.teacher_model:
            parser.error("Fresh distillation requires --teacher_model (use the original unpruned 3DGS checkpoint)")
    if not args.source_path or not args.model_path:
        parser.error("--source_path and --model_path are required")
    if args.train_test_exp:
        parser.error("SH distillation checkpoints do not carry exposure parameters; omit --train_test_exp")
    if args.new_max_sh < 0 or args.save_every < 0:
        parser.error("--new_max_sh and --save_every must be nonnegative")
    if (not np.isfinite(args.instance_roi_weight) or not np.isfinite(args.instance_edge_weight)
            or args.instance_roi_weight < 0 or args.instance_edge_weight < 0):
        parser.error("Instance ROI/edge weights must be finite and nonnegative")
    if args.instance_balanced_sampling and not args.instance_roi_dir:
        parser.error("--instance_balanced_sampling requires --instance_roi_dir")
    if any(not np.isfinite(getattr(args, name)) or getattr(args, name) < 0 for name in
           ("structure_boundary_weight", "structure_thin_weight", "structure_anchor_weight")):
        parser.error("Structure loss weights must be finite and nonnegative")
    if args.structure_selection_mode == "adaptive" and args.structure_anchor_weight != 0:
        parser.error("Adaptive coverage requires structure_anchor_weight=0")
    if args.structure_priors and not args.instance_roi_dir:
        parser.error("Structure distillation requires training-view structure ROIs")
    try:
        args.iterations = distillation_schedule(args.distill_start_iteration, args.distill_steps, args.iterations)
    except ValueError as error:
        parser.error(str(error))
    if args.position_lr_max_steps is None:
        args.position_lr_max_steps = args.iterations
    if args.position_lr_max_steps <= 0:
        parser.error("--position_lr_max_steps must be positive")
    args.source_path = os.path.abspath(args.source_path)
    args.model_path = os.path.abspath(args.model_path)
    args.start_checkpoint = os.path.abspath(args.start_checkpoint)
    args.teacher_model = os.path.abspath(args.teacher_model)
    if args.instance_metadata:
        args.instance_metadata = os.path.abspath(args.instance_metadata)
    elif resume is None and os.path.isfile(args.start_checkpoint + ".instances.npz"):
        args.instance_metadata = args.start_checkpoint + ".instances.npz"
    if args.structure_priors:
        args.structure_priors = os.path.abspath(args.structure_priors)
    if args.instance_roi_dir:
        args.instance_roi_dir = os.path.abspath(args.instance_roi_dir)
    if args.optimizer_type != "default":
        parser.error("SH distillation uses LightGaussian AdamW; --optimizer_type must be default")
    return args, model, optimization, pipeline, resume


def make_student_optimizer(student, state=None):
    # LightGaussian calls training_setup again after lowering SH, resetting its
    # optimizer. Preserve this stage boundary, using its AdamW defaults.
    groups = [{"params": group["params"], "lr": group["lr"], "name": group["name"]}
              for group in student.optimizer.param_groups]
    student.optimizer = torch.optim.AdamW(groups, lr=0.0, eps=1e-15, weight_decay=0.01)
    if state is not None:
        student.optimizer.load_state_dict(state)


def save_model(scene, student, iteration, args, scheduler, remaining, view_names, inputs, teacher_hash,
               instance_state=None, roi_provenance=None, structure_provenance=None):
    checkpoint_path = os.path.join(args.model_path, "chkpnt{}.pth".format(iteration))
    # Save optimizer-bearing checkpoint before rendering/evaluation can fail.
    atomic_torch_save((student.capture(), iteration), checkpoint_path)
    instance_record = None
    if instance_state is not None:
        instance_path = checkpoint_path + ".instances.npz"
        instance_state.save(instance_path, student.get_xyz)
        instance_record = {"path": instance_path, "sha256": sha256_file(instance_path)}
    metadata = {
        "format": "pwsr-sh-distill-v1", "checkpoint_sha256": sha256_file(checkpoint_path),
        "configuration": vars(args), "iteration": iteration,
        "completed_steps": iteration - args.distill_start_iteration,
        "optimizer": "AdamW", "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(), "remaining_views": list(remaining),
        "view_names": view_names, "inputs": inputs, "teacher_fingerprint": teacher_hash,
        "instance_metadata": instance_record, "instance_roi_provenance": roi_provenance,
        "structure_provenance": structure_provenance,
    }
    atomic_torch_save(metadata, checkpoint_path + ".state.pth")
    ply_path = os.path.join(args.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply")
    atomic_write(ply_path, student.save_ply)
    if instance_state is not None:
        instance_state.save(os.path.join(os.path.dirname(ply_path), "instance_priors.npz"), student.get_xyz)
        instance_state.save(os.path.join(args.model_path, "instance_priors.npz"), student.get_xyz)
    print("[ITER {}] Saved PLY, checkpoint and full resume sidecar".format(iteration), flush=True)
    return checkpoint_path, ply_path


@torch.no_grad()
def evaluate(scene, student, pipeline, background, iteration):
    from gaussian_renderer import render
    from utils.image_utils import psnr
    from utils.loss_utils import ssim
    cameras = scene.getTestCameras()
    if not cameras:
        raise ValueError("No test cameras available for final evaluation; use --eval or --skip_final_eval")
    total_psnr, total_ssim = 0.0, 0.0
    for camera in cameras:
        image = render(camera, student, pipeline, background, clamp_output=False)["render"].clamp(0, 1)
        target = camera.original_image.to(image.device).clamp(0, 1)
        total_psnr += psnr(image, target).mean().double().item()
        total_ssim += ssim(image, target).mean().double().item()
    values = {"iteration": iteration, "test_views": len(cameras), "representation": "float_tensor",
              "PSNR": total_psnr / len(cameras), "SSIM": total_ssim / len(cameras)}
    print("[ITER {}] Float test metrics: {}".format(iteration, values), flush=True)
    return values


def training(args, model_parameters, optimization_parameters, pipeline_parameters, resume=None):
    from gaussian_renderer import render
    from scene import Scene, GaussianModel
    from utils.loss_utils import l1_loss, ssim
    if not torch.cuda.is_available():
        raise RuntimeError("SH distillation requires a CUDA GPU and this repository's rasterizer")
    if resume is None and os.path.isdir(args.model_path) and any(
            name != "train.log" for name in os.listdir(args.model_path)):
        raise FileExistsError("Output already contains a distillation run; choose another directory or --resume_checkpoint")
    os.makedirs(args.model_path, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset = model_parameters.extract(args)
    optimization = optimization_parameters.extract(args)
    pipeline = pipeline_parameters.extract(args)

    input_model_path = args.resume_checkpoint if resume is not None else args.start_checkpoint
    student_parameters, stored_student_iteration = read_standard_checkpoint(input_model_path)
    teacher_parameters, stored_teacher_iteration = read_standard_checkpoint(args.teacher_model)
    if resume is not None and (student_parameters[0] != args.new_max_sh or
                              student_parameters[3].shape[1] != (args.new_max_sh + 1) ** 2 - 1):
        raise ValueError("Resume checkpoint SH degree/storage does not match saved target degree")
    student = GaussianModel(args.new_max_sh)
    scene = Scene(dataset, student, shuffle=False, initialize_gaussians=False)
    student.load_checkpoint_model(student_parameters, training_args=optimization,
                                  target_sh_degree=args.new_max_sh, restore_optimizer=False)
    if student.active_sh_degree != args.new_max_sh:
        raise ValueError("Student input must activate at least the target SH degree")
    make_student_optimizer(student, student_parameters[10] if resume is not None else None)
    if resume is not None and student.active_sh_degree != args.new_max_sh:
        raise ValueError("Resume checkpoint SH degree does not match saved target degree")
    teacher = GaussianModel(int(teacher_parameters[0]))
    teacher.load_checkpoint_model(teacher_parameters, training_args=None, restore_optimizer=False)
    if teacher.active_sh_degree <= args.new_max_sh:
        raise ValueError("Teacher must retain a higher active SH degree than the student")
    for name in PARAMETERS:
        getattr(teacher, name).requires_grad_(False)
        if getattr(teacher, name).data_ptr() == getattr(student, name).data_ptr():
            raise RuntimeError("Teacher and student unexpectedly share model storage")
    teacher.optimizer = None
    student._scaling.requires_grad_(args.enable_covariance)
    student._rotation.requires_grad_(args.enable_covariance)
    student._opacity.requires_grad_(args.enable_opacity)
    teacher_hash = model_fingerprint(teacher)
    count = student.get_xyz.shape[0]
    instance_state = None
    instance_path = args.instance_metadata
    if resume is not None:
        record = resume.get("instance_metadata")
        if record is not None:
            instance_path = args.resume_checkpoint + ".instances.npz"
            if not os.path.isfile(instance_path) or sha256_file(instance_path) != record["sha256"]:
                raise ValueError("Resume instance metadata is absent or differs from the saved checkpoint")
        elif instance_path:
            raise ValueError("Resume sidecar has no instance metadata; cannot silently add labels during continuation")
    if instance_path:
        from utils.instance_pruning import InstancePruningState
        instance_state = InstancePruningState.load_resume(instance_path, student.get_xyz)
    structure_state, anchor_trust = None, None
    if args.structure_priors:
        from utils.structure_pruning import StructurePruningState
        from utils.structure_losses import AnchorTrustRegion
        if instance_state is None:
            raise ValueError("Structure distillation requires original Gaussian lineage")
        instance_state.audit_floors()
        structure_state = StructurePruningState.load(args.structure_priors)
        if structure_state.source_xyz_sha256 != instance_state.source_xyz_sha256:
            raise ValueError("Structure priors refer to another source model")
        structure_state.audit_coverage(instance_state.original_ids, require_anchors=args.structure_selection_mode == "fixed")
        if args.structure_anchor_weight:
            anchor_trust = AnchorTrustRegion(structure_state)
    structure_provenance = structure_state.provenance if structure_state is not None else None
    if resume is not None and structure_provenance != resume.get("structure_provenance"):
        raise ValueError("Structure priors changed since the checkpoint")
    scheduler = ExponentialLR(student.optimizer, gamma=0.90)
    cameras = scene.getTrainCameras()
    if not cameras:
        raise ValueError("Dataset has no training cameras")
    view_names = [camera.image_name for camera in cameras]
    rois = None
    if args.instance_roi_dir:
        if structure_state is not None:
            from utils.structure_losses import TrainingStructureROIs
            rois = TrainingStructureROIs(args.instance_roi_dir, cameras)
        else:
            from utils.instance_roi import TrainingInstanceROIs
            rois = TrainingInstanceROIs(args.instance_roi_dir, cameras)
    roi_provenance = rois.provenance if rois is not None else None
    if resume is not None and roi_provenance != resume.get("instance_roi_provenance"):
        raise ValueError("Training-view instance ROI maps or image dimensions changed since checkpoint")
    sampling_weights = rois.sampling_weights(view_names) if args.instance_balanced_sampling else None
    remaining = []
    first_iteration = args.distill_start_iteration
    inputs = resume["inputs"] if resume is not None else {
        "student": checkpoint_provenance(args.start_checkpoint, stored_student_iteration),
        "teacher": checkpoint_provenance(args.teacher_model, stored_teacher_iteration)}
    if resume is not None:
        if stored_student_iteration != resume["iteration"]:
            raise ValueError("Resume checkpoint iteration disagrees with its sidecar")
        if teacher_hash != resume["teacher_fingerprint"]:
            raise ValueError("Teacher model changed since the checkpoint")
        if view_names != resume["view_names"]:
            raise ValueError("Training cameras or camera ordering changed since the checkpoint")
        scheduler.load_state_dict(resume["scheduler"])
        remaining = list(resume["remaining_views"])
        first_iteration = stored_student_iteration
        inputs = resume["inputs"]
        restore_rng_state(resume["rng"])
    if first_iteration < args.distill_start_iteration or first_iteration > args.iterations:
        raise ValueError("Resume iteration falls outside the configured distillation stage")
    del student_parameters, teacher_parameters

    dataset.sh_degree = args.new_max_sh
    atomic_write(os.path.join(args.model_path, "cfg_args"),
                 lambda path: _write_text(path, str(argparse.Namespace(**vars(dataset)))))
    manifest = {
        "stage": "checkpoint_SH_distillation", "status": "running", "inputs": inputs,
        "configuration": vars(args), "teacher_fingerprint": teacher_hash,
        "gaussian_count": count, "teacher_gaussian_count": teacher.get_xyz.shape[0],
        "teacher_sh_degree": teacher.active_sh_degree, "student_sh_degree": student.active_sh_degree,
        "optimizer": {"type": "AdamW", "eps": 1e-15, "weight_decay": 0.01,
                      "initialization": "resumed" if resume is not None else "fresh_stage"},
        "iteration_schedule": {"virtual_start": args.distill_start_iteration, "virtual_end": args.iterations,
                               "requested_optimizer_steps": args.distill_steps, "resumed_from": first_iteration},
        "importance_scores": "omitted: VecTree quantization is outside this SH distillation stage",
        "instance_metadata": instance_path,
        "instance_roi_provenance": roi_provenance,
        "structure_provenance": structure_provenance,
        "instance_roi_supervision": "training original views only; augmented views skip 2D ROI losses",
        "evaluation": "Float PSNR/SSIM here; LPIPS can be evaluated separately via metrics.py",
    }
    if resume is not None:
        manifest["last_checkpoint"] = args.resume_checkpoint
    manifest_path = os.path.join(args.model_path, "run_args.json")
    atomic_json_save(manifest, manifest_path)
    background = torch.tensor([1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
                              dtype=torch.float32, device="cuda")
    print("Checkpoint SH{} -> SH{}: {} Gaussians; stored input iteration {}, virtual {} -> {} ({} steps)".format(
        teacher.active_sh_degree, student.active_sh_degree, count, inputs["student"]["stored_iteration"],
        args.distill_start_iteration, args.iterations, args.distill_steps), flush=True)
    ema_loss = 0.0
    start_time = time.monotonic()
    metrics = {}
    metrics_path = os.path.join(args.model_path, "metrics_float.json")
    if resume is not None and os.path.isfile(metrics_path):
        with open(metrics_path, encoding="utf-8") as stream:
            metrics = json.load(stream)
    for iteration in range(first_iteration + 1, args.iterations + 1):
        student.update_learning_rate(iteration)
        if args.instance_balanced_sampling:
            camera = cameras[random.choices(range(len(cameras)), weights=sampling_weights, k=1)[0]]
        else:
            if not remaining:
                remaining = list(range(len(cameras)))
            camera = cameras[remaining.pop(random.randint(0, len(remaining) - 1))]
        original_view = not (args.augmented_view and iteration % 3)
        if not original_view:
            camera = augmented_camera(camera)
        with torch.no_grad():
            teacher_image = render(camera, teacher, pipeline, background, clamp_output=False)["render"]
        image = render(camera, student, pipeline, background, clamp_output=False)["render"]
        loss = ((1 - optimization.lambda_dssim) * l1_loss(image, teacher_image)
                + optimization.lambda_dssim * (1 - ssim(image, teacher_image)))
        if rois is not None and original_view:
            region_loss, edge_loss = rois.losses(camera.image_name, image, teacher_image)
            loss = loss + args.instance_roi_weight * region_loss + args.instance_edge_weight * edge_loss
        if structure_state is not None:
            if original_view:
                boundary_survival, thin_continuity = rois.structure_losses(camera.image_name, image, teacher_image)
                loss = loss + args.structure_boundary_weight * boundary_survival + args.structure_thin_weight * thin_continuity
            if args.structure_anchor_weight:
                loss = loss + args.structure_anchor_weight * anchor_trust.loss(student, instance_state.original_ids)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite distillation loss at iteration {}".format(iteration))
        loss.backward()
        student.optimizer.step()
        student.optimizer.zero_grad(set_to_none=True)
        # Apply decay after the completed 500-step block; unlike the historical
        # script, every labeled step performs an update, including the final one.
        if (iteration - args.distill_start_iteration) % 500 == 0:
            scheduler.step()
        if student.get_xyz.shape[0] != count:
            raise RuntimeError("SH distillation unexpectedly changed Gaussian count")
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        if not args.quiet and (iteration % 100 == 0 or iteration == args.iterations):
            print("[ITER {}] step {}/{} loss {:.7f} elapsed {:.1f}s".format(
                iteration, iteration - args.distill_start_iteration, args.distill_steps,
                ema_loss, time.monotonic() - start_time), flush=True)
        is_final = iteration == args.iterations
        should_evaluate = iteration in args.test_iterations or (is_final and not args.skip_final_eval)
        should_save = is_final or should_evaluate or (args.save_every > 0 and
                      (iteration - args.distill_start_iteration) % args.save_every == 0)
        if should_save:
            checkpoint_path, ply_path = save_model(scene, student, iteration, args, scheduler,
                                                   remaining, view_names, inputs, teacher_hash,
                                                   instance_state, roi_provenance, structure_provenance)
            manifest["last_checkpoint"] = checkpoint_path
            manifest["last_ply"] = ply_path
            manifest["completed_steps"] = iteration - args.distill_start_iteration
            if structure_state is not None:
                atomic_json_save(structure_state.audit_coverage(instance_state.original_ids, require_anchors=args.structure_selection_mode == "fixed"), os.path.join(os.path.dirname(ply_path), "structure_coverage.json"))
            atomic_json_save(manifest, manifest_path)
        if should_evaluate:
            metrics[str(iteration)] = evaluate(scene, student, pipeline, background, iteration)
            atomic_json_save(metrics, metrics_path)
    if first_iteration == args.iterations and not args.skip_final_eval:
        metrics[str(first_iteration)] = evaluate(scene, student, pipeline, background, first_iteration)
        atomic_json_save(metrics, metrics_path)
    teacher_unchanged = teacher_hash == model_fingerprint(teacher)
    if not teacher_unchanged or any(getattr(teacher, name).grad is not None for name in PARAMETERS):
        raise RuntimeError("Frozen teacher integrity check failed")
    manifest.update(status="complete", completed_steps=args.distill_steps, teacher_unchanged=True,
                    elapsed_seconds_this_run=time.monotonic() - start_time)
    atomic_json_save(manifest, manifest_path)
    print("Distillation complete: {} actual optimizer steps, frozen teacher verified.".format(args.distill_steps), flush=True)


def _write_text(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(value + "\n")


if __name__ == "__main__":
    arguments, model, optimization, pipeline, resume_state = parse_arguments()
    training(arguments, model, optimization, pipeline, resume_state)
