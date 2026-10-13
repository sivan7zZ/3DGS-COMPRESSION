"""Run pretrained 3DGS -> PWSR pruning -> checkpoint SH distillation -> evaluation."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def resolve(path):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else ROOT / value).resolve()


def build_plan(config, stage="all", stage3_checkpoint=None, python=None):
    python = python or sys.executable
    source = resolve(config["source_path"])
    output = resolve(config.get("output_root", "output/city2_unified"))
    allowed = (ROOT / "output").resolve()
    if allowed not in output.parents:
        raise ValueError("output_root must be a new subdirectory of this repository's output/")
    for historical in (allowed / "city2_stage3", allowed / "city2_stage3_sh2", allowed / "integration_backups"):
        if output == historical or historical in output.parents:
            raise ValueError("Historical outputs/backups cannot be used as pipeline output_root")
    if not source.is_dir():
        raise FileNotFoundError(str(source))
    stage3, sh2 = output / "stage3", output / "sh2"
    prune_end = int(config.get("prune_iterations", 25199))
    steps = int(config.get("distill_steps", 5000))
    distill_start = int(config.get("distill_start_iteration", 35000))
    if steps <= 0 or distill_start < 0:
        raise ValueError("Invalid distillation schedule")
    final = distill_start + steps
    if stage3_checkpoint and stage != "distill":
        raise ValueError("--stage3_checkpoint is for --stage distill; use --stage evaluate after exporting stage3 into the run")
    checkpoint = resolve(stage3_checkpoint) if stage3_checkpoint else stage3 / ("chkpnt%d.pth" % prune_end)
    teacher = resolve(config["teacher_checkpoint"]) if config.get("teacher_checkpoint") else None
    common = ["-s", str(source), "--eval", "--resolution", str(config.get("resolution", -1))]
    if config.get("white_background", False):
        common.append("--white_background")
    plan = []
    def command(label, script, arguments, destination):
        plan.append({"name": label, "command": [str(python), str(ROOT / script)] + arguments,
                     "output": str(destination)})
    if stage in ("all", "prune"):
        inputs = [("input_ply", "--start_pointcloud"), ("input_checkpoint", "--init_checkpoint")]
        chosen = [(key, flag) for key, flag in inputs if config.get(key)]
        if len(chosen) != 1:
            raise ValueError("Set exactly one input_ply or input_checkpoint")
        key, flag = chosen[0]
        initial = resolve(config[key])
        if not initial.is_file():
            raise FileNotFoundError(str(initial))
        arguments = common + ["-m", str(stage3), flag, str(initial), "--iterations", str(prune_end),
                      "--checkpoint_iterations", str(prune_end), "--save_iterations", str(prune_end),
                      "--test_iterations", str(prune_end), "--disable_viewer"]
        for name in ("pruning_target", "pruning_start_iter", "pruning_stop_iter", "pruning_interval",
                     "pruning_interval_maximum", "densify_until_iter", "beta"):
            if name in config:
                arguments.extend(["--" + name, str(config[name])])
        command("prune", "train_prune.py", arguments, stage3)
    if stage in ("all", "distill"):
        if teacher is None:
            raise ValueError("Set teacher_checkpoint to the original unpruned 3DGS checkpoint")
        if stage == "distill" and not checkpoint.is_file():
            raise FileNotFoundError(str(checkpoint))
        if not teacher.is_file():
            raise FileNotFoundError(str(teacher))
        command("distill", "distill_train.py", common + ["-m", str(sh2), "--start_checkpoint", str(checkpoint),
                "--teacher_model", str(teacher), "--distill_start_iteration", str(distill_start),
                "--distill_steps", str(steps), "--new_max_sh", str(config.get("new_max_sh", 2))], sh2)
    if stage in ("all", "evaluate"):
        for directory, iteration in ((stage3, prune_end), (sh2, final)):
            if stage == "evaluate" and not (directory / "point_cloud" / ("iteration_%d" % iteration) / "point_cloud.ply").is_file():
                raise FileNotFoundError("Missing model in " + str(directory))
            command("render_" + directory.name, "render.py", common + ["-m", str(directory), "--iteration", str(iteration), "--skip_train"], directory)
        command("metrics", "metrics.py", ["-m", str(stage3), str(sh2), "--device", "cuda"], output)
    return output, plan


def check_output_collisions(plan):
    for task in plan:
        if task["name"] in ("prune", "distill"):
            folder = Path(task["output"])
            if folder.exists() and any(folder.iterdir()):
                raise FileExistsError("Existing training output: %s. Use a new output_root or the stage's resume command." % folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "city.json"))
    parser.add_argument("--stage", choices=["all", "prune", "distill", "evaluate"], default="all")
    parser.add_argument("--output_root")
    parser.add_argument("--stage3_checkpoint")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without creating files")
    args = parser.parse_args(argv)
    with resolve(args.config).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if args.output_root:
        config["output_root"] = args.output_root
    output, plan = build_plan(config, args.stage, args.stage3_checkpoint)
    for task in plan:
        print(task["name"] + ": " + " ".join(shlex.quote(arg) for arg in task["command"]), flush=True)
    if args.dry_run:
        return
    check_output_collisions(plan)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "stage": args.stage, "tasks": plan, "status": "running", "completed": []}
    manifest_path = output / ("pipeline_" + args.stage + ".json")
    def save_manifest():
        temporary = manifest_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
        os.replace(str(temporary), str(manifest_path))
    save_manifest()
    try:
        for task in plan:
            destination = Path(task["output"])
            destination.mkdir(parents=True, exist_ok=True)
            log = output / (task["name"] + ".log")
            with log.open("a", encoding="utf-8") as stream:
                process = subprocess.Popen(task["command"], cwd=str(ROOT), stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                result = process.wait()
            if result:
                raise subprocess.CalledProcessError(result, task["command"])
            manifest["completed"].append(task["name"])
            save_manifest()
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        save_manifest()
        raise
    manifest["status"] = "complete"
    save_manifest()


if __name__ == "__main__":
    main()
