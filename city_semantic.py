#!/usr/bin/env python
"""Fit one City RGB semantic evaluator, freeze it, and apply to every render.

All model inference is 2D downstream evaluation; this does not add semantics to
the Gaussian PLY. Real test labels and compressed images never train this model.
"""
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from utils.city_semantic_data import CLASS_NAMES, prepare_dataset, sha256_file


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    import torchvision

    class Block(nn.Module):
        def __init__(self, incoming, outgoing):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True))

        def forward(self, value):
            return self.layers(value)

    class CityResNet34UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torchvision.models.resnet34(weights=None)
            self.encoder.fc = nn.Identity()
            self.decode4 = Block(512 + 256, 256)
            self.decode3 = Block(256 + 128, 128)
            self.decode2 = Block(128 + 64, 64)
            self.decode1 = Block(64 + 64, 64)
            self.decode0 = Block(64, 32)
            self.head = nn.Conv2d(32, 6, 1)

        def train(self, mode=True):
            super().train(mode)
            # ImageNet running statistics remain fixed with only six/eight views.
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
            return self

        def forward(self, rgb):
            e0 = self.encoder.relu(self.encoder.bn1(self.encoder.conv1(rgb)))
            e1 = self.encoder.layer1(self.encoder.maxpool(e0))
            e2 = self.encoder.layer2(e1)
            e3 = self.encoder.layer3(e2)
            e4 = self.encoder.layer4(e3)
            def up(value, skip, block):
                value = functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
                return block(torch.cat([value, skip], 1))
            value = up(e4, e3, self.decode4)
            value = up(value, e2, self.decode3)
            value = up(value, e1, self.decode2)
            value = up(value, e0, self.decode1)
            value = functional.interpolate(value, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            return self.head(self.decode0(value))

    return torch, nn, functional, CityResNet34UNet


def initialize(seed, pretrained_path, device):
    torch, nn, functional, model_class = torch_modules()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = model_class()
    state = torch.load(str(pretrained_path), map_location="cpu")
    state = {key: value for key, value in state.items() if not key.startswith("fc.")}
    missing, unexpected = model.encoder.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError("ImageNet ResNet34 initialization mismatch: %r %r" % (missing, unexpected))
    return model.to(device)


def load_frames(manifest, names):
    by_name = {row["frame"]: row for row in manifest["frames"]}
    result = []
    for name in names:
        row = by_name[name]
        if sha256_file(row["rgb_path"]) != row["rgb_sha256"] or sha256_file(row["label_path"]) != row["label_sha256"]:
            raise ValueError("Registered dataset provenance changed")
        rgb = np.asarray(Image.open(row["rgb_path"]).convert("RGB"))
        label = np.asarray(Image.open(row["label_path"]).convert("L"))
        result.append((name, rgb, label))
    return result


def training_batch(frames, rng, batch_size, crop_size, device):
    import torch
    rgbs, labels = [], []
    for _ in range(batch_size):
        _, rgb, label = frames[int(rng.randint(len(frames)))]
        height, width = label.shape
        if height < crop_size or width < crop_size:
            raise ValueError("Training crop exceeds registered native image")
        if rng.rand() < 0.5:
            class_ids = np.unique(label[label > 0])
            class_id = int(class_ids[int(rng.randint(len(class_ids)))])
            ys, xs = np.where(label == class_id)
            pick = int(rng.randint(len(ys)))
            y = int(np.clip(ys[pick] - rng.randint(crop_size), 0, height - crop_size))
            x = int(np.clip(xs[pick] - rng.randint(crop_size), 0, width - crop_size))
        else:
            y, x = int(rng.randint(height - crop_size + 1)), int(rng.randint(width - crop_size + 1))
        patch = rgb[y:y+crop_size, x:x+crop_size].copy()
        target = label[y:y+crop_size, x:x+crop_size].copy()
        k = int(rng.randint(4))
        patch, target = np.rot90(patch, k), np.rot90(target, k)
        if rng.rand() < 0.5:
            patch, target = patch[:, ::-1], target[:, ::-1]
        patch = patch.astype(np.float32) / 255.0
        gray = patch.mean(axis=2, keepdims=True)
        patch = (patch - gray) * rng.uniform(0.85, 1.15) + gray
        patch = np.clip(patch * rng.uniform(0.85, 1.15) + rng.uniform(-0.04, 0.04), 0, 1)
        patch = (patch - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        target = target.astype(np.int64) - 1
        target[target < 0] = 255
        rgbs.append(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        labels.append(np.ascontiguousarray(target))
    return torch.from_numpy(np.stack(rgbs)).to(device), torch.from_numpy(np.stack(labels)).to(device)


def tile_positions(length, tile, stride):
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def predict_native(model, rgb, device, tile=512, stride=384, batch_size=4):
    import torch
    import torch.nn.functional as functional
    if tile < 32 or stride <= 0 or stride > tile:
        raise ValueError("Invalid native inference tile/stride")
    height, width = rgb.shape[:2]
    padded = np.pad(rgb, ((0, max(0, tile-height)), (0, max(0, tile-width)), (0, 0)), mode="reflect")
    ys, xs = tile_positions(padded.shape[0], tile, stride), tile_positions(padded.shape[1], tile, stride)
    coordinates = [(y, x) for y in ys for x in xs]
    sums = np.zeros((6, padded.shape[0], padded.shape[1]), np.float32)
    count = np.zeros(padded.shape[:2], np.float32)
    model.eval()
    with torch.no_grad():
        for first in range(0, len(coordinates), batch_size):
            group = coordinates[first:first+batch_size]
            patches = np.stack([padded[y:y+tile, x:x+tile] for y, x in group]).astype(np.float32)/255.0
            patches = (patches - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
            tensor = torch.from_numpy(patches.transpose(0, 3, 1, 2)).to(device)
            with torch.cuda.amp.autocast(enabled=(str(device).startswith("cuda"))):
                logits = model(tensor)
            probabilities = functional.softmax(logits.float(), 1).cpu().numpy()
            for (y, x), probability in zip(group, probabilities):
                sums[:, y:y+tile, x:x+tile] += probability
                count[y:y+tile, x:x+tile] += 1
    if not np.all(count > 0):
        raise AssertionError("Native inference left uncovered pixels")
    return (sums[:, :height, :width] / count[None, :height, :width]).argmax(0).astype(np.uint8) + 1


def validation_miou(model, frames, device):
    confusion = np.zeros((6, 6), np.int64)
    for _, rgb, target in frames:
        prediction = predict_native(model, rgb, device)
        valid = target > 0
        confusion += np.bincount(((target[valid].astype(np.int64)-1)*6 + prediction[valid]-1), minlength=36).reshape(6, 6)
    intersection = np.diag(confusion)
    union = confusion.sum(0) + confusion.sum(1) - intersection
    present = union > 0
    return float(np.mean(intersection[present] / union[present])), confusion.tolist()


def train_run(manifest, names, validation_names, output, args, steps, phase):
    torch, nn, functional, _ = torch_modules()
    device = args.device
    model = initialize(args.seed, args.pretrained, device)
    frames = load_frames(manifest, names)
    validation = load_frames(manifest, validation_names)
    pixels = np.zeros(6, np.float64)
    for _, _, label in frames:
        pixels += np.bincount(label.ravel(), minlength=7)[1:7]
    frequency = pixels / pixels.sum()
    weights = np.clip(np.sqrt(frequency.mean()/np.maximum(frequency, 1e-9)), 0.5, 3.0)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    rng = np.random.RandomState(args.seed)
    history, best_miou, best_step, best_state = [], -1, 0, None
    start = time.time()
    metadata = {"architecture": "resnet34_unet_groupnorm_v1", "phase": phase,
                "classes": CLASS_NAMES, "ignore_label": 0, "seed": args.seed,
                "train_frames": names, "validation_frames": validation_names,
                "data_manifest_sha256": sha256_file(args.data_manifest),
                "imagenet_initialization": str(Path(args.pretrained).resolve()),
                "imagenet_sha256": sha256_file(args.pretrained),
                "test_frames_used_for_training_or_selection": [],
                "train_settings": {"steps": steps, "batch_size": args.batch_size, "crop": args.crop,
                                   "initial_lr": args.lr, "weight_decay": 1e-4,
                                   "loss": "train-only sqrt frequency weighted CE + 0.5 soft Dice, ignore zero",
                                   "class_weights": weights.cpu().numpy().tolist(),
                                   "augmentation": "native random crop, 50% class-balanced crop center, D4 flips/rotations and mild color jitter",
                                   "encoder_bn": "fixed ImageNet running statistics, trainable affine"},
                "inference_settings": {"tile": 512, "stride": 384, "batch_size": 4,
                                       "native_resolution": True, "overlap_aggregation": "mean softmax probability"}}
    for step in range(1, steps + 1):
        model.train()
        rgb, target = training_batch(frames, rng, args.batch_size, args.crop, device)
        lr = args.lr * (1.0 - (step-1)/float(steps)) ** 0.9
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            logits = model(rgb)
            ce = functional.cross_entropy(logits, target, weight=weights, ignore_index=255)
            valid = (target != 255).unsqueeze(1)
            safe_target = target.clamp(0, 5)
            truth = functional.one_hot(safe_target, 6).permute(0, 3, 1, 2).float() * valid
            probabilities = functional.softmax(logits.float(), 1) * valid
            intersection = (probabilities * truth).sum((0, 2, 3))
            denominator = (probabilities + truth).sum((0, 2, 3))
            dice = 1.0 - ((2*intersection+1) / (denominator+1)).mean()
            loss = ce + 0.5*dice
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % 25 == 0 or step == steps:
            row = {"step": step, "loss": float(loss.detach()), "ce": float(ce.detach()),
                   "dice_loss": float(dice.detach()), "lr": lr,
                   "elapsed_seconds": time.time()-start}
            print(json.dumps({"phase": phase, **row}), flush=True)
            history.append(row)
        if validation and (step % args.val_interval == 0 or step == steps):
            miou, confusion = validation_miou(model, validation, device)
            record = {"step": step, "validation_miou": miou, "confusion_matrix": confusion}
            history.append(record)
            print(json.dumps({"phase": phase, **record}), flush=True)
            if miou > best_miou:
                best_miou, best_step = miou, step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                torch.save({"state_dict": best_state, "metadata": dict(metadata, selected_step=best_step,
                            validation_miou=best_miou)}, str(output / "inner_best.pt"))
            json_write(output / (phase + "_history.json"), history)
    final_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if not validation:
        best_step, best_state = steps, final_state
    json_write(output / (phase + "_history.json"), history)
    del model, optimizer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"state_dict": best_state, "metadata": dict(metadata, selected_step=best_step,
            validation_miou=None if not validation else best_miou)}, best_step


def train(args):
    torch, _, _, _ = torch_modules()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; start this command only in the GPU queue")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.data_manifest).read_text())
    inner, selected = train_run(manifest, manifest["splits"]["inner_train"],
                                manifest["splits"]["inner_val"], output, args, args.steps, "inner_selection")
    if args.skip_refit:
        final = inner
    else:
        final, _ = train_run(manifest, manifest["splits"]["refit_train"], [],
                             output, args, selected, "refit_all8")
    final["metadata"]["selection_source"] = "inner_best.pt, validation mIoU only; no test labels"
    torch.save(final, str(output / "model.pt"))
    protocol = {"status": "complete", "checkpoint": str(output / "model.pt"),
                "checkpoint_sha256": sha256_file(output / "model.pt"),
                "inner_best_checkpoint": str(output / "inner_best.pt"),
                "inner_best_sha256": sha256_file(output / "inner_best.pt"),
                "selected_step": selected, "refit_all8": not args.skip_refit,
                "metadata": final["metadata"],
                "qualification": "Scene-specific 2D downstream evaluator, with only two annotated held-out views. Independent of all compression training. Not a 3D semantic-label or physical-instance ground truth."}
    json_write(output / "training_protocol.json", protocol)
    print(json.dumps(protocol), flush=True)


def infer(args):
    torch, _, _, model_class = torch_modules()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable; start this command only in the GPU queue")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = model_class()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    inputs = json.loads(Path(args.inputs_json).read_text())
    predictions, seen_ids = [], set()
    for item in inputs["images"]:
        if item["id"] in seen_ids:
            raise ValueError("Duplicate input image id")
        seen_ids.add(item["id"])
        rgb = np.asarray(Image.open(item["path"]).convert("RGB"))
        prediction = predict_native(model, rgb, args.device, args.tile, args.stride, args.batch_size)
        safe_id = hashlib.sha256(item["id"].encode("utf-8")).hexdigest()[:20]
        prediction_path = output / (safe_id + ".png")
        Image.fromarray(prediction, mode="L").save(str(prediction_path))
        record = {"id": item["id"], "image_path": str(Path(item["path"]).resolve()),
                  "image_sha256": sha256_file(item["path"]),
                  "prediction_path": str(prediction_path), "prediction_sha256": sha256_file(prediction_path),
                  "shape": list(prediction.shape)}
        predictions.append(record)
        print(json.dumps(record), flush=True)
    json_write(output / "predictions_manifest.json", {
        "schema": "city_semantic_predictions_v1", "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint), "metadata": checkpoint["metadata"],
        "inputs_json_sha256": sha256_file(args.inputs_json),
        "inference_settings": {"tile": args.tile, "stride": args.stride,
                               "batch_size": args.batch_size, "native_resolution": True,
                               "aggregation": "mean softmax probabilities"}, "predictions": predictions})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--city_source", default="/home/zhangys/gaussian-splatting（原生）/data/city")
    prepare.add_argument("--segmentation_source", default="/home/zhangys/semantic_data/3D-AS_dataset/City/scene0/segmentation")
    fit = sub.add_parser("train")
    fit.add_argument("--data_manifest", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--pretrained", default="/home/zhangys/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth")
    fit.add_argument("--steps", type=int, default=1500)
    fit.add_argument("--batch_size", type=int, default=4)
    fit.add_argument("--crop", type=int, default=384)
    fit.add_argument("--lr", type=float, default=2e-4)
    fit.add_argument("--seed", type=int, default=20261104)
    fit.add_argument("--val_interval", type=int, default=100)
    fit.add_argument("--skip_refit", action="store_true")
    fit.add_argument("--device", default="cuda")
    prediction = sub.add_parser("infer")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--inputs_json", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--tile", type=int, default=512)
    prediction.add_argument("--stride", type=int, default=384)
    prediction.add_argument("--batch_size", type=int, default=4)
    prediction.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "prepare":
        path = prepare_dataset(args.output, args.city_source, args.segmentation_source, Path(__file__).parent)
        print(str(path), flush=True)
    elif args.command == "train":
        train(args)
    else:
        infer(args)


if __name__ == "__main__":
    main()
