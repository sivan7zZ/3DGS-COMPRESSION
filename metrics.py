# Copyright (C) 2023, Inria. All rights reserved. See LICENSE.md.
"""Evaluate saved standard PNG renders; errors fail the pipeline."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from lpipsPyTorch import LPIPS
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.checkpoint_utils import atomic_json_save

@torch.no_grad()
def evaluate(model_paths, device="cuda"):
    device = torch.device(device)
    criterion = LPIPS("vgg", "0.1").to(device).eval()
    for model_path in model_paths:
        root = Path(model_path)
        test = root / "test"
        if not test.is_dir():
            raise FileNotFoundError(str(test))
        results, per_view = {}, {}
        for method in sorted(p for p in test.iterdir() if p.is_dir()):
            files = sorted((method / "renders").glob("*.png"))
            if not files:
                raise ValueError("No PNG renders in " + str(method))
            values = {"PSNR": {}, "SSIM": {}, "LPIPS": {}}
            for file in files:
                with Image.open(file) as im:
                    image = to_tensor(im.convert("RGB")).unsqueeze(0).to(device)
                with Image.open(method / "gt" / file.name) as im:
                    target = to_tensor(im.convert("RGB")).unsqueeze(0).to(device)
                if image.shape != target.shape:
                    raise ValueError("Render/GT size mismatch: " + file.name)
                values["PSNR"][file.name] = psnr(image, target).mean().item()
                values["SSIM"][file.name] = ssim(image, target).mean().item()
                values["LPIPS"][file.name] = criterion(image, target).mean().item()
            results[method.name] = {key: sum(v.values()) / len(v) for key, v in values.items()}
            per_view[method.name] = values
            print(str(root), method.name, results[method.name], flush=True)
        if not results:
            raise ValueError("No evaluation methods in " + str(test))
        atomic_json_save(results, root / "results.json")
        atomic_json_save(per_view, root / "per_view.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_paths", nargs="+", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    evaluate(args.model_paths, args.device)
