#!/usr/bin/env python3
"""Export the already-trained Final V3 curriculum model for Android CAM display.

The static batch-one ONNX graph returns both classification logits and the
post-normalization 10x10 feature map.  Android can form a class-activation map
from ``relu(sum_c(feature_map[c] * classifier_weight[class, c]))``, normalize
it, bilinearly resize it to the displayed image, and overlay it.  This is
equivalent to Grad-CAM for Mango Mamba's final GAP + linear classifier, up to a
positive scalar.  No training is performed by this exporter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.mango_mamba as mango_mamba_module
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref
from models.mango_mamba import mango_mamba_tiny


V3_CONFIG = {
    "num_classes": 7,
    "drop_path_rate": 0.01,
    "mixer_mode": "hybrid",
    "use_eca": True,
    "use_grn": True,
    "use_swiglu": True,
    "window_size": [8, 8, 5, 5],
    "inter_window_mamba_stages": [0],
}
CLASS_NAMES = ["Anthracnose", "Dieback", "Gall_Midge", "Healthy", "Powdery_Mildew", "Red_Rust", "Sooty_Mold"]
INPUT_SIZE = 320


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: Path) -> nn.Module:
    model = mango_mamba_tiny(**V3_CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.eval()


class LogitsAndFeatureMap(nn.Module):
    """Match model.forward while retaining the map immediately before GAP."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.model.patch_embed(image)
        for stage in self.model.stages:
            feature_map = stage(feature_map)
        feature_map = self.model.norm(feature_map)
        features = torch.flatten(self.model.avgpool(feature_map), 1)
        logits = self.model.head(features)
        return logits, feature_map


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Destination .onnx file")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    # The project model imports the CUDA-only scan function as a module global.
    # Swap that global for mamba-ssm's traceable reference implementation only
    # in this export process; no model/source/checkpoint file is modified.
    mango_mamba_module.selective_scan_fn = selective_scan_ref
    wrapped = LogitsAndFeatureMap(load_model(args.checkpoint)).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        torch_logits, torch_map = wrapped(dummy)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped, dummy, args.output, input_names=["image"], output_names=["logits", "feature_map"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    ort_logits, ort_map = session.run(["logits", "feature_map"], {"image": dummy.numpy()})
    logits_diff = float(np.abs(torch_logits.numpy() - ort_logits).max())
    map_diff = float(np.abs(torch_map.numpy() - ort_map).max())
    if max(logits_diff, map_diff) > args.atol:
        raise AssertionError(f"ONNX parity failure: logits={logits_diff}, feature_map={map_diff}, atol={args.atol}")

    stem = args.output.with_suffix("")
    head = wrapped.model.head
    contract = {
        "artifact": args.output.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "architecture": "Mango Mamba Final V3 curriculum Phase 3",
        "model_config": V3_CONFIG,
        "class_names": CLASS_NAMES,
        "input": {"name": "image", "shape": [1, 3, 320, 320], "layout": "NCHW", "dtype": "float32", "resize_shorter_side": 366, "center_crop": 320, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "outputs": {"logits": {"shape": [1, 7]}, "feature_map": {"shape": list(torch_map.shape)}},
        "cam": {"target_class": "use predicted argmax or a user-selected class index", "formula": "relu(sum_c(feature_map[0,c,y,x] * classifier.weight[target_class,c]))", "normalization": "divide by the map maximum when positive", "resize": "bilinear to the displayed image and overlay; it is a coarse attention map, not segmentation"},
        "classifier": {"weight": head.weight.detach().tolist(), "bias": head.bias.detach().tolist()},
        "validation": {"onnx_checker": "passed", "onnxruntime_provider": "CPUExecutionProvider", "max_abs_logit_difference": logits_diff, "max_abs_feature_map_difference": map_diff, "atol": args.atol},
    }
    manifest_path = Path(f"{stem}_android_cam_manifest.json")
    manifest_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(json.dumps({"onnx": str(args.output), "manifest": str(manifest_path), "validation": contract["validation"], "onnx_size_mib": os.path.getsize(args.output) / 2**20}, indent=2))


if __name__ == "__main__":
    main()
