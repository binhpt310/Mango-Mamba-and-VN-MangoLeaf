#!/usr/bin/env python3
"""Export a trained MangoMamba checkpoint to ONNX.

The DiseaseAwareMixer's selective-scan uses a custom CUDA kernel
(mamba_ssm.selective_scan_fn) that has no ONNX symbolic. For export, the
model is switched to mamba_ssm's pure-PyTorch reference scan
(selective_scan_ref) via models.mango_mamba.EXPORT_MODE, which produces an
identical (up to floating point) but fully traceable computation graph.

Note: window_partition/window_reverse recover the batch size via
`int(windows.shape[0] / ...)`, which the tracer bakes in as a constant.
The exported graph is therefore only valid for batch size 1 (fine for the
single-image edge/mobile inference this model targets) even though the
input has a nominal dynamic batch axis.

Usage:
    python3 export_onnx.py \
        --checkpoint output_phase4/combined_redrust_phase4-*/model_best_inference.pth \
        --num-classes 9 \
        --output output_phase4/mango_mamba_9class.onnx
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

import models.mango_mamba as mango_mamba_module
from models.mango_mamba import mango_mamba_tiny

CLASS_NAMES_BY_COUNT = {
    7: ['Anthracnose', 'Dieback', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Red Rust', 'Sooty Mold'],
    8: ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
        'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'],
    9: ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
        'Gall Midge', 'Healthy', 'Powdery Mildew', 'Red Rust', 'Sooty Mould'],
}


class LogitsOnly(nn.Module):
    """Wraps MangoMamba to return a single logits tensor (ONNX needs tensor outputs, not dicts)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)['logits']


def load_checkpoint(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


def resolve_checkpoint_path(pattern):
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matched: {pattern}")
    return matches[-1]


def main():
    parser = argparse.ArgumentParser(description='Export MangoMamba checkpoint to ONNX')
    parser.add_argument('--checkpoint', required=True,
                         help='Path (or glob pattern) to a .pth/.pth.tar checkpoint')
    parser.add_argument('--num-classes', type=int, required=True)
    parser.add_argument('--class-names', type=str, default=None,
                         help='Comma-separated class names, in label-index order. '
                              'Defaults to the built-in list for 7/8/9 classes.')
    parser.add_argument('--output', required=True, help='Output .onnx path')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--atol', type=float, default=1e-3,
                         help='Max allowed abs diff between PyTorch and ONNX Runtime logits')
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    print(f'Loading checkpoint: {checkpoint_path}')

    if args.class_names:
        class_names = [c.strip() for c in args.class_names.split(',')]
    else:
        class_names = CLASS_NAMES_BY_COUNT.get(args.num_classes)
    if class_names is None or len(class_names) != args.num_classes:
        raise ValueError(
            f'No default class names for num_classes={args.num_classes}; pass --class-names explicitly.'
        )

    model = mango_mamba_tiny(num_classes=args.num_classes, drop_path_rate=0.0)
    load_checkpoint(model, checkpoint_path)
    model.eval()

    # Switch the Mamba mixer to the traceable pure-PyTorch reference scan.
    mango_mamba_module.EXPORT_MODE = True

    wrapped = LogitsOnly(model)
    wrapped.eval()

    dummy_input = torch.randn(1, 3, args.image_size, args.image_size)

    with torch.no_grad():
        torch_logits = wrapped(dummy_input)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    print(f'Exporting to ONNX (opset {args.opset}, static batch=1)...')
    torch.onnx.export(
        wrapped,
        dummy_input,
        args.output,
        input_names=['image'],
        output_names=['logits'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f'Saved: {args.output}')

    # --- Verify numerical parity between PyTorch and ONNX Runtime ---
    # (keep EXPORT_MODE=True here too: the PyTorch reference must match what
    # was traced, and CPU tensors can't run the CUDA-only selective_scan_fn)
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print('ONNX model structurally valid (onnx.checker passed).')

    sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])

    # Graph is static-batch=1 (see module docstring), so only batch=1 inputs are valid.
    max_diff = 0.0
    rng = np.random.default_rng(42)
    test_inputs = [dummy_input.numpy()]
    test_inputs += [rng.standard_normal((1, 3, args.image_size, args.image_size)).astype(np.float32)
                    for _ in range(3)]

    for np_input in test_inputs:
        with torch.no_grad():
            torch_out = wrapped(torch.from_numpy(np_input)).numpy()
        onnx_out = sess.run(['logits'], {'image': np_input})[0]
        diff = np.abs(torch_out - onnx_out).max()
        max_diff = max(max_diff, diff)
        print(f'  batch={np_input.shape[0]}  max|PyTorch - ONNXRuntime| = {diff:.6g}')

    mango_mamba_module.EXPORT_MODE = False

    if max_diff > args.atol:
        raise AssertionError(f'Max diff {max_diff:.6g} exceeds tolerance {args.atol:.6g}')
    print(f'Parity check passed (max diff {max_diff:.6g} <= tol {args.atol:.6g}).')

    labels_path = os.path.splitext(args.output)[0] + '_labels.json'
    with open(labels_path, 'w') as f:
        json.dump({'class_names': class_names}, f, indent=2)
    print(f'Saved labels: {labels_path}')

    onnx_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f'ONNX file size: {onnx_size_mb:.2f} MB')


if __name__ == '__main__':
    main()
