#!/usr/bin/env python3
"""Check thesis imports and CUDA forward/backward without training or saving weights."""

import importlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys


def main():
    for name in ('torchvision', 'timm', 'einops', 'pytorch_grad_cam', 'matplotlib',
                 'seaborn', 'sklearn', 'yaml', 'mamba_ssm'):
        importlib.import_module(name)
    import torch
    import selective_scan_cuda
    from mango_mamba import mango_mamba_tiny

    if not torch.cuda.is_available():
        raise RuntimeError('The thesis training and analysis scripts require an NVIDIA CUDA GPU.')
    torch.manual_seed(42)
    checks = []
    for size, classes, windows, inter in ((224, 8, (8, 8, 14, 7), ()),
                                         (320, 7, (8, 8, 5, 5), (0,))):
        model = mango_mamba_tiny(
            num_classes=classes, mixer_mode='hybrid', use_eca=True, use_grn=True,
            use_swiglu=True, window_size=list(windows), inter_window_mamba_stages=inter,
        ).cuda().train()
        images = torch.randn(2, 3, size, size, device='cuda')
        with torch.amp.autocast('cuda'):
            logits = model(images)['logits']
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2, dtype=torch.long, device='cuda'))
        if logits.shape != (2, classes) or not torch.isfinite(loss).item():
            raise RuntimeError(f'Invalid model output at {size}px: {logits.shape}, loss={loss.item()}')
        loss.backward()
        scan_gradients = [p.grad for name, p in model.named_parameters() if name.endswith('A_log')]
        if not scan_gradients or not all(g is not None and torch.isfinite(g).all().item() for g in scan_gradients):
            raise RuntimeError(f'Missing or nonfinite selective-scan gradients at {size}px')
        model.eval()
        with torch.inference_mode():
            fp32_logits = model(images)['logits']
        if not torch.isfinite(fp32_logits).all().item():
            raise RuntimeError(f'Nonfinite FP32 inference at {size}px')
        checks.append({'resolution': size, 'classes': classes, 'amp_forward_backward': 'passed',
                       'selective_scan_gradients': 'passed', 'fp32_forward': 'passed'})
        del model, images, logits, loss, fp32_logits, scan_gradients
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(json.dumps({
        'python': sys.version.split()[0], 'machine': platform.machine(),
        'packages': {name: version(name) for name in ('torch', 'torchvision', 'timm', 'mamba-ssm', 'einops', 'grad-cam')},
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
        'selective_scan_extension': str(Path(selective_scan_cuda.__file__).resolve()),
        'checks': checks,
        'scope': 'Synthetic batches only; no dataset evaluation or experiment reproduction.',
    }, indent=2))


if __name__ == '__main__':
    main()
