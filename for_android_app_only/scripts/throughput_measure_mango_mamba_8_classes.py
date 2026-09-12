#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# MangoMamba Throughput Measurement Script
# Modified from MambaVision throughput_measure.py to work with MangoMamba models

import torch
import argparse
import time
import numpy as np
import os
from ptflops import get_model_complexity_info
from models.mango_mamba import mango_mamba_tiny

def load_mango_mamba_model(checkpoint_path, num_classes=8):
    """Load MangoMamba model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Create the model architecture
    model = mango_mamba_tiny(num_classes=num_classes)

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        # Lightweight checkpoint format (.pth)
        state_dict = checkpoint['model_state_dict']
        model_name = checkpoint.get('model_name', 'MangoMamba')
        print(f"Loaded lightweight checkpoint - Model: {model_name}")
    elif 'state_dict' in checkpoint:
        # Full checkpoint format (.pth.tar)
        state_dict = checkpoint['state_dict']
        print(f"Loaded full checkpoint - Epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        # Direct state dict
        state_dict = checkpoint
        print("Loaded direct state dict")

    # Load the state dict into the model
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Model weights loaded successfully!")
    except RuntimeError as e:
        print(f"Error loading model weights: {e}")
        # Try to load with strict=False for debugging
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        raise e

    return model

def measure_throughput(model, batch_size, resolution, channel_last=False, warmup_runs=10, test_runs=500):
    """Measure model throughput and inference time"""

    # Create input data
    input_data = torch.randn((batch_size, 3, resolution, resolution), device='cuda')

    # Use channel_last format if specified
    if channel_last:
        input_data = input_data.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)
        print("Using channel_last memory format")

    # Move model to GPU and set to eval mode
    model.cuda()
    model.eval()

    # Calculate model complexity
    try:
        macs, params = get_model_complexity_info(
            model,
            tuple([3, resolution, resolution]),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        print(f"Model stats: MACs: {macs:,}, Params: {params:,}")
        print(f"Model size: {params * 4 / 1024 / 1024:.2f} MB (float32)")
    except Exception as e:
        print(f"Warning: Could not calculate model complexity: {e}")
        macs, params = 0, 0

    # Warmup runs
    print(f"Warming up with {warmup_runs} runs...")
    with torch.cuda.amp.autocast():
        for i in range(warmup_runs):
            with torch.no_grad():
                output = model(input_data)
                # Handle MangoMamba output format (dict with 'logits' key)
                if isinstance(output, dict) and 'logits' in output:
                    logits = output['logits']

    # Ensure GPU synchronization before timing
    torch.cuda.synchronize()

    # Actual throughput measurement
    print(f"Measuring throughput with {test_runs} runs...")
    timer = []
    start_time = time.time()

    with torch.cuda.amp.autocast():
        for i in range(test_runs):
            start_time_loc = time.time()
            with torch.no_grad():
                output = model(input_data)
                # Handle MangoMamba output format
                if isinstance(output, dict) and 'logits' in output:
                    logits = output['logits']
            timer.append(time.time() - start_time_loc)

        torch.cuda.synchronize()

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    avg_time_per_batch = total_time / test_runs
    avg_time_per_sample = avg_time_per_batch / batch_size
    throughput_samples_per_sec = batch_size / avg_time_per_batch
    throughput_median = batch_size / np.median(timer)

    # Print results
    print("\n" + "="*60)
    print("THROUGHPUT MEASUREMENT RESULTS")
    print("="*60)
    print(f"Model Parameters: {params:,}")
    print(f"Model MACs: {macs:,}")
    print(f"Input Resolution: {resolution}x{resolution}")
    print(f"Batch Size: {batch_size}")
    print(f"Test Runs: {test_runs}")
    print(f"Channel Last: {channel_last}")
    print("-"*60)
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Time per Batch: {avg_time_per_batch*1000:.2f}ms")
    print(f"Average Time per Sample: {avg_time_per_sample*1000:.2f}ms")
    print(f"Throughput (Average): {throughput_samples_per_sec:.2f} samples/sec")
    print(f"Throughput (Median): {throughput_median:.2f} samples/sec")
    print(f"Inference Time per Sample: {avg_time_per_sample*1000:.2f}ms")
    print("="*60)

    return {
        'throughput_avg': throughput_samples_per_sec,
        'throughput_median': throughput_median,
        'inference_time_ms': avg_time_per_sample * 1000,
        'batch_time_ms': avg_time_per_batch * 1000,
        'total_time': total_time,
        'params': params,
        'macs': macs
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MangoMamba Throughput Measurement')
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to model checkpoint (.pth or .pth.tar)")
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Number of classes in the model")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Input image resolution")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for throughput measurement")
    parser.add_argument("--channel-last", action="store_true",
                        help="Use channel_last memory format")
    parser.add_argument("--warmup-runs", type=int, default=10,
                        help="Number of warmup runs")
    parser.add_argument("--test-runs", type=int, default=500,
                        help="Number of test runs for measurement")

    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This script requires a GPU.")
        exit(1)

    print(f"Using GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")

    try:
        # Load the model
        model = load_mango_mamba_model(args.checkpoint, args.num_classes)

        # Measure throughput
        results = measure_throughput(
            model=model,
            batch_size=args.batch_size,
            resolution=args.resolution,
            channel_last=args.channel_last,
            warmup_runs=args.warmup_runs,
            test_runs=args.test_runs
        )

        print(f"\nSummary for {os.path.basename(args.checkpoint)}:")
        print(f"  Inference Time: {results['inference_time_ms']:.2f}ms per sample")
        print(f"  Throughput: {results['throughput_avg']:.2f} samples/sec")
        print(f"  Model Size: {results['params'] * 4 / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"Error during throughput measurement: {e}")
        import traceback
        traceback.print_exc()
