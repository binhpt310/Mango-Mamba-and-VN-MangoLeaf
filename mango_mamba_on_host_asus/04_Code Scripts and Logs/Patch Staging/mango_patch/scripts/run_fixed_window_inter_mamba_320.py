#!/usr/bin/env python3
"""320px MangoLeafBD trial; all settings match the corrected 224px baseline."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "fixed_window_inter_mamba_320"
SEEDS = tuple(map(int, sys.argv[1:] or (123, 42, 2024, 3333)))


def command(seed: int) -> list[str]:
    return [
        sys.executable, "train_mango_mamba.py",
        "--data_dir", str(ROOT / "data" / "mangoleafBD"), "--num-classes", "8",
        "--input-size", "3", "320", "320", "--epochs", "100", "--patience-epochs", "20",
        "--warmup-epochs", "5", "--batch-size", "64", "--grad-accum-steps", "4", "--workers", "4",
        "--opt", "adamw", "--lr", "0.0005", "--weight-decay", "0.01", "--sched", "cosine",
        "--min-lr", "0.000005", "--smoothing", "0.01", "--drop-path-rate", "0.01",
        "--re-prob", "0.05", "--color-jitter", "0.05", "--hflip", "0.5", "--vflip", "0.2",
        "--early-stopping-monitor", "loss", "--seed", str(seed),
        "--output", str(OUTPUT / "mangoleafbd" / "fixed_window_inter_mamba"),
        "--experiment", f"fixed_window_inter_mamba_320_seed_{seed}", "--log-interval", "10",
        "--model", "mango_mamba_tiny", "--mixer-mode", "hybrid", "--use-eca", "--use-grn",
        "--use-swiglu", "--inter-window-mamba-stages", "0", "--window-size", "7", "7", "7", "7",
    ]


for seed in SEEDS:
    subprocess.run(command(seed), cwd=ROOT, check=True)
