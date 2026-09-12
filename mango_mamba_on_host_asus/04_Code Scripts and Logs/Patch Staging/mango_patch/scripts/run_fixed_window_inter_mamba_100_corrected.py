#!/usr/bin/env python3
"""Run the intended 100-epoch baseline with corrected trainer semantics."""

import json
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "fixed_window_inter_mamba_100_corrected"
SEEDS = (123, 42, 2024, 3333)
PROFILE = {
    "epochs": 100,
    "patience_epochs": 20,
    "warmup_epochs": 5,
    "batch_size": 64,
    "grad_accum_steps": 4,
    "lr": 0.0005,
    "weight_decay": 0.01,
    "smoothing": 0.01,
    "drop_path_rate": 0.01,
    "re_prob": 0.05,
    "color_jitter": 0.05,
    "checkpoint_monitor": "loss",
}


def command(seed: int) -> list[str]:
    return [
        sys.executable, "train_mango_mamba.py",
        "--data_dir", str(ROOT / "data" / "mangoleafBD"), "--num-classes", "8",
        "--epochs", "100", "--patience-epochs", "20", "--warmup-epochs", "5",
        "--batch-size", "64", "--grad-accum-steps", "4", "--workers", "4",
        "--opt", "adamw", "--lr", "0.0005", "--weight-decay", "0.01",
        "--sched", "cosine", "--min-lr", "0.000005", "--smoothing", "0.01",
        "--drop-path-rate", "0.01", "--re-prob", "0.05", "--color-jitter", "0.05",
        "--hflip", "0.5", "--vflip", "0.2", "--early-stopping-monitor", "loss",
        "--seed", str(seed), "--output", str(OUTPUT / "mangoleafbd" / "fixed_window_inter_mamba"),
        "--experiment", f"fixed_window_inter_mamba_corrected_seed_{seed}", "--log-interval", "10",
        "--model", "mango_mamba_tiny", "--mixer-mode", "hybrid",
        "--use-eca", "--use-grn", "--use-swiglu", "--inter-window-mamba-stages", "0",
        "--window-size", "7", "7", "7", "7",
    ]


def verify_checkpoint(seed: int) -> None:
    run = max((OUTPUT / "mangoleafbd" / "fixed_window_inter_mamba").glob(
        f"fixed_window_inter_mamba_corrected_seed_{seed}-*/training_results.json"
    ), key=lambda path: path.stat().st_mtime).parent
    result = json.loads((run / "training_results.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(run / "model_best.pth.tar", map_location="cpu", weights_only=False)
    saved_epoch = int(checkpoint["epoch"]) + 1
    reported_epoch = int(result["early_stopping"]["best_epoch"])
    if saved_epoch != reported_epoch:
        raise RuntimeError(f"seed {seed}: model_best epoch {saved_epoch} != reported epoch {reported_epoch}")


def main() -> None:
    if "--dry-run" in sys.argv:
        for seed in SEEDS:
            print("RUN", " ".join(command(seed)))
        return
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"output must be empty: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "seeds": SEEDS,
        "dataset": "MangoLeafBD",
        "profile": PROFILE,
        "checkpoint_selection": "minimum validation loss; saved epoch must equal reported best epoch",
    }, indent=2), encoding="utf-8")
    for seed in SEEDS:
        subprocess.run(command(seed), cwd=ROOT, check=True)
        verify_checkpoint(seed)


if __name__ == "__main__":
    main()
