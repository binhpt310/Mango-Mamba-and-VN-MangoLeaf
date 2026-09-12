#!/usr/bin/env python3
"""Run the nine clean-split ablation cases for one target dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "train_mango_mamba.py"
RESULTS = ROOT / "results" / "ablation_mangoleafbd_vn_100"

PROFILE = {
    "epochs": 100,
    "batch_size": 64,
    "grad_accum_steps": 4,
    "workers": 4,
    "lr": 0.0005,
    "warmup_epochs": 5,
    "patience_epochs": 20,
}

DATASETS = {
    "mangoleafbd": {
        "path": ROOT / "data" / "mangoleafBD",
        "num_classes": 8,
    },
    "vn_mangoleaf": {
        "path": ROOT / "data" / "vn_mangoleaf",
        "num_classes": 7,
    },
}


def features(
    mixer: str = "hybrid",
    *,
    eca: bool = False,
    grn: bool = False,
    swiglu: bool = False,
    fixed_window: bool = False,
) -> list[str]:
    flags = ["--mixer-mode", mixer]
    if eca:
        flags.append("--use-eca")
    if grn:
        flags.append("--use-grn")
    if swiglu:
        flags.append("--use-swiglu")
    window = ("7", "7", "7", "7") if fixed_window else ("8", "8", "14", "7")
    return [*flags, "--window-size", *window]


CASES = {
    "lka_features": features("pure_lka", eca=True, grn=True, swiglu=True),
    "mamba_features": features("pure_mamba", eca=True, grn=True, swiglu=True),
    "lka_minimal": features("pure_lka"),
    "hybrid_minimal": features("hybrid"),
    "no_eca": features("hybrid", grn=True, swiglu=True),
    "no_swiglu": features("hybrid", eca=True, grn=True),
    "no_grn": features("hybrid", eca=True, swiglu=True),
    "fixed_window": features("hybrid", eca=True, grn=True, swiglu=True, fixed_window=True),
    "original_window": features("hybrid", eca=True, grn=True, swiglu=True),
}


def run_case(dataset_name: str, seed: int, case_name: str) -> None:
    dataset = DATASETS[dataset_name]
    command = [
        sys.executable,
        str(TRAINER),
        "--data_dir",
        str(dataset["path"]),
        "--num-classes",
        str(dataset["num_classes"]),
        "--epochs",
        str(PROFILE["epochs"]),
        "--batch-size",
        str(PROFILE["batch_size"]),
        "--grad-accum-steps",
        str(PROFILE["grad_accum_steps"]),
        "--workers",
        str(PROFILE["workers"]),
        "--lr",
        str(PROFILE["lr"]),
        "--warmup-epochs",
        str(PROFILE["warmup_epochs"]),
        "--patience-epochs",
        str(PROFILE["patience_epochs"]),
        "--seed",
        str(seed),
        "--output",
        str(RESULTS / dataset_name / case_name),
        "--experiment",
        f"{case_name}_seed_{seed}",
        "--log-interval",
        "10",
        *CASES[case_name],
    ]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    if not TRAINER.is_file():
        raise RuntimeError(f"trainer not found: {TRAINER}")
    for split in ("train", "val", "test"):
        if not (dataset["path"] / split).is_dir():
            raise RuntimeError(f"missing {split} split: {dataset['path'] / split}")

    output = RESULTS / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "plan": "nine ablation cases on a clean-split target dataset",
        "dataset": args.dataset,
        "dataset_path": str(dataset["path"]),
        "num_classes": dataset["num_classes"],
        "seed": args.seed,
        "profile": PROFILE,
        "cases": list(CASES),
        "case_arguments": CASES,
        "window_policy": "[8,8,14,7] for every case except fixed_window, which is full Hybrid + ECA + GRN + SwiGLU with [7,7,7,7]",
    }
    (output / f"run_manifest_seed_{args.seed}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for case_name in CASES:
        run_case(args.dataset, args.seed, case_name)


if __name__ == "__main__":
    main()
