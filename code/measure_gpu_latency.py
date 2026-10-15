#!/usr/bin/env python3
"""Simple native-resolution latency and test-error analysis for one Final V3 checkpoint.

This script intentionally measures one already-trained Final V3 checkpoint only.
It does not train, export, quantize, or compare models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from timm.data import create_transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mango_mamba import mango_mamba_tiny


V3_CONFIG = {
    "num_classes": 8,
    "drop_path_rate": 0.01,
    "mixer_mode": "hybrid",
    "use_eca": True,
    "use_grn": True,
    "use_swiglu": True,
    "window_size": [8, 8, 5, 5],
    "inter_window_mamba_stages": [0],
}
INPUT_SIZE = 320
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model(checkpoint_path: Path) -> torch.nn.Module:
    model = mango_mamba_tiny(**V3_CONFIG)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.cuda().eval()


def benchmark(model: torch.nn.Module, warmup: int, samples: int) -> list[float]:
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device="cuda")
    with torch.inference_mode():
        for _ in range(warmup):
            model(example)
        torch.cuda.synchronize()
        timings = []
        for _ in range(samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(example)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)))
    return timings


def save_confusion_matrix(matrix: np.ndarray, names: list[str], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Image count")
    axis.set(
        xticks=np.arange(len(names)), yticks=np.arange(len(names)),
        xticklabels=names, yticklabels=names,
        ylabel="True class", xlabel="Predicted class", title="Final V3 Phase-1: held-out test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate(model: torch.nn.Module, test_dir: Path, output_dir: Path, batch_size: int, workers: int) -> dict:
    transform = create_transform(
        input_size=(3, INPUT_SIZE, INPUT_SIZE), is_training=False, interpolation="bicubic",
        mean=MEAN, std=STD, crop_pct=0.875,
    )
    dataset = ImageFolder(test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    targets: list[int] = []
    predictions: list[int] = []
    errors: list[dict[str, object]] = []

    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.cuda(non_blocking=True))["logits"]
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predicted = probabilities.argmax(axis=1)
            label_values = labels.numpy()
            targets.extend(label_values.tolist())
            predictions.extend(predicted.tolist())
            for index, (truth, guess, probabilities_row) in enumerate(zip(label_values, predicted, probabilities)):
                if int(truth) != int(guess):
                    errors.append({
                        "path": str(Path(dataset.samples[offset + index][0]).resolve()),
                        "true_index": int(truth),
                        "true_class": dataset.classes[int(truth)],
                        "predicted_index": int(guess),
                        "predicted_class": dataset.classes[int(guess)],
                        "predicted_confidence": float(probabilities_row[int(guess)]),
                        "true_class_probability": float(probabilities_row[int(truth)]),
                    })
            offset += len(label_values)

    errors.sort(key=lambda item: float(item["predicted_confidence"]), reverse=True)
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(dataset.classes))))
    report = classification_report(
        targets, predictions, labels=list(range(len(dataset.classes))), target_names=dataset.classes,
        output_dict=True, zero_division=0,
    )
    accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))

    save_confusion_matrix(matrix, dataset.classes, output_dir / "confusion_matrix_test.png")
    with (output_dir / "misclassifications.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "true_index", "true_class", "predicted_index", "predicted_class", "predicted_confidence", "true_class_probability"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    return {
        "test_images": len(targets),
        "test_accuracy": accuracy,
        "error_count": len(errors),
        "class_names": dataset.classes,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "misclassification_csv": "misclassifications.csv",
        "confusion_matrix_png": "confusion_matrix_test.png",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="MangoLeafBD root containing test/")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if not args.checkpoint.is_file() or not (args.data_dir / "test").is_dir():
        raise FileNotFoundError("checkpoint or MangoLeafBD test split is missing")

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    torch.backends.cudnn.benchmark = True
    model = load_model(args.checkpoint)
    timings = benchmark(model, args.warmup, args.samples)
    error_analysis = evaluate(model, args.data_dir / "test", args.output, args.eval_batch_size, args.workers)
    properties = torch.cuda.get_device_properties(0)

    summary = {
        "purpose": "Single-checkpoint Final V3 descriptive latency and held-out test error analysis",
        "selection_note": "Selected as strongest current checkpoint by previously recorded held-out test accuracy; descriptive post-hoc selection.",
        "timestamp_unix": time(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": V3_CONFIG,
        "input_size": [3, INPUT_SIZE, INPUT_SIZE],
        "latency_ms_model_forward_batch_1_fp32": {
            "warmup": args.warmup,
            "samples": args.samples,
            "mean": float(np.mean(timings)),
            "median": float(np.median(timings)),
            "p95": float(np.percentile(timings, 95)),
            "std": float(np.std(timings)),
            "raw_csv": "latency_samples_ms.csv",
        },
        "environment": {
            "gpu": properties.name,
            "gpu_memory_gib": properties.total_memory / 2**30,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "precision": "FP32",
        },
        "error_analysis": error_analysis,
    }
    with (args.output / "latency_samples_ms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "latency_ms"])
        writer.writerows(enumerate(timings, start=1))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
