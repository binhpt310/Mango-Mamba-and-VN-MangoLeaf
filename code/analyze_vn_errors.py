#!/usr/bin/env python3
"""Export reproducible VN-MangoLeaf errors and classifier CAM evidence for Final V3.

This is descriptive held-out-test analysis for one already-trained curriculum
checkpoint.  It does not retrain, tune on test data, or alter the checkpoint.
The CAM comes directly from the final normalized feature map and linear head:
for this GAP + linear classifier it is the Grad-CAM-equivalent class map, up to
a positive scalar, without requiring an extra backward pass.
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
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from timm.data import create_transform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mango_mamba import mango_mamba_tiny


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


def save_confusion_matrix(matrix: np.ndarray, names: list[str], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Image count")
    axis.set(
        xticks=np.arange(len(names)), yticks=np.arange(len(names)),
        xticklabels=names, yticklabels=names, ylabel="True class", xlabel="Predicted class",
        title="Final V3 curriculum Phase 3: VN-MangoLeaf held-out test",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def class_cam(feature_map: np.ndarray, weights: np.ndarray, class_index: int) -> np.ndarray:
    """Return ReLUed, normalized class-activation map at final feature resolution."""
    cam = np.einsum("c,chw->hw", weights[class_index], feature_map)
    cam = np.maximum(cam, 0)
    maximum = float(cam.max())
    return cam / maximum if maximum > 0 else cam


def save_error_panel(
    original_path: Path,
    output_path: Path,
    true_name: str,
    predicted_name: str,
    confidence: float,
    true_cam: np.ndarray,
    predicted_cam: np.ndarray,
) -> None:
    with Image.open(original_path) as image:
        rgb = image.convert("RGB")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Original image")
        axes[1].imshow(rgb)
        axes[1].imshow(true_cam, cmap="jet", alpha=0.46, interpolation="bilinear", extent=(0, rgb.width, rgb.height, 0))
        axes[1].set_title(f"True class CAM\n{true_name}")
        axes[2].imshow(rgb)
        axes[2].imshow(predicted_cam, cmap="jet", alpha=0.46, interpolation="bilinear", extent=(0, rgb.width, rgb.height, 0))
        axes[2].set_title(f"Predicted class CAM\n{predicted_name} ({confidence:.1%})")
        for axis in axes:
            axis.axis("off")
        fig.suptitle("Final V3 curriculum Phase 3: held-out VN-MangoLeaf error", y=0.98)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
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
    activations: list[torch.Tensor] = []

    def capture_norm(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output.detach().cpu())

    hook = model.norm.register_forward_hook(capture_norm)
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.cuda(non_blocking=True))["logits"]
            feature_maps = activations.pop().numpy()
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
                        "feature_map": feature_maps[index],
                    })
            offset += len(label_values)
    hook.remove()

    errors.sort(key=lambda item: float(item["predicted_confidence"]), reverse=True)
    error_images_dir = output_dir
    head_weights = model.head.weight.detach().cpu().numpy()
    csv_rows: list[dict[str, object]] = []
    for index, error in enumerate(errors, start=1):
        original = Path(str(error["path"]))
        stem = f"error_{index:03d}_true_{error['true_class']}_pred_{error['predicted_class']}"
        source_copy = error_images_dir / f"{stem}{original.suffix.lower()}"
        panel = error_images_dir / f"{stem}_cam.png"
        shutil.copy2(original, source_copy)
        feature_map = np.asarray(error.pop("feature_map"))
        save_error_panel(
            original, panel, str(error["true_class"]), str(error["predicted_class"]),
            float(error["predicted_confidence"]),
            class_cam(feature_map, head_weights, int(error["true_index"])),
            class_cam(feature_map, head_weights, int(error["predicted_index"])),
        )
        csv_rows.append({**error, "source_copy": str(source_copy.relative_to(output_dir)), "cam_panel": str(panel.relative_to(output_dir))})

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(dataset.classes))))
    report = classification_report(
        targets, predictions, labels=list(range(len(dataset.classes))), target_names=dataset.classes,
        output_dict=True, zero_division=0,
    )
    accuracy = float(np.mean(np.asarray(targets) == np.asarray(predictions)))
    save_confusion_matrix(matrix, dataset.classes, output_dir / "confusion_matrix_test.png")
    with (output_dir / "misclassifications.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "true_index", "true_class", "predicted_index", "predicted_class", "predicted_confidence", "true_class_probability", "source_copy", "cam_panel"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with (output_dir / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "precision", "recall", "f1-score", "support"])
        writer.writeheader()
        for name in dataset.classes:
            writer.writerow({"class": name, **report[name]})

    return {
        "test_images": len(targets), "test_accuracy": accuracy, "error_count": len(errors),
        "class_names": dataset.classes, "classification_report": report, "confusion_matrix": matrix.tolist(),
        "misclassification_csv": "misclassifications.csv", "per_class_metrics_csv": "per_class_metrics.csv",
        "confusion_matrix_png": "confusion_matrix_test.png", "error_images_directory": ".",
        "cam_method": "Final normalized feature map × linear classifier weight (GAP + linear Grad-CAM equivalent up to a positive scalar).",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True, help="VN-MangoLeaf root containing test/")
    parser.add_argument("--source-args", type=Path, required=True, help="args.yaml saved by the selected training run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this analysis")
    if not args.checkpoint.is_file() or not args.source_args.is_file() or not (args.data_dir / "test").is_dir():
        raise FileNotFoundError("checkpoint, source args, or VN-MangoLeaf test split is missing")

    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), args.output / Path(__file__).name)
    shutil.copy2(args.source_args, args.output / "source_args.yaml")
    torch.backends.cudnn.benchmark = True
    model = load_model(args.checkpoint)
    error_analysis = evaluate(model, args.data_dir / "test", args.output, args.eval_batch_size, args.workers)
    properties = torch.cuda.get_device_properties(0)
    summary = {
        "purpose": "Held-out VN-MangoLeaf error analysis for one Final V3 Phase-3 curriculum checkpoint",
        "selection_note": "Selected post-hoc as the strongest currently recorded Phase-3 curriculum seed by held-out test accuracy; descriptive only, not an unbiased selection protocol.",
        "timestamp_unix": time(), "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "model_config": V3_CONFIG, "input_size": [3, INPUT_SIZE, INPUT_SIZE],
        "environment": {"gpu": properties.name, "gpu_memory_gib": properties.total_memory / 2**30, "torch": torch.__version__, "cuda": torch.version.cuda, "precision": "FP32"},
        "error_analysis": error_analysis,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
