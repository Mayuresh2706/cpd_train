#!/usr/bin/env python3
"""
evaluate.py — Child Presence Detection (CPD) Model Evaluation

Evaluates the trained YOLOv8n-cls model on the test split and generates:
  • Overall / top-1 / top-5 accuracy
  • Per-class precision, recall, F1-score
  • Confusion matrix image
  • Sample predictions grid (16 random test images)

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model runs/classify/train/weights/best.pt
    python scripts/evaluate.py --data data/sviro_yolo/test --output results/
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = str(PROJECT_ROOT / "runs" / "classify" / "train" / "weights" / "best.pt")
DEFAULT_DATA = str(PROJECT_ROOT / "data" / "sviro_yolo" / "test")
DEFAULT_OUTPUT = str(PROJECT_ROOT / "results")

# YOLOv8 ImageFolder sorts classes alphabetically during training.
# The model's internal class order is: 0=adult_child, 1=child_only, 2=empty
CLASS_NAMES: List[str] = ["adult_child", "child_only", "empty"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate CPD YOLOv8n-cls model on the test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path to the test data directory.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Directory to save evaluation outputs.",
    )
    parser.add_argument(
        "--danger-threshold",
        type=float,
        default=0.0,
        help=(
            "Danger-biased threshold for 'child_only' class. "
            "If the model's probability for child_only exceeds this value, "
            "the prediction is forced to child_only regardless of top-1. "
            "Set to 0.0 to disable (default). Recommended: 0.15 for safety-critical use."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def collect_test_images(data_dir: Path) -> List[Tuple[Path, str]]:
    """Walk the test directory and return (image_path, class_name) pairs.

    Expects the standard ImageFolder layout::

        test/
          empty/
            img001.png
          adult_child/
            img002.png
          child_only/
            img003.png

    Args:
        data_dir: Root of the test split.

    Returns:
        List of (image_path, true_class_name) tuples.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    samples: List[Tuple[Path, str]] = []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in image_extensions:
                samples.append((img_path, class_name))
    return samples


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    class_names: List[str],
) -> Dict[str, object]:
    """Compute accuracy and per-class precision / recall / F1.

    Args:
        y_true: Ground-truth class names.
        y_pred: Predicted class names.
        class_names: Ordered list of class names.

    Returns:
        Dictionary with ``accuracy``, ``per_class`` (dict), and ``confusion_matrix`` (np.ndarray).
    """
    n = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n > 0 else 0.0

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for t, p in zip(y_true, y_pred):
        ti = name_to_idx.get(t, 0)
        pi = name_to_idx.get(p, 0)
        cm[ti, pi] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[i, :].sum()),
        }

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def compute_topk_accuracy(
    y_true: List[str],
    all_probs: List[np.ndarray],
    class_names: List[str],
    k: int = 5,
) -> float:
    """Compute top-k accuracy from probability vectors.

    Args:
        y_true: Ground-truth class names.
        all_probs: List of probability arrays (one per sample).
        class_names: Ordered class names matching the model output.
        k: Top-k value.

    Returns:
        Top-k accuracy as a float in [0, 1].
    """
    correct = 0
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    actual_k = min(k, len(class_names))

    for true_name, probs in zip(y_true, all_probs):
        true_idx = name_to_idx.get(true_name, -1)
        topk_indices = np.argsort(probs)[::-1][:actual_k]
        if true_idx in topk_indices:
            correct += 1
    return correct / len(y_true) if y_true else 0.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """Save a colour-coded confusion matrix as an image.

    Args:
        cm: Confusion matrix (rows=true, cols=predicted).
        class_names: Class labels.
        output_path: Destination file path.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", output_path)


def save_predictions_grid(
    samples: List[Tuple[Path, str]],
    predictions: List[str],
    confidences: List[float],
    output_path: Path,
    grid_size: int = 16,
) -> None:
    """Save a grid image of sample predictions.

    Args:
        samples: (image_path, true_class) pairs.
        predictions: Predicted class names.
        confidences: Prediction confidences.
        output_path: Destination file.
        grid_size: Number of images to include.
    """
    n = min(grid_size, len(samples))
    indices = random.sample(range(len(samples)), n)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes:
        ax.axis("off")

    for plot_idx, sample_idx in enumerate(indices):
        img_path, true_label = samples[sample_idx]
        pred_label = predictions[sample_idx]
        conf = confidences[sample_idx]

        img = Image.open(img_path).convert("RGB")
        axes[plot_idx].imshow(img)
        colour = "green" if pred_label == true_label else "red"
        axes[plot_idx].set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf:.1%})",
            fontsize=9,
            color=colour,
        )

    fig.suptitle("Sample Predictions", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Predictions grid saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run evaluation pipeline."""
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model weights not found: %s", model_path)
        sys.exit(1)

    logger.info("Loading model from %s …", model_path)
    model = YOLO(str(model_path))

    # Determine class names from the model
    model_class_names: List[str] = CLASS_NAMES
    if hasattr(model, "names") and model.names:
        model_class_names = (
            list(model.names.values())
            if isinstance(model.names, dict)
            else list(model.names)
        )
        logger.info("Model classes: %s", model_class_names)

    # --- Collect test images ---
    data_dir = Path(args.data)
    if not data_dir.exists():
        logger.error("Test data directory not found: %s", data_dir)
        sys.exit(1)

    samples = collect_test_images(data_dir)
    logger.info("Found %d test images in %s", len(samples), data_dir)
    if not samples:
        logger.error("No images found — aborting.")
        sys.exit(1)

    # --- Run inference ---
    y_true: List[str] = []
    y_pred: List[str] = []
    y_conf: List[float] = []
    all_probs: List[np.ndarray] = []

    # Danger-biased threshold setup
    danger_threshold = getattr(args, 'danger_threshold', 0.0)
    danger_class = "child_only"
    danger_idx = model_class_names.index(danger_class) if danger_class in model_class_names else -1
    if danger_threshold > 0.0:
        logger.info(
            "Danger-biased mode ENABLED: if P(%s) > %.2f, prediction is forced to '%s'",
            danger_class, danger_threshold, danger_class,
        )

    logger.info("Running inference on %d images (imgsz=%d) …", len(samples), args.imgsz)
    for img_path, true_label in samples:
        results = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            verbose=False,
        )
        result = results[0]
        probs = result.probs

        pred_idx = int(probs.top1)
        pred_name = model_class_names[pred_idx] if pred_idx < len(model_class_names) else str(pred_idx)
        confidence = float(probs.top1conf)
        prob_array = probs.data.cpu().numpy()

        # --- Danger-biased override ---
        # If the child_only probability exceeds the threshold, force the prediction
        # to child_only. This maximises recall for the safety-critical class.
        if danger_threshold > 0.0 and danger_idx >= 0:
            child_prob = float(prob_array[danger_idx])
            if child_prob >= danger_threshold:
                pred_name = danger_class
                confidence = child_prob

        y_true.append(true_label)
        y_pred.append(pred_name)
        y_conf.append(confidence)
        all_probs.append(prob_array)

    # --- Compute metrics ---
    metrics = compute_metrics(y_true, y_pred, model_class_names)
    top1_acc = metrics["accuracy"]
    top5_acc = compute_topk_accuracy(y_true, all_probs, model_class_names, k=5)

    # --- Print summary ---
    sep = "=" * 62
    logger.info("\n%s", sep)
    logger.info("  CPD Model Evaluation Results")
    logger.info("%s", sep)
    logger.info("  Model        : %s", model_path.name)
    logger.info("  Test samples : %d", len(samples))
    logger.info("  Top-1 Acc    : %.2f%%", top1_acc * 100)
    logger.info("  Top-5 Acc    : %.2f%%", top5_acc * 100)
    logger.info("%s", sep)
    logger.info("  %-14s  %9s  %9s  %9s  %7s", "Class", "Precision", "Recall", "F1-Score", "Support")
    logger.info("  %s", "-" * 56)
    for cls_name, cls_metrics in metrics["per_class"].items():
        logger.info(
            "  %-14s  %9.4f  %9.4f  %9.4f  %7d",
            cls_name,
            cls_metrics["precision"],
            cls_metrics["recall"],
            cls_metrics["f1"],
            cls_metrics["support"],
        )
    logger.info("%s\n", sep)

    # --- Save confusion matrix ---
    cm_path = output_dir / "confusion_matrix.png"
    save_confusion_matrix(metrics["confusion_matrix"], model_class_names, cm_path)

    # --- Save predictions grid ---
    grid_path = output_dir / "sample_predictions.png"
    save_predictions_grid(samples, y_pred, y_conf, grid_path, grid_size=16)

    # --- Save text report ---
    report_path = output_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("CPD YOLOv8n-cls — Evaluation Report\n")
        fh.write(f"{'=' * 50}\n")
        fh.write(f"Model       : {model_path.name}\n")
        fh.write(f"Test images : {len(samples)}\n")
        fh.write(f"Top-1 Acc   : {top1_acc:.4f}\n")
        fh.write(f"Top-5 Acc   : {top5_acc:.4f}\n\n")
        fh.write(f"{'Class':<15} {'Prec':>9} {'Recall':>9} {'F1':>9} {'Support':>8}\n")
        fh.write(f"{'-' * 52}\n")
        for cls_name, cls_m in metrics["per_class"].items():
            fh.write(
                f"{cls_name:<15} {cls_m['precision']:>9.4f} {cls_m['recall']:>9.4f} "
                f"{cls_m['f1']:>9.4f} {cls_m['support']:>8d}\n"
            )
        fh.write(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}\n")
    logger.info("Text report saved to %s", report_path)
    logger.info("✅ Evaluation complete. Results in %s", output_dir)


if __name__ == "__main__":
    main()
