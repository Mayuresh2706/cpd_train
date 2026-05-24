#!/usr/bin/env python3
"""
train.py — Child Presence Detection (CPD) YOLOv8n-cls Training Script

Trains a YOLOv8n classification model on the SVIRO-derived 3-class dataset
(empty / adult_child / child_only) using a two-stage transfer-learning strategy:

  Stage 1: Freeze backbone (first 10 layers), train with lr0=0.01
  Stage 2: Unfreeze all layers, fine-tune with lr0=0.001

Hardware target: NVIDIA RTX 4060 (8 GB VRAM).

Usage:
    python scripts/train.py
    python scripts/train.py --batch 32 --device 0
    python scripts/train.py --single-stage --epochs-stage1 80
    python scripts/train.py --resume runs/classify/train/weights/last.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

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
DEFAULT_DATA = str(PROJECT_ROOT / "data" / "sviro_yolo")
DEFAULT_PROJECT = str(PROJECT_ROOT / "runs" / "classify")
BEST_MODEL_REF_FILE = PROJECT_ROOT / "best_model_path.txt"
PRETRAINED_MODEL = "yolov8n-cls.pt"
FREEZE_LAYERS = 9  # Number of backbone layers to freeze in Stage 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n-cls for Child Presence Detection (CPD)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA,
        help="Path to the YOLO classification dataset root (must contain train/val/test subdirs).",
    )
    parser.add_argument(
        "--epochs-stage1",
        type=int,
        default=50,
        help="Number of epochs for Stage 1 (frozen backbone).",
    )
    parser.add_argument(
        "--epochs-stage2",
        type=int,
        default=30,
        help="Number of epochs for Stage 2 (full fine-tuning).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (16 is conservative for 8 GB VRAM).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (square).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to train on ('0' for GPU, 'cpu' for CPU).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from.",
    )
    parser.add_argument(
        "--single-stage",
        action="store_true",
        help="Skip two-stage strategy; train all layers for (stage1 + stage2) epochs.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _save_best_model_path(model_path: str) -> None:
    """Persist the best model path to a text file for downstream scripts."""
    try:
        BEST_MODEL_REF_FILE.write_text(model_path, encoding="utf-8")
        logger.info("Best model path saved to %s", BEST_MODEL_REF_FILE)
    except OSError as exc:
        logger.warning("Could not write best-model reference file: %s", exc)


def _print_summary(results, stage_name: str, elapsed: float) -> None:
    """Print a human-readable training summary."""
    sep = "=" * 60
    logger.info("\n%s", sep)
    logger.info("  %s  —  Training Summary", stage_name)
    logger.info("%s", sep)
    logger.info("  Elapsed time : %.1f min", elapsed / 60)
    if hasattr(results, "results_dict"):
        for key, value in results.results_dict.items():
            logger.info("  %-25s : %s", key, value)
    logger.info("%s\n", sep)


def train_stage(
    model: YOLO,
    *,
    data: str,
    epochs: int,
    lr0: float,
    freeze: Optional[int],
    batch: int,
    imgsz: int,
    device: str,
    project: str,
    name: str,
    patience: int = 30,
    workers: int = 0,
    resume: bool = False,
) -> object:
    """Run a single training stage and return the results object.

    Args:
        model: YOLO model instance.
        data: Path to dataset root.
        epochs: Maximum training epochs.
        lr0: Initial learning rate.
        freeze: Number of layers to freeze (``None`` to train all).
        batch: Batch size.
        imgsz: Image size.
        device: Training device identifier.
        project: Project directory for saving runs.
        name: Run name inside *project*.
        patience: Early-stopping patience (epochs without improvement).
        workers: Dataloader workers.
        resume: Whether to resume from the last checkpoint.

    Returns:
        Ultralytics results object.
    """
    train_kwargs: dict = dict(
        data=data,
        epochs=epochs,
        lr0=lr0,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        exist_ok=True,
        verbose=True,
        resume=resume,
    )
    if freeze is not None:
        train_kwargs["freeze"] = freeze

    start = time.perf_counter()
    results = model.train(**train_kwargs)
    elapsed = time.perf_counter() - start
    _print_summary(results, name, elapsed)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point for training."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  CPD YOLOv8n-cls Training")
    logger.info("=" * 60)
    logger.info("  Dataset       : %s", args.data)
    logger.info("  Image size    : %d", args.imgsz)
    logger.info("  Batch size    : %d", args.batch)
    logger.info("  Device        : %s", args.device)
    logger.info("  Two-stage     : %s", "No" if args.single_stage else "Yes")
    if not args.single_stage:
        logger.info("  Stage 1 epochs: %d  (freeze %d layers, lr=0.01)", args.epochs_stage1, FREEZE_LAYERS)
        logger.info("  Stage 2 epochs: %d  (all layers,        lr=0.001)", args.epochs_stage2)
    else:
        total = args.epochs_stage1 + args.epochs_stage2
        logger.info("  Total epochs  : %d  (single-stage, lr=0.01)", total)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Resume shortcut
    # ------------------------------------------------------------------
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        model = YOLO(args.resume)
        results = model.train(resume=True)
        best_path = str(Path(model.trainer.best) if hasattr(model, "trainer") else args.resume)
        _save_best_model_path(best_path)
        return

    # ------------------------------------------------------------------
    # Single-stage training
    # ------------------------------------------------------------------
    if args.single_stage:
        total_epochs = args.epochs_stage1 + args.epochs_stage2
        logger.info("🚀 Single-stage training for %d epochs …", total_epochs)
        model = YOLO(PRETRAINED_MODEL)
        results = train_stage(
            model,
            data=args.data,
            epochs=total_epochs,
            lr0=0.01,
            freeze=None,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=DEFAULT_PROJECT,
            name="train",
        )
        best_path = str(Path(DEFAULT_PROJECT) / "train" / "weights" / "best.pt")
        _save_best_model_path(best_path)
        logger.info("✅ Training complete. Best weights → %s", best_path)
        return

    # ------------------------------------------------------------------
    # Two-stage training
    # ------------------------------------------------------------------

    # Stage 1 — frozen backbone
    logger.info("🧊 Stage 1 — Frozen backbone (first %d layers) …", FREEZE_LAYERS)
    model = YOLO(PRETRAINED_MODEL)
    # train_stage(
    #     model,
    #     data=args.data,
    #     epochs=args.epochs_stage1,
    #     lr0=0.01,
    #     freeze=FREEZE_LAYERS,
    #     batch=args.batch,
    #     imgsz=args.imgsz,
    #     device=args.device,
    #     project=DEFAULT_PROJECT,
    #     name="stage1",
    # )

    stage1_best = Path(DEFAULT_PROJECT) / "stage1" / "weights" / "best.pt"
    if not stage1_best.exists():
        logger.error("Stage 1 best weights not found at %s — aborting.", stage1_best)
        sys.exit(1)

    # Stage 2 — full fine-tuning
    logger.info("🔥 Stage 2 — Full fine-tuning from Stage 1 best weights …")
    model = YOLO(str(stage1_best))
    train_stage(
        model,
        data=args.data,
        epochs=args.epochs_stage2,
        lr0=0.001,
        freeze=None,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=DEFAULT_PROJECT,
        name="stage2",
    )

    stage2_best = Path(DEFAULT_PROJECT) / "stage2" / "weights" / "best.pt"
    if not stage2_best.exists():
        logger.warning(
            "Stage 2 best weights not found — falling back to Stage 1 best."
        )
        stage2_best = stage1_best

    # Copy final best to a canonical location
    final_best = Path(DEFAULT_PROJECT) / "train" / "weights"
    final_best.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(str(stage2_best), str(final_best / "best.pt"))
    logger.info("📦 Final best weights copied to %s", final_best / "best.pt")
    _save_best_model_path(str(final_best / "best.pt"))

    logger.info("=" * 60)
    logger.info("  ✅  All training stages complete!")
    logger.info("  Best model : %s", final_best / "best.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
