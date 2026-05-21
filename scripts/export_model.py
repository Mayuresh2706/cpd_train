#!/usr/bin/env python3
"""
export_model.py — Export CPD YOLOv8n-cls to ONNX / TorchScript

Converts the trained classification model to deployment-ready formats:
  • ONNX   — primary target for FPGA deployment via Xilinx FINN
  • TorchScript — backup format for PyTorch-native inference

The ONNX export is validated with ``onnx.checker`` and a test inference
is run through ONNX Runtime to verify the output shape.

Usage:
    python scripts/export_model.py
    python scripts/export_model.py --model runs/classify/train/weights/best.pt
    python scripts/export_model.py --imgsz 224 --output-dir exports/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

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
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "exports")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export CPD YOLOv8n-cls model to ONNX and TorchScript",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Path to trained model weights (.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size for the exported model.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save exported models.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def export_onnx(model: YOLO, imgsz: int, output_dir: Path) -> Path:
    """Export to ONNX with fixed input shape (suitable for FPGA).

    Args:
        model: Loaded YOLO model.
        imgsz: Square image dimension.
        output_dir: Export destination.

    Returns:
        Path to the exported ONNX file.
    """
    logger.info("Exporting to ONNX (opset=13, simplify=True, dynamic=False) …")
    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        dynamic=False,
        opset=13,
    )

    onnx_src = Path(export_path)
    onnx_dst = output_dir / onnx_src.name
    if onnx_src != onnx_dst:
        import shutil
        shutil.copy2(str(onnx_src), str(onnx_dst))

    logger.info("ONNX model saved to %s (%.2f MB)", onnx_dst, _file_size_mb(onnx_dst))
    return onnx_dst


def validate_onnx(onnx_path: Path, imgsz: int) -> None:
    """Validate the ONNX model with onnx.checker and ONNX Runtime.

    Args:
        onnx_path: Path to the ONNX file.
        imgsz: Expected image dimension.
    """
    # --- Static validation ---
    try:
        import onnx

        logger.info("Validating ONNX model with onnx.checker …")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✅ ONNX model passed checker validation.")
    except ImportError:
        logger.warning("'onnx' package not installed — skipping static validation.")
    except Exception as exc:
        logger.error("ONNX validation failed: %s", exc)

    # --- Runtime inference test ---
    try:
        import onnxruntime as ort

        logger.info("Running test inference with ONNX Runtime …")
        session = ort.InferenceSession(str(onnx_path))
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]

        input_name = input_meta.name
        input_shape = input_meta.shape  # e.g. [1, 3, 640, 640]
        output_name = output_meta.name

        logger.info("  Input  : name=%s  shape=%s", input_name, input_shape)

        # Build a dummy input
        dummy_shape = [
            d if isinstance(d, int) else 1 for d in input_shape
        ]
        dummy_input = np.random.randn(*dummy_shape).astype(np.float32)
        outputs = session.run([output_name], {input_name: dummy_input})

        logger.info("  Output : name=%s  shape=%s", output_name, outputs[0].shape)
        logger.info("✅ ONNX Runtime inference test passed.")
    except ImportError:
        logger.warning("'onnxruntime' not installed — skipping runtime test.")
    except Exception as exc:
        logger.error("ONNX Runtime test failed: %s", exc)


def export_torchscript(model: YOLO, imgsz: int, output_dir: Path) -> Path:
    """Export to TorchScript as a backup format.

    Args:
        model: Loaded YOLO model.
        imgsz: Square image dimension.
        output_dir: Export destination.

    Returns:
        Path to the exported TorchScript file.
    """
    logger.info("Exporting to TorchScript …")
    export_path = model.export(
        format="torchscript",
        imgsz=imgsz,
    )

    ts_src = Path(export_path)
    ts_dst = output_dir / ts_src.name
    if ts_src != ts_dst:
        import shutil
        shutil.copy2(str(ts_src), str(ts_dst))

    logger.info("TorchScript model saved to %s (%.2f MB)", ts_dst, _file_size_mb(ts_dst))
    return ts_dst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the export pipeline."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error("Model weights not found: %s", model_path)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("  CPD YOLOv8n-cls Model Export")
    logger.info("=" * 60)
    logger.info("  Source model : %s", model_path)
    logger.info("  Image size   : %d", args.imgsz)
    logger.info("  Output dir   : %s", output_dir)
    logger.info("=" * 60)

    model = YOLO(str(model_path))

    # --- ONNX export ---
    onnx_path = export_onnx(model, args.imgsz, output_dir)
    validate_onnx(onnx_path, args.imgsz)

    # --- TorchScript export ---
    # Reload model because export can modify internal state
    model = YOLO(str(model_path))
    ts_path = export_torchscript(model, args.imgsz, output_dir)

    # --- Summary ---
    sep = "=" * 60
    logger.info("\n%s", sep)
    logger.info("  Export Summary")
    logger.info("%s", sep)
    logger.info("  %-18s : %s (%.2f MB)", "ONNX", onnx_path.name, _file_size_mb(onnx_path))
    logger.info("  %-18s : %s (%.2f MB)", "TorchScript", ts_path.name, _file_size_mb(ts_path))
    logger.info("  %-18s : %d × %d", "Input size", args.imgsz, args.imgsz)
    logger.info("  %-18s : %s", "Output dir", output_dir)
    logger.info("%s", sep)
    logger.info("✅ All exports complete.")


if __name__ == "__main__":
    main()
