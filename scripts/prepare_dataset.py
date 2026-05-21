#!/usr/bin/env python3
"""Prepare the SVIRO dataset for YOLOv8 3-class classification training.

This script converts the SVIRO dataset (full-scene rear-seat images) into a
YOLOv8-compatible ImageFolder structure with three scene-level classes:

    - **empty**       : No child present in any seat.
    - **adult_child** : At least one child AND at least one adult.
    - **child_only**  : At least one child AND NO adult (danger scenario).

SVIRO filename convention
-------------------------
``[car_name]_[split]_imageID_[id]_GT_[left]_[middle]_[right].png``

Car names may contain underscores (e.g. ``BMW_X5``, ``Tesla_Model_3``), so
the split token (``_train_`` / ``_test_``) is located by substring search.

Per-seat labels
~~~~~~~~~~~~~~~
0 = empty seat, 1 = infant in infant seat, 2 = child on child seat,
3 = adult, 4 = everyday object, 5 = empty infant seat, 6 = empty child seat.

Usage
-----
::

    python prepare_dataset.py -i data/sviro_raw -o data/sviro_yolo

Output
------
::

    <output>/
    ├── train/{empty,adult_child,child_only}/
    ├── val/{empty,adult_child,child_only}/
    ├── test/{empty,adult_child,child_only}/
    └── class_mapping.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SVIRO_SEAT_LABELS: Dict[int, str] = {
    0: "empty_seat",
    1: "infant_in_infant_seat",
    2: "child_on_child_seat",
    3: "adult",
    4: "everyday_object",
    5: "empty_infant_seat",
    6: "empty_child_seat",
}

CHILD_LABELS = {1, 2}
ADULT_LABEL = 3

TARGET_CLASSES = ["empty", "adult_child", "child_only"]

CLASS_MAPPING: Dict[str, object] = {
    "classes": TARGET_CLASSES,
    "class_indices": {name: idx for idx, name in enumerate(TARGET_CLASSES)},
    "description": {
        "empty": "No child present in any seat (labels 0, 3, 4, 5, 6 only).",
        "adult_child": "At least one child (label 1 or 2) AND at least one adult (label 3).",
        "child_only": "At least one child (label 1 or 2) AND NO adult — danger scenario.",
    },
    "sviro_seat_labels": SVIRO_SEAT_LABELS,
}

# Regex that matches the tail of a valid full-scene filename:
#   _imageID_<digits>_GT_<int>_<int>_<int>.png
_TAIL_PATTERN = re.compile(
    r"_imageID_(\d+)_GT_(\d+)_(\d+)_(\d+)\.png$", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("prepare_dataset")


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def parse_sviro_filename(filename: str) -> Optional[Dict[str, object]]:
    """Parse a SVIRO full-scene filename and return metadata.

    Parameters
    ----------
    filename : str
        The basename of the image file, e.g.
        ``BMW_X5_train_imageID_00001_GT_0_3_2.png``.

    Returns
    -------
    dict or None
        Dictionary with keys ``car_name``, ``split``, ``image_id``,
        ``left``, ``middle``, ``right``; or *None* if the filename does
        not match the expected pattern.
    """
    # Skip individual seat crops (they contain "seatPosition" in name).
    if "seatPosition" in filename:
        return None

    # Find the split token in the filename.
    for split_token in ("_train_", "_test_"):
        idx = filename.find(split_token)
        if idx != -1:
            car_name = filename[:idx]
            remainder = filename[idx + 1:]  # e.g. "train_imageID_00001_GT_0_3_2.png"
            split_name = split_token.strip("_")  # "train" or "test"
            break
    else:
        return None  # no split token found

    # Match the tail: imageID_<id>_GT_<l>_<m>_<r>.png
    tail_match = _TAIL_PATTERN.search(filename)
    if tail_match is None:
        return None

    image_id = int(tail_match.group(1))
    left = int(tail_match.group(2))
    middle = int(tail_match.group(3))
    right = int(tail_match.group(4))

    # Validate label values.
    for label in (left, middle, right):
        if label not in SVIRO_SEAT_LABELS:
            log.warning(
                "Unrecognised seat label %d in file '%s' — skipping.", label, filename
            )
            return None

    return {
        "car_name": car_name,
        "split": split_name,
        "image_id": image_id,
        "left": left,
        "middle": middle,
        "right": right,
    }


def determine_scene_class(left: int, middle: int, right: int) -> str:
    """Map three per-seat labels to a scene-level class.

    Parameters
    ----------
    left, middle, right : int
        SVIRO per-seat ground-truth labels.

    Returns
    -------
    str
        One of ``"empty"``, ``"adult_child"``, or ``"child_only"``.
    """
    seats = {left, middle, right}
    has_child = bool(seats & CHILD_LABELS)
    has_adult = ADULT_LABEL in seats

    if not has_child:
        return "empty"
    if has_adult:
        return "adult_child"
    return "child_only"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_images(
    root: Path, image_type: str = "rgb"
) -> List[Tuple[Path, Dict[str, object]]]:
    """Recursively find full-scene SVIRO images under *root*.

    Only images inside directories whose name contains
    ``<image_type>_wholeImage`` (e.g. ``rgb_wholeImage``,
    ``grayscale_wholeImage``) are collected.

    Parameters
    ----------
    root : Path
        Root directory of the raw SVIRO dataset.
    image_type : str
        ``"rgb"`` or ``"grayscale"``.

    Returns
    -------
    list of (Path, dict)
        Each element is a tuple of the image path and its parsed metadata.
    """
    target_dir_fragment = f"{image_type}_wholeImage".lower()
    results: List[Tuple[Path, Dict[str, object]]] = []
    skipped = 0

    log.info("Scanning '%s' for %s whole-image PNGs …", root, image_type)

    for png_path in sorted(root.rglob("*.png")):
        # Check that the image lives inside a *_wholeImage directory.
        if target_dir_fragment not in str(png_path.parent).lower():
            continue

        meta = parse_sviro_filename(png_path.name)
        if meta is None:
            skipped += 1
            continue

        results.append((png_path, meta))

    log.info(
        "Found %d valid images (%d skipped / unrecognised).", len(results), skipped
    )
    return results


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def split_train_val(
    images: List[Tuple[Path, Dict[str, object]]],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[
    List[Tuple[Path, Dict[str, object]]],
    List[Tuple[Path, Dict[str, object]]],
    List[Tuple[Path, Dict[str, object]]],
]:
    """Separate images into train / val / test splits.

    SVIRO provides a predefined train/test split encoded in the filename.
    The training portion is further split into train and val subsets.

    Parameters
    ----------
    images : list
        Output of :func:`discover_images`.
    val_fraction : float
        Fraction of training images to hold out for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (train, val, test) — each a list of ``(Path, metadata)`` tuples.
    """
    train_pool: List[Tuple[Path, Dict[str, object]]] = []
    test_set: List[Tuple[Path, Dict[str, object]]] = []

    for item in images:
        if item[1]["split"] == "train":
            train_pool.append(item)
        else:
            test_set.append(item)

    # Reproducible shuffle.
    rng = random.Random(seed)
    rng.shuffle(train_pool)

    val_count = int(len(train_pool) * val_fraction)
    val_set = train_pool[:val_count]
    train_set = train_pool[val_count:]

    log.info(
        "Split sizes — train: %d, val: %d, test: %d",
        len(train_set),
        len(val_set),
        len(test_set),
    )
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Output creation
# ---------------------------------------------------------------------------


def copy_images(
    items: List[Tuple[Path, Dict[str, object]]],
    output_dir: Path,
    split_name: str,
    use_symlink: bool = False,
) -> Counter:
    """Copy (or symlink) images into the ImageFolder structure.

    Parameters
    ----------
    items : list
        ``(path, metadata)`` tuples for one split.
    output_dir : Path
        Root output directory (e.g. ``data/sviro_yolo``).
    split_name : str
        ``"train"``, ``"val"``, or ``"test"``.
    use_symlink : bool
        If *True*, create symbolic links instead of copying files.

    Returns
    -------
    Counter
        Class counts for this split.
    """
    class_counts: Counter = Counter()

    for src_path, meta in items:
        scene_class = determine_scene_class(meta["left"], meta["middle"], meta["right"])
        class_counts[scene_class] += 1

        dest_dir = output_dir / split_name / scene_class
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / src_path.name

        if dest_path.exists():
            # Avoid overwriting — append car name to disambiguate.
            stem = f"{meta['car_name']}_{src_path.stem}"
            dest_path = dest_dir / f"{stem}{src_path.suffix}"

        if use_symlink:
            try:
                dest_path.symlink_to(src_path.resolve())
            except OSError as exc:
                log.warning(
                    "Symlink failed for '%s' (%s). Falling back to copy.",
                    src_path.name,
                    exc,
                )
                shutil.copy2(src_path, dest_path)
        else:
            shutil.copy2(src_path, dest_path)

    return class_counts


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(
    stats: Dict[str, Counter], total_images: int
) -> None:
    """Print a summary table of class distribution per split.

    Parameters
    ----------
    stats : dict
        Mapping of split name → Counter of class counts.
    total_images : int
        Overall image count before splitting.
    """
    line = "-" * 62
    print("\n" + line)
    print(f"{'DATASET SUMMARY':^62}")
    print(line)
    print(f"  Total source images found : {total_images}")
    print(line)
    header = f"  {'Split':<10}"
    for cls in TARGET_CLASSES:
        header += f"  {cls:>12}"
    header += f"  {'Total':>8}"
    print(header)
    print(line)

    grand_total: Counter = Counter()
    for split_name in ("train", "val", "test"):
        counts = stats.get(split_name, Counter())
        grand_total += counts
        row = f"  {split_name:<10}"
        split_total = sum(counts.values())
        for cls in TARGET_CLASSES:
            c = counts.get(cls, 0)
            pct = (c / split_total * 100) if split_total else 0.0
            row += f"  {c:>6} ({pct:4.1f}%)"
        row += f"  {split_total:>8}"
        print(row)

    print(line)
    overall = sum(grand_total.values())
    row = f"  {'TOTAL':<10}"
    for cls in TARGET_CLASSES:
        c = grand_total.get(cls, 0)
        pct = (c / overall * 100) if overall else 0.0
        row += f"  {c:>6} ({pct:4.1f}%)"
    row += f"  {overall:>8}"
    print(row)
    print(line)

    # Imbalance warning.
    if overall > 0:
        fracs = [grand_total.get(cls, 0) / overall for cls in TARGET_CLASSES]
        max_frac = max(fracs)
        min_frac = min(fracs)
        if min_frac > 0 and max_frac / min_frac > 5:
            log.warning(
                "Severe class imbalance detected (max/min ratio = %.1f). "
                "Consider using class weights, oversampling, or data augmentation.",
                max_frac / min_frac,
            )
        elif min_frac == 0:
            missing = [
                cls for cls in TARGET_CLASSES if grand_total.get(cls, 0) == 0
            ]
            log.warning(
                "The following classes have ZERO samples: %s. "
                "Check your dataset or class mapping.",
                ", ".join(missing),
            )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert SVIRO dataset to YOLOv8 3-class classification format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python prepare_dataset.py -i data/sviro_raw -o data/sviro_yolo\n"
            "  python prepare_dataset.py -i D:/datasets/SVIRO --symlink --image-type grayscale\n"
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data/sviro_raw"),
        help="Path to the raw SVIRO dataset root directory (default: data/sviro_raw).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/sviro_yolo"),
        help="Path for the output ImageFolder dataset (default: data/sviro_yolo).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of training data to use for validation (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val splitting (default: 42).",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links instead of copying files (saves disk space).",
    )
    parser.add_argument(
        "--image-type",
        choices=["rgb", "grayscale"],
        default="rgb",
        help="Which image modality to use: 'rgb' or 'grayscale' (default: rgb).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for dataset preparation.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments.  Defaults to ``sys.argv[1:]``.
    """
    parser = build_argparser()
    args = parser.parse_args(argv)

    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    val_split: float = args.val_split
    seed: int = args.seed
    use_symlink: bool = args.symlink
    image_type: str = args.image_type

    # --- Validate inputs ---------------------------------------------------
    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    if not 0.0 < val_split < 1.0:
        log.error("--val-split must be between 0 and 1 (exclusive). Got: %s", val_split)
        sys.exit(1)

    log.info("Input  : %s", input_dir)
    log.info("Output : %s", output_dir)
    log.info("Image type  : %s", image_type)
    log.info("Val split   : %.0f%%", val_split * 100)
    log.info("Seed        : %d", seed)
    log.info("Use symlink : %s", use_symlink)

    # --- Discover images ---------------------------------------------------
    images = discover_images(input_dir, image_type=image_type)

    if not images:
        log.error(
            "No valid images found under '%s'. "
            "Check that the directory contains SVIRO data with "
            "'%s_wholeImage' sub-folders.",
            input_dir,
            image_type,
        )
        sys.exit(1)

    total_images = len(images)

    # --- Split -------------------------------------------------------------
    train_set, val_set, test_set = split_train_val(
        images, val_fraction=val_split, seed=seed
    )

    # --- Create output directories -----------------------------------------
    for split_name in ("train", "val", "test"):
        for cls in TARGET_CLASSES:
            (output_dir / split_name / cls).mkdir(parents=True, exist_ok=True)

    # --- Copy / symlink images ---------------------------------------------
    stats: Dict[str, Counter] = {}

    log.info("Writing train split …")
    stats["train"] = copy_images(train_set, output_dir, "train", use_symlink)

    log.info("Writing val split …")
    stats["val"] = copy_images(val_set, output_dir, "val", use_symlink)

    log.info("Writing test split …")
    stats["test"] = copy_images(test_set, output_dir, "test", use_symlink)

    # --- Write class_mapping.json ------------------------------------------
    mapping_path = output_dir / "class_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(CLASS_MAPPING, fh, indent=2, default=str)
    log.info("Class mapping written to %s", mapping_path)

    # --- Summary -----------------------------------------------------------
    print_summary(stats, total_images)

    log.info("Done. Dataset ready at: %s", output_dir)


if __name__ == "__main__":
    main()
