#!/usr/bin/env python3
"""Download and extract the SVIRO dataset for Child Presence Detection training.

The SVIRO (Synthetic Vehicle Interior Rear-seat Occupancy) dataset contains
synthetic images of vehicle interiors with various occupant configurations
across 10 different car models.

Official website: https://sviro.kl.dfki.de/
Reference repo:   https://github.com/wesleylp/sviro-det

Usage examples:
    # Try auto-download all car models
    python scripts/download_sviro.py

    # Download a specific car model
    python scripts/download_sviro.py --car BMW_X5

    # Download multiple specific car models
    python scripts/download_sviro.py --car BMW_X5 Tesla_Model_3

    # Use an existing local copy of the dataset
    python scripts/download_sviro.py --path /existing/sviro/dir

    # Specify a custom output directory
    python scripts/download_sviro.py --output /custom/output/dir
"""

import argparse
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path(
    r"C:\Users\user\.gemini\antigravity\scratch\cpd_yolo\data\sviro_raw"
)

SVIRO_WEBSITE = "https://sviro.kl.dfki.de/"
SVIRO_DOWNLOAD_PAGE = "https://sviro.kl.dfki.de/download/"

CAR_MODELS: List[str] = [
    "BMW_X5",
    "Tesla_Model_3",
    "Hyundai_Tucson",
    "Lexus_GS_F",
    "Toyota_Hilux",
    "BMW_i3",
    "Mercedes_A-Class",
    "Renault_Zoe",
    "VW_Tiguan",
    "Ford_Escape",
]

# Known / candidate URL patterns that have historically been used by the
# DFKI server.  The script will try each pattern in order until one succeeds.
# {car} is replaced with the car-model name at runtime.
_URL_PATTERNS: List[str] = [
    # Pattern used by the wesleylp/sviro-det get_data.sh script
    "https://sviro.kl.dfki.de/data/{car}.zip",
    "https://sviro.kl.dfki.de/data/{car}.tar.gz",
    "https://sviro.kl.dfki.de/downloads/{car}.zip",
    "https://sviro.kl.dfki.de/downloads/{car}.tar.gz",
    # Alternate casing / slug variants
    "https://sviro.kl.dfki.de/data/{car_lower}.zip",
    "https://sviro.kl.dfki.de/data/{car_lower}.tar.gz",
]

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB per read

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
# Helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 hex-digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _open_browser(url: str) -> None:
    """Try to open *url* in the user's default web browser."""
    import webbrowser

    try:
        webbrowser.open(url)
        logger.info("Opened browser to %s", url)
    except Exception:
        logger.warning("Could not open browser automatically.")


def _resolve_download_url(car: str) -> Optional[str]:
    """Probe candidate URLs for *car* and return the first one that responds
    with HTTP 200 (or at least a non-404 status with a Content-Length header).
    Returns ``None`` if every candidate fails.
    """
    for pattern in _URL_PATTERNS:
        url = pattern.format(car=car, car_lower=car.lower())
        try:
            resp = requests.head(url, allow_redirects=True, timeout=15)
            if resp.status_code == 200:
                logger.info("Resolved download URL: %s", url)
                return url
        except requests.RequestException:
            continue
    return None


def _download_file(url: str, dest: Path, resume: bool = True) -> Path:
    """Download *url* to *dest* with a tqdm progress bar.

    If *resume* is ``True`` and *dest* already exists, the download resumes
    from where it left off (if the server supports Range requests).

    Returns the path to the downloaded file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    headers: dict[str, str] = {}
    mode = "wb"
    initial_size = 0

    if resume and dest.exists():
        initial_size = dest.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        mode = "ab"
        logger.info("Resuming download from byte %d", initial_size)

    resp = requests.get(url, headers=headers, stream=True, timeout=30)

    # If the server doesn't support Range, start from scratch.
    if resp.status_code == 200 and initial_size > 0:
        mode = "wb"
        initial_size = 0
    elif resp.status_code not in (200, 206):
        resp.raise_for_status()

    total_size = int(resp.headers.get("Content-Length", 0)) + initial_size

    with open(dest, mode) as fh, tqdm(
        total=total_size,
        initial=initial_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=dest.name,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                fh.write(chunk)
                pbar.update(len(chunk))

    logger.info("Downloaded %s (%s bytes)", dest.name, dest.stat().st_size)
    return dest


def _extract_archive(archive_path: Path, output_dir: Path) -> Path:
    """Extract a ZIP or tar.gz archive into *output_dir*.

    Returns the directory the contents were extracted into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        logger.info("Extracting ZIP: %s", archive_path.name)
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                zf.extract(member, output_dir)
    elif tarfile.is_tarfile(archive_path):
        logger.info("Extracting TAR: %s", archive_path.name)
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc=f"Extracting {archive_path.name}"):
                tf.extract(member, output_dir, filter="data")
    else:
        logger.error(
            "Unknown archive format for %s – skipping extraction.", archive_path.name
        )
        return output_dir

    logger.info("Extracted to %s", output_dir)
    return output_dir


def _print_manual_instructions(cars: List[str], output_dir: Path) -> None:
    """Print clear instructions for manual download when auto-download fails."""
    border = "=" * 72
    print(f"\n{border}")
    print("  MANUAL DOWNLOAD REQUIRED")
    print(border)
    print()
    print("  The automatic download could not locate the files on the DFKI")
    print("  server.  Please download the SVIRO dataset manually:")
    print()
    print(f"  1. Visit the SVIRO website : {SVIRO_WEBSITE}")
    print(f"  2. Go to the download page : {SVIRO_DOWNLOAD_PAGE}")
    print("  3. Follow the registration / request process if required.")
    print("  4. Download the following car model archives:")
    for car in cars:
        print(f"       - {car}")
    print()
    print("  5. Place the downloaded ZIP/tar.gz files into:")
    print(f"       {output_dir}")
    print()
    print("  6. Re-run this script to extract them:")
    print(f"       python {Path(__file__).name} --path {output_dir}")
    print()
    print(border)
    print()


# ---------------------------------------------------------------------------
# Per-car download pipeline
# ---------------------------------------------------------------------------


def download_car(car: str, output_dir: Path) -> bool:
    """Attempt to download and extract a single car model's data.

    Returns ``True`` if the data is ready (downloaded or already present),
    ``False`` if auto-download failed.
    """
    car_dir = output_dir / car
    if car_dir.exists() and any(car_dir.iterdir()):
        logger.info("Car model '%s' already present at %s – skipping.", car, car_dir)
        return True

    logger.info("Resolving download URL for '%s' …", car)
    url = _resolve_download_url(car)

    if url is None:
        logger.warning(
            "Could not find a download URL for '%s'. Manual download needed.", car
        )
        return False

    # Determine archive filename from URL
    archive_name = url.rsplit("/", 1)[-1]
    archive_path = output_dir / archive_name

    # Skip download if archive already exists and has non-zero size.
    if archive_path.exists() and archive_path.stat().st_size > 0:
        logger.info("Archive '%s' already on disk – skipping download.", archive_name)
    else:
        try:
            _download_file(url, archive_path, resume=True)
        except requests.RequestException as exc:
            logger.error("Download failed for '%s': %s", car, exc)
            return False

    # Integrity check – basic: file exists and is larger than 1 KiB
    if archive_path.stat().st_size < 1024:
        logger.error(
            "Downloaded file '%s' appears too small – possibly corrupt.", archive_name
        )
        return False

    logger.info("SHA-256: %s  %s", _sha256_file(archive_path), archive_name)

    # Extract
    try:
        _extract_archive(archive_path, output_dir)
    except Exception as exc:
        logger.error("Extraction failed for '%s': %s", archive_name, exc)
        return False

    return True


# ---------------------------------------------------------------------------
# Link / copy existing dataset
# ---------------------------------------------------------------------------


def link_existing(source: Path, output_dir: Path) -> None:
    """Copy or symlink an existing SVIRO download into *output_dir*."""
    source = source.resolve()
    if not source.exists():
        logger.error("Provided path does not exist: %s", source)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # If source IS the output dir, nothing to do.
    if source == output_dir.resolve():
        logger.info("Source and output directories are the same – nothing to copy.")
        return

    # Determine what's inside the source directory.
    found_cars: List[str] = []
    for item in source.iterdir():
        if item.is_dir() and item.name in CAR_MODELS:
            found_cars.append(item.name)

    if found_cars:
        logger.info(
            "Found %d car model(s) in source: %s", len(found_cars), ", ".join(found_cars)
        )
    else:
        # Maybe the source IS a single car-model directory.
        if source.name in CAR_MODELS:
            found_cars = [source.name]
            source = source.parent
            logger.info("Source appears to be a single car model directory: %s", source.name)
        else:
            logger.warning(
                "No recognised car-model directories found in %s. "
                "Will copy contents as-is.",
                source,
            )

    # Attempt symlink first (fast, no disk usage). Fall back to copy.
    for car in found_cars or []:
        src = source / car
        dst = output_dir / car
        if dst.exists():
            logger.info("'%s' already exists in output – skipping.", car)
            continue
        try:
            dst.symlink_to(src)
            logger.info("Symlinked %s -> %s", dst, src)
        except OSError:
            logger.info("Symlink not available – copying %s (may take a while) …", car)
            shutil.copytree(src, dst)
            logger.info("Copied %s", car)

    # Also look for archives that haven't been extracted yet.
    for item in source.iterdir():
        if item.is_file() and (
            item.suffix == ".zip" or item.name.endswith(".tar.gz")
        ):
            dest_archive = output_dir / item.name
            if not dest_archive.exists():
                shutil.copy2(item, dest_archive)
                logger.info("Copied archive %s", item.name)
            logger.info("Extracting %s …", item.name)
            _extract_archive(dest_archive, output_dir)

    logger.info("Existing dataset linked/copied to %s", output_dir)


# ---------------------------------------------------------------------------
# Scan & report
# ---------------------------------------------------------------------------


def scan_dataset(output_dir: Path) -> None:
    """Print a summary of which car models are available locally."""
    print("\n--- SVIRO Dataset Summary ---")
    print(f"Location: {output_dir}\n")
    total_images = 0
    for car in CAR_MODELS:
        car_dir = output_dir / car
        if car_dir.exists():
            # Count image files recursively
            n_images = sum(
                1
                for f in car_dir.rglob("*")
                if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
            )
            total_images += n_images
            status = f"✓  {n_images:>6,} images"
        else:
            status = "✗  not found"
        print(f"  {car:<22s} {status}")
    print(f"\n  Total images: {total_images:,}")
    print("-----------------------------\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and prepare the SVIRO dataset for CPD training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--car",
        nargs="+",
        choices=CAR_MODELS,
        default=None,
        metavar="CAR",
        help=(
            "Download only the specified car model(s). "
            f"Choices: {', '.join(CAR_MODELS)}"
        ),
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to an existing local SVIRO download to use instead of downloading.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for downloaded data (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open the SVIRO download page in the browser if auto-download fails.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the SVIRO dataset download script."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir: Path = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # --path: link / copy an existing dataset ----------------------------------
    if args.path is not None:
        link_existing(args.path, output_dir)
        scan_dataset(output_dir)
        return

    # Determine which cars to download ----------------------------------------
    cars_to_download: List[str] = args.car if args.car else list(CAR_MODELS)
    logger.info("Car models requested: %s", ", ".join(cars_to_download))

    # Download each car --------------------------------------------------------
    succeeded: List[str] = []
    failed: List[str] = []

    for car in cars_to_download:
        logger.info("=" * 60)
        logger.info("Processing car model: %s", car)
        logger.info("=" * 60)
        ok = download_car(car, output_dir)
        if ok:
            succeeded.append(car)
        else:
            failed.append(car)

    # Report -------------------------------------------------------------------
    scan_dataset(output_dir)

    if failed:
        _print_manual_instructions(failed, output_dir)
        if args.browser:
            _open_browser(SVIRO_DOWNLOAD_PAGE)

    if succeeded:
        logger.info(
            "Successfully obtained %d/%d car model(s).",
            len(succeeded),
            len(cars_to_download),
        )
    if failed:
        logger.warning(
            "%d car model(s) could not be auto-downloaded: %s",
            len(failed),
            ", ".join(failed),
        )
        sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
