"""Path builders for split CSVs and coronary image files."""

from __future__ import annotations

from pathlib import Path


def normalize_artery(artery: str) -> str:
    """Normalize artery values to the CSV convention: LCA or RCA."""
    normalized = artery.upper()
    if normalized.startswith("DATA_"):
        normalized = normalized.removeprefix("DATA_")
    return normalized


def artery_data_dir(artery: str) -> str:
    """Return the DATA_<artery> directory name for an artery value."""
    return f"DATA_{normalize_artery(artery)}"


def build_image_path(
    data_root: str | Path,
    task: str,
    artery: str,
    label: str,
    filename: str,
) -> Path:
    """Build an image path from CSV row fields."""
    return Path(data_root) / task / artery_data_dir(artery) / label / filename


def build_split_csv_path(
    split_root: str | Path,
    task: str,
    artery: str,
    csv_filename: str,
) -> Path:
    """Build a split CSV path from split root, task, artery, and filename."""
    return Path(split_root) / task / artery_data_dir(artery) / "labels" / csv_filename
