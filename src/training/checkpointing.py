"""Checkpoint and history helpers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(checkpoint: dict[str, Any], path: str | Path) -> Path:
    """Save a checkpoint dictionary, creating the parent directory if needed."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dictionary."""
    return torch.load(Path(path), map_location=map_location)


def save_history_csv(history: list[dict[str, Any]], path: str | Path) -> Path:
    """Write epoch history rows to CSV."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"]

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    return csv_path
