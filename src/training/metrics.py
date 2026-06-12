"""Lightweight metric helpers for classification training."""

from __future__ import annotations

import torch
from torch import Tensor


class AverageMeter:
    """Track a weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def accuracy_from_logits(logits: Tensor, targets: Tensor) -> float:
    """Return mean classification accuracy for raw logits."""
    if targets.numel() == 0:
        return 0.0
    predictions = torch.argmax(logits, dim=1)
    return (predictions == targets).float().mean().item()
