"""Multi-teacher distillation loss helpers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DEFAULT_TEMPERATURE = 4.0
DEFAULT_ALPHAS = {
    "occlusion": 0.1,
    "frame_quality": 0.1,
    "dominance": 0.1,
}


def _validate_temperature(temperature: float) -> None:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")


def _validate_logits(student_logits: Tensor, teacher_logits: Tensor) -> None:
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student_logits and teacher_logits must have the same shape, "
            f"got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}."
        )


def temperature_scaled_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Tensor:
    """Return temperature-scaled KL distillation loss for raw logits."""
    _validate_temperature(temperature)
    _validate_logits(student_logits, teacher_logits)
    return (
        F.kl_div(
            F.log_softmax(student_logits / temperature, dim=1),
            F.softmax(teacher_logits.detach() / temperature, dim=1),
            reduction="batchmean",
        )
        * (temperature**2)
    )


def distillation_ce_kl_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    alpha: float,
    temperature: float,
    ce_loss_fn: nn.Module | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Combine supervised cross-entropy with teacher KL distillation loss."""
    if not 0 <= alpha <= 1:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}.")
    _validate_temperature(temperature)
    _validate_logits(student_logits, teacher_logits)

    ce_loss = (ce_loss_fn or nn.CrossEntropyLoss())(student_logits, targets)
    kl_loss = temperature_scaled_kl_loss(student_logits, teacher_logits, temperature)
    total_loss = (1 - alpha) * ce_loss + alpha * kl_loss
    components = {
        "ce_loss": float(ce_loss.detach().cpu()),
        "kl_loss": float(kl_loss.detach().cpu()),
        "total_loss": float(total_loss.detach().cpu()),
        "alpha": float(alpha),
        "temperature": float(temperature),
    }
    return total_loss, components


def multi_task_distillation_losses(
    student_outputs: dict[str, Tensor],
    teacher_outputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    alphas: dict[str, float],
    temperature: float,
    ce_loss_fn: nn.Module | None = None,
    reduction: str = "mean",
) -> tuple[Tensor, dict[str, Any]]:
    """Compute CE+KL losses for tasks present in all required dictionaries."""
    if reduction not in {"mean", "sum"}:
        raise ValueError("reduction must be 'mean' or 'sum'.")

    task_losses: list[Tensor] = []
    logs: dict[str, Any] = {"tasks": {}}

    for task in ("occlusion", "frame_quality", "dominance"):
        if task not in student_outputs or task not in teacher_outputs or task not in targets:
            continue
        loss, components = distillation_ce_kl_loss(
            student_logits=student_outputs[task],
            teacher_logits=teacher_outputs[task],
            targets=targets[task],
            alpha=alphas.get(task, DEFAULT_ALPHAS[task]),
            temperature=temperature,
            ce_loss_fn=ce_loss_fn,
        )
        task_losses.append(loss)
        logs["tasks"][task] = components

    if not task_losses:
        raise ValueError("No overlapping tasks found for multi-task distillation loss.")

    stacked_losses = torch.stack(task_losses)
    total_loss = stacked_losses.mean() if reduction == "mean" else stacked_losses.sum()
    logs["total_loss"] = float(total_loss.detach().cpu())
    logs["reduction"] = reduction
    logs["temperature"] = float(temperature)
    return total_loss, logs
