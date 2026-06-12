"""Teacher checkpoint loading and forward helpers for MTD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.models.teachers import SingleFrameTeacher, VideoTeacher
from src.utils.paths import artery_data_dir, normalize_artery


FOLDER_TO_MODEL_TASK = {
    "occlusion": "occlusion",
    "framequality": "frame_quality",
    "dominance": "dominance",
}
MODEL_TO_FOLDER_TASK = {value: key for key, value in FOLDER_TO_MODEL_TASK.items()}
TASK_INPUT_KEYS = {
    "occlusion": "occlusion_images",
    "frame_quality": "frame_quality_images",
    "dominance": "dominance_images",
}


def folder_task_name(task: str) -> str:
    """Map internal model/loss task names to checkpoint folder task names."""
    normalized = task.lower()
    if normalized in MODEL_TO_FOLDER_TASK:
        return MODEL_TO_FOLDER_TASK[normalized]
    if normalized in FOLDER_TO_MODEL_TASK:
        return normalized
    raise ValueError(f"Unsupported task '{task}'.")


def model_task_name(task: str) -> str:
    """Map checkpoint folder task names to internal model/loss task names."""
    normalized = task.lower()
    if normalized in FOLDER_TO_MODEL_TASK:
        return FOLDER_TO_MODEL_TASK[normalized]
    if normalized in MODEL_TO_FOLDER_TASK:
        return normalized
    raise ValueError(f"Unsupported task '{task}'.")


def resolve_teacher_checkpoint_path(
    teacher_root: str | Path,
    task: str,
    artery: str,
    fold: int,
    backbone: str,
    checkpoint_name: str = "best.pt",
) -> Path:
    """Resolve a teacher checkpoint path from Step 3's output layout."""
    folder_task = folder_task_name(task)
    return (
        Path(teacher_root)
        / folder_task
        / artery_data_dir(artery)
        / f"fold_{fold}"
        / backbone
        / checkpoint_name
    )


def freeze_teacher(model: nn.Module) -> nn.Module:
    """Set eval mode and freeze all teacher parameters."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def _instantiate_teacher(task: str, backbone: str) -> nn.Module:
    folder_task = folder_task_name(task)
    if folder_task == "occlusion":
        return VideoTeacher(backbone_name=backbone, pretrained=False)
    if folder_task in {"framequality", "dominance"}:
        return SingleFrameTeacher(backbone_name=backbone, pretrained=False)
    raise ValueError(f"Unsupported teacher task '{task}'.")


def _metadata_value(checkpoint: dict[str, Any], key: str) -> Any:
    return checkpoint.get(key)


def _validate_metadata(
    checkpoint: dict[str, Any],
    task: str,
    artery: str,
    fold: int,
    backbone: str,
) -> None:
    expected = {
        "task": folder_task_name(task),
        "artery": normalize_artery(artery),
        "fold": fold,
        "backbone": backbone,
    }
    for key, expected_value in expected.items():
        actual_value = _metadata_value(checkpoint, key)
        if actual_value is None:
            continue
        if key == "task":
            actual_value = folder_task_name(str(actual_value))
        elif key == "artery":
            actual_value = normalize_artery(str(actual_value))
        elif key == "fold":
            actual_value = int(actual_value)
        if actual_value != expected_value:
            raise ValueError(
                f"Checkpoint metadata mismatch for '{key}': "
                f"expected {expected_value!r}, found {actual_value!r}."
            )


def load_teacher_from_checkpoint(
    teacher_root: str | Path,
    task: str,
    artery: str,
    fold: int,
    backbone: str,
    device: torch.device | str,
    checkpoint_name: str = "best.pt",
) -> tuple[str, nn.Module, dict[str, Any]]:
    """Load one frozen/eval teacher and return its internal task name."""
    checkpoint_path = resolve_teacher_checkpoint_path(
        teacher_root=teacher_root,
        task=task,
        artery=artery,
        fold=fold,
        backbone=backbone,
        checkpoint_name=checkpoint_name,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'.")
    _validate_metadata(checkpoint, task=task, artery=artery, fold=fold, backbone=backbone)

    model = _instantiate_teacher(task, backbone)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    freeze_teacher(model)
    return model_task_name(task), model, checkpoint


def load_teacher_bundle(
    teacher_root: str | Path,
    tasks: list[str] | tuple[str, ...],
    artery: str,
    fold: int,
    backbone: str,
    device: torch.device | str,
    checkpoint_name: str = "best.pt",
) -> dict[str, nn.Module]:
    """Load requested teachers into a bundle keyed by internal task names."""
    normalized_artery = normalize_artery(artery)
    teachers: dict[str, nn.Module] = {}
    for task in tasks:
        folder_task = folder_task_name(task)
        if folder_task == "occlusion" and normalized_artery != "RCA":
            continue
        model_task, teacher, _ = load_teacher_from_checkpoint(
            teacher_root=teacher_root,
            task=folder_task,
            artery=normalized_artery,
            fold=fold,
            backbone=backbone,
            device=device,
            checkpoint_name=checkpoint_name,
        )
        teachers[model_task] = teacher
    return teachers


def run_teacher_bundle(
    teachers: dict[str, nn.Module],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run matching teachers under no_grad and return logits by internal task key."""
    outputs: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for task, teacher in teachers.items():
            model_task = model_task_name(task)
            input_key = TASK_INPUT_KEYS[model_task]
            if input_key not in batch or batch[input_key].numel() == 0:
                continue
            outputs[model_task] = teacher(batch[input_key])
    return outputs
