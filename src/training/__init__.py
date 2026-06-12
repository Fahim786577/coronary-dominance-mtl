"""Training utilities for coronary dominance reproducibility code."""

from src.training.checkpointing import load_checkpoint, save_checkpoint, save_history_csv
from src.training.distillation import (
    DEFAULT_ALPHAS,
    DEFAULT_TEMPERATURE,
    distillation_ce_kl_loss,
    multi_task_distillation_losses,
    temperature_scaled_kl_loss,
)
from src.training.metrics import AverageMeter, accuracy_from_logits
from src.training.teacher_loading import (
    folder_task_name,
    freeze_teacher,
    load_teacher_bundle,
    load_teacher_from_checkpoint,
    model_task_name,
    resolve_teacher_checkpoint_path,
    run_teacher_bundle,
)
from src.training.teacher_trainer import TeacherTrainer

__all__ = [
    "AverageMeter",
    "DEFAULT_ALPHAS",
    "DEFAULT_TEMPERATURE",
    "TeacherTrainer",
    "accuracy_from_logits",
    "distillation_ce_kl_loss",
    "folder_task_name",
    "freeze_teacher",
    "load_checkpoint",
    "load_teacher_bundle",
    "load_teacher_from_checkpoint",
    "model_task_name",
    "multi_task_distillation_losses",
    "resolve_teacher_checkpoint_path",
    "run_teacher_bundle",
    "save_checkpoint",
    "save_history_csv",
    "temperature_scaled_kl_loss",
]
