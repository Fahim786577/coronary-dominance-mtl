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
from src.training.mtl_trainer import MTLTrainer, MTLTrainerConfig
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
from src.training.twophase import (
    compute_task_gradients,
    get_shared_parameter_items,
    project_against_priority,
    run_twophase_phase1_step,
    run_twophase_phase2_step,
    select_priority_task,
)

__all__ = [
    "AverageMeter",
    "DEFAULT_ALPHAS",
    "DEFAULT_TEMPERATURE",
    "MTLTrainer",
    "MTLTrainerConfig",
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
    "compute_task_gradients",
    "get_shared_parameter_items",
    "project_against_priority",
    "resolve_teacher_checkpoint_path",
    "run_twophase_phase1_step",
    "run_twophase_phase2_step",
    "run_teacher_bundle",
    "save_checkpoint",
    "save_history_csv",
    "select_priority_task",
    "temperature_scaled_kl_loss",
]
