"""Baseline and MTD supervised MTL trainer."""

from __future__ import annotations

import csv
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.training.checkpointing import save_checkpoint
from src.training.distillation import DEFAULT_ALPHAS, DEFAULT_TEMPERATURE, distillation_ce_kl_loss
from src.training.metrics import AverageMeter, accuracy_from_logits
from src.training.teacher_loading import run_teacher_bundle
from src.training.twophase import (
    TASK_INPUT_KEYS,
    run_twophase_phase1_step,
    run_twophase_phase2_step,
)


TASKS = ("occlusion", "frame_quality", "dominance")


@dataclass
class MTLTrainerConfig:
    """Runtime controls for MTL training."""

    max_epochs: int
    early_stopping_patience: int
    gradient_clip_max_norm: float | None = 1.0
    task_weights: dict[str, float] | None = None
    use_mtd: bool = False
    mtd_temperature: float = DEFAULT_TEMPERATURE
    mtd_alphas: dict[str, float] | None = None
    use_twophase: bool = False
    twophase_mode: str = "practical"
    twophase_priority_source: str = "shared_grad_norm"
    twophase_projection: bool = True
    twophase_eps: float = 1e-12
    twophase_seed: int = 42


def move_batch_to_device(
    inputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Move nested MTL input/target dictionaries to the target device."""
    return (
        {key: value.to(device, non_blocking=True) for key, value in inputs.items()},
        {key: value.to(device, non_blocking=True) for key, value in targets.items()},
    )


class MTLTrainer:
    """Train CoronaryTemporalMTL with baseline, MTD, and practical TwoPhase modes."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
        device: torch.device,
        output_dir: str | Path,
        config: MTLTrainerConfig,
        checkpoint_metadata: Mapping[str, Any],
        teachers: dict[str, nn.Module] | None = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config
        self.task_weights = config.task_weights or {task: 1.0 for task in TASKS}
        self.mtd_alphas = config.mtd_alphas or dict(DEFAULT_ALPHAS)
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.teachers = teachers or {}
        if self.config.use_mtd and not self.teachers:
            raise ValueError("MTD training requires at least one loaded teacher.")
        self.twophase_rng = random.Random(self.config.twophase_seed)
        self.history: list[dict[str, Any]] = []
        self.best_val_accuracy = -1.0
        self.best_val_loss = float("inf")

    def _mode_name(self) -> str:
        if self.config.use_mtd and self.config.use_twophase:
            return "mtl_mtd_twophase_practical"
        if self.config.use_mtd:
            return "mtl_mtd"
        if self.config.use_twophase:
            return "mtl_twophase_practical"
        return "baseline_mtl"

    def _compute_supervised_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, float], dict[str, dict[str, float]]]:
        task_losses: dict[str, Tensor] = {}
        task_accuracies: dict[str, float] = {}
        weighted_losses: list[Tensor] = []

        for task in TASKS:
            if task not in outputs or task not in targets:
                continue
            loss = self.criterion(outputs[task], targets[task])
            task_losses[task] = loss
            task_accuracies[task] = accuracy_from_logits(outputs[task].detach(), targets[task])
            weighted_losses.append(loss * self.task_weights.get(task, 1.0))

        if not weighted_losses:
            raise ValueError("No task losses could be computed for this MTL batch.")
        return torch.stack(weighted_losses).mean(), task_losses, task_accuracies, {}

    def _compute_mtd_losses(
        self,
        outputs: dict[str, Tensor],
        teacher_outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, float], dict[str, dict[str, float]]]:
        task_losses: dict[str, Tensor] = {}
        task_accuracies: dict[str, float] = {}
        task_components: dict[str, dict[str, float]] = {}
        weighted_losses: list[Tensor] = []

        for task in TASKS:
            if task not in outputs or task not in teacher_outputs or task not in targets:
                continue
            loss, components = distillation_ce_kl_loss(
                student_logits=outputs[task],
                teacher_logits=teacher_outputs[task],
                targets=targets[task],
                alpha=self.mtd_alphas.get(task, DEFAULT_ALPHAS[task]),
                temperature=self.config.mtd_temperature,
                ce_loss_fn=self.criterion,
            )
            task_losses[task] = loss
            task_accuracies[task] = accuracy_from_logits(outputs[task].detach(), targets[task])
            task_components[task] = components
            weighted_losses.append(loss * self.task_weights.get(task, 1.0))

        if not weighted_losses:
            raise ValueError("No MTD task losses could be computed for this MTL batch.")
        return torch.stack(weighted_losses).mean(), task_losses, task_accuracies, task_components

    def _compute_single_task_loss(
        self,
        task: str,
        inputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Any]]:
        outputs = self.model(inputs)
        if task not in outputs or task not in targets:
            raise ValueError(f"Task '{task}' is missing from model outputs or targets.")

        if self.config.use_mtd:
            teacher_outputs = run_teacher_bundle(self.teachers, inputs)
            if task not in teacher_outputs:
                raise RuntimeError(f"MTD TwoPhase step is missing teacher output for task '{task}'.")
            loss, components = distillation_ce_kl_loss(
                student_logits=outputs[task],
                teacher_logits=teacher_outputs[task],
                targets=targets[task],
                alpha=self.mtd_alphas.get(task, DEFAULT_ALPHAS[task]),
                temperature=self.config.mtd_temperature,
                ce_loss_fn=self.criterion,
            )
            metrics: dict[str, Any] = {
                "loss": components["total_loss"],
                "accuracy": accuracy_from_logits(outputs[task].detach(), targets[task]),
                "ce_loss": components["ce_loss"],
                "kl_loss": components["kl_loss"],
                "mtd_loss": components["total_loss"],
            }
            return loss * self.task_weights.get(task, 1.0), metrics

        loss = self.criterion(outputs[task], targets[task])
        return loss * self.task_weights.get(task, 1.0), {
            "loss": float(loss.detach().cpu()),
            "accuracy": accuracy_from_logits(outputs[task].detach(), targets[task]),
            "ce_loss": None,
            "kl_loss": None,
            "mtd_loss": None,
        }

    @staticmethod
    def _active_tasks_for_batch(inputs: dict[str, Tensor], targets: dict[str, Tensor]) -> list[str]:
        return [
            task
            for task in TASKS
            if task in targets and TASK_INPUT_KEYS[task] in inputs and inputs[TASK_INPUT_KEYS[task]].numel() > 0
        ]

    def _run_twophase_epoch(self, loader: DataLoader, epoch: int) -> dict[str, Any]:
        self.model.train(True)
        total_loss_meter = AverageMeter()
        task_loss_meters = {task: AverageMeter() for task in TASKS}
        task_accuracy_meters = {task: AverageMeter() for task in TASKS}
        ce_loss_meters = {task: AverageMeter() for task in TASKS}
        kl_loss_meters = {task: AverageMeter() for task in TASKS}
        mtd_loss_meters = {task: AverageMeter() for task in TASKS}
        phase1_steps = 0
        phase2_steps = 0
        priority_counts = {task: 0 for task in TASKS}
        phase2_probability = epoch / max(1, self.config.max_epochs)

        for inputs, targets in loader:
            inputs, targets = move_batch_to_device(inputs, targets, self.device)
            active_tasks = self._active_tasks_for_batch(inputs, targets)
            if not active_tasks:
                continue

            if self.twophase_rng.uniform(0.0, 1.0) >= phase2_probability:
                result = run_twophase_phase1_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    inputs=inputs,
                    targets=targets,
                    active_tasks=active_tasks,
                    compute_task_loss=self._compute_single_task_loss,
                    gradient_clip_max_norm=self.config.gradient_clip_max_norm,
                )
                phase1_steps += 1
            else:
                result = run_twophase_phase2_step(
                    model=self.model,
                    optimizer=self.optimizer,
                    inputs=inputs,
                    targets=targets,
                    active_tasks=active_tasks,
                    compute_task_loss=self._compute_single_task_loss,
                    gradient_clip_max_norm=self.config.gradient_clip_max_norm,
                    projection=self.config.twophase_projection,
                    eps=self.config.twophase_eps,
                )
                phase2_steps += 1
                if result["priority_task"] in priority_counts:
                    priority_counts[result["priority_task"]] += 1

            batch_size = next(iter(targets.values())).size(0)
            total_loss_meter.update(result["loss"], batch_size)
            for task, metrics in result["task_metrics"].items():
                task_loss_meters[task].update(metrics["loss"], batch_size)
                task_accuracy_meters[task].update(metrics["accuracy"], batch_size)
                if metrics.get("ce_loss") is not None:
                    ce_loss_meters[task].update(metrics["ce_loss"], batch_size)
                if metrics.get("kl_loss") is not None:
                    kl_loss_meters[task].update(metrics["kl_loss"], batch_size)
                if metrics.get("mtd_loss") is not None:
                    mtd_loss_meters[task].update(metrics["mtd_loss"], batch_size)

        available_accuracies = [
            meter.average for meter in task_accuracy_meters.values() if meter.count > 0
        ]
        if not available_accuracies:
            raise ValueError("No task accuracies were computed during TwoPhase training.")
        mean_accuracy = sum(available_accuracies) / len(available_accuracies)
        return {
            "loss": total_loss_meter.average,
            "mean_accuracy": mean_accuracy,
            "task_losses": {
                task: meter.average if meter.count > 0 else None
                for task, meter in task_loss_meters.items()
            },
            "task_accuracies": {
                task: meter.average if meter.count > 0 else None
                for task, meter in task_accuracy_meters.items()
            },
            "mtd_components": {
                task: {
                    "ce_loss": ce_loss_meters[task].average if ce_loss_meters[task].count > 0 else None,
                    "kl_loss": kl_loss_meters[task].average if kl_loss_meters[task].count > 0 else None,
                    "mtd_loss": mtd_loss_meters[task].average if mtd_loss_meters[task].count > 0 else None,
                }
                for task in TASKS
            },
            "twophase": {
                "phase1_steps": phase1_steps,
                "phase2_steps": phase2_steps,
                "priority_counts": priority_counts,
            },
        }

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int = 1) -> dict[str, Any]:
        if train and self.config.use_twophase:
            return self._run_twophase_epoch(loader, epoch)

        self.model.train(train)
        total_loss_meter = AverageMeter()
        task_loss_meters = {task: AverageMeter() for task in TASKS}
        task_accuracy_meters = {task: AverageMeter() for task in TASKS}
        ce_loss_meters = {task: AverageMeter() for task in TASKS}
        kl_loss_meters = {task: AverageMeter() for task in TASKS}
        mtd_loss_meters = {task: AverageMeter() for task in TASKS}

        for inputs, targets in loader:
            inputs, targets = move_batch_to_device(inputs, targets, self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(inputs)
                if train and self.config.use_mtd:
                    teacher_outputs = run_teacher_bundle(self.teachers, inputs)
                    missing_teachers = [
                        task
                        for task in TASKS
                        if task in outputs and task in targets and task not in teacher_outputs
                    ]
                    if missing_teachers:
                        missing = ", ".join(missing_teachers)
                        raise RuntimeError(f"MTD training is missing teacher outputs for: {missing}.")
                    total_loss, task_losses, task_accuracies, task_components = self._compute_mtd_losses(
                        outputs,
                        teacher_outputs,
                        targets,
                    )
                else:
                    total_loss, task_losses, task_accuracies, task_components = (
                        self._compute_supervised_losses(outputs, targets)
                    )

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    if self.config.gradient_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_max_norm,
                        )
                    self.optimizer.step()

            batch_size = next(iter(targets.values())).size(0)
            total_loss_meter.update(total_loss.item(), batch_size)
            for task, loss in task_losses.items():
                task_loss_meters[task].update(loss.item(), batch_size)
                task_accuracy_meters[task].update(task_accuracies[task], batch_size)
            for task, components in task_components.items():
                ce_loss_meters[task].update(components["ce_loss"], batch_size)
                kl_loss_meters[task].update(components["kl_loss"], batch_size)
                mtd_loss_meters[task].update(components["total_loss"], batch_size)

        available_accuracies = [
            meter.average for meter in task_accuracy_meters.values() if meter.count > 0
        ]
        mean_accuracy = sum(available_accuracies) / len(available_accuracies)
        return {
            "loss": total_loss_meter.average,
            "mean_accuracy": mean_accuracy,
            "task_losses": {
                task: meter.average if meter.count > 0 else None
                for task, meter in task_loss_meters.items()
            },
            "task_accuracies": {
                task: meter.average if meter.count > 0 else None
                for task, meter in task_accuracy_meters.items()
            },
            "mtd_components": {
                task: {
                    "ce_loss": ce_loss_meters[task].average if ce_loss_meters[task].count > 0 else None,
                    "kl_loss": kl_loss_meters[task].average if kl_loss_meters[task].count > 0 else None,
                    "mtd_loss": mtd_loss_meters[task].average if mtd_loss_meters[task].count > 0 else None,
                }
                for task in TASKS
            },
            "twophase": {
                "phase1_steps": None,
                "phase2_steps": None,
                "priority_counts": {task: None for task in TASKS},
            },
        }

    def _history_row(self, epoch: int, train_metrics: dict[str, Any], val_metrics: dict[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mean_accuracy": train_metrics["mean_accuracy"],
            "val_mean_accuracy": val_metrics["mean_accuracy"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        for task in TASKS:
            row[f"train_{task}_loss"] = train_metrics["task_losses"][task]
            row[f"val_{task}_loss"] = val_metrics["task_losses"][task]
            row[f"train_{task}_accuracy"] = train_metrics["task_accuracies"][task]
            row[f"val_{task}_accuracy"] = val_metrics["task_accuracies"][task]
            row[f"train_{task}_ce_loss"] = train_metrics["mtd_components"][task]["ce_loss"]
            row[f"train_{task}_kl_loss"] = train_metrics["mtd_components"][task]["kl_loss"]
            row[f"train_{task}_mtd_loss"] = train_metrics["mtd_components"][task]["mtd_loss"]
        row["train_twophase_phase1_steps"] = train_metrics["twophase"]["phase1_steps"]
        row["train_twophase_phase2_steps"] = train_metrics["twophase"]["phase2_steps"]
        for task in TASKS:
            row[f"train_twophase_priority_{task}_count"] = train_metrics["twophase"]["priority_counts"][task]
        return row

    def _checkpoint_payload(self, epoch: int, config_dict: Mapping[str, Any]) -> dict[str, Any]:
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "epoch": epoch,
            "best_val_accuracy": self.best_val_accuracy,
            "best_val_loss": self.best_val_loss,
            **self.checkpoint_metadata,
            "mode": self._mode_name(),
            "use_mtd": self.config.use_mtd,
            "use_twophase": self.config.use_twophase,
            "config": dict(config_dict),
        }

    @staticmethod
    def _save_history(history: list[dict[str, Any]], path: Path) -> None:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "train_mean_accuracy",
            "val_mean_accuracy",
            "train_occlusion_loss",
            "train_frame_quality_loss",
            "train_dominance_loss",
            "val_occlusion_loss",
            "val_frame_quality_loss",
            "val_dominance_loss",
            "train_occlusion_accuracy",
            "train_frame_quality_accuracy",
            "train_dominance_accuracy",
            "val_occlusion_accuracy",
            "val_frame_quality_accuracy",
            "val_dominance_accuracy",
            "learning_rate",
        ]
        mtd_fieldnames = [
            "train_occlusion_ce_loss",
            "train_occlusion_kl_loss",
            "train_occlusion_mtd_loss",
            "train_frame_quality_ce_loss",
            "train_frame_quality_kl_loss",
            "train_frame_quality_mtd_loss",
            "train_dominance_ce_loss",
            "train_dominance_kl_loss",
            "train_dominance_mtd_loss",
        ]
        if any(row.get(field) is not None for row in history for field in mtd_fieldnames):
            fieldnames.extend(mtd_fieldnames)
        twophase_fieldnames = [
            "train_twophase_phase1_steps",
            "train_twophase_phase2_steps",
            "train_twophase_priority_occlusion_count",
            "train_twophase_priority_frame_quality_count",
            "train_twophase_priority_dominance_count",
        ]
        if any(row.get(field) is not None for row in history for field in twophase_fieldnames):
            fieldnames.extend(twophase_fieldnames)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in history:
                writer.writerow({key: "" if value is None else value for key, value in row.items()})

    @staticmethod
    def _print_task_metrics(prefix: str, metrics: dict[str, Any]) -> None:
        """Print task-wise loss and accuracy for tasks present in the epoch."""
        for task in TASKS:
            loss = metrics["task_losses"][task]
            accuracy = metrics["task_accuracies"][task]
            if loss is None or accuracy is None:
                continue
            components = metrics["mtd_components"][task]
            if components["ce_loss"] is not None and components["kl_loss"] is not None:
                print(
                    f"  {prefix:<5} {task:<14} loss={loss:.4f} "
                    f"ce={components['ce_loss']:.4f} kl={components['kl_loss']:.4f} "
                    f"acc={accuracy:.4f}"
                )
            else:
                print(f"  {prefix:<5} {task:<14} loss={loss:.4f} acc={accuracy:.4f}")

    def fit(self, config_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Run baseline MTL training and checkpoint best/last models."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        epochs_without_improvement = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step(val_metrics["mean_accuracy"])

            row = self._history_row(epoch, train_metrics, val_metrics)
            self.history.append(row)

            improved = val_metrics["mean_accuracy"] > self.best_val_accuracy
            tied_with_lower_loss = (
                val_metrics["mean_accuracy"] == self.best_val_accuracy
                and val_metrics["loss"] < self.best_val_loss
            )
            if improved or tied_with_lower_loss:
                self.best_val_accuracy = val_metrics["mean_accuracy"]
                self.best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            payload = self._checkpoint_payload(epoch=epoch, config_dict=config_dict)
            save_checkpoint(payload, self.output_dir / "last.pt")
            if improved or tied_with_lower_loss:
                save_checkpoint(payload, self.output_dir / "best.pt")

            phase_text = ""
            if train_metrics["twophase"]["phase1_steps"] is not None:
                phase_text = (
                    f" phase1_steps={train_metrics['twophase']['phase1_steps']}"
                    f" phase2_steps={train_metrics['twophase']['phase2_steps']}"
                )
            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_mean_acc={train_metrics['mean_accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_mean_acc={val_metrics['mean_accuracy']:.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.6g}"
                f"{phase_text}"
            )
            self._print_task_metrics("train", train_metrics)
            self._print_task_metrics("val", val_metrics)

            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        self._save_history(self.history, self.output_dir / "history.csv")
        return self.history
