"""Baseline supervised MTL trainer."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.training.checkpointing import save_checkpoint
from src.training.metrics import AverageMeter, accuracy_from_logits


TASKS = ("occlusion", "frame_quality", "dominance")


@dataclass
class MTLTrainerConfig:
    """Runtime controls for baseline MTL training."""

    max_epochs: int
    early_stopping_patience: int
    gradient_clip_max_norm: float | None = 1.0
    task_weights: dict[str, float] | None = None


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
    """Train a CoronaryTemporalMTL model with supervised CE losses only."""

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
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.history: list[dict[str, Any]] = []
        self.best_val_accuracy = -1.0
        self.best_val_loss = float("inf")

    def _compute_losses(
        self,
        outputs: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, float]]:
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
        return torch.stack(weighted_losses).mean(), task_losses, task_accuracies

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, Any]:
        self.model.train(train)
        total_loss_meter = AverageMeter()
        task_loss_meters = {task: AverageMeter() for task in TASKS}
        task_accuracy_meters = {task: AverageMeter() for task in TASKS}

        for inputs, targets in loader:
            inputs, targets = move_batch_to_device(inputs, targets, self.device)

            with torch.set_grad_enabled(train):
                outputs = self.model(inputs)
                total_loss, task_losses, task_accuracies = self._compute_losses(outputs, targets)

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
            "mode": "baseline_mtl",
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
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in history:
                writer.writerow({key: "" if value is None else value for key, value in row.items()})

    def fit(self, config_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Run baseline MTL training and checkpoint best/last models."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        epochs_without_improvement = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)

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

            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_mean_acc={train_metrics['mean_accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_mean_acc={val_metrics['mean_accuracy']:.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.6g}"
            )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        self._save_history(self.history, self.output_dir / "history.csv")
        return self.history
