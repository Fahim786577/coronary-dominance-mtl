"""Training loop for independent teacher models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.checkpointing import save_checkpoint, save_history_csv
from src.training.metrics import AverageMeter, accuracy_from_logits


@dataclass
class TrainerConfig:
    """Runtime controls for teacher training."""

    max_epochs: int
    early_stopping_patience: int
    gradient_clip_max_norm: float | None = 1.0


class TeacherTrainer:
    """Train and validate one independent teacher model."""

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
        config: TrainerConfig,
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
        self.checkpoint_metadata = dict(checkpoint_metadata)
        self.history: list[dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_val_accuracy = 0.0

    def _run_epoch(self, loader: DataLoader, train: bool) -> tuple[float, float]:
        self.model.train(train)
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.config.gradient_clip_max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.gradient_clip_max_norm,
                        )
                    self.optimizer.step()

            batch_size = labels.size(0)
            loss_meter.update(loss.item(), batch_size)
            accuracy_meter.update(accuracy_from_logits(logits.detach(), labels), batch_size)

        return loss_meter.average, accuracy_meter.average

    def _checkpoint_payload(
        self,
        epoch: int,
        config_dict: Mapping[str, Any],
    ) -> dict[str, Any]:
        scheduler_state = self.scheduler.state_dict() if self.scheduler is not None else None
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            **self.checkpoint_metadata,
            "config": dict(config_dict),
        }

    def fit(self, config_dict: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Run training, save best/last checkpoints, and write history.csv."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        epochs_without_improvement = 0

        for epoch in range(1, self.config.max_epochs + 1):
            train_loss, train_accuracy = self._run_epoch(self.train_loader, train=True)
            val_loss, val_accuracy = self._run_epoch(self.val_loader, train=False)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            learning_rate = self.optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": learning_rate,
            }
            self.history.append(row)

            improved = val_loss < self.best_val_loss
            if improved:
                self.best_val_loss = val_loss
                self.best_val_accuracy = val_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            payload = self._checkpoint_payload(epoch=epoch, config_dict=config_dict)
            save_checkpoint(payload, self.output_dir / "last.pt")
            if improved:
                save_checkpoint(payload, self.output_dir / "best.pt")

            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} "
                f"lr={learning_rate:.6g}"
            )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch} epochs.")
                break

        save_history_csv(self.history, self.output_dir / "history.csv")
        return self.history
