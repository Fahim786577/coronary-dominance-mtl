"""Checkpoint evaluation and CSV prediction export for CoronaryTemporalMTL."""

from __future__ import annotations

import csv
import json
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data.label_maps import get_class_id, get_label_map
from src.data.mtl_dataset import (
    build_mtl_transform,
    evenly_spaced_indices,
    frame_sort_key,
    load_grayscale_image,
    load_split_rows,
    split_csv_filename,
)
from src.evaluation.metrics import binary_classification_metrics, mean_metric
from src.models import CoronaryTemporalMTL
from src.models.backbones import normalize_backbone_name
from src.training.checkpointing import load_checkpoint
from src.utils.paths import build_image_path, build_split_csv_path, normalize_artery


EVALUATION_MODES = (
    "baseline",
    "mtd",
    "twophase",
    "mtd_twophase",
    "baseline_transfer",
    "mtd_transfer",
    "twophase_transfer",
    "mtd_twophase_transfer",
)

TASK_TO_FOLDER = {
    "occlusion": "occlusion",
    "frame_quality": "framequality",
    "dominance": "dominance",
}
TASK_TO_INPUT_KEY = {
    "occlusion": "occlusion_images",
    "frame_quality": "frame_quality_images",
    "dominance": "dominance_images",
}
MODE_METADATA_ALIASES = {
    "baseline": {"baseline", "baseline_mtl"},
    "mtd": {"mtd", "mtl_mtd"},
    "twophase": {"twophase", "mtl_twophase_practical"},
    "mtd_twophase": {"mtd_twophase", "mtl_mtd_twophase_practical"},
    "baseline_transfer": {"baseline", "baseline_mtl"},
    "mtd_transfer": {"mtd", "mtl_mtd"},
    "twophase_transfer": {"twophase", "mtl_twophase_practical"},
    "mtd_twophase_transfer": {"mtd_twophase", "mtl_mtd_twophase_practical"},
}
FRAME_SUFFIX_PATTERN = re.compile(r"_frame_(\d+)(?=\.[^.]+$|$)", re.IGNORECASE)

PREDICTION_COLUMNS = (
    "task",
    "artery",
    "split",
    "fold",
    "study_id",
    "video_id",
    "filename",
    "frame_filenames",
    "label",
    "true_class",
    "pred_class",
    "pred_label",
    "prob_class_0",
    "prob_class_1",
    "logit_class_0",
    "logit_class_1",
    "image_path",
    "total_frames",
)

METRIC_COLUMNS = (
    "task",
    "loss",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "tn",
    "fp",
    "fn",
    "tp",
    "sample_count",
)


@dataclass(frozen=True)
class LoadedMTLCheckpoint:
    """Loaded model, checkpoint payload, and metadata validation warnings."""

    model: CoronaryTemporalMTL
    checkpoint: dict[str, Any]
    warnings: list[str]


def available_tasks_for_artery(artery: str) -> tuple[str, ...]:
    """Return internal task keys that should be evaluated for an artery."""
    normalized = normalize_artery(artery)
    if normalized == "RCA":
        return "occlusion", "frame_quality", "dominance"
    if normalized == "LCA":
        return "frame_quality", "dominance"
    raise ValueError(f"artery must be RCA or LCA, got '{artery}'.")


def reverse_label_map(task: str) -> dict[int, str]:
    """Return class-id to label-name mapping for an internal task key."""
    folder_task = TASK_TO_FOLDER[task]
    return {class_id: label for label, class_id in get_label_map(folder_task).items()}


def derive_evaluation_video_id(filename: str) -> str:
    """Derive a video/group id by removing a trailing _frame_N suffix."""
    return FRAME_SUFFIX_PATTERN.sub("", Path(filename).stem)


def resolve_mtl_checkpoint_path(
    checkpoint_root: str | Path,
    mode: str,
    artery: str,
    fold: int,
    backbone: str,
    checkpoint_name: str = "best.pt",
) -> Path:
    """Resolve a student checkpoint from the training output layout."""
    if mode not in EVALUATION_MODES:
        valid_modes = ", ".join(EVALUATION_MODES)
        raise ValueError(f"Unsupported mode '{mode}'. Expected one of: {valid_modes}.")

    return (
        Path(checkpoint_root)
        / mode
        / f"DATA_{normalize_artery(artery)}"
        / f"fold_{fold}"
        / normalize_backbone_name(backbone)
        / checkpoint_name
    )


def resolve_evaluation_output_dir(
    output_root: str | Path,
    mode: str,
    artery: str,
    fold: int,
    backbone: str,
    split: str,
) -> Path:
    """Resolve the default Step 8 evaluation output directory."""
    return (
        Path(output_root)
        / mode
        / f"DATA_{normalize_artery(artery)}"
        / f"fold_{fold}"
        / normalize_backbone_name(backbone)
        / split
    )


def _metadata_warning(message: str, validation_warnings: list[str]) -> None:
    validation_warnings.append(message)
    warnings.warn(message, stacklevel=3)


def _validate_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    artery: str,
    fold: int,
    backbone: str,
    mode: str,
    allow_mismatch: bool,
) -> list[str]:
    validation_warnings: list[str] = []
    expected_values = {
        "artery": normalize_artery(artery),
        "fold": int(fold),
        "backbone": normalize_backbone_name(backbone),
    }

    for key, expected_value in expected_values.items():
        if key not in checkpoint:
            _metadata_warning(f"Checkpoint metadata is missing '{key}'. Using CLI value {expected_value!r}.", validation_warnings)
            continue

        actual_value = checkpoint[key]
        if key == "artery":
            actual_value = normalize_artery(str(actual_value))
        elif key == "fold":
            actual_value = int(actual_value)
        elif key == "backbone":
            actual_value = normalize_backbone_name(str(actual_value))

        if actual_value != expected_value:
            message = f"Checkpoint metadata mismatch for '{key}': expected {expected_value!r}, found {actual_value!r}."
            if allow_mismatch:
                _metadata_warning(message, validation_warnings)
            else:
                raise ValueError(message)

    checkpoint_mode = checkpoint.get("mode")
    if checkpoint_mode is None:
        _metadata_warning(f"Checkpoint metadata is missing 'mode'. Using CLI mode {mode!r}.", validation_warnings)
    elif str(checkpoint_mode) not in MODE_METADATA_ALIASES[mode]:
        message = f"Checkpoint metadata mismatch for 'mode': CLI mode {mode!r}, checkpoint mode {checkpoint_mode!r}."
        if allow_mismatch:
            _metadata_warning(message, validation_warnings)
        else:
            raise ValueError(message)

    if mode.endswith("_transfer"):
        transfer_learning = checkpoint.get("transfer_learning")
        if transfer_learning is None:
            _metadata_warning("Transfer checkpoint metadata is missing 'transfer_learning'.", validation_warnings)
        elif transfer_learning is not True:
            message = f"Expected transfer checkpoint metadata transfer_learning=True, found {transfer_learning!r}."
            if allow_mismatch:
                _metadata_warning(message, validation_warnings)
            else:
                raise ValueError(message)

    return validation_warnings


def load_mtl_model_from_checkpoint(
    checkpoint_path: str | Path,
    backbone: str,
    device: torch.device,
    artery: str,
    fold: int,
    mode: str,
    allow_metadata_mismatch: bool = False,
) -> LoadedMTLCheckpoint:
    """Reconstruct CoronaryTemporalMTL and load a saved student checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'.")

    validation_warnings = _validate_checkpoint_metadata(
        checkpoint=checkpoint,
        artery=artery,
        fold=fold,
        backbone=backbone,
        mode=mode,
        allow_mismatch=allow_metadata_mismatch,
    )

    model = CoronaryTemporalMTL(
        backbone_name=normalize_backbone_name(backbone),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return LoadedMTLCheckpoint(model=model, checkpoint=checkpoint, warnings=validation_warnings)


class CoronaryTaskInferenceDataset(Dataset):
    """Per-task dataset for evaluation without MTL sample cycling."""

    def __init__(
        self,
        data_root: str | Path,
        split_root: str | Path,
        task: str,
        artery: str,
        split: str,
        fold: int,
        image_size: int = 512,
        clip_length: int = 15,
        mean: float = 0.5485,
        std: float = 0.1407,
    ) -> None:
        if task not in TASK_TO_FOLDER:
            valid_tasks = ", ".join(sorted(TASK_TO_FOLDER))
            raise ValueError(f"Unsupported task '{task}'. Expected one of: {valid_tasks}.")

        self.data_root = Path(data_root)
        self.split_root = Path(split_root)
        self.task = task
        self.folder_task = TASK_TO_FOLDER[task]
        self.artery = normalize_artery(artery)
        self.split = split
        self.fold = fold
        self.clip_length = clip_length
        self.transform = build_mtl_transform(image_size=image_size, mean=mean, std=std)
        self.samples = self._load_samples()

    def _csv_path(self) -> Path:
        return build_split_csv_path(
            self.split_root,
            task=self.folder_task,
            artery=self.artery,
            csv_filename=split_csv_filename(self.task, self.split, self.fold),
        )

    def _load_samples(self) -> list[dict[str, Any]]:
        if self.task == "occlusion":
            if self.artery != "RCA":
                return []
            return self._load_occlusion_samples()
        return self._load_frame_samples()

    def _load_frame_samples(self) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for row in load_split_rows(self._csv_path()):
            label = row["label"]
            filename = row["filename"]
            image_path = build_image_path(
                self.data_root,
                task=self.folder_task,
                artery=row["artery"],
                label=label,
                filename=filename,
            )
            samples.append(
                {
                    "task": self.task,
                    "image_path": image_path,
                    "label_id": get_class_id(self.folder_task, label),
                    "metadata": {
                        "task": self.task,
                        "artery": normalize_artery(row["artery"]),
                        "split": row["split"],
                        "fold": row["fold"],
                        "study_id": row["study_id"],
                        "video_id": derive_evaluation_video_id(filename),
                        "filename": filename,
                        "frame_filenames": filename,
                        "label": label,
                        "image_path": str(image_path),
                        "total_frames": 1,
                    },
                }
            )

        if not samples:
            raise ValueError(f"No samples found for {self.task} {self.artery} {self.split}.")
        return samples

    def _load_occlusion_samples(self) -> list[dict[str, Any]]:
        grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in load_split_rows(self._csv_path()):
            video_id = derive_evaluation_video_id(row["filename"]) or row["study_id"]
            grouped_rows[video_id].append(row)

        samples: list[dict[str, Any]] = []
        for video_id, rows in sorted(grouped_rows.items()):
            sorted_rows = sorted(rows, key=frame_sort_key)
            labels = {row["label"] for row in sorted_rows}
            if len(labels) != 1:
                labels_text = ", ".join(sorted(labels))
                raise ValueError(f"Occlusion group '{video_id}' has mixed labels: {labels_text}.")

            label = sorted_rows[0]["label"]
            sample_indices = evenly_spaced_indices(len(sorted_rows), self.clip_length)
            sampled_rows = [sorted_rows[index] for index in sample_indices]
            frame_paths = [
                build_image_path(
                    self.data_root,
                    task="occlusion",
                    artery=row["artery"],
                    label=label,
                    filename=row["filename"],
                )
                for row in sampled_rows
            ]
            samples.append(
                {
                    "task": "occlusion",
                    "frame_paths": frame_paths,
                    "label_id": get_class_id("occlusion", label),
                    "metadata": {
                        "task": "occlusion",
                        "artery": normalize_artery(sorted_rows[0]["artery"]),
                        "split": sorted_rows[0]["split"],
                        "fold": sorted_rows[0]["fold"],
                        "study_id": sorted_rows[0]["study_id"],
                        "video_id": video_id,
                        "filename": video_id,
                        "frame_filenames": ";".join(row["filename"] for row in sampled_rows),
                        "label": label,
                        "image_path": ";".join(str(path) for path in frame_paths),
                        "total_frames": len(sorted_rows),
                    },
                }
            )

        if not samples:
            raise ValueError(f"No occlusion clips found for {self.artery} {self.split}.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        if self.task == "occlusion":
            frames = [
                load_grayscale_image(path, self.transform)
                for path in sample["frame_paths"]
            ]
            image_tensor = torch.stack(frames, dim=0)
        else:
            image_tensor = load_grayscale_image(sample["image_path"], self.transform)

        return {
            "task": sample["task"],
            "image": image_tensor,
            "target": int(sample["label_id"]),
            "metadata": sample["metadata"],
        }


def collate_task_batch(batch: list[dict[str, Any]]) -> tuple[dict[str, Tensor], Tensor, list[dict[str, Any]]]:
    """Collate one task-specific evaluation batch into the model input dictionary."""
    if not batch:
        raise ValueError("Cannot collate an empty evaluation batch.")

    task = str(batch[0]["task"])
    if any(item["task"] != task for item in batch):
        raise ValueError("A task evaluation batch cannot mix tasks.")

    input_key = TASK_TO_INPUT_KEY[task]
    inputs = {input_key: torch.stack([item["image"] for item in batch], dim=0)}
    targets = torch.tensor([int(item["target"]) for item in batch], dtype=torch.long)
    metadata = [dict(item["metadata"]) for item in batch]
    return inputs, targets, metadata


def _move_inputs_to_device(inputs: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def evaluate_task(
    model: CoronaryTemporalMTL,
    dataset: CoronaryTaskInferenceDataset,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate one task-specific dataset and return metrics plus prediction rows."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_task_batch,
    )
    task = dataset.task
    class_names = reverse_label_map(task)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []
    prediction_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for inputs, targets, metadatas in loader:
            inputs = _move_inputs_to_device(inputs, device)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            if task not in outputs:
                raise RuntimeError(f"Model output is missing expected task '{task}'.")

            logits = outputs[task]
            total_loss += float(criterion(logits, targets).detach().cpu())
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            batch_targets = targets.detach().cpu().tolist()
            batch_predictions = predictions.detach().cpu().tolist()
            batch_probs = probabilities.detach().cpu().tolist()
            batch_logits = logits.detach().cpu().tolist()

            targets_all.extend(int(value) for value in batch_targets)
            predictions_all.extend(int(value) for value in batch_predictions)

            for metadata, target_id, prediction_id, probs, raw_logits in zip(
                metadatas,
                batch_targets,
                batch_predictions,
                batch_probs,
                batch_logits,
            ):
                row = {
                    **metadata,
                    "fold": int(metadata["fold"]),
                    "true_class": int(target_id),
                    "pred_class": int(prediction_id),
                    "pred_label": class_names[int(prediction_id)],
                    "prob_class_0": float(probs[0]),
                    "prob_class_1": float(probs[1]),
                    "logit_class_0": float(raw_logits[0]),
                    "logit_class_1": float(raw_logits[1]),
                }
                prediction_rows.append(row)

    average_loss = total_loss / len(targets_all) if targets_all else 0.0
    metrics = binary_classification_metrics(targets_all, predictions_all, loss=average_loss)
    return metrics, prediction_rows


def _checkpoint_metadata_summary(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "epoch",
        "best_val_accuracy",
        "best_val_loss",
        "artery",
        "fold",
        "backbone",
        "mode",
        "use_mtd",
        "use_twophase",
        "twophase_mode",
        "twophase_is_paper_faithful",
        "twophase_priority_source",
        "twophase_projection",
        "transfer_learning",
        "transfer_source_checkpoint",
        "transfer_source_artery",
        "transfer_target_artery",
        "transfer_load_scope",
    )
    return {key: checkpoint[key] for key in keys if key in checkpoint}


def evaluate_checkpoint(
    data_root: str | Path,
    split_root: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    mode: str,
    artery: str,
    fold: int,
    split: str,
    backbone: str,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 0,
    image_size: int = 512,
    clip_length: int = 15,
    mean: float = 0.5485,
    std: float = 0.1407,
    allow_metadata_mismatch: bool = False,
    write_metrics: bool = True,
    write_predictions: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a checkpoint against labeled split CSVs and write Step 8 artifacts."""
    normalized_artery = normalize_artery(artery)
    normalized_backbone = normalize_backbone_name(backbone)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loaded = load_mtl_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        backbone=normalized_backbone,
        device=device,
        artery=normalized_artery,
        fold=fold,
        mode=mode,
        allow_metadata_mismatch=allow_metadata_mismatch,
    )

    active_tasks = available_tasks_for_artery(normalized_artery)
    task_metrics: dict[str, dict[str, Any]] = {}
    prediction_rows: list[dict[str, Any]] = []

    for task in active_tasks:
        dataset = CoronaryTaskInferenceDataset(
            data_root=data_root,
            split_root=split_root,
            task=task,
            artery=normalized_artery,
            split=split,
            fold=fold,
            image_size=image_size,
            clip_length=clip_length,
            mean=mean,
            std=std,
        )
        metrics, rows = evaluate_task(
            model=loaded.model,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        task_metrics[task] = metrics
        prediction_rows.extend(rows)

    report = {
        "checkpoint_path": str(Path(checkpoint_path)),
        "checkpoint_metadata": _checkpoint_metadata_summary(loaded.checkpoint),
        "checkpoint_metadata_warnings": loaded.warnings,
        "mode": mode,
        "artery": normalized_artery,
        "fold": int(fold),
        "backbone": normalized_backbone,
        "split": split,
        "active_tasks": list(active_tasks),
        "per_task": task_metrics,
        "overall": {
            "mean_task_accuracy": mean_metric(task_metrics, "accuracy"),
            "mean_task_balanced_accuracy": mean_metric(task_metrics, "balanced_accuracy"),
        },
    }

    if write_predictions:
        write_predictions_csv(prediction_rows, output_path / "predictions.csv")
    if write_metrics:
        write_metrics_json(report, output_path / "metrics.json")
        write_metrics_csv(task_metrics, output_path / "metrics.csv")

    return report, prediction_rows


def write_predictions_csv(prediction_rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write prediction rows using the Step 8 predictions.csv schema."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in prediction_rows:
            writer.writerow({column: row.get(column, "") for column in PREDICTION_COLUMNS})
    return output_path


def write_metrics_json(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write metrics.json with explicit confusion matrices and checkpoint context."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2)
    return output_path


def write_metrics_csv(task_metrics: Mapping[str, Mapping[str, Any]], path: str | Path) -> Path:
    """Write flattened per-task metrics to metrics.csv."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for task, metrics in task_metrics.items():
            writer.writerow({"task": task, **{column: metrics.get(column) for column in METRIC_COLUMNS if column != "task"}})
    return output_path
