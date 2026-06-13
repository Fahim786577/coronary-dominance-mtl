"""Multi-task dataset for baseline coronary MTL training."""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.label_maps import get_class_id
from src.utils.paths import build_image_path, build_split_csv_path, normalize_artery


CSV_PREFIXES = {
    "occlusion": "occlusion",
    "frame_quality": "framequality",
    "dominance": "dom",
}
FOLDER_TASKS = {
    "occlusion": "occlusion",
    "frame_quality": "framequality",
    "dominance": "dominance",
}
FRAME_NUMBER_PATTERN = re.compile(r"_frame_(\d+)(?=\.[^.]+$)", re.IGNORECASE)
REQUIRED_COLUMNS = ("filename", "label", "study_id", "artery", "split", "task", "fold")


def build_mtl_transform(image_size: int, mean: float, std: float) -> transforms.Compose:
    """Build the grayscale resize/tensor/normalize transform used for MTL."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )


def split_csv_filename(task: str, split: str, fold: int) -> str:
    """Return the expected split CSV filename for an internal task key."""
    return f"{CSV_PREFIXES[task]}_{split}_labels_fold_{fold}.csv"


def load_split_rows(csv_path: str | Path) -> list[dict[str, str]]:
    """Load split CSV rows and validate required columns."""
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return list(reader)


def derive_video_id(filename: str) -> str:
    """Derive a stable video ID by removing a trailing frame suffix."""
    return FRAME_NUMBER_PATTERN.sub("", Path(filename).stem)


def frame_sort_key(row: dict[str, str]) -> tuple[int, int | str]:
    """Sort by parsed frame number when possible, otherwise filename."""
    match = FRAME_NUMBER_PATTERN.search(row["filename"])
    if match is not None:
        return 0, int(match.group(1))
    return 1, row["filename"]


def evenly_spaced_indices(length: int, clip_length: int) -> list[int]:
    """Return deterministic frame indices with padding or even sampling."""
    if length <= 0:
        raise ValueError("Cannot sample from an empty frame group.")
    if length == clip_length:
        return list(range(length))
    if length < clip_length:
        return list(range(length)) + [length - 1] * (clip_length - length)
    if clip_length == 1:
        return [0]
    return [round(i * (length - 1) / (clip_length - 1)) for i in range(clip_length)]


def load_grayscale_image(path: Path, transform: transforms.Compose) -> Tensor:
    """Load one grayscale image and apply transforms."""
    with Image.open(path) as image:
        image = image.convert("L")
        return transform(image)


class CoronaryMTLDataset(Dataset):
    """Combined MTL dataset that cycles task-specific CSV samples."""

    def __init__(
        self,
        data_root: str | Path,
        split_root: str | Path,
        artery: str,
        split: str,
        fold: int,
        image_size: int = 512,
        clip_length: int = 15,
        mean: float = 0.5485,
        std: float = 0.1407,
        seed: int = 42,
        training: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.split_root = Path(split_root)
        self.artery = normalize_artery(artery)
        self.split = split
        self.fold = fold
        self.image_size = image_size
        self.clip_length = clip_length
        self.seed = seed
        self.training = training
        self.transform = build_mtl_transform(image_size=image_size, mean=mean, std=std)
        self.tasks = self._available_tasks()
        self.samples = self._load_all_samples()
        self.length = max(len(samples) for samples in self.samples.values())

    def _available_tasks(self) -> tuple[str, ...]:
        if self.artery == "RCA":
            return "occlusion", "frame_quality", "dominance"
        if self.artery == "LCA":
            return "frame_quality", "dominance"
        raise ValueError("artery must be RCA or LCA.")

    def _csv_path(self, task: str) -> Path:
        return build_split_csv_path(
            self.split_root,
            task=FOLDER_TASKS[task],
            artery=self.artery,
            csv_filename=split_csv_filename(task, self.split, self.fold),
        )

    def _load_frame_samples(self, task: str) -> list[dict[str, Any]]:
        folder_task = FOLDER_TASKS[task]
        samples: list[dict[str, Any]] = []
        for row in load_split_rows(self._csv_path(task)):
            label = row["label"]
            samples.append(
                {
                    "image_path": build_image_path(
                        self.data_root,
                        task=folder_task,
                        artery=row["artery"],
                        label=label,
                        filename=row["filename"],
                    ),
                    "label": get_class_id(folder_task, label),
                }
            )
        if not samples:
            raise ValueError(f"No samples found for {task} {self.artery} {self.split}.")
        return samples

    def _load_occlusion_samples(self) -> list[dict[str, Any]]:
        grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in load_split_rows(self._csv_path("occlusion")):
            group_id = derive_video_id(row["filename"]) or row["study_id"]
            grouped_rows[group_id].append(row)

        samples: list[dict[str, Any]] = []
        for group_id, rows in sorted(grouped_rows.items()):
            labels = {row["label"] for row in rows}
            if len(labels) != 1:
                raise ValueError(
                    f"Occlusion group '{group_id}' has mixed labels: {', '.join(sorted(labels))}."
                )
            label = rows[0]["label"]
            frame_paths = [
                build_image_path(
                    self.data_root,
                    task="occlusion",
                    artery=row["artery"],
                    label=label,
                    filename=row["filename"],
                )
                for row in sorted(rows, key=frame_sort_key)
            ]
            samples.append(
                {
                    "frame_paths": frame_paths,
                    "label": get_class_id("occlusion", label),
                }
            )
        if not samples:
            raise ValueError(f"No occlusion clips found for {self.artery} {self.split}.")
        return samples

    def _load_all_samples(self) -> dict[str, list[dict[str, Any]]]:
        samples: dict[str, list[dict[str, Any]]] = {}
        for task in self.tasks:
            if task == "occlusion":
                samples[task] = self._load_occlusion_samples()
            else:
                samples[task] = self._load_frame_samples(task)

        if self.training:
            rng = random.Random(self.seed)
            for task_samples in samples.values():
                rng.shuffle(task_samples)
        return samples

    def __len__(self) -> int:
        return self.length

    def _sample_for_task(self, task: str, index: int) -> dict[str, Any]:
        task_samples = self.samples[task]
        return task_samples[index % len(task_samples)]

    def _load_occlusion_clip(self, sample: dict[str, Any]) -> Tensor:
        indices = evenly_spaced_indices(len(sample["frame_paths"]), self.clip_length)
        frames = [
            load_grayscale_image(sample["frame_paths"][frame_index], self.transform)
            for frame_index in indices
        ]
        return torch.stack(frames, dim=0)

    def __getitem__(self, index: int) -> tuple[dict[str, Tensor], dict[str, int]]:
        inputs: dict[str, Tensor] = {}
        targets: dict[str, int] = {}

        if "occlusion" in self.tasks:
            sample = self._sample_for_task("occlusion", index)
            inputs["occlusion_images"] = self._load_occlusion_clip(sample)
            targets["occlusion"] = sample["label"]

        if "frame_quality" in self.tasks:
            sample = self._sample_for_task("frame_quality", index)
            inputs["frame_quality_images"] = load_grayscale_image(sample["image_path"], self.transform)
            targets["frame_quality"] = sample["label"]

        if "dominance" in self.tasks:
            sample = self._sample_for_task("dominance", index)
            inputs["dominance_images"] = load_grayscale_image(sample["image_path"], self.transform)
            targets["dominance"] = sample["label"]

        return inputs, targets
