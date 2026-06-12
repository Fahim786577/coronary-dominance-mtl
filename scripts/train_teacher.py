"""Train one independent teacher model from CSV splits."""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.coronary_dataset import CoronaryDataset
from src.data.label_maps import LABEL_MAPS, get_class_id
from src.models.teachers import SingleFrameTeacher, VideoTeacher
from src.training.teacher_trainer import TeacherTrainer, TrainerConfig
from src.utils.paths import build_image_path, build_split_csv_path, normalize_artery


SPLIT_PREFIXES = {
    "dominance": "dom",
    "framequality": "framequality",
    "occlusion": "occlusion",
}
VALID_TASK_ARTERIES = {
    "dominance": {"RCA", "LCA"},
    "framequality": {"RCA", "LCA"},
    "occlusion": {"RCA"},
}
FRAME_NUMBER_PATTERN = re.compile(r"_frame_(\d+)(?=\.[^.]+$)", re.IGNORECASE)


def parse_bool(value: str | bool) -> bool:
    """Parse CLI booleans such as true/false, yes/no, and 1/0."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def csv_filename(task: str, split: str, fold: int) -> str:
    return f"{SPLIT_PREFIXES[task]}_{split}_labels_fold_{fold}.csv"


def split_csv_path(split_root: str | Path, task: str, artery: str, split: str, fold: int) -> Path:
    return build_split_csv_path(
        split_root,
        task=task,
        artery=artery,
        csv_filename=csv_filename(task, split, fold),
    )


def build_image_transform(image_size: int, mean: float, std: float) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )


def derive_video_id(filename: str) -> str:
    """Derive a stable video ID by removing the trailing frame suffix when present."""
    return FRAME_NUMBER_PATTERN.sub("", Path(filename).stem)


def frame_sort_key(row: dict[str, Any]) -> tuple[int, int | str]:
    match = FRAME_NUMBER_PATTERN.search(row["filename"])
    if match is not None:
        return 0, int(match.group(1))
    return 1, row["filename"]


def evenly_spaced_indices(length: int, clip_length: int) -> list[int]:
    if length <= 0:
        raise ValueError("Cannot sample from an empty frame group.")
    if length == clip_length:
        return list(range(length))
    if length < clip_length:
        return list(range(length)) + [length - 1] * (clip_length - length)
    if clip_length == 1:
        return [0]
    return [round(i * (length - 1) / (clip_length - 1)) for i in range(clip_length)]


class OcclusionClipDataset(Dataset):
    """CSV-backed occlusion dataset returning one clip per video/study."""

    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        clip_length: int,
        transform: transforms.Compose,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.clip_length = clip_length
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self) -> list[dict[str, Any]]:
        grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            required = {"filename", "label", "study_id", "artery", "split", "task", "fold"}
            missing = sorted(required.difference(reader.fieldnames or []))
            if missing:
                raise ValueError(f"{self.csv_path} is missing required columns: {', '.join(missing)}")

            for row in reader:
                if row["task"] != "occlusion":
                    raise ValueError(f"{self.csv_path} contains non-occlusion task '{row['task']}'.")
                group_id = derive_video_id(row["filename"]) or row["study_id"]
                grouped_rows[group_id].append(row)

        samples: list[dict[str, Any]] = []
        for group_id, rows in sorted(grouped_rows.items()):
            labels = {row["label"] for row in rows}
            if len(labels) != 1:
                raise ValueError(
                    f"{self.csv_path} group '{group_id}' has mixed labels: {', '.join(sorted(labels))}."
                )
            rows = sorted(rows, key=frame_sort_key)
            label = rows[0]["label"]
            class_id = get_class_id("occlusion", label)
            frame_paths = [
                build_image_path(
                    self.data_root,
                    task="occlusion",
                    artery=row["artery"],
                    label=label,
                    filename=row["filename"],
                )
                for row in rows
            ]
            samples.append(
                {
                    "group_id": group_id,
                    "frame_paths": frame_paths,
                    "label": class_id,
                }
            )

        if not samples:
            raise ValueError(f"{self.csv_path} did not contain any occlusion clips.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        sample = self.samples[index]
        indices = evenly_spaced_indices(len(sample["frame_paths"]), self.clip_length)
        frames: list[Tensor] = []

        for frame_index in indices:
            image_path = sample["frame_paths"][frame_index]
            with Image.open(image_path) as image:
                image = image.convert("L")
                frames.append(self.transform(image))

        return torch.stack(frames, dim=0), sample["label"]


def validate_task_artery(task: str, artery: str) -> str:
    artery = normalize_artery(artery)
    valid_arteries = VALID_TASK_ARTERIES[task]
    if artery not in valid_arteries:
        valid = ", ".join(sorted(valid_arteries))
        raise ValueError(f"Task '{task}' supports artery values: {valid}. Got '{artery}'.")
    return artery


def default_values(task: str) -> dict[str, float | int | str]:
    if task == "occlusion":
        return {
            "batch_size": 8,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "mean": 0.5461,
            "std": 0.1453,
            "scheduler_patience": 3,
            "scheduler_factor": 0.2,
            "early_stopping_patience": 100,
            "clip_length": 15,
        }
    return {
        "batch_size": 16,
        "lr": 1e-3,
        "weight_decay": 0.01,
        "mean": 0.5504,
        "std": 0.1613,
        "scheduler_patience": 5,
        "scheduler_factor": 0.3,
        "early_stopping_patience": 30,
        "clip_length": 15,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one independent coronary teacher model.")
    parser.add_argument("--task", required=True, choices=sorted(LABEL_MAPS))
    parser.add_argument("--artery", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--backbone", required=True, choices=["resnet18", "mobilenet_v2", "densenet121"])
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=None)
    parser.add_argument("--pretrained", type=parse_bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mean", type=float, default=None)
    parser.add_argument("--std", type=float, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--scheduler_patience", type=int, default=None)
    parser.add_argument("--scheduler_factor", type=float, default=None)
    parser.add_argument("--gradient_clip_max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def apply_task_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = default_values(args.task)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def create_datasets(args: argparse.Namespace) -> tuple[Dataset, Dataset]:
    transform = build_image_transform(args.image_size, args.mean, args.std)
    train_csv = split_csv_path(args.split_root, args.task, args.artery, "train", args.fold)
    val_csv = split_csv_path(args.split_root, args.task, args.artery, "val", args.fold)

    if args.task == "occlusion":
        return (
            OcclusionClipDataset(train_csv, args.data_root, args.clip_length, transform),
            OcclusionClipDataset(val_csv, args.data_root, args.clip_length, transform),
        )

    return (
        CoronaryDataset(train_csv, args.data_root, transform=transform),
        CoronaryDataset(val_csv, args.data_root, transform=transform),
    )


def create_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.task == "occlusion":
        return VideoTeacher(backbone_name=args.backbone, pretrained=args.pretrained)
    return SingleFrameTeacher(backbone_name=args.backbone, pretrained=args.pretrained)


def create_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    if args.task == "occlusion":
        return torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["split_root"] = str(args.split_root)
    config["output_dir"] = str(args.output_dir)
    return config


def main() -> int:
    args = apply_task_defaults(parse_args())
    args.artery = validate_task_artery(args.task, args.artery)
    set_seed(args.seed)

    device = resolve_device(args.device)
    train_dataset, val_dataset = create_datasets(args)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(args).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = create_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    checkpoint_dir = (
        args.output_dir
        / args.task
        / f"DATA_{args.artery}"
        / f"fold_{args.fold}"
        / args.backbone
    )
    metadata = {
        "task": args.task,
        "artery": args.artery,
        "fold": args.fold,
        "backbone": args.backbone,
        "label_map": dict(LABEL_MAPS[args.task]),
    }
    trainer = TeacherTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=checkpoint_dir,
        config=TrainerConfig(
            max_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_max_norm=args.gradient_clip_max_norm,
        ),
        checkpoint_metadata=metadata,
    )

    print(f"Training {args.task} teacher on {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    trainer.fit(config_dict(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
