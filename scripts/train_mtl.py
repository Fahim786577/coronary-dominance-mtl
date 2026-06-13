"""Train baseline or MTD CoronaryTemporalMTL student models."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mtl_dataset import CoronaryMTLDataset
from src.models import CoronaryTemporalMTL
from src.training.mtl_trainer import MTLTrainer, MTLTrainerConfig
from src.training.teacher_loading import load_teacher_bundle, resolve_teacher_checkpoint_path
from src.utils.paths import normalize_artery


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline supervised MTL student model.")
    parser.add_argument("--artery", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--backbone", required=True, choices=["resnet18", "mobilenet_v2", "densenet121"])
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=15)
    parser.add_argument("--pretrained", type=parse_bool, default=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mean", type=float, default=0.5485)
    parser.add_argument("--std", type=float, default=0.1407)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--scheduler_patience", type=int, default=5)
    parser.add_argument("--scheduler_factor", type=float, default=0.2)
    parser.add_argument("--gradient_clip_max_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task_weight_occlusion", type=float, default=1.0)
    parser.add_argument("--task_weight_frame_quality", type=float, default=1.0)
    parser.add_argument("--task_weight_dominance", type=float, default=1.0)
    parser.add_argument("--use_mtd", type=parse_bool, default=False)
    parser.add_argument("--use_twophase", type=parse_bool, default=False)
    parser.add_argument("--teacher_root", type=Path, default=Path("outputs/teachers"))
    parser.add_argument("--teacher_checkpoint_name", default="best.pt")
    parser.add_argument("--mtd_temperature", type=float, default=4.0)
    parser.add_argument("--mtd_alpha_occlusion", type=float, default=0.1)
    parser.add_argument("--mtd_alpha_frame_quality", type=float, default=0.1)
    parser.add_argument("--mtd_alpha_dominance", type=float, default=0.1)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    args.artery = normalize_artery(args.artery)
    if args.artery not in {"RCA", "LCA"}:
        raise ValueError(f"artery must be RCA or LCA, got '{args.artery}'.")
    if args.use_twophase:
        raise NotImplementedError("TwoPhase training is Step 6 and is not implemented in Step 5B.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def config_dict(args: argparse.Namespace) -> dict[str, Any]:
    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["split_root"] = str(args.split_root)
    config["output_dir"] = str(args.output_dir)
    config["teacher_root"] = str(args.teacher_root)
    return config


def mtd_alphas(args: argparse.Namespace) -> dict[str, float]:
    return {
        "occlusion": args.mtd_alpha_occlusion,
        "frame_quality": args.mtd_alpha_frame_quality,
        "dominance": args.mtd_alpha_dominance,
    }


def load_mtd_teachers(
    args: argparse.Namespace,
    tasks: tuple[str, ...],
    device: torch.device,
) -> dict[str, torch.nn.Module]:
    """Load required MTD teachers, raising a task-specific path error if missing."""
    for task in tasks:
        checkpoint_path = resolve_teacher_checkpoint_path(
            teacher_root=args.teacher_root,
            task=task,
            artery=args.artery,
            fold=args.fold,
            backbone=args.backbone,
            checkpoint_name=args.teacher_checkpoint_name,
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Missing teacher checkpoint for "
                f"task='{task}', artery='{args.artery}', fold={args.fold}, "
                f"backbone='{args.backbone}'. Expected: {checkpoint_path}"
            )

    return load_teacher_bundle(
        teacher_root=args.teacher_root,
        tasks=tasks,
        artery=args.artery,
        fold=args.fold,
        backbone=args.backbone,
        device=device,
        checkpoint_name=args.teacher_checkpoint_name,
    )


def main() -> int:
    args = validate_args(parse_args())
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_dataset = CoronaryMTLDataset(
        data_root=args.data_root,
        split_root=args.split_root,
        artery=args.artery,
        split="train",
        fold=args.fold,
        image_size=args.image_size,
        clip_length=args.clip_length,
        mean=args.mean,
        std=args.std,
        seed=args.seed,
        training=True,
    )
    val_dataset = CoronaryMTLDataset(
        data_root=args.data_root,
        split_root=args.split_root,
        artery=args.artery,
        split="val",
        fold=args.fold,
        image_size=args.image_size,
        clip_length=args.clip_length,
        mean=args.mean,
        std=args.std,
        seed=args.seed,
        training=False,
    )

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

    model = CoronaryTemporalMTL(
        backbone_name=args.backbone,
        pretrained=args.pretrained,
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )
    task_weights = {
        "occlusion": args.task_weight_occlusion,
        "frame_quality": args.task_weight_frame_quality,
        "dominance": args.task_weight_dominance,
    }
    alphas = mtd_alphas(args)
    teachers = load_mtd_teachers(args, train_dataset.tasks, device) if args.use_mtd else None

    checkpoint_dir = (
        args.output_dir
        / ("mtd" if args.use_mtd else "baseline")
        / f"DATA_{args.artery}"
        / f"fold_{args.fold}"
        / args.backbone
    )
    trainer = MTLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=checkpoint_dir,
        config=MTLTrainerConfig(
            max_epochs=args.epochs,
            early_stopping_patience=args.early_stopping_patience,
            gradient_clip_max_norm=args.gradient_clip_max_norm,
            task_weights=task_weights,
            use_mtd=args.use_mtd,
            mtd_temperature=args.mtd_temperature,
            mtd_alphas=alphas,
        ),
        checkpoint_metadata={
            "artery": args.artery,
            "fold": args.fold,
            "backbone": args.backbone,
            "use_mtd": args.use_mtd,
            **(
                {
                    "temperature": args.mtd_temperature,
                    "mtd_alphas": alphas,
                    "teacher_root": str(args.teacher_root),
                    "teacher_checkpoint_name": args.teacher_checkpoint_name,
                }
                if args.use_mtd
                else {}
            ),
        },
        teachers=teachers,
    )

    mode_label = "MTL + MTD" if args.use_mtd else "baseline MTL"
    print(f"Training {mode_label} on {device}")
    print(f"Artery: {args.artery} | Tasks: {', '.join(train_dataset.tasks)}")
    if args.use_mtd:
        print(f"Teacher root: {args.teacher_root}")
        print(f"Teacher checkpoint: {args.teacher_checkpoint_name}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    trainer.fit(config_dict(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
