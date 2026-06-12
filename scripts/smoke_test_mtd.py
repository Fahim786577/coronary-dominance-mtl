"""Smoke test Multi-Teacher Distillation components without training."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models import CoronaryTemporalMTL, SingleFrameTeacher, VideoTeacher
from src.training.distillation import DEFAULT_ALPHAS, DEFAULT_TEMPERATURE, multi_task_distillation_losses
from src.training.teacher_loading import freeze_teacher, load_teacher_bundle, run_teacher_bundle
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
    parser = argparse.ArgumentParser(description="Smoke test MTD utilities.")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "mobilenet_v2", "densenet121"])
    parser.add_argument("--pretrained", type=parse_bool, default=False)
    parser.add_argument("--teacher_root", type=Path, default=None)
    parser.add_argument("--artery", default="RCA")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint_name", default="best.pt")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--alpha_occlusion", type=float, default=DEFAULT_ALPHAS["occlusion"])
    parser.add_argument("--alpha_frame_quality", type=float, default=DEFAULT_ALPHAS["frame_quality"])
    parser.add_argument("--alpha_dominance", type=float, default=DEFAULT_ALPHAS["dominance"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--clip_length", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_batch(batch_size: int, clip_length: int, image_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "occlusion_images": torch.randn(batch_size, clip_length, 1, image_size, image_size, device=device),
        "frame_quality_images": torch.randn(batch_size, 1, image_size, image_size, device=device),
        "dominance_images": torch.randn(batch_size, 1, image_size, image_size, device=device),
    }


def make_targets(batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "occlusion": torch.randint(0, 2, (batch_size,), device=device),
        "frame_quality": torch.randint(0, 2, (batch_size,), device=device),
        "dominance": torch.randint(0, 2, (batch_size,), device=device),
    }


def make_alphas(args: argparse.Namespace) -> dict[str, float]:
    return {
        "occlusion": args.alpha_occlusion,
        "frame_quality": args.alpha_frame_quality,
        "dominance": args.alpha_dominance,
    }


def synthetic_teachers(backbone: str, pretrained: bool, device: torch.device) -> dict[str, nn.Module]:
    teachers: dict[str, nn.Module] = {
        "occlusion": VideoTeacher(backbone_name=backbone, pretrained=pretrained).to(device),
        "frame_quality": SingleFrameTeacher(backbone_name=backbone, pretrained=pretrained).to(device),
        "dominance": SingleFrameTeacher(backbone_name=backbone, pretrained=pretrained).to(device),
    }
    for teacher in teachers.values():
        freeze_teacher(teacher)
    return teachers


def teacher_grad_count(teachers: dict[str, nn.Module]) -> int:
    return sum(
        1
        for teacher in teachers.values()
        for parameter in teacher.parameters()
        if parameter.grad is not None
    )


def all_teachers_frozen(teachers: dict[str, nn.Module]) -> bool:
    return all(
        not parameter.requires_grad
        for teacher in teachers.values()
        for parameter in teacher.parameters()
    )


def run_mtd_pass(
    student: CoronaryTemporalMTL,
    teachers: dict[str, nn.Module],
    batch: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    alphas: dict[str, float],
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    student_outputs = student(batch)
    print("student forward succeeded:", {task: tuple(logits.shape) for task, logits in student_outputs.items()})

    teacher_outputs = run_teacher_bundle(teachers, batch)
    print("teacher forward succeeded:", {task: tuple(logits.shape) for task, logits in teacher_outputs.items()})

    total_loss, logs = multi_task_distillation_losses(
        student_outputs=student_outputs,
        teacher_outputs=teacher_outputs,
        targets=targets,
        alphas=alphas,
        temperature=temperature,
    )
    for task, components in logs["tasks"].items():
        print(
            f"{task}: CE={components['ce_loss']:.6f} "
            f"KL={components['kl_loss']:.6f} total={components['total_loss']:.6f}"
        )
    print(f"total MTD loss={logs['total_loss']:.6f}")
    return total_loss, logs


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    artery = normalize_artery(args.artery)
    alphas = make_alphas(args)

    student = CoronaryTemporalMTL(backbone_name=args.backbone, pretrained=args.pretrained).to(device)
    student.train()
    student.zero_grad(set_to_none=True)

    if args.teacher_root is None:
        print("Running synthetic MTD smoke test.")
        teachers = synthetic_teachers(args.backbone, args.pretrained, device)
    else:
        print("Running checkpoint-based MTD smoke test.")
        tasks = ["occlusion", "framequality", "dominance"] if artery == "RCA" else ["framequality", "dominance"]
        teachers = load_teacher_bundle(
            teacher_root=args.teacher_root,
            tasks=tasks,
            artery=artery,
            fold=args.fold,
            backbone=args.backbone,
            device=device,
            checkpoint_name=args.checkpoint_name,
        )

    if not teachers:
        raise RuntimeError("No teachers were available for the MTD smoke test.")
    print("teachers frozen:", all_teachers_frozen(teachers))

    batch = make_batch(args.batch_size, args.clip_length, args.image_size, device)
    targets = make_targets(args.batch_size, device)
    total_loss, _ = run_mtd_pass(
        student=student,
        teachers=teachers,
        batch=batch,
        targets=targets,
        alphas=alphas,
        temperature=args.temperature,
    )

    total_loss.backward()
    student_grad_count = sum(1 for parameter in student.parameters() if parameter.grad is not None)
    teacher_grads = teacher_grad_count(teachers)
    print("student backward succeeded:", student_grad_count > 0)
    print("teacher gradients produced:", teacher_grads)

    if student_grad_count == 0:
        raise RuntimeError("Student backward pass did not produce gradients.")
    if teacher_grads != 0:
        raise RuntimeError("Teacher gradients were produced during MTD smoke test.")
    if not all_teachers_frozen(teachers):
        raise RuntimeError("One or more teacher parameters are not frozen.")

    print("MTD smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
