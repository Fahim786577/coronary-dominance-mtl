"""Evaluate CoronaryTemporalMTL checkpoints on labeled split CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation import (
    EVALUATION_MODES,
    evaluate_checkpoint,
    resolve_evaluation_output_dir,
    resolve_mtl_checkpoint_path,
)
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
    parser = argparse.ArgumentParser(description="Evaluate a CoronaryTemporalMTL checkpoint.")
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--split_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--checkpoint_root", type=Path, default=Path("outputs/mtl"))
    parser.add_argument("--artery", required=True)
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--backbone", required=True, choices=["resnet18", "mobilenet_v2", "densenet121"])
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--mode", default="baseline", choices=EVALUATION_MODES)
    parser.add_argument("--checkpoint_name", default="best.pt")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--clip_length", type=int, default=15)
    parser.add_argument("--mean", type=float, default=0.5485)
    parser.add_argument("--std", type=float, default=0.1407)
    parser.add_argument("--pretrained", type=parse_bool, default=False)
    parser.add_argument("--allow_metadata_mismatch", type=parse_bool, default=False)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    """Resolve auto/cpu/cuda device strings."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main() -> int:
    args = parse_args()
    artery = normalize_artery(args.artery)
    device = resolve_device(args.device)

    if args.pretrained:
        print("Note: --pretrained is accepted for CLI consistency, but evaluation reconstructs checkpoints with pretrained=False.")

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = resolve_mtl_checkpoint_path(
            checkpoint_root=args.checkpoint_root,
            mode=args.mode,
            artery=artery,
            fold=args.fold,
            backbone=args.backbone,
            checkpoint_name=args.checkpoint_name,
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    evaluation_dir = resolve_evaluation_output_dir(
        output_root=args.output_dir,
        mode=args.mode,
        artery=artery,
        fold=args.fold,
        backbone=args.backbone,
        split=args.split,
    )

    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Output directory: {evaluation_dir}")

    report, prediction_rows = evaluate_checkpoint(
        data_root=args.data_root,
        split_root=args.split_root,
        checkpoint_path=checkpoint_path,
        output_dir=evaluation_dir,
        mode=args.mode,
        artery=artery,
        fold=args.fold,
        split=args.split,
        backbone=args.backbone,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        clip_length=args.clip_length,
        mean=args.mean,
        std=args.std,
        allow_metadata_mismatch=args.allow_metadata_mismatch,
        write_metrics=True,
        write_predictions=True,
    )

    print("Evaluation complete.")
    print(f"Active tasks: {', '.join(report['active_tasks'])}")
    print(f"Prediction rows: {len(prediction_rows)}")
    print(f"Mean task accuracy: {report['overall']['mean_task_accuracy']:.4f}")
    print(f"Mean task balanced accuracy: {report['overall']['mean_task_balanced_accuracy']:.4f}")
    print(f"Metrics JSON: {evaluation_dir / 'metrics.json'}")
    print(f"Metrics CSV: {evaluation_dir / 'metrics.csv'}")
    print(f"Predictions CSV: {evaluation_dir / 'predictions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
