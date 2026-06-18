"""Run paper-style integrated inference for one RCA/LCA frame-folder pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.integrated_inference import (
    load_model_from_checkpoint,
    resolve_device,
    run_single_pair_integrated_inference,
    write_integrated_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run integrated RCA/LCA inference for a single frame-folder pair."
    )
    parser.add_argument("--input_mode", default="single_pair", choices=["single_pair"])
    parser.add_argument("--rca_frame_dir", required=True, type=Path)
    parser.add_argument("--lca_frame_dir", required=True, type=Path)
    parser.add_argument("--rca_checkpoint", required=True, type=Path)
    parser.add_argument("--lca_checkpoint", required=True, type=Path)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "mobilenet_v2", "densenet121"])
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip_length", type=int, default=15)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--mean", type=float, default=0.5485)
    parser.add_argument("--std", type=float, default=0.1407)
    parser.add_argument("--occlusion_threshold", type=float, default=0.5)
    parser.add_argument("--frame_quality_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.rca_checkpoint.is_file():
        raise FileNotFoundError(f"RCA checkpoint not found: {args.rca_checkpoint}")
    if not args.lca_checkpoint.is_file():
        raise FileNotFoundError(f"LCA checkpoint not found: {args.lca_checkpoint}")

    print(f"Loading RCA checkpoint: {args.rca_checkpoint}")
    rca_model = load_model_from_checkpoint(
        checkpoint_path=args.rca_checkpoint,
        backbone=args.backbone,
        device=device,
    )
    print(f"Loading LCA checkpoint: {args.lca_checkpoint}")
    lca_model = load_model_from_checkpoint(
        checkpoint_path=args.lca_checkpoint,
        backbone=args.backbone,
        device=device,
    )

    result = run_single_pair_integrated_inference(
        rca_frame_dir=args.rca_frame_dir,
        lca_frame_dir=args.lca_frame_dir,
        rca_model=rca_model,
        lca_model=lca_model,
        device=device,
        clip_length=args.clip_length,
        image_size=args.image_size,
        mean=args.mean,
        std=args.std,
        occlusion_threshold=args.occlusion_threshold,
        frame_quality_threshold=args.frame_quality_threshold,
        batch_size=args.batch_size,
    )
    final_json_path, frame_csv_path = write_integrated_outputs(result, args.output_dir)

    final_prediction = result.final_prediction
    print("Integrated inference complete.")
    print(f"Route used: {final_prediction['route_used']}")
    print(f"RCA occluded: {final_prediction['rca_occluded']}")
    print(f"Occlusion probability: {final_prediction['occlusion_probability']:.4f}")
    print(f"Final prediction: {final_prediction['pred_label']} ({final_prediction['final_confidence']:.4f})")
    print(f"Final JSON: {final_json_path}")
    print(f"Frame CSV: {frame_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
