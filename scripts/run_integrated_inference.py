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
    run_multi_sequence_study_integrated_inference,
    run_single_pair_integrated_inference,
    write_integrated_outputs,
    write_multi_sequence_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated RCA/LCA inference.")
    parser.add_argument("--input_mode", default="single_pair", choices=["single_pair", "multi_sequence_study"])
    parser.add_argument("--rca_frame_dir", type=Path)
    parser.add_argument("--lca_frame_dir", type=Path)
    parser.add_argument("--rca_study_dir", type=Path)
    parser.add_argument("--lca_study_dir", type=Path)
    parser.add_argument("--sequence_pair_policy", default="trim_to_min", choices=["trim_to_min", "strict_equal"])
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


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate mode-specific required arguments."""
    if args.input_mode == "single_pair":
        missing = [
            name
            for name in ("rca_frame_dir", "lca_frame_dir")
            if getattr(args, name) is None
        ]
        if missing:
            missing_text = ", ".join(f"--{name}" for name in missing)
            raise ValueError(f"--input_mode single_pair requires: {missing_text}")
    elif args.input_mode == "multi_sequence_study":
        missing = [
            name
            for name in ("rca_study_dir", "lca_study_dir")
            if getattr(args, name) is None
        ]
        if missing:
            missing_text = ", ".join(f"--{name}" for name in missing)
            raise ValueError(f"--input_mode multi_sequence_study requires: {missing_text}")
    return args


def main() -> int:
    args = validate_args(parse_args())
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

    if args.input_mode == "single_pair":
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
    else:
        result = run_multi_sequence_study_integrated_inference(
            rca_study_dir=args.rca_study_dir,
            lca_study_dir=args.lca_study_dir,
            rca_model=rca_model,
            lca_model=lca_model,
            device=device,
            sequence_pair_policy=args.sequence_pair_policy,
            clip_length=args.clip_length,
            image_size=args.image_size,
            mean=args.mean,
            std=args.std,
            occlusion_threshold=args.occlusion_threshold,
            frame_quality_threshold=args.frame_quality_threshold,
            batch_size=args.batch_size,
        )
        final_json_path, pair_csv_path, pair_frame_csv_path, pairing_report_csv_path = (
            write_multi_sequence_outputs(result, args.output_dir)
        )

        final_prediction = result.study_final_prediction
        print("Multi-sequence integrated inference complete.")
        print(f"Sequence pairs: {final_prediction['num_sequence_pairs']}")
        print(f"Valid pairs: {final_prediction['num_valid_pairs']}")
        print(f"Final prediction: {final_prediction['pred_label']} ({final_prediction['final_confidence']:.4f})")
        print(f"Study final JSON: {final_json_path}")
        print(f"Pair CSV: {pair_csv_path}")
        print(f"Pair frame CSV: {pair_frame_csv_path}")
        print(f"Pairing report CSV: {pairing_report_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
