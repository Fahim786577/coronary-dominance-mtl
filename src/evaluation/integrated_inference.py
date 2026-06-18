"""Single-pair integrated inference for the coronary dominance MTL system."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from src.data.mtl_dataset import build_mtl_transform, evenly_spaced_indices, load_grayscale_image
from src.models import CoronaryTemporalMTL
from src.models.backbones import normalize_backbone_name
from src.training.checkpointing import load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FRAME_NUMBER_PATTERN = re.compile(r"_frame_(\d+)(?=\.[^.]+$|$)", re.IGNORECASE)

FRAME_PREDICTION_COLUMNS = (
    "artery",
    "frame_path",
    "frame_filename",
    "frame_quality_pred",
    "frame_quality_prob_informative",
    "used_for_dominance",
    "dominance_pred",
    "dominance_prob_rightdom",
    "dominance_prob_leftdom",
)

DOMINANCE_LABELS = {0: "rightdom", 1: "leftdom"}
FRAME_QUALITY_LABELS = {0: "noninformative", 1: "informative"}


@dataclass
class FrameRecord:
    """One loaded frame and its metadata."""

    path: Path
    tensor: Tensor


@dataclass
class FrameQualityResult:
    """Frame-quality prediction for one frame."""

    pred_class: int
    prob_informative: float

    @property
    def pred_label(self) -> str:
        return FRAME_QUALITY_LABELS[self.pred_class]


@dataclass
class DominanceResult:
    """Dominance prediction for one retained frame."""

    pred_class: int
    prob_rightdom: float
    prob_leftdom: float

    @property
    def pred_label(self) -> str:
        return DOMINANCE_LABELS[self.pred_class]


@dataclass
class IntegratedInferenceResult:
    """Final single-pair inference result and frame-level rows."""

    final_prediction: dict[str, Any]
    frame_rows: list[dict[str, Any]]


def resolve_device(device_arg: str) -> torch.device:
    """Resolve auto/cpu/cuda device strings."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    backbone: str = "resnet18",
    device: torch.device | str = "cpu",
) -> CoronaryTemporalMTL:
    """Load a CoronaryTemporalMTL checkpoint for inference."""
    resolved_device = torch.device(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=resolved_device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain 'model_state_dict'.")

    model = CoronaryTemporalMTL(
        backbone_name=normalize_backbone_name(backbone),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model


def _frame_sort_key(path: Path) -> tuple[int, int | str]:
    match = FRAME_NUMBER_PATTERN.search(path.name)
    if match is not None:
        return 0, int(match.group(1))
    return 1, path.name


def list_frame_paths(frame_dir: str | Path) -> list[Path]:
    """List image frames from a directory in deterministic frame order."""
    directory = Path(frame_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Frame directory not found: {directory}")

    paths = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ValueError(f"No image frames found in {directory}.")
    return sorted(paths, key=_frame_sort_key)


def load_frame_records(
    frame_dir: str | Path,
    image_size: int = 512,
    mean: float = 0.5485,
    std: float = 0.1407,
) -> list[FrameRecord]:
    """Load all frames in grayscale with the current MTL inference transform."""
    transform = build_mtl_transform(image_size=image_size, mean=mean, std=std)
    return [
        FrameRecord(path=path, tensor=load_grayscale_image(path, transform))
        for path in list_frame_paths(frame_dir)
    ]


def _batch_tensors(tensors: list[Tensor], batch_size: int) -> list[Tensor]:
    return [
        torch.stack(tensors[index : index + batch_size], dim=0)
        for index in range(0, len(tensors), batch_size)
    ]


def detect_rca_occlusion(
    model: nn.Module,
    rca_frames: list[FrameRecord],
    device: torch.device,
    clip_length: int,
) -> tuple[float, list[int]]:
    """Run RCA occlusion detection using one deterministic evenly spaced clip."""
    if not rca_frames:
        raise ValueError("RCA occlusion detection requires at least one RCA frame.")

    indices = evenly_spaced_indices(len(rca_frames), clip_length)
    clip = torch.stack([rca_frames[index].tensor for index in indices], dim=0).unsqueeze(0)
    clip = clip.to(device, non_blocking=True)

    with torch.no_grad():
        outputs = model({"occlusion_images": clip})
        if "occlusion" not in outputs:
            raise RuntimeError("RCA model did not return an occlusion output.")
        probabilities = torch.softmax(outputs["occlusion"], dim=1)
        occluded_probability = float(probabilities[0, 1].detach().cpu())

    return occluded_probability, indices


def run_frame_quality(
    model: nn.Module,
    frames: list[FrameRecord],
    device: torch.device,
    batch_size: int = 16,
) -> list[FrameQualityResult]:
    """Run frame-quality inference on single frames."""
    results: list[FrameQualityResult] = []
    with torch.no_grad():
        for batch in _batch_tensors([frame.tensor for frame in frames], batch_size):
            batch = batch.to(device, non_blocking=True)
            outputs = model({"frame_quality_images": batch})
            if "frame_quality" not in outputs:
                raise RuntimeError("Model did not return a frame_quality output.")
            probabilities = torch.softmax(outputs["frame_quality"], dim=1)
            predictions = torch.argmax(outputs["frame_quality"], dim=1)
            for pred_class, probs in zip(predictions.detach().cpu().tolist(), probabilities.detach().cpu().tolist()):
                results.append(
                    FrameQualityResult(
                        pred_class=int(pred_class),
                        prob_informative=float(probs[1]),
                    )
                )
    return results


def run_dominance(
    model: nn.Module,
    frames: list[FrameRecord],
    device: torch.device,
    batch_size: int = 16,
) -> list[DominanceResult]:
    """Run frame-level dominance inference on retained frames."""
    results: list[DominanceResult] = []
    with torch.no_grad():
        for batch in _batch_tensors([frame.tensor for frame in frames], batch_size):
            batch = batch.to(device, non_blocking=True)
            outputs = model({"dominance_images": batch})
            if "dominance" not in outputs:
                raise RuntimeError("Model did not return a dominance output.")
            probabilities = torch.softmax(outputs["dominance"], dim=1)
            predictions = torch.argmax(outputs["dominance"], dim=1)
            for pred_class, probs in zip(predictions.detach().cpu().tolist(), probabilities.detach().cpu().tolist()):
                results.append(
                    DominanceResult(
                        pred_class=int(pred_class),
                        prob_rightdom=float(probs[0]),
                        prob_leftdom=float(probs[1]),
                    )
                )
    return results


def majority_vote(dominance_results: list[DominanceResult]) -> tuple[int, float, dict[str, Any]]:
    """Vote final dominance with deterministic tie-breaking."""
    if not dominance_results:
        raise ValueError("Cannot majority vote without dominance predictions.")

    vote_rightdom = sum(result.pred_class == 0 for result in dominance_results)
    vote_leftdom = sum(result.pred_class == 1 for result in dominance_results)
    mean_prob_rightdom = sum(result.prob_rightdom for result in dominance_results) / len(dominance_results)
    mean_prob_leftdom = sum(result.prob_leftdom for result in dominance_results) / len(dominance_results)

    if vote_rightdom > vote_leftdom:
        pred_class = 0
    elif vote_leftdom > vote_rightdom:
        pred_class = 1
    elif mean_prob_leftdom > mean_prob_rightdom:
        pred_class = 1
    else:
        pred_class = 0

    final_confidence = mean_prob_rightdom if pred_class == 0 else mean_prob_leftdom
    details = {
        "vote_rightdom": int(vote_rightdom),
        "vote_leftdom": int(vote_leftdom),
        "mean_prob_rightdom": float(mean_prob_rightdom),
        "mean_prob_leftdom": float(mean_prob_leftdom),
    }
    return pred_class, float(final_confidence), details


def run_single_pair_integrated_inference(
    rca_frame_dir: str | Path,
    lca_frame_dir: str | Path,
    rca_model: nn.Module,
    lca_model: nn.Module,
    device: torch.device,
    clip_length: int = 15,
    image_size: int = 512,
    mean: float = 0.5485,
    std: float = 0.1407,
    occlusion_threshold: float = 0.5,
    frame_quality_threshold: float = 0.5,
    batch_size: int = 16,
) -> IntegratedInferenceResult:
    """Run the paper-style integrated inference pathway for one RCA/LCA pair."""
    rca_frames = load_frame_records(rca_frame_dir, image_size=image_size, mean=mean, std=std)
    lca_frames = load_frame_records(lca_frame_dir, image_size=image_size, mean=mean, std=std)

    occlusion_probability, occlusion_indices = detect_rca_occlusion(
        model=rca_model,
        rca_frames=rca_frames,
        device=device,
        clip_length=clip_length,
    )
    rca_occluded = occlusion_probability >= occlusion_threshold

    if rca_occluded:
        route_used = "LCA"
        selected_frames = lca_frames
        selected_model = lca_model
    else:
        route_used = "RCA"
        selected_frames = rca_frames
        selected_model = rca_model

    frame_quality_results = run_frame_quality(
        model=selected_model,
        frames=selected_frames,
        device=device,
        batch_size=batch_size,
    )
    informative_indices = [
        index
        for index, result in enumerate(frame_quality_results)
        if result.prob_informative >= frame_quality_threshold
    ]
    fallback_all_frames = len(informative_indices) == 0
    retained_indices = list(range(len(selected_frames))) if fallback_all_frames else informative_indices
    retained_frames = [selected_frames[index] for index in retained_indices]

    dominance_results = run_dominance(
        model=selected_model,
        frames=retained_frames,
        device=device,
        batch_size=batch_size,
    )
    pred_class, final_confidence, vote_details = majority_vote(dominance_results)
    pred_label = DOMINANCE_LABELS[pred_class]

    dominance_by_selected_index = {
        selected_index: result
        for selected_index, result in zip(retained_indices, dominance_results)
    }
    frame_rows: list[dict[str, Any]] = []
    for index, (frame, quality_result) in enumerate(zip(selected_frames, frame_quality_results)):
        dominance_result = dominance_by_selected_index.get(index)
        frame_rows.append(
            {
                "artery": route_used,
                "frame_path": str(frame.path),
                "frame_filename": frame.path.name,
                "frame_quality_pred": quality_result.pred_label,
                "frame_quality_prob_informative": quality_result.prob_informative,
                "used_for_dominance": dominance_result is not None,
                "dominance_pred": dominance_result.pred_label if dominance_result is not None else "",
                "dominance_prob_rightdom": (
                    dominance_result.prob_rightdom if dominance_result is not None else ""
                ),
                "dominance_prob_leftdom": (
                    dominance_result.prob_leftdom if dominance_result is not None else ""
                ),
            }
        )

    final_prediction = {
        "route_used": route_used,
        "rca_occluded": bool(rca_occluded),
        "occlusion_probability": float(occlusion_probability),
        "occlusion_clip_frame_indices": [int(index) for index in occlusion_indices],
        "num_rca_frames": len(rca_frames),
        "num_lca_frames": len(lca_frames),
        "num_selected_frames": len(selected_frames),
        "num_informative_frames": len(informative_indices),
        "fallback_all_frames": bool(fallback_all_frames),
        **vote_details,
        "pred_class": int(pred_class),
        "pred_label": pred_label,
        "final_confidence": float(final_confidence),
    }
    return IntegratedInferenceResult(final_prediction=final_prediction, frame_rows=frame_rows)


def write_integrated_outputs(
    result: IntegratedInferenceResult,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write final_prediction.json and frame_predictions.csv."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_json_path = output_path / "final_prediction.json"
    with final_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(result.final_prediction, json_file, indent=2)

    frame_csv_path = output_path / "frame_predictions.csv"
    with frame_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FRAME_PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.frame_rows:
            writer.writerow({column: row.get(column, "") for column in FRAME_PREDICTION_COLUMNS})

    return final_json_path, frame_csv_path
