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

from src.data.label_maps import get_class_id
from src.data.mtl_dataset import build_mtl_transform, evenly_spaced_indices, load_grayscale_image
from src.evaluation.metrics import binary_classification_metrics
from src.models import CoronaryTemporalMTL
from src.models.backbones import normalize_backbone_name
from src.training.checkpointing import load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
FRAME_NUMBER_PATTERN = re.compile(r"_frame_(\d+)(?=\.[^.]+$|$)", re.IGNORECASE)
NATURAL_SORT_PATTERN = re.compile(r"(\d+)")

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
PAIR_PREDICTION_COLUMNS = (
    "pair_index",
    "rca_sequence_dir",
    "lca_sequence_dir",
    "status",
    "reason",
    "route_used",
    "rca_occluded",
    "occlusion_probability",
    "num_rca_frames",
    "num_lca_frames",
    "num_selected_frames",
    "num_informative_frames",
    "fallback_all_frames",
    "vote_rightdom",
    "vote_leftdom",
    "mean_prob_rightdom",
    "mean_prob_leftdom",
    "pred_class",
    "pred_label",
    "final_confidence",
)
PAIR_FRAME_PREDICTION_COLUMNS = ("pair_index", *FRAME_PREDICTION_COLUMNS)
SEQUENCE_PAIRING_REPORT_COLUMNS = (
    "sequence_type",
    "sequence_index",
    "sequence_dir",
    "paired",
    "pair_index",
    "status",
    "reason",
)
INTEGRATED_PREDICTION_COLUMNS = (
    "study_id",
    "label",
    "true_class",
    "pred_class",
    "pred_label",
    "correct",
    "subset",
    "status",
    "reason",
    "num_rca_sequences",
    "num_lca_sequences",
    "num_sequence_pairs",
    "num_valid_pairs",
    "sequence_pair_policy",
    "num_ignored_rca_sequences",
    "num_ignored_lca_sequences",
    "vote_rightdom",
    "vote_leftdom",
    "mean_prob_rightdom",
    "mean_prob_leftdom",
    "final_confidence",
)
ALL_PAIR_PREDICTION_COLUMNS = (
    "study_id",
    "subset",
    "pair_index",
    "rca_sequence_dir",
    "lca_sequence_dir",
    "status",
    "reason",
    "route_used",
    "rca_occluded",
    "occlusion_probability",
    "pred_class",
    "pred_label",
    "final_confidence",
)
ALL_PAIR_FRAME_PREDICTION_COLUMNS = ("study_id", "subset", *PAIR_FRAME_PREDICTION_COLUMNS)
ALL_SEQUENCE_PAIRING_REPORT_COLUMNS = ("study_id", "subset", *SEQUENCE_PAIRING_REPORT_COLUMNS)
METRIC_COLUMNS = (
    "accuracy",
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
    "confusion_matrix",
)
MANIFEST_REQUIRED_COLUMNS = ("study_id", "label", "rca_study_dir", "lca_study_dir")

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


@dataclass
class MultiSequenceInferenceResult:
    """Final multi-sequence study result and output table rows."""

    study_final_prediction: dict[str, Any]
    pair_rows: list[dict[str, Any]]
    pair_frame_rows: list[dict[str, Any]]
    pairing_report_rows: list[dict[str, Any]]


@dataclass
class ManifestCohortInferenceResult:
    """Cohort-level integrated inference result and combined output rows."""

    prediction_rows: list[dict[str, Any]]
    metrics: dict[str, Any]
    subset_metrics: dict[str, dict[str, Any]]
    all_pair_rows: list[dict[str, Any]]
    all_pair_frame_rows: list[dict[str, Any]]
    all_pairing_report_rows: list[dict[str, Any]]
    has_subset: bool


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


def natural_sort_key(path: Path) -> list[int | str]:
    """Return a deterministic natural/alphanumeric sort key for paths."""
    parts = NATURAL_SORT_PATTERN.split(path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def discover_sequence_dirs(study_dir: str | Path) -> list[Path]:
    """Discover immediate sequence subdirectories in natural sort order."""
    directory = Path(study_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Study sequence directory not found: {directory}")
    return sorted([path for path in directory.iterdir() if path.is_dir()], key=natural_sort_key)


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


def has_valid_image_files(frame_dir: str | Path) -> bool:
    """Return whether a sequence directory contains at least one valid image file."""
    directory = Path(frame_dir)
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        for path in directory.iterdir()
    )


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


def pair_sequence_dirs(
    rca_study_dir: str | Path,
    lca_study_dir: str | Path,
    sequence_pair_policy: str = "trim_to_min",
) -> tuple[list[tuple[int, Path, Path]], list[dict[str, Any]], dict[str, int]]:
    """Pair RCA/LCA sequence directories deterministically."""
    if sequence_pair_policy not in {"trim_to_min", "strict_equal"}:
        raise ValueError("sequence_pair_policy must be 'trim_to_min' or 'strict_equal'.")

    rca_sequences = discover_sequence_dirs(rca_study_dir)
    lca_sequences = discover_sequence_dirs(lca_study_dir)
    num_rca_sequences = len(rca_sequences)
    num_lca_sequences = len(lca_sequences)

    if sequence_pair_policy == "strict_equal" and num_rca_sequences != num_lca_sequences:
        raise ValueError(
            "strict_equal sequence pairing requires equal sequence counts: "
            f"RCA={num_rca_sequences}, LCA={num_lca_sequences}."
        )

    num_pairs = min(num_rca_sequences, num_lca_sequences)
    pairs = [
        (pair_index, rca_sequences[pair_index], lca_sequences[pair_index])
        for pair_index in range(num_pairs)
    ]

    report_rows: list[dict[str, Any]] = []
    for pair_index, rca_dir, lca_dir in pairs:
        report_rows.append(
            {
                "sequence_type": "RCA",
                "sequence_index": pair_index,
                "sequence_dir": str(rca_dir),
                "paired": True,
                "pair_index": pair_index,
                "status": "paired",
                "reason": "",
            }
        )
        report_rows.append(
            {
                "sequence_type": "LCA",
                "sequence_index": pair_index,
                "sequence_dir": str(lca_dir),
                "paired": True,
                "pair_index": pair_index,
                "status": "paired",
                "reason": "",
            }
        )

    for sequence_index, sequence_dir in enumerate(rca_sequences[num_pairs:], start=num_pairs):
        report_rows.append(
            {
                "sequence_type": "RCA",
                "sequence_index": sequence_index,
                "sequence_dir": str(sequence_dir),
                "paired": False,
                "pair_index": "",
                "status": "ignored_extra_sequence",
                "reason": "No matching LCA sequence under trim_to_min policy.",
            }
        )
    for sequence_index, sequence_dir in enumerate(lca_sequences[num_pairs:], start=num_pairs):
        report_rows.append(
            {
                "sequence_type": "LCA",
                "sequence_index": sequence_index,
                "sequence_dir": str(sequence_dir),
                "paired": False,
                "pair_index": "",
                "status": "ignored_extra_sequence",
                "reason": "No matching RCA sequence under trim_to_min policy.",
            }
        )

    summary = {
        "num_rca_sequences": num_rca_sequences,
        "num_lca_sequences": num_lca_sequences,
        "num_sequence_pairs": num_pairs,
        "num_ignored_rca_sequences": max(0, num_rca_sequences - num_pairs),
        "num_ignored_lca_sequences": max(0, num_lca_sequences - num_pairs),
    }
    return pairs, report_rows, summary


def _pair_row_from_result(
    pair_index: int,
    rca_sequence_dir: Path,
    lca_sequence_dir: Path,
    result: IntegratedInferenceResult,
) -> dict[str, Any]:
    final_prediction = result.final_prediction
    metric_columns = PAIR_PREDICTION_COLUMNS[5:]
    return {
        "pair_index": pair_index,
        "rca_sequence_dir": str(rca_sequence_dir),
        "lca_sequence_dir": str(lca_sequence_dir),
        "status": "valid",
        "reason": "",
        **{column: final_prediction.get(column, "") for column in metric_columns},
    }


def _skipped_pair_row(
    pair_index: int,
    rca_sequence_dir: Path,
    lca_sequence_dir: Path,
    reason: str,
) -> dict[str, Any]:
    return {
        "pair_index": pair_index,
        "rca_sequence_dir": str(rca_sequence_dir),
        "lca_sequence_dir": str(lca_sequence_dir),
        "status": "skipped",
        "reason": reason,
    }


def _mark_pairing_report_skipped(
    report_rows: list[dict[str, Any]],
    pair_index: int,
    reason: str,
) -> None:
    for row in report_rows:
        if row.get("pair_index") == pair_index:
            row["status"] = "skipped"
            row["reason"] = reason


def aggregate_pair_predictions(pair_rows: list[dict[str, Any]]) -> tuple[int, float, dict[str, Any]]:
    """Aggregate valid pair-level predictions into one study-level decision."""
    valid_rows = [row for row in pair_rows if row.get("status") == "valid"]
    if not valid_rows:
        raise ValueError("No valid sequence pairs remain after pairing and validation.")

    vote_rightdom = sum(int(row["pred_class"]) == 0 for row in valid_rows)
    vote_leftdom = sum(int(row["pred_class"]) == 1 for row in valid_rows)
    mean_prob_rightdom = sum(float(row["mean_prob_rightdom"]) for row in valid_rows) / len(valid_rows)
    mean_prob_leftdom = sum(float(row["mean_prob_leftdom"]) for row in valid_rows) / len(valid_rows)

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


def run_multi_sequence_study_integrated_inference(
    rca_study_dir: str | Path,
    lca_study_dir: str | Path,
    rca_model: nn.Module,
    lca_model: nn.Module,
    device: torch.device,
    sequence_pair_policy: str = "trim_to_min",
    clip_length: int = 15,
    image_size: int = 512,
    mean: float = 0.5485,
    std: float = 0.1407,
    occlusion_threshold: float = 0.5,
    frame_quality_threshold: float = 0.5,
    batch_size: int = 16,
) -> MultiSequenceInferenceResult:
    """Run integrated inference across paired RCA/LCA sequences for one study."""
    pairs, pairing_report_rows, pairing_summary = pair_sequence_dirs(
        rca_study_dir=rca_study_dir,
        lca_study_dir=lca_study_dir,
        sequence_pair_policy=sequence_pair_policy,
    )

    pair_rows: list[dict[str, Any]] = []
    pair_frame_rows: list[dict[str, Any]] = []
    pair_predictions: list[dict[str, Any]] = []

    for pair_index, rca_sequence_dir, lca_sequence_dir in pairs:
        rca_has_frames = has_valid_image_files(rca_sequence_dir)
        lca_has_frames = has_valid_image_files(lca_sequence_dir)
        if not rca_has_frames or not lca_has_frames:
            missing = []
            if not rca_has_frames:
                missing.append("RCA sequence has no valid image files")
            if not lca_has_frames:
                missing.append("LCA sequence has no valid image files")
            reason = "; ".join(missing)
            pair_rows.append(_skipped_pair_row(pair_index, rca_sequence_dir, lca_sequence_dir, reason))
            _mark_pairing_report_skipped(pairing_report_rows, pair_index, reason)
            continue

        result = run_single_pair_integrated_inference(
            rca_frame_dir=rca_sequence_dir,
            lca_frame_dir=lca_sequence_dir,
            rca_model=rca_model,
            lca_model=lca_model,
            device=device,
            clip_length=clip_length,
            image_size=image_size,
            mean=mean,
            std=std,
            occlusion_threshold=occlusion_threshold,
            frame_quality_threshold=frame_quality_threshold,
            batch_size=batch_size,
        )
        pair_row = _pair_row_from_result(pair_index, rca_sequence_dir, lca_sequence_dir, result)
        pair_rows.append(pair_row)
        pair_predictions.append(
            {
                "pair_index": pair_index,
                "rca_sequence_dir": str(rca_sequence_dir),
                "lca_sequence_dir": str(lca_sequence_dir),
                **result.final_prediction,
            }
        )
        for frame_row in result.frame_rows:
            pair_frame_rows.append({"pair_index": pair_index, **frame_row})

    valid_pair_rows = [row for row in pair_rows if row.get("status") == "valid"]
    pred_class, final_confidence, vote_details = aggregate_pair_predictions(pair_rows)
    study_final_prediction = {
        **pairing_summary,
        "num_valid_pairs": len(valid_pair_rows),
        "sequence_pair_policy": sequence_pair_policy,
        **vote_details,
        "pred_class": int(pred_class),
        "pred_label": DOMINANCE_LABELS[pred_class],
        "final_confidence": float(final_confidence),
        "pair_predictions": pair_predictions,
    }
    return MultiSequenceInferenceResult(
        study_final_prediction=study_final_prediction,
        pair_rows=pair_rows,
        pair_frame_rows=pair_frame_rows,
        pairing_report_rows=pairing_report_rows,
    )


def write_multi_sequence_outputs(
    result: MultiSequenceInferenceResult,
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path]:
    """Write multi-sequence study outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    final_json_path = output_path / "study_final_prediction.json"
    with final_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(result.study_final_prediction, json_file, indent=2)

    pair_csv_path = output_path / "pair_predictions.csv"
    with pair_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PAIR_PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.pair_rows:
            writer.writerow({column: row.get(column, "") for column in PAIR_PREDICTION_COLUMNS})

    pair_frame_csv_path = output_path / "pair_frame_predictions.csv"
    with pair_frame_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PAIR_FRAME_PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.pair_frame_rows:
            writer.writerow({column: row.get(column, "") for column in PAIR_FRAME_PREDICTION_COLUMNS})

    pairing_report_csv_path = output_path / "sequence_pairing_report.csv"
    with pairing_report_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SEQUENCE_PAIRING_REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.pairing_report_rows:
            writer.writerow({column: row.get(column, "") for column in SEQUENCE_PAIRING_REPORT_COLUMNS})

    return final_json_path, pair_csv_path, pair_frame_csv_path, pairing_report_csv_path


def _csv_value(value: Any) -> Any:
    """Convert nested values to portable CSV cells."""
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return value


def _read_manifest_rows(manifest_csv: str | Path) -> tuple[list[dict[str, str]], bool]:
    manifest_path = Path(manifest_csv)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in MANIFEST_REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            missing_text = ", ".join(missing_columns)
            raise ValueError(f"Manifest CSV is missing required columns: {missing_text}")
        has_subset = "subset" in fieldnames
        return [dict(row) for row in reader], has_subset


def _manifest_prediction_row(
    manifest_row: dict[str, str],
    true_class: int | str,
    status: str,
    reason: str,
    has_subset: bool,
    sequence_pair_policy: str,
    final_prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_prediction = final_prediction or {}
    pred_class = final_prediction.get("pred_class", "")
    correct = ""
    if status == "valid" and pred_class != "" and true_class != "":
        correct = int(pred_class) == int(true_class)

    return {
        "study_id": (manifest_row.get("study_id") or "").strip(),
        "label": (manifest_row.get("label") or "").strip(),
        "true_class": true_class,
        "pred_class": pred_class,
        "pred_label": final_prediction.get("pred_label", ""),
        "correct": correct,
        "subset": (manifest_row.get("subset") or "").strip() if has_subset else "",
        "status": status,
        "reason": reason,
        "num_rca_sequences": final_prediction.get("num_rca_sequences", ""),
        "num_lca_sequences": final_prediction.get("num_lca_sequences", ""),
        "num_sequence_pairs": final_prediction.get("num_sequence_pairs", ""),
        "num_valid_pairs": final_prediction.get("num_valid_pairs", ""),
        "sequence_pair_policy": final_prediction.get("sequence_pair_policy", sequence_pair_policy),
        "num_ignored_rca_sequences": final_prediction.get("num_ignored_rca_sequences", ""),
        "num_ignored_lca_sequences": final_prediction.get("num_ignored_lca_sequences", ""),
        "vote_rightdom": final_prediction.get("vote_rightdom", ""),
        "vote_leftdom": final_prediction.get("vote_leftdom", ""),
        "mean_prob_rightdom": final_prediction.get("mean_prob_rightdom", ""),
        "mean_prob_leftdom": final_prediction.get("mean_prob_leftdom", ""),
        "final_confidence": final_prediction.get("final_confidence", ""),
    }


def _prefixed_rows(
    rows: list[dict[str, Any]],
    study_id: str,
    subset: str,
) -> list[dict[str, Any]]:
    return [{"study_id": study_id, "subset": subset, **row} for row in rows]


def _compute_subset_metrics(
    prediction_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    subset_metrics: dict[str, dict[str, Any]] = {}
    subsets = sorted({str(row.get("subset", "")) for row in prediction_rows if row.get("status") == "valid"})
    for subset in subsets:
        subset_rows = [
            row
            for row in prediction_rows
            if row.get("status") == "valid" and str(row.get("subset", "")) == subset
        ]
        targets = [int(row["true_class"]) for row in subset_rows]
        predictions = [int(row["pred_class"]) for row in subset_rows]
        subset_metrics[subset] = binary_classification_metrics(targets, predictions)
    return subset_metrics


def run_manifest_cohort_integrated_inference(
    manifest_csv: str | Path,
    rca_model: nn.Module,
    lca_model: nn.Module,
    device: torch.device,
    invalid_study_policy: str = "skip",
    sequence_pair_policy: str = "trim_to_min",
    clip_length: int = 15,
    image_size: int = 512,
    mean: float = 0.5485,
    std: float = 0.1407,
    occlusion_threshold: float = 0.5,
    frame_quality_threshold: float = 0.5,
    batch_size: int = 16,
) -> ManifestCohortInferenceResult:
    """Run integrated inference for every study in a manifest CSV."""
    if invalid_study_policy not in {"skip", "error"}:
        raise ValueError("invalid_study_policy must be 'skip' or 'error'.")

    manifest_rows, has_subset = _read_manifest_rows(manifest_csv)
    prediction_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_pair_frame_rows: list[dict[str, Any]] = []
    all_pairing_report_rows: list[dict[str, Any]] = []
    targets: list[int] = []
    predictions: list[int] = []

    for row_number, manifest_row in enumerate(manifest_rows, start=2):
        study_id = (manifest_row.get("study_id") or "").strip()
        subset = (manifest_row.get("subset") or "").strip() if has_subset else ""
        true_class: int | str = ""
        try:
            label = (manifest_row.get("label") or "").strip()
            true_class = get_class_id("dominance", label)

            rca_study_dir = (manifest_row.get("rca_study_dir") or "").strip()
            lca_study_dir = (manifest_row.get("lca_study_dir") or "").strip()
            if not rca_study_dir:
                raise ValueError("missing rca_study_dir")
            if not lca_study_dir:
                raise ValueError("missing lca_study_dir")

            study_result = run_multi_sequence_study_integrated_inference(
                rca_study_dir=rca_study_dir,
                lca_study_dir=lca_study_dir,
                rca_model=rca_model,
                lca_model=lca_model,
                device=device,
                sequence_pair_policy=sequence_pair_policy,
                clip_length=clip_length,
                image_size=image_size,
                mean=mean,
                std=std,
                occlusion_threshold=occlusion_threshold,
                frame_quality_threshold=frame_quality_threshold,
                batch_size=batch_size,
            )
            final_prediction = study_result.study_final_prediction
            pred_class = int(final_prediction["pred_class"])

            prediction_rows.append(
                _manifest_prediction_row(
                    manifest_row=manifest_row,
                    true_class=true_class,
                    status="valid",
                    reason="",
                    has_subset=has_subset,
                    sequence_pair_policy=sequence_pair_policy,
                    final_prediction=final_prediction,
                )
            )
            targets.append(int(true_class))
            predictions.append(pred_class)
            all_pair_rows.extend(_prefixed_rows(study_result.pair_rows, study_id, subset))
            all_pair_frame_rows.extend(_prefixed_rows(study_result.pair_frame_rows, study_id, subset))
            all_pairing_report_rows.extend(_prefixed_rows(study_result.pairing_report_rows, study_id, subset))
        except Exception as exc:
            reason = f"row {row_number}: {exc}"
            if invalid_study_policy == "error":
                raise RuntimeError(f"Invalid study '{study_id or '<missing study_id>'}' in manifest: {reason}") from exc
            prediction_rows.append(
                _manifest_prediction_row(
                    manifest_row=manifest_row,
                    true_class=true_class,
                    status="invalid",
                    reason=reason,
                    has_subset=has_subset,
                    sequence_pair_policy=sequence_pair_policy,
                )
            )

    metrics = binary_classification_metrics(targets, predictions)
    subset_metrics = _compute_subset_metrics(prediction_rows) if has_subset else {}
    return ManifestCohortInferenceResult(
        prediction_rows=prediction_rows,
        metrics=metrics,
        subset_metrics=subset_metrics,
        all_pair_rows=all_pair_rows,
        all_pair_frame_rows=all_pair_frame_rows,
        all_pairing_report_rows=all_pairing_report_rows,
        has_subset=has_subset,
    )


def write_manifest_outputs(
    result: ManifestCohortInferenceResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write cohort-level integrated inference outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "predictions": output_path / "integrated_predictions.csv",
        "metrics_csv": output_path / "integrated_metrics.csv",
        "metrics_json": output_path / "integrated_metrics.json",
        "pair_predictions": output_path / "all_pair_predictions.csv",
        "pair_frame_predictions": output_path / "all_pair_frame_predictions.csv",
        "pairing_report": output_path / "all_sequence_pairing_report.csv",
    }

    with paths["predictions"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=INTEGRATED_PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.prediction_rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in INTEGRATED_PREDICTION_COLUMNS})

    with paths["metrics_csv"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerow({column: _csv_value(result.metrics.get(column, "")) for column in METRIC_COLUMNS})

    metrics_json = {"overall": result.metrics}
    if result.has_subset:
        metrics_json["subsets"] = result.subset_metrics
    with paths["metrics_json"].open("w", encoding="utf-8") as json_file:
        json.dump(metrics_json, json_file, indent=2)

    if result.has_subset:
        subset_path = output_path / "subset_metrics.csv"
        paths["subset_metrics"] = subset_path
        with subset_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=("subset", *METRIC_COLUMNS), extrasaction="ignore")
            writer.writeheader()
            for subset, metrics in result.subset_metrics.items():
                writer.writerow(
                    {
                        "subset": subset,
                        **{column: _csv_value(metrics.get(column, "")) for column in METRIC_COLUMNS},
                    }
                )

    with paths["pair_predictions"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ALL_PAIR_PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.all_pair_rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in ALL_PAIR_PREDICTION_COLUMNS})

    with paths["pair_frame_predictions"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ALL_PAIR_FRAME_PREDICTION_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.all_pair_frame_rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in ALL_PAIR_FRAME_PREDICTION_COLUMNS})

    with paths["pairing_report"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ALL_SEQUENCE_PAIRING_REPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in result.all_pairing_report_rows:
            writer.writerow({column: _csv_value(row.get(column, "")) for column in ALL_SEQUENCE_PAIRING_REPORT_COLUMNS})

    return paths
