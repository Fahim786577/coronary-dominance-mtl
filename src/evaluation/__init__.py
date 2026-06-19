"""Evaluation and inference helpers for coronary MTL checkpoints."""

from src.evaluation.inference import (
    EVALUATION_MODES,
    CoronaryTaskInferenceDataset,
    available_tasks_for_artery,
    evaluate_checkpoint,
    evaluate_task,
    load_mtl_model_from_checkpoint,
    resolve_evaluation_output_dir,
    resolve_mtl_checkpoint_path,
    write_metrics_csv,
    write_metrics_json,
    write_predictions_csv,
)
from src.evaluation.integrated_inference import (
    IntegratedInferenceResult,
    ManifestCohortInferenceResult,
    MultiSequenceInferenceResult,
    load_model_from_checkpoint,
    run_manifest_cohort_integrated_inference,
    run_multi_sequence_study_integrated_inference,
    run_single_pair_integrated_inference,
    write_integrated_outputs,
    write_manifest_outputs,
    write_multi_sequence_outputs,
)
from src.evaluation.metrics import binary_classification_metrics

__all__ = [
    "EVALUATION_MODES",
    "CoronaryTaskInferenceDataset",
    "IntegratedInferenceResult",
    "ManifestCohortInferenceResult",
    "MultiSequenceInferenceResult",
    "available_tasks_for_artery",
    "binary_classification_metrics",
    "evaluate_checkpoint",
    "evaluate_task",
    "load_model_from_checkpoint",
    "load_mtl_model_from_checkpoint",
    "resolve_evaluation_output_dir",
    "resolve_mtl_checkpoint_path",
    "run_manifest_cohort_integrated_inference",
    "run_multi_sequence_study_integrated_inference",
    "run_single_pair_integrated_inference",
    "write_integrated_outputs",
    "write_manifest_outputs",
    "write_multi_sequence_outputs",
    "write_metrics_csv",
    "write_metrics_json",
    "write_predictions_csv",
]
