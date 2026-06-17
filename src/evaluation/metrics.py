"""Binary classification metrics for evaluation without sklearn."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return 0.0 for undefined ratios to keep JSON/CSV outputs portable."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def confusion_counts(
    targets: Sequence[int],
    predictions: Sequence[int],
) -> dict[str, int]:
    """Compute binary confusion counts with class 1 treated as positive."""
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have the same length.")

    tn = fp = fn = tp = 0
    for target, prediction in zip(targets, predictions):
        target_id = int(target)
        prediction_id = int(prediction)
        if target_id == 1 and prediction_id == 1:
            tp += 1
        elif target_id == 0 and prediction_id == 1:
            fp += 1
        elif target_id == 1 and prediction_id == 0:
            fn += 1
        elif target_id == 0 and prediction_id == 0:
            tn += 1
        else:
            raise ValueError(
                f"Binary metrics expect class IDs 0 or 1, got target={target_id}, prediction={prediction_id}."
            )

    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def binary_classification_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    loss: float | None = None,
) -> dict[str, Any]:
    """Return binary metrics for one task.

    Undefined ratios use 0.0 instead of NaN so the result can be written
    consistently to both JSON and CSV.
    """
    counts = confusion_counts(targets, predictions)
    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]
    sample_count = tn + fp + fn + tp

    accuracy = _safe_divide(tp + tn, sample_count)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    balanced_accuracy = (recall + specificity) / 2.0
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    mcc_denominator = math.sqrt(
        float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    )
    mcc = _safe_divide((tp * tn) - (fp * fn), mcc_denominator)

    return {
        "loss": loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sample_count": sample_count,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def mean_metric(task_metrics: dict[str, dict[str, Any]], metric_name: str) -> float:
    """Return the mean of a metric across tasks with at least one sample."""
    values = [
        float(metrics[metric_name])
        for metrics in task_metrics.values()
        if int(metrics.get("sample_count", 0)) > 0 and metrics.get(metric_name) is not None
    ]
    if not values:
        return 0.0
    return sum(values) / len(values)
