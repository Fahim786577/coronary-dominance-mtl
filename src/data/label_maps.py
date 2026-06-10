"""Central label maps for coronary dominance multi-task datasets."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


LABEL_MAPS: dict[str, dict[str, int]] = {
    "occlusion": {
        "nonoccluded": 0,
        "occluded": 1,
    },
    "framequality": {
        "noninformative": 0,
        "informative": 1,
    },
    "dominance": {
        "rightdom": 0,
        "leftdom": 1,
    },
}


def get_label_map(task: str) -> Mapping[str, int]:
    """Return the label-to-class-id map for a supported task."""
    try:
        return MappingProxyType(LABEL_MAPS[task])
    except KeyError as exc:
        valid_tasks = ", ".join(sorted(LABEL_MAPS))
        raise ValueError(f"Unsupported task '{task}'. Expected one of: {valid_tasks}.") from exc


def get_class_id(task: str, label: str) -> int:
    """Return the integer class ID for a task label."""
    label_map = get_label_map(task)
    try:
        return label_map[label]
    except KeyError as exc:
        valid_labels = ", ".join(sorted(label_map))
        raise ValueError(
            f"Unsupported label '{label}' for task '{task}'. Expected one of: {valid_labels}."
        ) from exc


def get_supported_tasks() -> tuple[str, ...]:
    """Return supported task names."""
    return tuple(LABEL_MAPS)
