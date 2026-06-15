"""RCA to LCA transfer initialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.utils.paths import normalize_artery


SUPPORTED_TRANSFER_SCOPES = {"all", "shared", "shared_and_common_heads"}
SHARED_PREFIXES = ("backbone.", "extractor.")
COMMON_HEAD_PREFIXES = ("frame_quality_head.", "dominance_head.")


def load_transfer_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a student checkpoint for transfer initialization."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Transfer checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=map_location)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Transfer checkpoint {path} does not contain 'model_state_dict'.")
    return checkpoint


def _scope_allows_key(key: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "shared":
        return key.startswith(SHARED_PREFIXES)
    if scope == "shared_and_common_heads":
        return key.startswith(SHARED_PREFIXES + COMMON_HEAD_PREFIXES)
    raise NotImplementedError(f"Unsupported transfer load scope: {scope}")


def _validate_transfer_metadata(
    checkpoint: dict[str, Any],
    source_artery: str,
    target_artery: str,
    fold: int,
    backbone: str,
    allow_backbone_mismatch: bool,
    allow_fold_mismatch: bool,
) -> None:
    if normalize_artery(target_artery) != "LCA":
        raise ValueError("RCA -> LCA transfer requires --artery LCA.")

    checkpoint_artery = checkpoint.get("artery")
    if checkpoint_artery is not None and normalize_artery(str(checkpoint_artery)) != normalize_artery(source_artery):
        raise ValueError(
            "Transfer source artery mismatch: "
            f"expected {normalize_artery(source_artery)!r}, found {normalize_artery(str(checkpoint_artery))!r}."
        )

    checkpoint_fold = checkpoint.get("fold")
    if checkpoint_fold is not None and int(checkpoint_fold) != fold and not allow_fold_mismatch:
        raise ValueError(
            f"Transfer fold mismatch: expected fold {fold}, found fold {checkpoint_fold}. "
            "Use --allow_transfer_fold_mismatch true to override."
        )

    checkpoint_backbone = checkpoint.get("backbone")
    if checkpoint_backbone is not None and str(checkpoint_backbone) != backbone and not allow_backbone_mismatch:
        raise ValueError(
            f"Transfer backbone mismatch: expected {backbone!r}, found {checkpoint_backbone!r}. "
            "Use --allow_transfer_backbone_mismatch true to override."
        )


def filter_transfer_state_dict(
    source_state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
    load_scope: str = "shared_and_common_heads",
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Filter source weights to keys allowed by scope and compatible with target model."""
    if load_scope not in SUPPORTED_TRANSFER_SCOPES:
        raise NotImplementedError(f"Unsupported transfer load scope: {load_scope}")

    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in source_state_dict.items():
        if not _scope_allows_key(key, load_scope):
            skipped.append(key)
            continue
        target_value = target_state_dict.get(key)
        if target_value is None or target_value.shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value

    return filtered, skipped


def apply_transfer_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
    source_artery: str = "RCA",
    target_artery: str = "LCA",
    fold: int = 1,
    backbone: str = "resnet18",
    load_scope: str = "shared_and_common_heads",
    allow_backbone_mismatch: bool = False,
    allow_fold_mismatch: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Initialize a student model from an RCA checkpoint and report loaded/skipped keys."""
    checkpoint = load_transfer_checkpoint(checkpoint_path, map_location=map_location)
    _validate_transfer_metadata(
        checkpoint=checkpoint,
        source_artery=source_artery,
        target_artery=target_artery,
        fold=fold,
        backbone=backbone,
        allow_backbone_mismatch=allow_backbone_mismatch,
        allow_fold_mismatch=allow_fold_mismatch,
    )

    target_state_dict = model.state_dict()
    filtered_state_dict, skipped_keys = filter_transfer_state_dict(
        source_state_dict=checkpoint["model_state_dict"],
        target_state_dict=target_state_dict,
        load_scope=load_scope,
    )
    if not filtered_state_dict:
        raise RuntimeError(
            "Transfer initialization loaded no compatible keys. "
            f"Checkpoint: {checkpoint_path}, scope: {load_scope}."
        )

    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    skipped = sorted(set(skipped_keys + list(load_result.missing_keys) + list(load_result.unexpected_keys)))
    loaded = sorted(filtered_state_dict)

    return {
        "checkpoint": checkpoint,
        "loaded_keys": loaded,
        "skipped_keys": skipped,
        "loaded_key_count": len(loaded),
        "skipped_key_count": len(skipped),
    }
