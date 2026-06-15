"""Practical TwoPhase-inspired optimization helpers.

This module implements a pragmatic approximation inspired by
connection-strength-based TwoPhase optimization. It is not a paper-faithful
implementation because the current model does not include task-specific batch
normalization or channel-wise task-priority modules.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn


TASK_INPUT_KEYS = {
    "occlusion": "occlusion_images",
    "frame_quality": "frame_quality_images",
    "dominance": "dominance_images",
}

TaskLossFn = Callable[[str, dict[str, Tensor], dict[str, Tensor]], tuple[Tensor, dict[str, Any]]]


def get_shared_parameter_items(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return shared feature-extractor parameters for practical projection."""
    shared_items: list[tuple[str, nn.Parameter]] = []
    seen: set[int] = set()

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not (name.startswith("backbone.") or name.startswith("extractor.")):
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            continue
        seen.add(parameter_id)
        shared_items.append((name, parameter))

    if not shared_items:
        raise RuntimeError(
            "Could not identify shared feature-extractor parameters for TwoPhase projection."
        )
    return shared_items


def flatten_grads(grads: list[Tensor | None], params: list[nn.Parameter]) -> Tensor:
    """Flatten gradients, substituting zeros for unused parameters."""
    if len(grads) != len(params):
        raise ValueError("grads and params must have the same length.")
    flat_parts = [
        torch.zeros_like(parameter).flatten() if grad is None else grad.flatten()
        for grad, parameter in zip(grads, params)
    ]
    if not flat_parts:
        raise RuntimeError("Cannot flatten an empty gradient list.")
    return torch.cat(flat_parts)


def unflatten_to_params(flat_grad: Tensor, params: list[nn.Parameter]) -> None:
    """Write a flattened gradient vector back into matching parameter grads."""
    offset = 0
    for parameter in params:
        numel = parameter.numel()
        grad_piece = flat_grad[offset : offset + numel].view_as(parameter).clone()
        parameter.grad = grad_piece
        offset += numel
    if offset != flat_grad.numel():
        raise ValueError("Flat gradient size does not match target parameters.")


def _active_task_inputs(task: str, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    input_key = TASK_INPUT_KEYS[task]
    if input_key not in inputs:
        raise KeyError(f"Missing input key '{input_key}' for task '{task}'.")
    return {input_key: inputs[input_key]}


def _task_gradients_by_name(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    }


def compute_task_gradients(
    model: nn.Module,
    task: str,
    inputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    compute_task_loss: TaskLossFn,
    shared_items: list[tuple[str, nn.Parameter]],
) -> dict[str, Any]:
    """Compute one task's loss, metrics, and gradients."""
    model.zero_grad(set_to_none=True)
    task_inputs = _active_task_inputs(task, inputs)
    loss, metrics = compute_task_loss(task, task_inputs, targets)
    loss.backward()

    shared_names = [name for name, _ in shared_items]
    shared_params = [parameter for _, parameter in shared_items]
    shared_grads = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in shared_params
    ]
    shared_flat = flatten_grads(shared_grads, shared_params)

    return {
        "task": task,
        "loss": loss.detach(),
        "metrics": metrics,
        "shared_names": shared_names,
        "shared_flat_grad": shared_flat,
        "param_grads": _task_gradients_by_name(model),
    }


def select_priority_task(task_gradients: dict[str, Tensor]) -> tuple[str, dict[str, float]]:
    """Select the task with the largest global shared-gradient norm."""
    if not task_gradients:
        raise ValueError("No task gradients were provided for priority selection.")
    norms = {
        task: float(torch.linalg.vector_norm(gradient).detach().cpu())
        for task, gradient in task_gradients.items()
    }
    priority_task = max(norms, key=norms.get)
    return priority_task, norms


def project_against_priority(
    task_grad: Tensor,
    priority_grad: Tensor,
    eps: float = 1e-12,
) -> Tensor:
    """Project a task gradient away from a conflicting priority gradient."""
    dot_product = torch.dot(task_grad, priority_grad)
    if dot_product >= 0:
        return task_grad
    priority_norm_sq = torch.dot(priority_grad, priority_grad)
    if priority_norm_sq <= eps:
        return task_grad
    return task_grad - (dot_product / (priority_norm_sq + eps)) * priority_grad


def _combine_non_shared_gradients(
    task_results: list[dict[str, Any]],
    shared_names: set[str],
) -> dict[str, Tensor]:
    combined: dict[str, Tensor] = {}
    for result in task_results:
        for name, gradient in result["param_grads"].items():
            if name in shared_names:
                continue
            if name not in combined:
                combined[name] = gradient.clone()
            else:
                combined[name] = combined[name] + gradient
    return combined


def _write_named_gradients(
    model: nn.Module,
    named_gradients: dict[str, Tensor],
) -> None:
    parameter_map = dict(model.named_parameters())
    for name, gradient in named_gradients.items():
        parameter = parameter_map.get(name)
        if parameter is not None and parameter.requires_grad:
            parameter.grad = gradient.clone()


def _summarize_task_metrics(task_results: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    task_metrics: dict[str, Any] = {}
    losses = []
    for result in task_results:
        task = result["task"]
        metrics = dict(result["metrics"])
        task_metrics[task] = metrics
        losses.append(float(metrics["loss"]))
    return sum(losses) / len(losses), task_metrics


def run_twophase_phase1_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    active_tasks: list[str],
    compute_task_loss: TaskLossFn,
    gradient_clip_max_norm: float | None,
) -> dict[str, Any]:
    """Run practical Phase 1: sequential task-wise optimizer updates."""
    task_results: list[dict[str, Any]] = []

    for task in active_tasks:
        optimizer.zero_grad(set_to_none=True)
        task_inputs = _active_task_inputs(task, inputs)
        loss, metrics = compute_task_loss(task, task_inputs, targets)
        loss.backward()
        if gradient_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
        optimizer.step()
        task_results.append({"task": task, "metrics": metrics})

    loss_value, task_metrics = _summarize_task_metrics(task_results)
    return {
        "phase": 1,
        "loss": loss_value,
        "task_metrics": task_metrics,
        "priority_task": None,
    }


def run_twophase_phase2_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: dict[str, Tensor],
    targets: dict[str, Tensor],
    active_tasks: list[str],
    compute_task_loss: TaskLossFn,
    gradient_clip_max_norm: float | None,
    projection: bool = True,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Run practical Phase 2: global shared-gradient priority projection."""
    shared_items = get_shared_parameter_items(model)
    shared_names = {name for name, _ in shared_items}
    shared_params = [parameter for _, parameter in shared_items]
    task_results = [
        compute_task_gradients(model, task, inputs, targets, compute_task_loss, shared_items)
        for task in active_tasks
    ]

    task_shared_grads = {
        result["task"]: result["shared_flat_grad"]
        for result in task_results
    }
    priority_task, shared_norms = select_priority_task(task_shared_grads)
    priority_grad = task_shared_grads[priority_task]

    projected_shared_grads: list[Tensor] = []
    for task, gradient in task_shared_grads.items():
        if task == priority_task or not projection:
            projected_shared_grads.append(gradient)
        else:
            projected_shared_grads.append(project_against_priority(gradient, priority_grad, eps=eps))

    optimizer.zero_grad(set_to_none=True)
    shared_total = torch.stack(projected_shared_grads).sum(dim=0)
    unflatten_to_params(shared_total, shared_params)
    _write_named_gradients(model, _combine_non_shared_gradients(task_results, shared_names))

    if gradient_clip_max_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
    optimizer.step()

    loss_value, task_metrics = _summarize_task_metrics(task_results)
    return {
        "phase": 2,
        "loss": loss_value,
        "task_metrics": task_metrics,
        "priority_task": priority_task,
        "shared_gradient_norms": shared_norms,
    }
