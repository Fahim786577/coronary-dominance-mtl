"""Backbone builders for coronary dominance models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models


def _resnet18_without_weights() -> nn.Module:
    """Build ResNet18 without relying on a specific torchvision API generation."""
    try:
        return models.resnet18(weights=None)
    except TypeError:
        return models.resnet18(pretrained=False)


def _build_resnet18(pretrained: bool) -> tuple[nn.Module, bool]:
    """Build torchvision ResNet18, falling back to random weights if needed."""
    if not pretrained:
        return _resnet18_without_weights(), False

    try:
        weights = models.ResNet18_Weights.DEFAULT
        return models.resnet18(weights=weights), True
    except (AttributeError, TypeError):
        try:
            return models.resnet18(pretrained=True), True
        except Exception:
            return _resnet18_without_weights(), False
    except Exception:
        return _resnet18_without_weights(), False


def _replace_first_conv_with_grayscale(model: nn.Module, pretrained_loaded: bool) -> None:
    """Replace RGB conv1 with a single-channel convolution."""
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    if pretrained_loaded and old_conv.weight.shape[1] == 3:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
    else:
        nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
        if new_conv.bias is not None:
            nn.init.zeros_(new_conv.bias)

    model.conv1 = new_conv


def build_resnet18_grayscale(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return a grayscale ResNet18 convolutional feature extractor and feature dim."""
    model, pretrained_loaded = _build_resnet18(pretrained)
    _replace_first_conv_with_grayscale(model, pretrained_loaded)
    feature_dim = model.fc.in_features
    extractor = nn.Sequential(*list(model.children())[:-2])
    return extractor, feature_dim


BackboneBuilder = Callable[[bool], tuple[nn.Module, int]]
