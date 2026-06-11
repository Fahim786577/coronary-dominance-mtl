"""Backbone builders for coronary dominance models."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torchvision import models


ModelConstructor = Callable[..., nn.Module]
BackboneBuilder = Callable[[bool], tuple[nn.Module, int]]


def _build_without_weights(constructor: ModelConstructor) -> nn.Module:
    """Build a torchvision model without relying on one API generation."""
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)


def _build_torchvision_model(
    constructor: ModelConstructor,
    weights_class_name: str,
    pretrained: bool,
) -> tuple[nn.Module, bool]:
    """Build a torchvision model, falling back to random weights if needed."""
    if not pretrained:
        return _build_without_weights(constructor), False

    try:
        weights_class = getattr(models, weights_class_name)
        return constructor(weights=weights_class.DEFAULT), True
    except (AttributeError, TypeError):
        try:
            return constructor(pretrained=True), True
        except Exception:
            return _build_without_weights(constructor), False
    except Exception:
        return _build_without_weights(constructor), False


def _make_grayscale_conv(old_conv: nn.Conv2d, pretrained_loaded: bool) -> nn.Conv2d:
    """Create a single-channel conv initialized from an RGB conv when possible."""
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

    return new_conv


def _replace_resnet_conv1(model: nn.Module, pretrained_loaded: bool) -> None:
    """Replace ResNet conv1 with a grayscale convolution."""
    model.conv1 = _make_grayscale_conv(model.conv1, pretrained_loaded)


def _replace_mobilenet_first_conv(model: nn.Module, pretrained_loaded: bool) -> None:
    """Replace MobileNetV2 first convolution with a grayscale convolution."""
    old_conv = model.features[0][0]
    model.features[0][0] = _make_grayscale_conv(old_conv, pretrained_loaded)


def _replace_densenet_first_conv(model: nn.Module, pretrained_loaded: bool) -> None:
    """Replace DenseNet121 first convolution with a grayscale convolution."""
    old_conv = model.features.conv0
    model.features.conv0 = _make_grayscale_conv(old_conv, pretrained_loaded)


def _build_resnet18(pretrained: bool) -> tuple[nn.Module, bool]:
    return _build_torchvision_model(models.resnet18, "ResNet18_Weights", pretrained)


def _build_mobilenet_v2(pretrained: bool) -> tuple[nn.Module, bool]:
    return _build_torchvision_model(models.mobilenet_v2, "MobileNet_V2_Weights", pretrained)


def _build_densenet121(pretrained: bool) -> tuple[nn.Module, bool]:
    return _build_torchvision_model(models.densenet121, "DenseNet121_Weights", pretrained)


def build_resnet18_grayscale(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return a grayscale ResNet18 convolutional feature extractor and feature dim."""
    model, pretrained_loaded = _build_resnet18(pretrained)
    _replace_resnet_conv1(model, pretrained_loaded)
    feature_dim = model.fc.in_features
    extractor = nn.Sequential(*list(model.children())[:-2])
    return extractor, feature_dim


def build_mobilenet_v2_grayscale(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return a grayscale MobileNetV2 feature extractor and feature dim."""
    model, pretrained_loaded = _build_mobilenet_v2(pretrained)
    _replace_mobilenet_first_conv(model, pretrained_loaded)
    return model.features, 1280


def build_densenet121_grayscale(pretrained: bool = True) -> tuple[nn.Module, int]:
    """Return a grayscale DenseNet121 feature extractor and feature dim."""
    model, pretrained_loaded = _build_densenet121(pretrained)
    _replace_densenet_first_conv(model, pretrained_loaded)
    extractor = nn.Sequential(model.features, nn.ReLU(inplace=True))
    return extractor, 1024


BACKBONE_BUILDERS: dict[str, BackboneBuilder] = {
    "resnet18": build_resnet18_grayscale,
    "mobilenet_v2": build_mobilenet_v2_grayscale,
    "densenet121": build_densenet121_grayscale,
}

BACKBONE_ALIASES = {
    "resnet": "resnet18",
    "mobilenetv2": "mobilenet_v2",
    "mobile_net_v2": "mobilenet_v2",
    "densenet": "densenet121",
}


def normalize_backbone_name(backbone_name: str) -> str:
    """Normalize supported backbone names and aliases."""
    normalized = backbone_name.lower().replace("-", "_")
    return BACKBONE_ALIASES.get(normalized, normalized)


def build_backbone(backbone_name: str = "resnet18", pretrained: bool = True) -> tuple[nn.Module, int]:
    """Build a supported grayscale feature extractor and return its feature dim."""
    normalized_name = normalize_backbone_name(backbone_name)
    try:
        builder = BACKBONE_BUILDERS[normalized_name]
    except KeyError as exc:
        valid_names = ", ".join(sorted(BACKBONE_BUILDERS))
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Expected one of: {valid_names}."
        ) from exc

    return builder(pretrained)
