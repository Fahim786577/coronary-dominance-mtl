"""Temporal multi-task ResNet18 model for coronary tasks."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn

from src.models.backbones import build_resnet18_grayscale


class CoronaryTemporalMTL(nn.Module):
    """Shared ResNet18 extractor with temporal and single-frame task heads."""

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 5,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone, feature_dim = build_resnet18_grayscale(pretrained=pretrained)
        self.extractor = self.backbone
        self.feature_dim = feature_dim

        self.spatial_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_model = nn.LSTM(
            input_size=feature_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0.0,
        )
        self.occlusion_head = nn.Sequential(
            nn.Linear(lstm_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )
        self.frame_quality_head = self._build_single_frame_head(feature_dim, num_classes)
        self.dominance_head = self._build_single_frame_head(feature_dim, num_classes)

    @staticmethod
    def _build_single_frame_head(feature_dim: int, num_classes: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _has_images(images: Tensor | None) -> bool:
        return images is not None and images.numel() > 0

    def _extract_single_frame_features(self, images: Tensor) -> Tensor:
        features = self.extractor(images)
        features = self.spatial_pooling(features)
        return features.flatten(1)

    def _forward_occlusion(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise ValueError("occlusion_images must have shape [B, S, 1, H, W].")

        batch_size, sequence_length = images.shape[:2]
        frame_features = self._extract_single_frame_features(
            images.reshape(batch_size * sequence_length, *images.shape[2:])
        )
        sequence_features = frame_features.reshape(batch_size, sequence_length, self.feature_dim)
        temporal_out, _ = self.temporal_model(sequence_features)
        return self.occlusion_head(temporal_out[:, -1, :])

    def _forward_single_frame(self, images: Tensor, head: nn.Module, name: str) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"{name} must have shape [B, 1, H, W].")
        return head(self._extract_single_frame_features(images))

    def forward(self, images: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Return raw logits for each task present in the input dictionary."""
        outputs: dict[str, Tensor] = {}

        occlusion_images = images.get("occlusion_images")
        if self._has_images(occlusion_images):
            outputs["occlusion"] = self._forward_occlusion(occlusion_images)

        frame_quality_images = images.get("frame_quality_images")
        if self._has_images(frame_quality_images):
            outputs["frame_quality"] = self._forward_single_frame(
                frame_quality_images,
                self.frame_quality_head,
                "frame_quality_images",
            )

        dominance_images = images.get("dominance_images")
        if self._has_images(dominance_images):
            outputs["dominance"] = self._forward_single_frame(
                dominance_images,
                self.dominance_head,
                "dominance_images",
            )

        return outputs
