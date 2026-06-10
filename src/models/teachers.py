"""Single-task teacher model scaffolds for later distillation work."""

from __future__ import annotations

from torch import Tensor, nn

from src.models.backbones import build_resnet18_grayscale


class SingleFrameTeacher(nn.Module):
    """ResNet18 single-frame teacher for frame quality or dominance."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone, feature_dim = build_resnet18_grayscale(pretrained=pretrained)
        self.extractor = self.backbone
        self.spatial_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw logits for images shaped [B, 1, H, W]."""
        features = self.extractor(images)
        features = self.spatial_pooling(features).flatten(1)
        return self.classifier(features)


class VideoTeacher(nn.Module):
    """ResNet18 plus LSTM teacher for occlusion clips."""

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
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return raw logits for clips shaped [B, S, 1, H, W]."""
        if images.ndim != 5:
            raise ValueError("VideoTeacher expects images shaped [B, S, 1, H, W].")

        batch_size, sequence_length = images.shape[:2]
        frame_features = self.extractor(images.reshape(batch_size * sequence_length, *images.shape[2:]))
        frame_features = self.spatial_pooling(frame_features).flatten(1)
        sequence_features = frame_features.reshape(batch_size, sequence_length, self.feature_dim)
        temporal_out, _ = self.temporal_model(sequence_features)
        return self.classifier(temporal_out[:, -1, :])
