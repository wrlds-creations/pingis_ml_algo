"""Four-class CNN candidates for onset-anchored bounce classification."""

from __future__ import annotations

import torch
from torch import nn


class BounceCandidateCnn(nn.Module):
    def __init__(self, num_classes: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        channels = (1, 16, 32, 64)
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            blocks.extend(
                (
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                )
            )
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, num_classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ResidualBounceCandidateCnn(nn.Module):
    """Mobile-sized residual CNN supporting one or two frontend channels."""

    def __init__(
        self,
        num_classes: int = 4,
        input_channels: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        widths = (24, 48, 96)
        stages: list[nn.Module] = []
        current_channels = input_channels
        for stage_index, width in enumerate(widths):
            stages.extend(
                (
                    nn.Conv2d(
                        current_channels,
                        width,
                        kernel_size=3,
                        stride=1 if stage_index == 0 else 2,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(width),
                    nn.SiLU(inplace=True),
                    ResidualBlock(width, dropout=0.05),
                )
            )
            current_channels = width
        self.features = nn.Sequential(*stages)
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.maximum_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1] * 2, 96),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        pooled = torch.cat(
            (self.average_pool(features), self.maximum_pool(features)),
            dim=1,
        )
        return self.classifier(pooled)
