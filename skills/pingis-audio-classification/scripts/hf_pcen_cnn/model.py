"""Small four-class CNN used for the frontend ablation."""

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
