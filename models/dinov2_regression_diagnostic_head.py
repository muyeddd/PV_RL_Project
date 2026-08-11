"""Fixed regression heads for the frozen-DINOv2 Fold4 diagnostic."""

import torch
import torch.nn as nn


class DINOv2DiagnosticRegressionHead(nn.Module):
    """384 -> 128 -> 1, optionally followed by the v1 Sigmoid."""

    def __init__(self, output_activation: str):
        super().__init__()
        if output_activation not in {"Sigmoid", "Linear"}:
            raise ValueError("output_activation must be Sigmoid or Linear")
        layers: list[nn.Module] = [
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        ]
        if output_activation == "Sigmoid":
            layers.append(nn.Sigmoid())
        self.output_activation = output_activation
        self.regressor = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 384:
            raise ValueError(f"Expected [N, 384] features, got {tuple(features.shape)}")
        return self.regressor(features)
