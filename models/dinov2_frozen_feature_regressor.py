"""Small regression head for frozen DINOv2-S/14 CLS features."""

import torch
import torch.nn as nn


class DINOv2FrozenFeatureRegressor(nn.Module):
    """384 -> 128 -> 1 regressor; consumes no auxiliary inputs."""

    def __init__(self, input_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        if input_dim != 384 or hidden_dim != 128 or dropout != 0.3:
            raise ValueError("Frozen regression v1 requires input=384, hidden=128, dropout=0.3")
        self.regressor = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != 384:
            raise ValueError(f"Expected feature tensor [N, 384], got {tuple(features.shape)}")
        return self.regressor(features)
