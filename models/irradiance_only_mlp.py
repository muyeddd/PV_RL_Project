import torch
import torch.nn as nn


class IrradianceOnlyMLP(nn.Module):
    """Small regressor that uses only the raw irradiance scalar."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, irradiance):
        if irradiance.ndim != 1:
            raise ValueError(
                "irradiance must have shape [batch], "
                f"got {tuple(irradiance.shape)}"
            )
        output = self.regressor(irradiance.unsqueeze(1))
        return torch.sigmoid(output)
