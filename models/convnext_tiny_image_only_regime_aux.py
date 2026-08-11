"""ConvNeXt-Tiny image-only regressor with a diagnostic regime auxiliary head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from models.convnext_tiny_image_only import (
    CLASSIFIER_INPUT_FEATURES,
    OFFICIAL_CHECKPOINT_FILENAME,
    OFFICIAL_CHECKPOINT_SHA256,
    OFFICIAL_WEIGHTS,
    _load_official_state_dict,
    verify_official_checkpoint,
)


class RegimeAuxiliaryHead(nn.Module):
    """Baseline-width shared representation feeding regression and regime heads."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(CLASSIFIER_INPUT_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.regression = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.regime = nn.Linear(128, 3)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.shared(features)
        return self.regression(shared), self.regime(shared)


class SolarConvNeXtTinyImageOnlyRegimeAux(nn.Module):
    """Fully trainable baseline ConvNeXt with one unweighted auxiliary task."""

    def __init__(self, dropout: float = 0.3, use_pretrained: bool = True):
        super().__init__()
        if dropout != 0.3:
            raise ValueError("The regime-aware v1 experiment requires dropout=0.3")

        backbone = convnext_tiny(weights=None)
        if use_pretrained:
            provenance = verify_official_checkpoint()
            _load_official_state_dict(backbone, Path(str(provenance["path"])))
        else:
            provenance = {
                "weights_enum": None,
                "path": None,
                "filename": None,
                "size_bytes": None,
                "sha256": None,
            }

        original_classifier = backbone.classifier
        if not isinstance(original_classifier, nn.Sequential):
            raise RuntimeError("Unexpected torchvision ConvNeXt classifier type")
        if len(original_classifier) != 3:
            raise RuntimeError("Unexpected torchvision ConvNeXt classifier length")
        if not isinstance(original_classifier[1], nn.Flatten):
            raise RuntimeError("ConvNeXt final flatten layer is missing")
        if not isinstance(original_classifier[2], nn.Linear):
            raise RuntimeError("ConvNeXt final classification layer is not Linear")
        if original_classifier[2].in_features != CLASSIFIER_INPUT_FEATURES:
            raise RuntimeError(
                "Unexpected ConvNeXt-Tiny classifier input dimension: "
                f"{original_classifier[2].in_features}"
            )

        # Preserve torchvision's final LayerNorm2d and Flatten exactly as in the
        # baseline; replace only the ImageNet Linear with the minimal dual head.
        original_classifier[2] = RegimeAuxiliaryHead(dropout=dropout)
        self.backbone = backbone
        self.pretrained_provenance = provenance

        for parameter in self.parameters():
            parameter.requires_grad_(True)

    @property
    def auxiliary_head(self) -> RegimeAuxiliaryHead:
        return self.backbone.classifier[2]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone(images)


__all__ = [
    "OFFICIAL_CHECKPOINT_FILENAME",
    "OFFICIAL_CHECKPOINT_SHA256",
    "OFFICIAL_WEIGHTS",
    "RegimeAuxiliaryHead",
    "SolarConvNeXtTinyImageOnlyRegimeAux",
    "verify_official_checkpoint",
]
