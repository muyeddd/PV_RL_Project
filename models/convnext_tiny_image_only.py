"""Isolated ConvNeXt-Tiny image-only regressor for normalized PV loss L."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


OFFICIAL_WEIGHTS = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
OFFICIAL_CHECKPOINT_FILENAME = "convnext_tiny-983f1562.pth"
OFFICIAL_CHECKPOINT_SHA256 = (
    "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"
)
CLASSIFIER_INPUT_FEATURES = 768


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def official_checkpoint_path() -> Path:
    """Return the expected torch-hub cache path without accessing the network."""

    enum_filename = Path(urlparse(OFFICIAL_WEIGHTS.url).path).name
    if enum_filename != OFFICIAL_CHECKPOINT_FILENAME:
        raise RuntimeError(
            "Installed torchvision ConvNeXt-Tiny weight filename changed: "
            f"expected {OFFICIAL_CHECKPOINT_FILENAME}, got {enum_filename}"
        )
    return Path(torch.hub.get_dir()) / "checkpoints" / enum_filename


def verify_official_checkpoint(path: Path | None = None) -> dict[str, object]:
    """Verify the exact cached official checkpoint; never download a weight file."""

    checkpoint_path = Path(path) if path is not None else official_checkpoint_path()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "ConvNeXt-Tiny official weights are not present in the local torch cache: "
            f"{checkpoint_path}. Downloading is intentionally disabled."
        )
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256.lower() != OFFICIAL_CHECKPOINT_SHA256:
        raise RuntimeError(
            "ConvNeXt-Tiny cached checkpoint SHA256 mismatch: "
            f"expected {OFFICIAL_CHECKPOINT_SHA256}, got {actual_sha256}"
        )
    return {
        "weights_enum": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "path": str(checkpoint_path.resolve()),
        "filename": checkpoint_path.name,
        "size_bytes": checkpoint_path.stat().st_size,
        "sha256": actual_sha256.lower(),
    }


def _load_official_state_dict(backbone: nn.Module, checkpoint_path: Path) -> None:
    """Load the already-verified cache file directly, with no URL fallback."""

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state_dict, strict=True)


class SolarConvNeXtTinyImageOnly(nn.Module):
    """Fully trainable ConvNeXt-Tiny with the baseline-compatible regression head."""

    def __init__(self, dropout: float = 0.3, use_pretrained: bool = True):
        super().__init__()
        if dropout != 0.3:
            raise ValueError("The v1 challenger requires dropout=0.3")

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

        # Preserve torchvision's final LayerNorm2d and Flatten exactly. Only the
        # ImageNet 1000-class Linear is replaced by the matched regression head.
        original_classifier[2] = nn.Sequential(
            nn.Linear(CLASSIFIER_INPUT_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.backbone = backbone
        self.pretrained_provenance = provenance

        # The v1 experiment is full-backbone fine-tuning, not frozen features.
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    @property
    def regression_head(self) -> nn.Sequential:
        return self.backbone.classifier[2]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

