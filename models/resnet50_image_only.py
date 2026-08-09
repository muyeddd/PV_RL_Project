import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SolarResNet50ImageOnly(nn.Module):
    """Image-only ResNet50 regressor for normalized PV loss prediction."""

    def __init__(self, dropout=0.3, use_pretrained=True):
        super().__init__()

        if use_pretrained:
            try:
                backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
            except Exception:
                backbone = resnet50(pretrained=True)
        else:
            backbone = resnet50(weights=None)

        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.regressor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, images):
        image_features = self.backbone(images)
        output = self.regressor(image_features)
        return torch.sigmoid(output)
