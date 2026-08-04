import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class SolarEfficientNetB0WithI(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        # 预训练 EfficientNet-B0
        try:
            backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except Exception:
            backbone = efficientnet_b0(pretrained=True)

        # EfficientNet 的分类头替换掉
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # I 分支
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # 融合回归头
        self.head = nn.Sequential(
            nn.Linear(in_features + 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_i):
        img_feat = self.backbone(x_img)
        i_feat = self.i_branch(x_i.unsqueeze(1))
        feat = torch.cat([img_feat, i_feat], dim=1)
        out = self.head(feat)
        return torch.sigmoid(out)