import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SolarResNet50WithI(nn.Module):
    def __init__(self, dropout=0.3, use_pretrained=True):
        super().__init__()

        # 预训练 ResNet50
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

        # I 分支
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(in_features + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_i):
        img_feat = self.backbone(x_img)
        i_feat = self.i_branch(x_i.unsqueeze(1))
        feat = torch.cat([img_feat, i_feat], dim=1)
        out = self.regressor(feat)
        return torch.sigmoid(out)