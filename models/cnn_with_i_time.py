import torch
import torch.nn as nn
import torchvision.models as models


class SolarResNetWithITime(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=True)

        # 去掉原来的分类头，只保留图像特征
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # I 分支
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # 时间分支
        self.time_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # 融合回归头
        self.head = nn.Sequential(
            nn.Linear(in_features + 16 + 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_i, x_time):
        img_feat = self.backbone(x_img)          # [B, 512]
        i_feat = self.i_branch(x_i.unsqueeze(1)) # [B, 16]
        t_feat = self.time_branch(x_time.unsqueeze(1))  # [B, 16]

        feat = torch.cat([img_feat, i_feat, t_feat], dim=1)
        out = self.head(feat)

        return torch.sigmoid(out)