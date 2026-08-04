import torch
import torch.nn as nn
import torchvision.models as models

class SolarResNetWithI(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()

        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(pretrained=True)

        # 去掉原始分类头，保留图像特征
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # I 分支：处理辐照度标量
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU()
        )

        # 融合后回归头
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