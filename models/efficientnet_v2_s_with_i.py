import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class SolarEfficientNetV2SWithI(nn.Module):
    def __init__(self, dropout=0.3, use_pretrained=True):
        super().__init__()

        # EfficientNetV2-S backbone
        if use_pretrained:
            try:
                backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
            except Exception:
                backbone = efficientnet_v2_s(pretrained=True)
        else:
            backbone = efficientnet_v2_s(weights=None)

        # torchvision 的 EfficientNet classifier 通常是 Sequential(Dropout, Linear)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Identity()
        self.backbone = backbone

        # I 分支
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True),
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(in_features + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_img, x_i):
        img_feat = self.backbone(x_img)
        i_feat = self.i_branch(x_i.unsqueeze(1))
        feat = torch.cat([img_feat, i_feat], dim=1)
        out = self.regressor(feat)
        return torch.sigmoid(out)