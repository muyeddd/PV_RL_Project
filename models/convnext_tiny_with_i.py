import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class SolarConvNeXtTinyWithI(nn.Module):
    def __init__(self, dropout=0.3, use_pretrained=True):
        super().__init__()

        # 1) ConvNeXt-Tiny backbone
        if use_pretrained:
            try:
                weights = ConvNeXt_Tiny_Weights.DEFAULT
                backbone = convnext_tiny(weights=weights)
            except Exception:
                backbone = convnext_tiny(pretrained=True)
        else:
            backbone = convnext_tiny(weights=None)

        # 2) 取出 backbone 最后一层输入维度
        # torchvision 的 convnext_tiny classifier 结构是:
        # classifier = [norm, flatten, linear]
        in_features = backbone.classifier[2].in_features

        # 3) 去掉原始 1000 类分类头
        backbone.classifier[2] = nn.Identity()
        self.backbone = backbone

        # 4) I 分支
        self.i_branch = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, 16),
            nn.ReLU(inplace=True)
        )

        # 5) 回归头
        self.regressor = nn.Sequential(
            nn.Linear(in_features + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x_img, x_i):
        img_feat = self.backbone(x_img)          # [B, C]
        i_feat = self.i_branch(x_i.unsqueeze(1)) # [B, 16]
        feat = torch.cat([img_feat, i_feat], dim=1)
        out = self.regressor(feat)
        return torch.sigmoid(out)