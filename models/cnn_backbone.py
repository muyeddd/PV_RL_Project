import torch
import torch.nn as nn
import torchvision.models as models

class SolarResNet(nn.Module):
    def __init__(self):
        super(SolarResNet, self).__init__()

        self.backbone = models.resnet18(pretrained=True)

        # 改最后一层：回归任务（输出1个值 L）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)