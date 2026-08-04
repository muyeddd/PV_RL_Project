import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image

from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from models.resnet50_with_i import SolarResNet50WithI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SolarResNet50WithI(dropout=0.3)
model.load_state_dict(
    torch.load(
        r"E:\PV_RL_Project\outputs\models_ckpt\best_resnet50_with_i.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

print("Model loaded.")