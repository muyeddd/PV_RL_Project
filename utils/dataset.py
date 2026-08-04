import os
from PIL import Image
from torch.utils.data import Dataset
import torch

from utils.parser import parse_filename


class SolarDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.files = sorted(os.listdir(img_dir))
        self.transform = transform

    def parse_time_feature(self, time_str):
        """
        time_str 例子：
        Fri_Jun_16_10__0__11_2017

        我们提取 hour, minute, second，转成一个连续数值特征：
        hour + minute/60 + second/3600
        """
        parts = time_str.split("_")

        # parts 可能是：
        # ['Fri', 'Jun', '16', '10', '', '0', '', '11', '2017']
        hour = int(parts[3])
        minute = int(parts[5])
        second = int(parts[7])

        time_float = hour + minute / 60.0 + second / 3600.0
        return time_float

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        path = os.path.join(self.img_dir, filename)

        image = Image.open(path).convert("RGB")

        time_str, L, I = parse_filename(filename)
        time_float = self.parse_time_feature(time_str)

        if self.transform:
            image = self.transform(image)

        L = torch.tensor(L, dtype=torch.float32)
        I = torch.tensor(I, dtype=torch.float32)
        time_feat = torch.tensor(time_float, dtype=torch.float32)

        return image, L, I, time_feat