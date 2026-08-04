import os
from PIL import Image
import matplotlib.pyplot as plt

project_root = r"E:\PV_RL_Project"
img_dir = os.path.join(project_root, "data", "raw", "PanelImages")

files = os.listdir(img_dir)
sample_file = files[0]
sample_path = os.path.join(img_dir, sample_file)

print("正在读取文件：", sample_file)

img = Image.open(sample_path).convert("RGB")

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.title(sample_file)
plt.axis("off")
plt.show()