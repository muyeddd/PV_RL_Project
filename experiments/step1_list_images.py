import os

img_dir = r"E:\PV_RL_Project\data\raw\PanelImages"

files = os.listdir(img_dir)

print("图片数量:", len(files))
print("前5个文件:")
print(files[:5])