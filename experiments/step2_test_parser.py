import os
import re

def parse_filename(filename):
    """
    解析文件名：
    solar_Fri_Jun_16_10__0__11_2017_L_0.906153208302_I_0.321592156863.jpg
    """
    name = filename.replace(".jpg", "")
    parts = name.split("_")

    # 最后两项分别是 L 和 I 的数值
    # 例如：..._L_0.906153..._I_0.321592...
    try:
        L_index = parts.index("L")
        I_index = parts.index("I")
        L = float(parts[L_index + 1])
        I = float(parts[I_index + 1])
    except Exception as e:
        raise ValueError(f"文件名解析失败: {filename}") from e

    # 时间信息放在 solar 后面，L 前面
    time_info = "_".join(parts[1:L_index])

    return {
        "L": L,
        "I": I,
        "time": time_info,
        "filename": filename
    }


project_root = r"E:\PV_RL_Project"
img_dir = os.path.join(project_root, "data", "raw", "PanelImages")

files = os.listdir(img_dir)

print("总图片数量:", len(files))
print("前5个文件:")

for f in files[:5]:
    print("\n原始文件名:", f)
    print("解析结果:", parse_filename(f))