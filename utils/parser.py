import os
import re
# test git
def parse_filename(filename):
    """
    解析类似下面的文件名：
    solar_Fri_Jun_16_10__0__11_2017_L_0.906153208302_I_0.321592156863.jpg

    返回：
    time_str, L, I
    """
    name = os.path.basename(filename)
    name = name.replace(".jpg", "")

    pattern = r"^solar_(.+?)_L_([0-9eE+\-.]+)_I_([0-9eE+\-.]+)$"
    match = re.match(pattern, name)

    if match is None:
        raise ValueError(f"无法解析文件名: {filename}")

    time_str = match.group(1)
    L = float(match.group(2))
    I = float(match.group(3))

    return time_str, L, I