import sys
import os

project_root = r"E:\PV_RL_Project"
sys.path.insert(0, project_root)

from utils.parser import parse_filename

sample = "solar_Fri_Jun_16_10__0__11_2017_L_0.906153208302_I_0.321592156863.jpg"

time_str, L, I = parse_filename(sample)

print("time_str:", time_str)
print("L:", L)
print("I:", I)