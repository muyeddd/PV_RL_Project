import pandas as pd
import os

files = [
    "conformal_summary.csv",
    "coverage_by_bins_90.csv",
    "test_conformal_predictions.csv",
    "mc_calibration_predictions.csv",
    "mc_test_predictions.csv",
    "mc_summary.csv",
    "mc_conformal_summary.csv",
    "mc_conformal_coverage_by_bins_90.csv",
    "mc_conformal_test_prediction.csv",
    "mc_vs_mc_conformal_compare.csv",
]

for file in files:
    if not os.path.exists(file):
        print(f"\n[缺失] {file}")
        continue

    print("\n" + "=" * 80)
    print(f"文件: {file}")

    df = pd.read_csv(file)
    print("行列数:", df.shape)
    print("列名:")
    print(list(df.columns))

    print("\n前5行:")
    print(df.head())

    print("\n数值列统计:")
    print(df.describe(include="all"))