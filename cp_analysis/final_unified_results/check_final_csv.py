import pandas as pd

calib = pd.read_csv("mc_calibration_predictions.csv")
test = pd.read_csv("mc_test_predictions.csv")

required = [
    "filename", "y_true", "pred_mean", "pred_std", "I",
    "lower_mc_90", "upper_mc_90", "width_mc_90"
]

for name, df in [("calib", calib), ("test", test)]:
    print("\n==============================")
    print(name)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head())

    missing = [c for c in required if c not in df.columns]
    if missing:
        print("缺少列:", missing)

    print("\n数值范围:")
    print(df[["y_true", "pred_mean", "pred_std", "I"]].describe())

    print("\n重复文件名数量:", df["filename"].duplicated().sum())

overlap = set(calib["filename"]) & set(test["filename"])
print("\n校准集和测试集重合样本数:", len(overlap))

if len(overlap) > 0:
    print("警告：校准集和测试集存在重合文件！")
else:
    print("校准集和测试集无重合，正常。")