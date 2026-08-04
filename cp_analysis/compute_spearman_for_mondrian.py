import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv("mondrian_test_predictions.csv")

df["abs_error"] = (df["true_L"] - df["pred_L"]).abs()

methods = [
    ("Split CP", "split_width"),
    ("Irradiance-Mondrian CP", "i_mondrian_width"),
    ("Pred-L-Mondrian CP", "pred_mondrian_width"),
]

rows = []

for method, width_col in methods:
    rho, p_value = spearmanr(df[width_col], df["abs_error"])

    rows.append({
        "method": method,
        "spearman_width_error": rho,
        "p_value": p_value,
        "mean_width": df[width_col].mean(),
        "median_width": df[width_col].median(),
    })

out = pd.DataFrame(rows)
out.to_csv("mondrian_spearman.csv", index=False, encoding="utf-8-sig")

print(out)