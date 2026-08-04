import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

test = pd.read_csv("mc_test_predictions.csv")

y_true = test["y_true"].values
y_pred = test["pred_mean"].values

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

out = pd.DataFrame([{
    "model": "ResNet50+I (MC mean)",
    "test_samples": len(test),
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

out.to_csv("final_point_metrics.csv", index=False, encoding="utf-8-sig")

print(out)