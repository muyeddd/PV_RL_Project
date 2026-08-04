Final unified experiment protocol

Project:
PV soiling-induced power loss trustworthy prediction

Base directory:
E:\PV_RL_Project\cp_analysis\final_unified_results

Base prediction files:
mc_calibration_predictions.csv
mc_test_predictions.csv

Calibration samples:
6863

Test samples:
6864

Ground truth:
y_true

Point prediction:
pred_mean

Uncertainty:
pred_std

Irradiance:
I

Raw MC interval:
lower_mc_90, upper_mc_90

Final base predictor:
ResNet50+I with MC mean prediction

Final point metrics:
see final_point_metrics.csv

Final interval metrics:
see final_interval_summary.csv

Final risk-coverage metrics:
see final_risk_summary.csv

Important note:
Earlier model comparison results, such as RMSE=0.0722 and MAE=0.0362, are used only for model selection.
The final trustworthy prediction analysis uses the unified mc_calibration_predictions.csv and mc_test_predictions.csv files.