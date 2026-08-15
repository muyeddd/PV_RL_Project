"""Paper1 Stage6 frozen final evaluation for RANDOM_TEST then sealed dates.

Importing this module reads no formal data and performs no inference. Formal
execution is restricted to two fixed targets and frozen model/protocol inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments import build_paper1_clean_random_v1 as split_builder
from experiments import run_paper1_cqr_stage3a1_inference_v1 as stage3a1
from experiments import run_paper1_cqr_stage3a2_intervals_v1 as stage3a2
from experiments import run_paper1_decision_stage4a_rule_v1 as stage4a
from experiments import run_paper1_economic_stage5a_normalized_v1 as stage5a
from experiments import run_paper1_uq_stage1a_inference_v1 as stage1a
from experiments import run_paper1_uq_stage1b_intervals_v1 as stage1b

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "paper1_clean_random_v1"
STAGE = "final_stage6_v1"
TARGET_RANDOM_TEST = "random_test"
TARGET_SEALED_DATES = "sealed_dates"
TARGET_ORDER = (TARGET_RANDOM_TEST, TARGET_SEALED_DATES)
RANDOM_ROLE = "RANDOM_TEST"
SEALED_ROLE = "SEALED_DATES"
EXPECTED_N = {TARGET_RANDOM_TEST: 2582, TARGET_SEALED_DATES: 8855}
SEALED_DATES = ("2017-06-15", "2017-06-24", "2017-06-30")
ALPHA = 0.10
TARGET_COVERAGE = 0.90
QHAT = 0.004862844288256299
TAU_GRID = (0.05, 0.10, 0.15, 0.20)
REFERENCE_TAU = 0.15
METHOD_ORDER = stage4a.METHOD_ORDER
POINT_CHECKPOINT = stage1a.CLEAN_CHECKPOINT
POINT_CHECKPOINT_SHA256 = "97f3ec016cf99f83a78e28e2b4aca24787203f105243447d908da739c295de23"
CQR_CHECKPOINT = stage3a1.SOURCE_CQR_CHECKPOINT
CQR_CHECKPOINT_SHA256 = "fd5deea62c867fcffe3791f768752da9dc3a39a1c146244b1e225d6b40b0da80"
QHAT_ARTIFACT = stage3a2.OUTPUT_DIR / "cqr_conformal_calibration.json"
RANDOM_MANIFEST = split_builder.OUTPUT_DIR / "random_test.csv"
SEALED_SOURCE_MANIFEST = split_builder.SOURCE_DATE_MANIFEST
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / PROTOCOL / STAGE
MANIFEST_COLUMNS = ("sample_id", "date", "timestamp", "image_path", "role")
BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
BIN_LABELS = ("[0.0,0.2]", "(0.2,0.4]", "(0.4,0.6]", "(0.6,0.8]", "(0.8,1.0]")
TOL = 1e-12
FORBIDDEN_FIELDS = frozenset({"winner", "best", "rank", "ranking", "recommended", "selected_method", "optimal_tau"})

PREDICTION_COLUMNS = MANIFEST_COLUMNS + ("target", "true_L", "irradiance", "point_pred", "q05", "q50", "q95", "lower", "upper", "width", "covered", "lower_clipped", "upper_clipped")
PREDICTION_METRIC_COLUMNS = ("target", "scope", "date", "model", "N", "R2", "RMSE", "MAE", "bias")
INTERVAL_METRIC_COLUMNS = ("target", "scope", "date", "method", "N", "alpha", "target_coverage", "PICP", "MPIW", "median_width", "coverage_error", "mean_interval_score_alpha_0p10", "lower_clipped_n", "upper_clipped_n")
CONDITIONAL_COLUMNS = ("target", "scope", "date", "binning_variable", "bin_label", "N", "PICP", "MPIW")
DECISION_ACTION_COLUMNS = stage4a.ACTION_COLUMNS
DECISION_METRIC_COLUMNS = ("target", "scope", "date") + stage4a.METRIC_COLUMNS
ECONOMIC_SAMPLE_COLUMNS = stage5a.SAMPLE_REGRET_COLUMNS
ECONOMIC_METRIC_COLUMNS = ("target", "scope", "date") + stage5a.METRIC_COLUMNS
BREAK_EVEN_COLUMNS = ("target", "scope", "date") + stage5a.BREAK_EVEN_COLUMNS


def _resolved(path: Path) -> Path:
    return Path(path).resolve()


def project_relative(path: Path) -> str:
    return str(_resolved(path).relative_to(PROJECT_ROOT.resolve())).replace(os.sep, "/")


def validate_constants() -> None:
    if PROTOCOL != stage4a.PROTOCOL or PROTOCOL != stage5a.PROTOCOL:
        raise ValueError("Frozen protocol drift")
    if TARGET_ORDER != ("random_test", "sealed_dates"):
        raise ValueError("Final target order drift")
    if ALPHA != stage3a2.ALPHA or ALPHA != 0.10 or QHAT != 0.004862844288256299:
        raise ValueError("Frozen alpha/qhat drift")
    if TAU_GRID != stage4a.TAU_GRID or METHOD_ORDER != stage4a.METHOD_ORDER:
        raise ValueError("Frozen decision protocol drift")
    if BIN_EDGES != tuple(stage1b.PRED_L_BINS) or BIN_EDGES != tuple(stage1b.IRRADIANCE_BINS):
        raise ValueError("Frozen conditional bins drift")
    if POINT_CHECKPOINT_SHA256 != stage1a.CLEAN_CHECKPOINT_SHA256 or CQR_CHECKPOINT_SHA256 != stage3a1.SOURCE_CQR_CHECKPOINT_SHA256:
        raise ValueError("Frozen checkpoint SHA drift")


def validate_target(target: str) -> str:
    validate_constants()
    if target not in TARGET_ORDER:
        raise PermissionError(f"Unauthorized Stage6 target: {target!r}")
    return target


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_file(path: Path, expected_sha256: str, authorized_path: Path) -> str:
    candidate = _resolved(path)
    if candidate != _resolved(authorized_path):
        raise PermissionError(f"Unauthorized checkpoint path: {candidate}")
    observed = sha256_file(candidate)
    if observed.lower() != expected_sha256.lower():
        raise ValueError(f"Checkpoint SHA256 mismatch: expected {expected_sha256}, observed {observed}")
    return observed.lower()


def validate_qhat_artifact(value: Mapping[str, Any]) -> float:
    if not math.isclose(float(value.get("alpha")), ALPHA, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Frozen qhat artifact alpha mismatch")
    qhat = float(value.get("qhat"))
    if not math.isclose(qhat, QHAT, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Frozen qhat mismatch")
    return qhat


def load_frozen_qhat(path: Path = QHAT_ARTIFACT) -> float:
    if _resolved(path) != _resolved(QHAT_ARTIFACT):
        raise PermissionError("Unauthorized qhat artifact")
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("qhat artifact must be a mapping")
    return validate_qhat_artifact(payload)


def validate_random_completion_provenance(value: Mapping[str, Any]) -> None:
    expected = {"protocol": PROTOCOL, "stage": STAGE, "target": TARGET_RANDOM_TEST, "formal_final_evaluation": True, "random_test_accessed": True, "sealed_final_dates_accessed": False, "point_checkpoint_sha256_verified": True, "cqr_checkpoint_sha256_verified": True}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise PermissionError("Valid completed RANDOM_TEST Stage6 provenance is required")


def require_random_completion(path: Path | None = None) -> None:
    marker = path or (OUTPUT_ROOT / TARGET_RANDOM_TEST / "provenance.json")
    if _resolved(marker) != _resolved(OUTPUT_ROOT / TARGET_RANDOM_TEST / "provenance.json"):
        raise PermissionError("Unauthorized RANDOM_TEST completion marker")
    if not Path(marker).is_file():
        raise FileNotFoundError("RANDOM_TEST Stage6 provenance must exist before sealed execution")
    with Path(marker).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("RANDOM_TEST provenance must be a mapping")
    validate_random_completion_provenance(value)


def validate_manifest(frame: pd.DataFrame, target: str, *, enforce_expected_n: bool = True) -> pd.DataFrame:
    validate_target(target)
    if tuple(frame.columns) != MANIFEST_COLUMNS or frame.empty:
        raise ValueError("Final manifest schema/emptiness guard failed")
    if enforce_expected_n and len(frame) != EXPECTED_N[target]:
        raise ValueError(f"{target} N guard failed: expected {EXPECTED_N[target]}, got {len(frame)}")
    if frame["sample_id"].isna().any() or frame["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique")
    if frame["image_path"].isna().any() or frame["image_path"].duplicated().any():
        raise ValueError("image_path must be unique")
    dates = tuple(pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d"))
    if target == TARGET_RANDOM_TEST:
        if set(frame["role"].astype(str)) != {RANDOM_ROLE}:
            raise PermissionError("RANDOM_TEST role guard failed")
        if set(dates) & set(SEALED_DATES):
            raise PermissionError("RANDOM_TEST must exclude sealed dates")
    else:
        if set(frame["role"].astype(str)) != {SEALED_ROLE}:
            raise PermissionError("SEALED_DATES role guard failed")
        if set(dates) != set(SEALED_DATES):
            raise PermissionError("Exact sealed date set is required")
        if RANDOM_ROLE in set(frame["role"].astype(str)):
            raise PermissionError("RANDOM_TEST role forbidden in sealed mode")
    return frame.loc[:, MANIFEST_COLUMNS].copy()


def load_manifest(target: str) -> pd.DataFrame:
    target = validate_target(target)
    if target == TARGET_RANDOM_TEST:
        frame = pd.read_csv(RANDOM_MANIFEST, usecols=list(MANIFEST_COLUMNS))
    else:
        source = pd.read_csv(SEALED_SOURCE_MANIFEST, usecols=["filename", "timestamp", "date"])
        source = source.loc[source["date"].astype(str).isin(SEALED_DATES)].copy()
        frame = pd.DataFrame({"sample_id": [split_builder.sample_id_from_timestamp(str(v)) for v in source["timestamp"]], "date": source["date"].astype(str), "timestamp": source["timestamp"].astype(str), "image_path": source["filename"].astype(str), "role": SEALED_ROLE})
    return validate_manifest(frame.loc[:, MANIFEST_COLUMNS], target)


def prepare_records(manifest: pd.DataFrame, target: str, stats: Mapping[str, Any], *, enforce_expected_n: bool = True) -> pd.DataFrame:
    guarded = validate_manifest(manifest, target, enforce_expected_n=enforce_expected_n)
    frozen = stage3a1.validate_irradiance_stats(stats)
    result = stage3a1.cqr_train.point_train.attach_development_values(guarded)
    result["irradiance"] = result["irradiance_raw"].astype(float)
    result["irradiance_normalized"] = (result["irradiance"] - float(frozen["mean"])) / float(frozen["std_ddof0"])
    values = result[["true_L", "irradiance", "irradiance_normalized"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Final truth/irradiance values must be finite")
    return result.loc[:, MANIFEST_COLUMNS + ("true_L", "irradiance", "irradiance_normalized")]


def conformalize(q05: Sequence[float], q95: Sequence[float], qhat: float = QHAT) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lo = np.asarray(q05, dtype=np.float64); hi = np.asarray(q95, dtype=np.float64)
    if lo.ndim != 1 or hi.shape != lo.shape or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError("Invalid CQR interval inputs")
    if not math.isclose(float(qhat), QHAT, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Only frozen qhat is authorized")
    raw_lo = lo - qhat; raw_hi = hi + qhat
    return np.clip(raw_lo, 0.0, 1.0), np.clip(raw_hi, 0.0, 1.0), raw_lo < 0.0, raw_hi > 1.0


def build_predictions(records: pd.DataFrame, point: Sequence[float], quantiles: np.ndarray, target: str, *, enforce_expected_n: bool = True) -> pd.DataFrame:
    n = len(records); point_values = np.asarray(point, dtype=np.float64)
    record_values = records.loc[:, ("true_L", "irradiance")].to_numpy(dtype=np.float64)
    if not np.isfinite(record_values).all():
        raise ValueError("Final truth/irradiance values must be finite")
    q = stage3a1.validate_quantile_array(np.asarray(quantiles), n)
    if point_values.shape != (n,) or not np.isfinite(point_values).all():
        raise ValueError("Invalid point predictions")
    lower, upper, lower_clip, upper_clip = conformalize(q[:, 0], q[:, 2])
    result = records.loc[:, MANIFEST_COLUMNS + ("true_L", "irradiance")].copy()
    result["target"] = validate_target(target); result["point_pred"] = point_values
    result["q05"] = q[:, 0]; result["q50"] = q[:, 1]; result["q95"] = q[:, 2]
    result["lower"] = lower; result["upper"] = upper; result["width"] = upper - lower
    result["covered"] = (result["true_L"] >= lower) & (result["true_L"] <= upper)
    result["lower_clipped"] = lower_clip; result["upper_clipped"] = upper_clip
    if enforce_expected_n and n != EXPECTED_N[target]:
        raise ValueError("Final prediction N guard failed")
    return result.loc[:, PREDICTION_COLUMNS]


def prediction_metrics(truth: Sequence[float], prediction: Sequence[float]) -> dict[str, float]:
    y = np.asarray(truth, dtype=np.float64); p = np.asarray(prediction, dtype=np.float64)
    if y.ndim != 1 or y.size == 0 or p.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Invalid prediction metric inputs")
    residual = p - y; sst = float(np.square(y - y.mean()).sum())
    return {"R2": float(1.0 - np.square(residual).sum() / sst) if sst > 0 else float("nan"), "RMSE": float(np.sqrt(np.square(residual).mean())), "MAE": float(np.abs(residual).mean()), "bias": float(residual.mean())}


def interval_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["true_L"].to_numpy(float); lo = frame["lower"].to_numpy(float); hi = frame["upper"].to_numpy(float)
    covered = (y >= lo) & (y <= hi); width = hi - lo; picp = float(covered.mean())
    return {"N": len(frame), "alpha": ALPHA, "target_coverage": TARGET_COVERAGE, "PICP": picp, "MPIW": float(width.mean()), "median_width": float(np.median(width)), "coverage_error": abs(picp - TARGET_COVERAGE), "mean_interval_score_alpha_0p10": float(stage1b.standard_interval_score(y, lo, hi).mean()), "lower_clipped_n": int(frame["lower_clipped"].astype(bool).sum()), "upper_clipped_n": int(frame["upper_clipped"].astype(bool).sum())}


def scopes(predictions: pd.DataFrame, target: str) -> list[tuple[str, str, pd.DataFrame]]:
    result = [("pooled", "", predictions)]
    if target == TARGET_SEALED_DATES:
        result.extend(("per_date", date, predictions.loc[predictions["date"].astype(str) == date]) for date in SEALED_DATES)
    return result


def build_prediction_metric_table(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for scope, date, group in scopes(predictions, target):
        for model, column in (("point_pred", "point_pred"), ("q50", "q50")):
            rows.append({"target": target, "scope": scope, "date": date, "model": model, "N": len(group), **prediction_metrics(group["true_L"], group[column])})
    return pd.DataFrame(rows).loc[:, PREDICTION_METRIC_COLUMNS]


def build_interval_metric_table(predictions: pd.DataFrame, target: str) -> pd.DataFrame:
    return pd.DataFrame([{"target": target, "scope": scope, "date": date, "method": "cqr_v1", **interval_metrics(group)} for scope, date, group in scopes(predictions, target)]).loc[:, INTERVAL_METRIC_COLUMNS]


def conditional_coverage(predictions: pd.DataFrame, target: str, variable: str) -> pd.DataFrame:
    if variable not in ("q50", "irradiance"):
        raise ValueError("Unauthorized conditional variable")
    labels = pd.cut(predictions[variable], bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True, right=True)
    rows = []
    for label in BIN_LABELS:
        group = predictions.loc[np.asarray(labels.astype(object) == label, dtype=bool)]
        rows.append({"target": target, "scope": "pooled", "date": "", "binning_variable": variable, "bin_label": label, "N": len(group), "PICP": float(group["covered"].mean()) if len(group) else None, "MPIW": float(group["width"].mean()) if len(group) else None})
    return pd.DataFrame(rows).loc[:, CONDITIONAL_COLUMNS]


def build_decision_actions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        for tau in TAU_GRID:
            part = predictions.loc[:, ("sample_id", "date", "timestamp", "image_path", "role", "true_L", "point_pred", "q50", "lower", "upper")].copy()
            part["tau"] = tau; part["method"] = method; part["oracle_action"] = stage4a.oracle_actions(part["true_L"], tau)
            if method == stage4a.POINT_THRESHOLD: pred = stage4a.point_threshold_actions(part["point_pred"], tau)
            elif method == stage4a.CQR_Q50_THRESHOLD: pred = stage4a.cqr_q50_threshold_actions(part["q50"], tau)
            else: pred = stage4a.cqr_interval_tristate_actions(part["lower"], part["upper"], tau)
            part["predicted_action"] = pred; rows.append(part.loc[:, DECISION_ACTION_COLUMNS])
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["sample_id", "method", "tau"]).any() or len(result) != len(predictions) * 12:
        raise ValueError("Final decision action expansion failed")
    return result


def build_decision_metrics(actions: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    groups = [("pooled", "", actions)]
    if target == TARGET_SEALED_DATES: groups += [("per_date", date, actions.loc[actions["date"].astype(str) == date]) for date in SEALED_DATES]
    for scope, date, all_rows in groups:
        for method in METHOD_ORDER:
            for tau in TAU_GRID:
                group = all_rows.loc[(all_rows["method"] == method) & (all_rows["tau"] == tau)]
                rows.append({"target": target, "scope": scope, "date": date, "method": method, "tau": tau, "evaluation_role": RANDOM_ROLE if target == TARGET_RANDOM_TEST else SEALED_ROLE, **stage4a.summarize_action_group(group["oracle_action"], group["predicted_action"])})
    result = pd.DataFrame(rows).loc[:, DECISION_METRIC_COLUMNS]
    if not np.allclose(result["auto_decision_coverage"], 1.0 - result["review_rate"], atol=TOL, rtol=0): raise ValueError("ADC/review identity failed")
    return result


def build_economic_sample(actions: pd.DataFrame) -> pd.DataFrame:
    components = [stage5a.sample_regret_components(row.true_L, row.tau, row.predicted_action) for row in actions.itertuples(index=False)]
    return pd.concat([actions.reset_index(drop=True), pd.DataFrame(components)], axis=1).loc[:, ECONOMIC_SAMPLE_COLUMNS]


def build_economic_metrics(sample: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []; groups = [("pooled", "", sample)]
    if target == TARGET_SEALED_DATES: groups += [("per_date", date, sample.loc[sample["date"].astype(str) == date]) for date in SEALED_DATES]
    role = RANDOM_ROLE if target == TARGET_RANDOM_TEST else SEALED_ROLE
    for scope, date, all_rows in groups:
        for method in METHOD_ORDER:
            for tau in TAU_GRID:
                group = all_rows.loc[(all_rows["method"] == method) & (all_rows["tau"] == tau)]
                rows.append({"target": target, "scope": scope, "date": date, "method": method, "tau": tau, "evaluation_role": role, **stage5a.summarize_economic_group(group)})
    result = pd.DataFrame(rows).loc[:, ECONOMIC_METRIC_COLUMNS]
    if not np.allclose(result["mean_total_cost_r0"], result["oracle_mean_cost"] + result["mean_regret_r0"], atol=TOL, rtol=0): raise ValueError("Economic cost identity failed")
    return result


def build_break_even(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for (scope, date), block in metrics.groupby(["scope", "date"], sort=False, dropna=False):
        indexed = block.set_index(["method", "tau"])
        for tau in TAU_GRID:
            point = indexed.loc[(stage4a.POINT_THRESHOLD, tau)]; q50 = indexed.loc[(stage4a.CQR_Q50_THRESHOLD, tau)]; cqr = indexed.loc[(stage4a.CQR_INTERVAL_TRISTATE, tau)]
            rr = float(cqr["review_rate"]); c0 = float(cqr["mean_regret_r0"])
            rp = stage5a.analytic_break_even_ratio(float(point["mean_regret_r0"]), c0, rr, tau); rq = stage5a.analytic_break_even_ratio(float(q50["mean_regret_r0"]), c0, rr, tau)
            if rr > 0:
                if not math.isclose(stage5a.cqr_mean_regret_at_review_ratio(c0, rr, rp, tau), float(point["mean_regret_r0"]), abs_tol=TOL, rel_tol=0) or not math.isclose(stage5a.cqr_mean_regret_at_review_ratio(c0, rr, rq, tau), float(q50["mean_regret_r0"]), abs_tol=TOL, rel_tol=0): raise ValueError("Break-even substitution failed")
            rows.append({"target": target, "scope": scope, "date": date, "tau": tau, "evaluation_role": cqr["evaluation_role"], "N": int(cqr["N"]), "cqr_method": stage4a.CQR_INTERVAL_TRISTATE, "review_n": int(cqr["review_n"]), "review_rate": rr, "cqr_mean_regret_r0": c0, "point_mean_regret_r0": float(point["mean_regret_r0"]), "q50_mean_regret_r0": float(q50["mean_regret_r0"]), "break_even_vs_point": rp, "break_even_vs_q50": rq})
    return pd.DataFrame(rows).loc[:, BREAK_EVEN_COLUMNS]


def ensure_no_forbidden_fields(*frames: pd.DataFrame) -> None:
    for frame in frames:
        if {str(c).lower() for c in frame.columns} & FORBIDDEN_FIELDS: raise ValueError("Forbidden winner/ranking field")


def output_dir_for(target: str) -> Path:
    return OUTPUT_ROOT / validate_target(target)


def ensure_output_available(path: Path) -> None:
    if Path(path).exists() and any(Path(path).iterdir()): raise FileExistsError(f"Refusing overwrite: {path}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle: json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n")


def make_config(target: str) -> dict[str, Any]:
    return {"protocol": PROTOCOL, "stage": STAGE, "target": validate_target(target), "formal_execution_order": list(TARGET_ORDER), "expected_N": EXPECTED_N[target], "sealed_dates": list(SEALED_DATES), "alpha": ALPHA, "qhat": QHAT, "qhat_source": project_relative(QHAT_ARTIFACT), "tau_grid": list(TAU_GRID), "reference_tau": REFERENCE_TAU, "primary_reference_tau": 0.15, "method_order": list(METHOD_ORDER), "primary_final_question": "whether CQR tri-state reduces automatic decision error and normalized decision regret versus both point and q50 thresholds, while explicitly reporting review/auto-decision coverage", "other_taus_are_sensitivity_scenarios_only": True, "tau_selected_from_final": False, "point_checkpoint": project_relative(POINT_CHECKPOINT), "point_checkpoint_sha256_expected": POINT_CHECKPOINT_SHA256, "cqr_checkpoint": project_relative(CQR_CHECKPOINT), "cqr_checkpoint_sha256_expected": CQR_CHECKPOINT_SHA256, "conditional_bin_edges": list(BIN_EDGES), "conditional_bin_labels": list(BIN_LABELS)}


def make_provenance(target: str, *, point_verified: bool, cqr_verified: bool, random_completed: bool) -> dict[str, Any]:
    return {**make_config(target), "formal_final_evaluation": True, "method_development_performed": False, "training_performed": False, "finetuning_performed": False, "point_checkpoint_sha256_verified": point_verified, "cqr_checkpoint_sha256_verified": cqr_verified, "qhat_recomputed_from_final": False, "final_recalibration_performed": False, "mc_dropout_performed": False, "new_risk_score_developed": False, "method_selected_from_final": False, "winner_declared": False, "currency_used": False, "gansu_price_used_in_core_evaluation": False, "random_test_accessed": target == TARGET_RANDOM_TEST, "sealed_final_dates_accessed": target == TARGET_SEALED_DATES, "random_test_stage6_completed_before_sealed": random_completed if target == TARGET_SEALED_DATES else False, "sealed_date_recalibration_performed": False}


def write_outputs(target: str, output_dir: Path, predictions: pd.DataFrame, pred_metrics: pd.DataFrame, int_metrics: pd.DataFrame, cond_q50: pd.DataFrame, cond_i: pd.DataFrame, actions: pd.DataFrame, decision_metrics: pd.DataFrame, sample_econ: pd.DataFrame, econ_metrics: pd.DataFrame, break_even: pd.DataFrame, config: Mapping[str, Any], provenance: Mapping[str, Any]) -> None:
    if _resolved(output_dir) != _resolved(output_dir_for(target)): raise PermissionError("Unauthorized output path")
    ensure_output_available(output_dir); ensure_no_forbidden_fields(predictions, pred_metrics, int_metrics, actions, decision_metrics, sample_econ, econ_metrics, break_even)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = ((predictions,"predictions.csv"),(pred_metrics.loc[pred_metrics["scope"]=="pooled"],"prediction_metrics.csv"),(int_metrics.loc[int_metrics["scope"]=="pooled"],"interval_metrics.csv"),(cond_q50,"conditional_coverage_q50.csv"),(cond_i,"conditional_coverage_irradiance.csv"),(actions,"decision_actions.csv"),(decision_metrics.loc[decision_metrics["scope"]=="pooled"],"decision_metrics.csv"),(sample_econ,"economic_sample_regrets.csv"),(econ_metrics.loc[econ_metrics["scope"]=="pooled"],"economic_metrics.csv"),(break_even.loc[break_even["scope"]=="pooled"],"break_even_review_cost.csv"))
    for frame, name in pairs: frame.to_csv(output_dir / name, index=False, mode="x")
    if target == TARGET_SEALED_DATES:
        for frame, name in ((pred_metrics,"prediction_metrics_by_date.csv"),(int_metrics,"interval_metrics_by_date.csv"),(decision_metrics,"decision_metrics_by_date.csv"),(econ_metrics,"economic_metrics_by_date.csv"),(break_even,"break_even_review_cost_by_date.csv")):
            frame.loc[frame["scope"]=="per_date"].to_csv(output_dir / name, index=False, mode="x")
    _write_json(output_dir / "config.json", config); _write_json(output_dir / "provenance.json", provenance)


def prepare_frozen_runtime(target: str) -> dict[str, Any]:
    """Complete every manifest-free preflight check and runtime construction."""
    target = validate_target(target)
    random_completed = False
    if target == TARGET_SEALED_DATES: require_random_completion(); random_completed = True
    qhat = load_frozen_qhat(); assert qhat == QHAT
    verify_checkpoint_file(POINT_CHECKPOINT, POINT_CHECKPOINT_SHA256, POINT_CHECKPOINT); verify_checkpoint_file(CQR_CHECKPOINT, CQR_CHECKPOINT_SHA256, CQR_CHECKPOINT)
    point_checkpoint = stage1a.load_clean_checkpoint(); cqr_checkpoint = stage3a1.load_verified_cqr_checkpoint()
    stats = stage3a1.load_train_irradiance_stats(); transform = stage3a1.build_inference_transform(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    point_model = stage1a.build_inference_model(point_checkpoint, device); cqr_model = stage3a1.build_inference_model(cqr_checkpoint, device)
    return {"target": target, "random_completed": random_completed, "stats": stats, "transform": transform, "device": device, "point_model": point_model, "cqr_model": cqr_model}


def run(target: str, batch_size: int = 64) -> dict[str, Any]:
    target = validate_target(target); output_dir = output_dir_for(target); ensure_output_available(output_dir)
    runtime = prepare_frozen_runtime(target)
    manifest = load_manifest(target); records = prepare_records(manifest, target, runtime["stats"]); dataset = stage3a1.Stage3A1InferenceDataset(records, runtime["transform"])
    device = runtime["device"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    point = stage1a.predict_deterministic(runtime["point_model"], loader, device); quantiles = stage3a1.predict_deterministic(runtime["cqr_model"], loader, device)
    predictions = build_predictions(records, point, quantiles, target); pm = build_prediction_metric_table(predictions, target); im = build_interval_metric_table(predictions, target); cq = conditional_coverage(predictions, target, "q50"); ci = conditional_coverage(predictions, target, "irradiance")
    actions = build_decision_actions(predictions); dm = build_decision_metrics(actions, target); sample = build_economic_sample(actions); em = build_economic_metrics(sample, target); be = build_break_even(em, target)
    config = make_config(target); provenance = make_provenance(target, point_verified=True, cqr_verified=True, random_completed=runtime["random_completed"])
    write_outputs(target, output_dir, predictions, pm, im, cq, ci, actions, dm, sample, em, be, config, provenance)
    return {"config": config, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--target", required=True); parser.add_argument("--batch-size", type=int, default=64); args = parser.parse_args(); print(json.dumps(run(args.target, args.batch_size), indent=2, sort_keys=True))


if __name__ == "__main__": main()
