#!/usr/bin/env python
"""P2-1D-1A-v1: Daily GHI Energy-Value Proxy Audit.

Read-only inputs; execution creates one immutable output directory. Raw timestamps
are preserved. Interval-ending timestamps minus one minute assign energy intervals
to calendar days; this interval semantics does not modify source data.
Nonfinite minutes are reported, never filled. Finite-minute energy is an observed
partial integral, not a coverage-corrected estimate. No reward or RL is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = "P2-1D-1A-v1"
TITLE = "Daily GHI Energy-Value Proxy Audit"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_1d_1a_daily_ghi_energy_value_audit_v1"
MASTER_LEDGER = BASE / "p2_1a_timeline_intervention_audit_v1/environment_master_ledger.csv"
BRIDGE = BASE / "p2_0c_1b_common_temp_power_bridge_v1/daily_common_temp_power_bridge.csv"
DATA = Path("C:/Users/muye/PV_RL_Data/wapp_malanville")
SOURCES = {
    "YEAR1": (DATA / "Solar-Measurements_Benin-Malanville_QC.csv",
              "7f15922f01de97eb6a8b1477f0357e1dd3460c2918a64f7f007622a08063bed3"),
    "YEAR2": (DATA / "Solar-Measurements_Benin-Malanville_QC_Year2.csv",
              "d85310c0a722184502714845abec945f64c854529b2470c57f3303447eb4fc52"),
}
CALENDAR = pd.date_range("2021-08-09", "2023-08-08", freq="D")
REQUIRED = {"Timestamp", "GHI", "DNI", "DHI", "ModA", "ModB",
            "Cleaning", "Precipitation", "TModA", "TModB"}
MARKERS = ["rain_day", "modb_manual_cleaning_day", "scheduled_maintenance",
           "hidden_state_invalid_day"]
PROJECTION = (
    "negative-to-zero projection is a physical irradiance projection "
    "for energy integration only, not row deletion or source-data modification."
)
GATES = ["source_sha256_pass", "timeline_pass", "730_day_pass",
         "daily_minute_count_pass", "daily_GHI_finite_99pct_pass",
         "daily_energy_positive_finite_pass", "global_normalization_pass",
         "intervention_retention_pass", "information_role_no_leakage_pass",
         "interval_timestamp_exact_span_pass", "interval_assignment_row_preservation_pass"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"finite_count": 0}
    quantiles = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {"finite_count": int(x.size), "min": float(x.min()),
            "max": float(x.max()), "mean": float(x.mean()),
            **dict(zip(["P01", "P05", "median", "P95", "P99"],
                       map(float, quantiles)))}


def raw_stats(values, high_threshold: float) -> dict:
    x = np.asarray(values, dtype=float)
    n = len(x)
    return {**stats(x), "minute_count": n,
            "negative_GHI_count": int((x < 0).sum()),
            "negative_GHI_fraction": float((x < 0).sum() / n) if n else None,
            "zero_GHI_count": int((x == 0).sum()),
            "NaN_count": int(np.isnan(x).sum()),
            "Inf_count": int(np.isinf(x).sum()),
            "positive_Inf_count": int(np.isposinf(x).sum()),
            "negative_Inf_count": int(np.isneginf(x).sum()),
            "nonfinite_count": int((~np.isfinite(x)).sum()),
            "high_GHI_count": int((x > high_threshold).sum()),
            "statistics_scope": "finite numeric raw GHI; nonfinite counted separately"}


def correlations(frame: pd.DataFrame) -> dict:
    pair = frame[["H_GHI_Wh_m2", "H_ModA_Wh_m2"]]
    pair = pair.loc[np.isfinite(pair).all(axis=1)]
    result = {"daily_correlation_pair_count": len(pair),
              "daily_Pearson_H_GHI_H_ModA": None,
              "daily_Spearman_H_GHI_H_ModA": None}
    if len(pair) > 1 and (pair.nunique() > 1).all():
        result["daily_Pearson_H_GHI_H_ModA"] = float(pair.corr().iloc[0, 1])
        result["daily_Spearman_H_GHI_H_ModA"] = float(
            pair.rank(method="average").corr().iloc[0, 1])
    return result


def dates(index) -> list[str]:
    return pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()


def strict_bool(series: pd.Series) -> pd.Series:
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        raise ValueError(f"Ambiguous boolean column: {series.name}")
    return mapped.astype(bool)


def timestamp_diagnostics(series: pd.Series) -> dict:
    stamps = pd.DatetimeIndex(series.dropna())
    unique = stamps.unique().sort_values()
    span = (pd.date_range(unique.min(), unique.max(), freq="min")
            if len(unique) else pd.DatetimeIndex([]))
    missing = span.difference(unique)
    return {
        "rows": len(series), "parse_failure_count": int(series.isna().sum()),
        "first_timestamp": str(unique.min()), "last_timestamp": str(unique.max()),
        "calendar_date_count": int(unique.normalize().nunique()),
        "unique_timestamp_count": len(unique),
        "duplicate_timestamp_count": int(stamps.duplicated().sum()),
        "duplicate_timestamps": stamps[stamps.duplicated(keep=False)].unique().astype(str).tolist(),
        "non_one_minute_spacing_count_in_source_order": int(
            (series.diff().iloc[1:] != pd.Timedelta(minutes=1)).sum()),
        "off_minute_grid_count": int((stamps != stamps.floor("min")).sum()),
        "missing_minute_timestamp_count_in_observed_span": len(missing),
        "missing_minute_timestamps_in_observed_span": missing.astype(str).tolist(),
        "unparsed_timestamp_row_indices": series.index[series.isna()].tolist(),
    }


def load_marker_calendar(path: Path) -> pd.DataFrame:
    source = pd.read_csv(path, encoding="utf-8-sig")
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    if source["date"].isna().any() or source["date"].duplicated().any():
        raise ValueError("Missing or duplicate marker dates")
    if not source["date"].eq(source["date"].dt.normalize()).all():
        raise ValueError("Marker dates are not midnight calendar labels")
    return source.set_index("date").sort_index()


def align_interventions(daily: pd.DataFrame) -> dict:
    authoritative = {"path": str(MASTER_LEDGER), "status": "unavailable",
                     "exact_calendar_pass": False, "marker_parsing_unambiguous": False}
    flags = None
    if MASTER_LEDGER.is_file():
        authoritative["sha256"] = sha256(MASTER_LEDGER)
        try:
            source = load_marker_calendar(MASTER_LEDGER)
            authoritative["missing_dates"] = dates(CALENDAR.difference(source.index))
            authoritative["unexpected_dates"] = dates(source.index.difference(CALENDAR))
            authoritative["exact_calendar_pass"] = bool(source.index.equals(CALENDAR))
            if not authoritative["exact_calendar_pass"]:
                raise ValueError("Master ledger must have the exact 730-day calendar")
            parsed = pd.DataFrame(index=source.index)
            for col in MARKERS[:3]:
                parsed[col] = strict_bool(source[col])
            # state_valid is authoritative; bridge_valid has different semantics.
            parsed["hidden_state_invalid_day"] = ~strict_bool(source["state_valid"])
            flags = parsed
            authoritative.update(status="PASS", marker_parsing_unambiguous=True)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            authoritative.update(status="FAIL", reason=str(exc))

    cross_check = {"path": str(BRIDGE), "status": "unavailable",
                   "required_for_intervention_retention": False,
                   "bridge_valid_diagnostic": {"status": "unavailable"}}
    if BRIDGE.is_file():
        cross_check["sha256"] = sha256(BRIDGE)
        try:
            bridge = load_marker_calendar(BRIDGE)
            # Record bridge validity independently; never infer hidden validity.
            if "bridge_valid" in bridge:
                try:
                    bridge_valid = strict_bool(bridge["bridge_valid"])
                    cross_check["bridge_valid_diagnostic"] = {
                        "status": "available", "true_count": int(bridge_valid.sum()),
                        "false_dates": dates(bridge.index[~bridge_valid])}
                except ValueError as exc:
                    cross_check["bridge_valid_diagnostic"] = {"status": "ambiguous", "reason": str(exc)}
            if flags is None:
                cross_check["status"] = "NOT_EVALUATED_MASTER_UNAVAILABLE_OR_INVALID"
            else:
                cross_check["missing_dates"] = dates(flags.index.difference(bridge.index))
                cross_check["unexpected_dates"] = dates(bridge.index.difference(flags.index))
                common = flags.index.intersection(bridge.index)
                mismatches = {}
                for col in MARKERS[:3]:
                    marker = strict_bool(bridge[col])
                    mismatches[col] = dates(common[flags.loc[common, col] != marker.loc[common]])
                cross_check["marker_mismatch_dates"] = mismatches
                cross_check["status"] = "PASS" if not (
                    cross_check["missing_dates"] or cross_check["unexpected_dates"]
                    or any(mismatches.values())) else "FAIL"
        except (ValueError, KeyError, TypeError, OSError) as exc:
            cross_check.update(status="FAIL", reason=str(exc))

    for col in MARKERS:
        daily[col] = pd.Series(pd.NA, index=daily.index, dtype="boolean")
    unambiguous = flags is not None
    result = {"authoritative_source": authoritative, "bridge_cross_check": cross_check,
              "alignment_unambiguous": unambiguous}
    if unambiguous:
        for col in MARKERS:
            daily[col] = flags[col].reindex(daily.index).astype("boolean")
            marked = flags.index[flags[col]]
            retained = marked.intersection(daily.index[daily["minute_count"] > 0])
            result[col] = {"count": len(marked), "dates": dates(marked),
                           "retained_count": len(retained),
                           "missing_dates": dates(marked.difference(retained))}
    special = pd.Timestamp("2022-06-29")
    row = daily.loc[special]
    resource_valid = bool(row["minute_count"] == 1440
                          and row["finite_GHI_fraction"] >= 0.99
                          and np.isfinite(row["H_GHI_Wh_m2"])
                          and row["H_GHI_Wh_m2"] > 0)
    hidden_invalid = bool(row["hidden_state_invalid_day"]) if unambiguous else None
    normalization_generated = "E_clean_GHI" in daily
    normalized_energy_retained = bool(np.isfinite(row["E_clean_GHI"])
                                      and row["E_clean_GHI"] > 0) if normalization_generated else None
    special_pass = bool(row["minute_count"] > 0 and hidden_invalid is True
                        and (not resource_valid or not normalization_generated
                             or normalized_energy_retained))
    result["2022-06-29"] = {
        "retained": bool(row["minute_count"] > 0),
        "GHI_resource_valid": resource_valid,
        "H_GHI_Wh_m2": float(row["H_GHI_Wh_m2"]),
        "hidden_state_invalid_day": hidden_invalid,
        "hidden_state_invalid_source": "master ledger: ~state_valid",
        "normalization_generated": normalization_generated,
        "E_clean_GHI": float(row["E_clean_GHI"]) if normalization_generated else None,
        "normalized_energy_retained": normalized_energy_retained,
        "normalization_note": "E_clean_GHI requires the global daily GHI validity gate; hidden validity is not used.",
        "pass": special_pass,
    }
    result["pass"] = bool(unambiguous and all(
        not result[col]["missing_dates"] for col in MARKERS)
        and special_pass)
    return result


def run(args: argparse.Namespace, out: Path) -> dict:
    manifest = {
        "stage": STAGE, "title": TITLE, "encoding": "cp1252",
        "csv_header_row": 1, "explicitly_skipped_unit_row": 2,
        "required_columns": sorted(REQUIRED), "expected_rows_per_year": 525600,
        "expected_combined_rows": 1051200, "expected_calendar_days": 730,
        "calendar_start": "2021-08-09", "calendar_end": "2023-08-08",
        "raw_timestamp_preserved": True,
        "timestamp_semantics": "1-minute interval-ending",
        "timestamp_semantics_description": "1-minute interval-ending timestamp",
        "interval_assignment": "raw Timestamp minus 1 minute",
        "daily_assignment": "interval_timestamp = raw_timestamp - 1 minute",
        "interval_shift_is_for_energy_interval_assignment_only": True,
        "interval_assignment_note": "Time-interval assignment semantics, not source-data modification; no row deletion or duplication.",
        "expected_interval_first": "2021-08-09 00:00",
        "expected_interval_last": "2023-08-08 23:59",
        "expected_unique_intervals": 1051200,
        "intervention_authoritative_source": str(MASTER_LEDGER),
        "hidden_state_invalid_day_definition": "~state_valid from master ledger only",
        "bridge_cross_check_source": str(BRIDGE),
        "bridge_cross_check_role": "DIAGNOSTIC_ONLY; compare rain/manual-clean/maintenance markers",
        "bridge_valid_role": "BRIDGE_DIAGNOSTIC_ONLY; no equality requirement with state_valid",
        "year_label": "frozen calendar periods; source_year_labels separately recorded",
        "GHI_physical": "max(GHI, 0)", "projection_statement": PROJECTION,
        "upper_clipping": False, "imputation": False, "date_deletion": False,
        "energy_formula": "sum(finite GHI_physical minutes) / 60",
        "energy_units": "Wh/m2/day", "nonfinite_policy": (
            "Retain rows and dates; report nonfinite coverage. Energy is an observed "
            "finite-minute partial integral without rescaling; all-nonfinite days are NaN."),
        "daily_finite_threshold": 0.99,
        "ModA_role": "DIAGNOSTIC_ONLY", "ModA_physical": "max(ModA, 0)",
        "H_ModA_formula": "sum(finite ModA_physical minutes) / 60",
        "ModB_used_for_energy": False, "soiling_state_used_for_energy": False,
        "Cleaning_used_to_modify_GHI": False,
        "hidden_state_validity_controls_resource": False,
        "normalization": "single global mean over the entire valid 730-day calendar",
        "normalization_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY",
        "forbidden_in_pre_action_observation": True,
        "no_future_GHI_provided_to_agent": True,
        "reward_implemented": False, "cleaning_cost_implemented": False,
        "Gym_implemented": False, "PPO_implemented": False,
        "high_GHI_report_threshold_W_m2": args.high_ghi_threshold,
        "high_GHI_threshold_is_reporting_only": True,
        "required_gates": GATES,
    }
    records, frames = {}, []
    for label, (path, expected_hash) in SOURCES.items():
        actual_hash = sha256(path)
        # Units can contain malformed delimiters: skip by physical line number.
        frame = pd.read_csv(path, encoding="cp1252", header=0, skiprows=[1],
                            low_memory=False)
        missing = REQUIRED.difference(frame.columns)
        if missing:
            raise ValueError(f"{label}: missing required columns {sorted(missing)}")
        stamp = pd.to_datetime(frame["Timestamp"], format="%Y-%m-%d %H:%M",
                               errors="coerce")
        interval_stamp = stamp - pd.Timedelta(minutes=1)
        ghi = pd.to_numeric(frame["GHI"], errors="coerce")
        moda = pd.to_numeric(frame["ModA"], errors="coerce")
        records[label] = {
            "path": str(path), "expected_sha256": expected_hash,
            "actual_sha256": actual_hash, "sha256_pass": actual_hash == expected_hash,
            "rows": len(frame), "rows_pass": len(frame) == 525600,
            "timestamp_parse_failure_count": int(stamp.isna().sum()),
            "raw_timestamp_diagnostics": timestamp_diagnostics(stamp),
            "interval_timestamp_diagnostics": timestamp_diagnostics(interval_stamp),
            "numeric_GHI_parse_failure_count": int((frame["GHI"].notna() & ghi.isna()).sum()),
            "raw_GHI": raw_stats(ghi, args.high_ghi_threshold),
        }
        # Construct a separate numeric/physical audit layer; never assign to raw GHI.
        frames.append(pd.DataFrame({"Timestamp": frame["Timestamp"].copy(),
                                    "raw_timestamp": stamp,
                                    "source_year_label": label,
                                    "ghi_numeric": ghi, "moda_numeric": moda}))
    manifest["sources"] = records
    minute = pd.concat(frames, ignore_index=True)
    rows_before_assignment = len(minute)
    minute["interval_timestamp"] = minute["raw_timestamp"] - pd.Timedelta(minutes=1)
    interval_index = pd.DatetimeIndex(minute["interval_timestamp"])
    unique = interval_index.dropna().unique().sort_values()
    expected_minutes = pd.date_range(CALENDAR[0], CALENDAR[-1] + pd.Timedelta(hours=23, minutes=59), freq="min")
    exact_intervals = bool(len(interval_index) == 1051200
                           and interval_index.is_unique
                           and interval_index.equals(expected_minutes))
    row_preservation = bool(len(minute) == rows_before_assignment
                            and len(minute) == sum(r["rows"] for r in records.values()))
    raw_diagnostics = timestamp_diagnostics(minute["raw_timestamp"])
    interval_diagnostics = timestamp_diagnostics(minute["interval_timestamp"])
    timeline = {
        "combined_rows": len(minute),
        "raw_timestamp_diagnostics": raw_diagnostics,
        "interval_timestamp_diagnostics": interval_diagnostics,
        "calendar_comparison_basis": "interval_timestamp",
        "missing_expected_calendar_minutes": expected_minutes.difference(unique).astype(str).tolist(),
        "outside_expected_calendar_minutes": unique.difference(expected_minutes).astype(str).tolist(),
        "interval_timestamp_exact_span_pass": exact_intervals,
        "rows_before_interval_assignment": rows_before_assignment,
        "rows_after_interval_assignment": len(minute),
        "interval_assignment_row_preservation_pass": row_preservation,
    }
    minute["date"] = minute["interval_timestamp"].dt.normalize()
    for prefix, numeric in [("GHI", "ghi_numeric"), ("ModA", "moda_numeric")]:
        minute[f"{prefix}_physical"] = np.maximum(minute[numeric], 0)
        minute[f"finite_{prefix}"] = np.isfinite(minute[numeric])
        minute[f"{prefix}_integration"] = minute[f"{prefix}_physical"].where(minute[f"finite_{prefix}"])
    minute["negative_GHI"] = minute["ghi_numeric"] < 0
    minute["zero_GHI"] = minute["ghi_numeric"] == 0
    minute["NaN_GHI"] = minute["ghi_numeric"].isna()
    minute["Inf_GHI"] = np.isinf(minute["ghi_numeric"])
    minute["high_GHI"] = minute["ghi_numeric"] > args.high_ghi_threshold
    group = minute.groupby("date", sort=True)
    observed_dates = pd.DatetimeIndex(group.size().index)
    # Union preserves unexpected dates AND exposes entirely absent expected dates.
    calendar = CALENDAR.union(observed_dates).sort_values()
    daily = pd.DataFrame(index=calendar)
    daily.index.name = "date"
    daily["year_label"] = np.where(daily.index <= pd.Timestamp("2022-08-08"), "YEAR1", "YEAR2")
    daily.loc[~daily.index.isin(CALENDAR), "year_label"] = "OUTSIDE_EXPECTED_CALENDAR"
    daily["source_year_labels"] = group["source_year_label"].agg(lambda s: "|".join(sorted(s.unique()))).reindex(calendar).fillna("")
    daily["minute_count"] = group.size().reindex(calendar, fill_value=0)
    for output, column in [("finite_GHI_count", "finite_GHI"), ("finite_ModA_count", "finite_ModA"),
                           ("negative_GHI_count", "negative_GHI"), ("zero_GHI_count", "zero_GHI"),
                           ("NaN_GHI_count", "NaN_GHI"), ("Inf_GHI_count", "Inf_GHI"),
                           ("high_GHI_count", "high_GHI")]:
        daily[output] = group[column].sum().reindex(calendar, fill_value=0).astype(int)
    denominator = daily["minute_count"].replace(0, np.nan)
    daily["finite_GHI_fraction"] = (daily["finite_GHI_count"] / denominator).fillna(0)
    daily["finite_ModA_fraction"] = (daily["finite_ModA_count"] / denominator).fillna(0)
    daily["negative_GHI_fraction"] = daily["negative_GHI_count"] / denominator
    for prefix in ["GHI", "ModA"]:
        daily[f"H_{prefix}_Wh_m2"] = group[f"{prefix}_integration"].sum(min_count=1).reindex(calendar) / 60
    daily["raw_GHI_max"] = group["ghi_numeric"].max().reindex(calendar)
    daily["raw_GHI_min"] = group["ghi_numeric"].min().reindex(calendar)
    daily["energy_is_partial"] = daily["finite_GHI_count"] < 1440
    minute_ok = daily["minute_count"].eq(1440)
    finite_ok = daily["finite_GHI_fraction"].ge(0.99)
    energy_ok = np.isfinite(daily["H_GHI_Wh_m2"]) & daily["H_GHI_Wh_m2"].gt(0)
    summary = {
        "stage": STAGE, "title": TITLE, "sources": records, "timeline": timeline,
        "source_sha256_pass": all(r["sha256_pass"] for r in records.values()),
        "timeline_pass": bool(all(r["rows_pass"] for r in records.values())
                              and exact_intervals and row_preservation
                              and raw_diagnostics["parse_failure_count"] == 0
                              and raw_diagnostics["non_one_minute_spacing_count_in_source_order"] == 0),
        "interval_timestamp_exact_span_pass": exact_intervals,
        "interval_assignment_row_preservation_pass": row_preservation,
        "730_day_pass": bool(len(observed_dates) == 730 and set(observed_dates) == set(CALENDAR)),
        "observed_calendar_date_count": len(observed_dates), "output_calendar_date_count": len(daily),
        "absent_expected_dates": dates(CALENDAR.difference(observed_dates)),
        "unexpected_dates": dates(observed_dates.difference(CALENDAR)),
        "daily_minute_count_pass": bool(minute_ok.all()),
        "daily_GHI_finite_99pct_pass": bool(finite_ok.all()),
        "daily_energy_positive_finite_pass": bool(energy_ok.all()),
        "failed_minute_count_dates": dates(daily.index[~minute_ok]),
        "failed_finite_GHI_dates": dates(daily.index[~finite_ok]),
        "failed_energy_dates": dates(daily.index[~energy_ok]),
        "raw_GHI": raw_stats(minute["ghi_numeric"], args.high_ghi_threshold),
        "high_GHI_dates": dates(daily.index[daily["high_GHI_count"] > 0]),
        "ModA_diagnostic": correlations(daily),
        "information_role_no_leakage_pass": bool(
            manifest["E_clean_GHI_role"] == "REALIZED_REWARD_SETTLEMENT_ONLY"
            and manifest["forbidden_in_pre_action_observation"]
            and manifest["no_future_GHI_provided_to_agent"]),
    }
    summary["daily_resource_data_quality_pass"] = bool(finite_ok.all())
    validity = all(summary[k] for k in GATES[:6])
    summary["daily_GHI_validity_gate_pass"] = validity
    summary["global_normalization_pass"] = False
    summary["normalization_status"] = "NOT_GENERATED_DAILY_VALIDITY_FAILED"
    if validity:
        global_mean = float(daily["H_GHI_Wh_m2"].mean())
        if np.isfinite(global_mean) and global_mean > 0:
            daily["E_clean_GHI"] = daily["H_GHI_Wh_m2"] / global_mean
            summary["global_mean_H_GHI"] = global_mean
            summary["global_normalization_pass"] = bool(
                np.isfinite(daily["E_clean_GHI"]).all()
                and np.isclose(daily["E_clean_GHI"].mean(), 1.0, rtol=1e-12, atol=1e-12))
            summary["normalization_status"] = "GENERATED"
            summary["E_clean_GHI"] = stats(daily["E_clean_GHI"])
            summary["E_clean_GHI"].update({f"{label}_mean": float(
                daily.loc[daily["year_label"] == label, "E_clean_GHI"].mean())
                for label in SOURCES})
    retention = align_interventions(daily)
    summary["intervention_retention"] = retention
    summary["intervention_retention_pass"] = retention["pass"]
    summary["stage_pass"] = all(summary[k] for k in GATES)
    summary["failed_gates"] = [k for k in GATES if not summary[k]]
    years = []
    for label, block in daily.groupby("year_label", sort=True):
        row = {"year_label": label, "calendar_day_count": len(block),
               "minute_count": int(block["minute_count"].sum()),
               "finite_GHI_count": int(block["finite_GHI_count"].sum()),
               **{f"H_GHI_{k}": v for k, v in stats(block["H_GHI_Wh_m2"]).items()},
               **correlations(block)}
        if "E_clean_GHI" in daily:
            row.update({f"E_clean_GHI_{k}": v for k, v in stats(block["E_clean_GHI"]).items()})
        years.append(row)
    proxy_columns = ["year_label", "source_year_labels", "minute_count", "finite_GHI_count",
                     "finite_GHI_fraction", "negative_GHI_count", "H_GHI_Wh_m2"]
    if "E_clean_GHI" in daily:
        proxy_columns.append("E_clean_GHI")
    daily[proxy_columns].to_csv(out / "daily_energy_value_proxy.csv", encoding="utf-8-sig")
    daily.to_csv(out / "daily_resource_diagnostics.csv", encoding="utf-8-sig")
    pd.DataFrame(years).to_csv(out / "year_summary.csv", index=False, encoding="utf-8-sig")
    write_json(out / "protocol_manifest.json", manifest)
    write_json(out / "audit_summary.json", summary)
    return summary


def write_json(path: Path, value) -> None:
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    with path.open("x", encoding="utf-8") as stream:
        json.dump(clean(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=f"{STAGE}: {TITLE}")
    parser.add_argument("--high-ghi-threshold", type=float, default=1500.0,
                        help="W/m2 reporting threshold only; never clips or gates energy.")
    args = parser.parse_args()
    if not np.isfinite(args.high_ghi_threshold) or args.high_ghi_threshold <= 0:
        parser.error("--high-ghi-threshold must be positive and finite")
    # Exclusive directory creation also prevents simultaneous runs overwriting.
    # On failure leave the directory intact as an immutable failed attempt.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    try:
        summary = run(args, OUTPUT)
    except Exception as exc:
        failure = {"stage": STAGE, "stage_pass": False,
                   "execution_error": f"{type(exc).__name__}: {exc}",
                   "gate_status": "NOT_EVALUATED_DUE_TO_EXECUTION_ERROR",
                   **{key: False for key in GATES}}
        if not (OUTPUT / "audit_summary.json").exists():
            write_json(OUTPUT / "audit_summary.json", failure)
        raise
    print(json.dumps({"stage": STAGE, "stage_pass": summary["stage_pass"],
                      "failed_gates": summary["failed_gates"], "output": str(OUTPUT)}))
    return 0 if summary["stage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
