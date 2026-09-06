#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1A
WAPP chronological timeline + intervention audit.

Build one authoritative 730-calendar-day environment ledger before any
counterfactual simulator, reward model, Gym environment, or PPO training.

This stage does NOT estimate a transition model and does NOT train RL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


START_DATE = pd.Timestamp("2021-08-09")
END_DATE = pd.Timestamp("2023-08-08")
EXPECTED_CALENDAR_DAYS = 730
EXPECTED_VALID_STATES = 729
EXPECTED_INVALID_STATES = 1
EXPECTED_MODB_CLEANS = 26
EXPECTED_RAIN_DAYS = 143
EXPECTED_RAIN_EVENTS = 69
EXPECTED_MAINTENANCE_DATES = {
    "2022-05-19",
    "2022-09-17",
    "2023-03-07",
}
EXPECTED_PERCEPTION_SEEDS = {
    20260906: "DEV",
    20260907: "FORMAL_EVAL",
    20260908: "FORMAL_EVAL",
    20260909: "FORMAL_EVAL",
    20260910: "FORMAL_EVAL",
    20260911: "FORMAL_EVAL",
}


def parse_bool_series(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any():
            raise RuntimeError(f"{name}: numeric boolean column contains NaN.")
        if (~x.isin([0, 1])).any():
            raise RuntimeError(f"{name}: numeric boolean column outside 0/1.")
        return x.astype(int).astype(bool)

    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    x = s.astype(str).str.strip().str.lower()
    mapped = x.map(mapping)
    if mapped.isna().any():
        vals = sorted(x[mapped.isna()].unique().tolist())[:10]
        raise RuntimeError(f"{name}: cannot parse boolean values {vals}")
    return mapped.astype(bool)


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q = x.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "q01": float(q.loc[0.01]),
        "q05": float(q.loc[0.05]),
        "q25": float(q.loc[0.25]),
        "q50": float(q.loc[0.50]),
        "q75": float(q.loc[0.75]),
        "q95": float(q.loc[0.95]),
        "q99": float(q.loc[0.99]),
        "max": float(x.max()),
    }


def load_daily(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "C_WAPP", "S_soil", "rain_mm_day", "rain_day",
        "has_any_cleaning_pulse", "modb_manual_cleaning_day",
        "scheduled_maintenance",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Daily reconstruction missing: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if df["date"].duplicated().any():
        raise RuntimeError("Daily reconstruction contains duplicate dates.")
    for col in [
        "rain_day", "has_any_cleaning_pulse",
        "modb_manual_cleaning_day", "scheduled_maintenance"
    ]:
        df[col] = parse_bool_series(df[col], col)
    for col in ["C_WAPP", "S_soil", "rain_mm_day"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_bridge(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "C_WAPP", "S_soil", "S_soil_observed", "S_soil_physical",
        "L_power_proxy", "bridge_valid", "power_bridge_model",
        "rain_mm_day", "rain_day", "has_any_cleaning_pulse",
        "modb_manual_cleaning_day", "scheduled_maintenance",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Power bridge missing: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    if df["date"].duplicated().any():
        raise RuntimeError("Power bridge contains duplicate dates.")
    for col in [
        "bridge_valid", "rain_day", "has_any_cleaning_pulse",
        "modb_manual_cleaning_day", "scheduled_maintenance"
    ]:
        df[col] = parse_bool_series(df[col], col)
    for col in [
        "C_WAPP", "S_soil", "S_soil_observed", "S_soil_physical",
        "L_power_proxy", "rain_mm_day"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_cleaning_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "cleaning_date", "pre_date", "S_pre", "S_cleaning_day",
        "reset_magnitude", "preclean_positive_soiling",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Cleaning audit missing: {sorted(missing)}")
    df["cleaning_date"] = pd.to_datetime(
        df["cleaning_date"], errors="raise"
    ).dt.normalize()
    df["pre_date"] = pd.to_datetime(
        df["pre_date"], errors="coerce"
    ).dt.normalize()
    if df["cleaning_date"].duplicated().any():
        raise RuntimeError("Cleaning audit contains duplicate cleaning dates.")
    return df


def load_rain_audit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "rain_event_id", "start_date", "end_date", "rain_days",
        "rain_mm_total", "pre_date", "post_date", "resolved",
        "cleaning_confounded",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Rain audit missing: {sorted(missing)}")
    for col in ["start_date", "end_date", "pre_date", "post_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
    for col in ["resolved", "cleaning_confounded"]:
        df[col] = parse_bool_series(df[col], col)
    return df


def load_perception_bank(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "L_true", "q50", "lower", "upper", "width",
        "perception_seed", "trajectory_role",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Perception bank missing: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["perception_seed"] = pd.to_numeric(
        df["perception_seed"], errors="raise"
    ).astype(int)
    for col in ["L_true", "q50", "lower", "upper", "width"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def exact_date_set(series: pd.Series) -> set[str]:
    return set(pd.to_datetime(series).dt.strftime("%Y-%m-%d"))


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-1A chronological timeline + intervention audit."
    )
    p.add_argument("--daily-reconstruction", required=True, type=Path)
    p.add_argument("--cleaning-audit", required=True, type=Path)
    p.add_argument("--rain-audit", required=True, type=Path)
    p.add_argument("--power-bridge", required=True, type=Path)
    p.add_argument("--perception-bank", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1a_timeline_intervention_audit_v1"
        ),
    )
    args = p.parse_args()

    paths = {
        "daily": args.daily_reconstruction.expanduser().resolve(),
        "cleaning": args.cleaning_audit.expanduser().resolve(),
        "rain": args.rain_audit.expanduser().resolve(),
        "bridge": args.power_bridge.expanduser().resolve(),
        "perception": args.perception_bank.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    print("[1/9] Load frozen P2-0B/P2-0C assets")
    daily = load_daily(paths["daily"])
    bridge = load_bridge(paths["bridge"])
    cleaning = load_cleaning_audit(paths["cleaning"])
    rain = load_rain_audit(paths["rain"])
    perception = load_perception_bank(paths["perception"])

    print("[2/9] Audit 730-day chronological calendar")
    calendar = pd.DataFrame({"date": pd.date_range(START_DATE, END_DATE, freq="D")})
    calendar["calendar_index"] = np.arange(len(calendar), dtype=int)

    calendar_gate = bool(
        len(calendar) == EXPECTED_CALENDAR_DAYS
        and calendar["date"].is_unique
        and calendar["date"].min() == START_DATE
        and calendar["date"].max() == END_DATE
        and (calendar["date"].diff().dropna() == pd.Timedelta(days=1)).all()
    )
    daily_date_gate = bool(
        len(daily) == EXPECTED_CALENDAR_DAYS
        and exact_date_set(daily["date"]) == exact_date_set(calendar["date"])
    )
    bridge_date_gate = bool(
        len(bridge) == EXPECTED_CALENDAR_DAYS
        and exact_date_set(bridge["date"]) == exact_date_set(calendar["date"])
    )

    print("[3/9] Reconcile daily reconstruction with frozen power bridge")
    dcols = [
        "date", "C_WAPP", "S_soil", "rain_mm_day", "rain_day",
        "has_any_cleaning_pulse", "modb_manual_cleaning_day",
        "scheduled_maintenance",
    ]
    bcols = [
        "date", "S_soil_observed", "S_soil_physical", "L_power_proxy",
        "bridge_valid", "power_bridge_model",
    ]
    ledger = calendar.merge(
        daily[dcols], on="date", how="left", validate="one_to_one"
    ).merge(
        bridge[bcols], on="date", how="left", validate="one_to_one"
    )

    observed_match = np.allclose(
        ledger["S_soil"].to_numpy(dtype=float),
        ledger["S_soil_observed"].to_numpy(dtype=float),
        atol=1e-12, rtol=0.0, equal_nan=True,
    )

    valid_count = int(ledger["bridge_valid"].astype(bool).sum())
    invalid_count = int((~ledger["bridge_valid"].astype(bool)).sum())
    state_count_gate = bool(
        valid_count == EXPECTED_VALID_STATES
        and invalid_count == EXPECTED_INVALID_STATES
    )
    missing_state_dates = (
        ledger.loc[~ledger["bridge_valid"].astype(bool), "date"]
        .dt.strftime("%Y-%m-%d").tolist()
    )

    print("[4/9] Audit authoritative interventions and rain events")
    clean_dates_daily = exact_date_set(
        ledger.loc[
            ledger["modb_manual_cleaning_day"].astype(bool), "date"
        ]
    )
    clean_dates_audit = exact_date_set(cleaning["cleaning_date"])
    cleaning_gate = bool(
        len(cleaning) == EXPECTED_MODB_CLEANS
        and len(clean_dates_daily) == EXPECTED_MODB_CLEANS
        and clean_dates_daily == clean_dates_audit
    )

    rain_days_count = int(ledger["rain_day"].astype(bool).sum())
    rain_days_gate = bool(rain_days_count == EXPECTED_RAIN_DAYS)

    maintenance_dates = exact_date_set(
        ledger.loc[
            ledger["scheduled_maintenance"].astype(bool), "date"
        ]
    )
    maintenance_gate = bool(
        maintenance_dates == EXPECTED_MAINTENANCE_DATES
    )

    rain_event_range_gate = bool(
        len(rain) == EXPECTED_RAIN_EVENTS
        and rain["start_date"].notna().all()
        and rain["end_date"].notna().all()
        and (rain["start_date"] >= START_DATE).all()
        and (rain["end_date"] <= END_DATE).all()
        and (rain["start_date"] <= rain["end_date"]).all()
    )

    print("[5/9] Align frozen 3C perception bank to hidden-state dates")
    valid_dates = set(
        ledger.loc[ledger["bridge_valid"].astype(bool), "date"].tolist()
    )

    alignment_rows = []
    perception_gate = True
    lmatch_gate = True
    duplicate_gate = True
    role_gate = True

    seeds_found = sorted(perception["perception_seed"].unique().tolist())
    if set(seeds_found) != set(EXPECTED_PERCEPTION_SEEDS):
        perception_gate = False

    bridge_map = ledger.set_index("date")["L_power_proxy"]

    for seed in seeds_found:
        g = perception[perception["perception_seed"].eq(seed)].copy()
        duplicated = bool(g["date"].duplicated().any())
        duplicate_gate &= not duplicated

        date_set = set(g["date"].tolist())
        date_match = bool(
            len(g) == EXPECTED_VALID_STATES and date_set == valid_dates
        )

        expected_role = EXPECTED_PERCEPTION_SEEDS.get(seed)
        roles = set(g["trajectory_role"].astype(str))
        this_role_gate = bool(
            expected_role is not None and roles == {expected_role}
        )
        role_gate &= this_role_gate

        merged = g[["date", "L_true"]].merge(
            bridge_map.rename("L_power_proxy"),
            left_on="date",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        diff = np.abs(
            merged["L_true"].to_numpy(dtype=float)
            - merged["L_power_proxy"].to_numpy(dtype=float)
        )
        max_abs_l_diff = float(np.nanmax(diff))
        this_lmatch = bool(
            np.isfinite(diff).all() and max_abs_l_diff <= 1e-12
        )
        lmatch_gate &= this_lmatch
        perception_gate &= date_match

        alignment_rows.append({
            "perception_seed": int(seed),
            "expected_role": expected_role,
            "rows": int(len(g)),
            "unique_dates": int(g["date"].nunique()),
            "date_set_matches_valid_hidden_states": date_match,
            "duplicate_dates": duplicated,
            "role_matches_frozen_manifest": this_role_gate,
            "max_abs_L_true_minus_L_power_proxy": max_abs_l_diff,
            "L_match_pass": this_lmatch,
        })

    perception_alignment = pd.DataFrame(alignment_rows)
    perception_primary_gate = bool(
        perception_gate and duplicate_gate and role_gate and lmatch_gate
    )

    print("[6/9] Mark audit-only Year1/Year2 periods")
    year1_end = START_DATE + pd.Timedelta(days=364)
    year2_start = year1_end + pd.Timedelta(days=1)
    ledger["audit_period"] = np.where(
        ledger["date"] <= year1_end, "YEAR1", "YEAR2"
    )
    ledger["state_valid"] = ledger["bridge_valid"].astype(bool)
    ledger["perception_available"] = ledger["date"].isin(valid_dates)

    ledger["days_since_prev_valid"] = np.nan
    prev_valid_date = None
    for i, row in ledger.iterrows():
        if bool(row["state_valid"]):
            if prev_valid_date is not None:
                ledger.loc[i, "days_since_prev_valid"] = int(
                    (row["date"] - prev_valid_date).days
                )
            prev_valid_date = row["date"]

    print("[7/9] Build mutually-exclusive t->t+1 transition ledger")
    transition_rows = []

    for i in range(len(ledger) - 1):
        src = ledger.iloc[i]
        dst = ledger.iloc[i + 1]

        src_valid = bool(src["state_valid"])
        dst_valid = bool(dst["state_valid"])
        dst_manual = bool(dst["modb_manual_cleaning_day"])
        src_manual = bool(src["modb_manual_cleaning_day"])
        maintenance_adjacent = bool(
            src["scheduled_maintenance"] or dst["scheduled_maintenance"]
        )
        rain_adjacent = bool(src["rain_day"] or dst["rain_day"])

        if not (src_valid and dst_valid):
            category = "INVALID_GAP"
        elif dst_manual:
            category = "MANUAL_CLEAN_CONTAMINATED"
        elif maintenance_adjacent:
            category = "MAINTENANCE_ADJACENT"
        elif rain_adjacent:
            category = "RAIN_AFFECTED"
        else:
            category = "DRY_NATURAL"

        delta_s_obs = np.nan
        delta_s_phys = np.nan
        delta_l = np.nan
        if src_valid and dst_valid:
            delta_s_obs = float(dst["S_soil_observed"] - src["S_soil_observed"])
            delta_s_phys = float(dst["S_soil_physical"] - src["S_soil_physical"])
            delta_l = float(dst["L_power_proxy"] - src["L_power_proxy"])

        transition_rows.append({
            "transition_index": i,
            "source_date": src["date"],
            "dest_date": dst["date"],
            "calendar_gap_days": int((dst["date"] - src["date"]).days),
            "source_state_valid": src_valid,
            "dest_state_valid": dst_valid,
            "source_manual_clean_day": src_manual,
            "dest_manual_clean_day": dst_manual,
            "source_rain_day": bool(src["rain_day"]),
            "dest_rain_day": bool(dst["rain_day"]),
            "source_scheduled_maintenance": bool(src["scheduled_maintenance"]),
            "dest_scheduled_maintenance": bool(dst["scheduled_maintenance"]),
            "rain_adjacent": rain_adjacent,
            "maintenance_adjacent": maintenance_adjacent,
            "transition_class": category,
            "natural_transition_candidate": bool(category == "DRY_NATURAL"),
            "delta_S_observed": delta_s_obs,
            "delta_S_physical": delta_s_phys,
            "delta_L_power_proxy": delta_l,
        })

    transitions = pd.DataFrame(transition_rows)

    class_count_gate = bool(
        len(transitions) == EXPECTED_CALENDAR_DAYS - 1
        and transitions["transition_class"].notna().all()
    )

    dry = transitions[transitions["transition_class"].eq("DRY_NATURAL")]
    dry_integrity_gate = bool(
        len(dry) > 0
        and dry["source_state_valid"].all()
        and dry["dest_state_valid"].all()
        and (~dry["dest_manual_clean_day"]).all()
        and (~dry["rain_adjacent"]).all()
        and (~dry["maintenance_adjacent"]).all()
        and (dry["calendar_gap_days"] == 1).all()
    )

    print("[8/9] Build summaries and evaluate P2-1A gates")
    class_summary = (
        transitions.groupby("transition_class", sort=False)
        .agg(
            transitions=("transition_index", "count"),
            mean_delta_S_observed=("delta_S_observed", "mean"),
            median_delta_S_observed=("delta_S_observed", "median"),
            mean_delta_L=("delta_L_power_proxy", "mean"),
            median_delta_L=("delta_L_power_proxy", "median"),
        )
        .reset_index()
    )

    year_summary = (
        ledger.groupby("audit_period")
        .agg(
            calendar_days=("date", "count"),
            valid_states=("state_valid", "sum"),
            rain_days=("rain_day", "sum"),
            modb_manual_clean_days=("modb_manual_cleaning_day", "sum"),
            generic_cleaning_pulse_days=("has_any_cleaning_pulse", "sum"),
            scheduled_maintenance_days=("scheduled_maintenance", "sum"),
        )
        .reset_index()
    )

    all_primary = bool(
        calendar_gate
        and daily_date_gate
        and bridge_date_gate
        and observed_match
        and state_count_gate
        and cleaning_gate
        and rain_days_gate
        and maintenance_gate
        and rain_event_range_gate
        and perception_primary_gate
        and class_count_gate
        and dry_integrity_gate
    )

    summary = {
        "stage": "P2-1A",
        "audit_only": True,
        "counterfactual_environment_built": False,
        "rl_started": False,
        "chronological_split_frozen": False,
        "calendar": {
            "start_date": START_DATE.strftime("%Y-%m-%d"),
            "end_date": END_DATE.strftime("%Y-%m-%d"),
            "calendar_days": int(len(calendar)),
            "calendar_gate_pass": calendar_gate,
            "daily_reconstruction_date_gate_pass": daily_date_gate,
            "power_bridge_date_gate_pass": bridge_date_gate,
        },
        "hidden_state": {
            "valid_states": valid_count,
            "invalid_states": invalid_count,
            "missing_or_invalid_state_dates": missing_state_dates,
            "expected_valid_states": EXPECTED_VALID_STATES,
            "state_count_gate_pass": state_count_gate,
            "daily_S_soil_matches_bridge_S_soil_observed": bool(observed_match),
        },
        "interventions": {
            "authoritative_modb_clean_days": int(
                ledger["modb_manual_cleaning_day"].sum()
            ),
            "cleaning_audit_rows": int(len(cleaning)),
            "cleaning_date_set_exact_match": bool(
                clean_dates_daily == clean_dates_audit
            ),
            "cleaning_gate_pass": cleaning_gate,
            "generic_cleaning_pulse_days_diagnostic": int(
                ledger["has_any_cleaning_pulse"].sum()
            ),
            "scheduled_maintenance_dates": sorted(maintenance_dates),
            "maintenance_gate_pass": maintenance_gate,
        },
        "rain": {
            "rain_days": rain_days_count,
            "rain_days_gate_pass": rain_days_gate,
            "rain_event_rows": int(len(rain)),
            "resolved_rain_events": int(rain["resolved"].sum()),
            "cleaning_confounded_rain_events": int(
                rain["cleaning_confounded"].sum()
            ),
            "rain_event_range_gate_pass": rain_event_range_gate,
            "rain_mm_day_distribution": qstats(
                ledger.loc[ledger["rain_day"].astype(bool), "rain_mm_day"]
            ),
        },
        "perception_alignment": {
            "seeds_found": seeds_found,
            "expected_seeds": sorted(EXPECTED_PERCEPTION_SEEDS.keys()),
            "bank_rows": int(len(perception)),
            "alignment_gate_pass": perception_primary_gate,
            "duplicate_date_gate_pass": duplicate_gate,
            "role_gate_pass": role_gate,
            "L_alignment_gate_pass": lmatch_gate,
        },
        "transition_audit": {
            "calendar_transitions": int(len(transitions)),
            "class_counts": {
                str(k): int(v)
                for k, v in transitions["transition_class"]
                .value_counts().to_dict().items()
            },
            "dry_natural_candidates": int(
                transitions["natural_transition_candidate"].sum()
            ),
            "class_count_gate_pass": class_count_gate,
            "dry_integrity_gate_pass": dry_integrity_gate,
        },
        "year_labels": {
            "YEAR1": {
                "start": START_DATE.strftime("%Y-%m-%d"),
                "end": year1_end.strftime("%Y-%m-%d"),
            },
            "YEAR2": {
                "start": year2_start.strftime("%Y-%m-%d"),
                "end": END_DATE.strftime("%Y-%m-%d"),
            },
            "split_status": "AUDIT_LABEL_ONLY_NOT_FROZEN_FOR_RL",
        },
        "primary_gates": {
            "calendar_pass": calendar_gate,
            "daily_date_pass": daily_date_gate,
            "bridge_date_pass": bridge_date_gate,
            "state_reconciliation_pass": bool(
                observed_match and state_count_gate
            ),
            "manual_clean_intervention_pass": cleaning_gate,
            "rain_pass": bool(rain_days_gate and rain_event_range_gate),
            "maintenance_pass": maintenance_gate,
            "perception_alignment_pass": perception_primary_gate,
            "transition_classification_pass": bool(
                class_count_gate and dry_integrity_gate
            ),
            "all_primary_gates_pass": all_primary,
        },
        "next_step_if_pass": (
            "Freeze P2-1A master ledger, then P2-1B must extract/model "
            "natural dynamics using uncontaminated transitions, with dry "
            "accumulation separated from rain response. Do not train PPO yet."
        ),
        "next_step_if_fail": (
            "Do not build a simulator. Resolve timeline, event, hidden-state, "
            "or perception-date inconsistencies first."
        ),
    }

    print("[9/9] Write P2-1A authoritative audit assets")
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger.to_csv(
        out_dir / "environment_master_ledger.csv",
        index=False,
        encoding="utf-8-sig",
    )
    transitions.to_csv(
        out_dir / "transition_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    class_summary.to_csv(
        out_dir / "transition_category_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    year_summary.to_csv(
        out_dir / "year_event_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    perception_alignment.to_csv(
        out_dir / "perception_alignment_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "environment_master_ledger.csv")
    print(out_dir / "transition_audit.csv")
    print(out_dir / "transition_category_summary.csv")
    print(out_dir / "year_event_summary.csv")
    print(out_dir / "perception_alignment_audit.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: P2-1A audit only. Do NOT fit counterfactual dynamics, "
        "build Gym, or train PPO until audit_summary.json is reviewed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
