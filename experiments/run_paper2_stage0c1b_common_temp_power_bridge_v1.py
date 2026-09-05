#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-0C-1B
Common-temperature PVWatts semantic bridge.

Purpose
-------
Construct a conservative, physically explicit relative electrical power-loss
PROXY for connecting the frozen WAPP soiling state to the Paper1/DeepSolarEye
label domain.

This stage is NOT an energy-yield/reward model. Its sole purpose is semantic
alignment for the perception emulator.

Why this model
--------------
P2-0C-1A rejected a direct TModA/TModB differential-temperature correction
because the two temperature channels showed a persistent multi-degree offset
and the candidate bridge produced many negative "soiling power-loss" days.

For the primary semantic bridge we therefore use a common-temperature
counterfactual. Under the standard PVWatts DC form

    P_dc = (G_eff / G_ref) * P_dc0 * [1 + gamma * (T_cell - T_ref)]

compare a clean and soiled counterfactual at the same module temperature:

    G_soil = (1 - S_phys) * G_clean

Then the common P_dc0 and temperature factor cancel:

    P_soil / P_clean = 1 - S_phys

so the relative electrical power-loss proxy is

    L_power_proxy = 1 - P_soil/P_clean = S_phys

This is NOT a silent identity assumption: it is the result of an explicit
first-order PVWatts ratio under the common-temperature counterfactual.

Important semantics
-------------------
- S_soil_observed remains unchanged.
- Only the physical bridge state is projected to [0,1]:
      S_phys = clip(S_soil_observed, 0, 1)
- L_power_proxy is NOT measured electrical power at Malanville.
- No TModB channel is used.
- Differential thermal effects are deliberately deferred to sensitivity /
  environment-shift analysis rather than inserted as an unvalidated point
  correction.
- RL reward / energy-yield modeling is a later, separate stage.

Inputs
------
P2-0B-5.5b daily_cleanliness_reconstruction.csv
P2-0C-1A audit_summary.json (documents rejected temperature-difference bridge)

Outputs
-------
daily_common_temp_power_bridge.csv
audit_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_DAYS = 730
EXPECTED_VALID_DAYS = 729
EXPECTED_CLEAN_DAYS = 26
MIN_COVERAGE = 0.95


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


def load_reconstruction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date",
        "cycle_id",
        "C_WAPP",
        "S_soil",
        "modb_manual_cleaning_day",
        "rain_day",
        "rain_mm_day",
        "observational_clipping_applied",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Missing 5.5b columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) != EXPECTED_DAYS:
        raise RuntimeError(f"Expected {EXPECTED_DAYS} rows, found {len(df)}")
    if df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in 5.5b reconstruction.")

    expected = pd.date_range("2021-08-09", "2023-08-08", freq="D")
    if not np.array_equal(
        df["date"].to_numpy(dtype="datetime64[ns]"),
        expected.to_numpy(dtype="datetime64[ns]"),
    ):
        raise RuntimeError("5.5b reconstruction does not exactly span report period.")

    if df["observational_clipping_applied"].fillna(False).astype(bool).any():
        raise RuntimeError("Upstream observational clipping detected.")

    c = pd.to_numeric(df["C_WAPP"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(df["S_soil"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(c) & np.isfinite(s)
    if not np.allclose(1.0 - c[finite], s[finite], rtol=0.0, atol=1e-12):
        raise RuntimeError("Upstream S_soil != 1 - C_WAPP.")

    return df


def load_rejected_bridge_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    if obj.get("stage") != "P2-0C-1A":
        raise RuntimeError("Rejected-bridge summary is not from P2-0C-1A.")

    bridge = obj.get("power_bridge", {})
    neg = bridge.get("candidate_negative_days")
    clean = bridge.get("candidate_clean_day_distribution", {})
    delta_t = obj.get("temperature_audit", {}).get(
        "estimated_clean_day_deltaT_bias_C"
    )

    if neg is None or clean.get("n") is None or delta_t is None:
        raise RuntimeError("P2-0C-1A summary is missing required rejection diagnostics.")

    return obj


def main() -> int:
    p = argparse.ArgumentParser(
        description="P2-0C-1B common-temperature PVWatts semantic bridge."
    )
    p.add_argument("--daily-cleanliness", required=True, type=Path)
    p.add_argument("--rejected-bridge-summary", required=True, type=Path)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_0c_1b_common_temp_power_bridge_v1"
        ),
    )
    args = p.parse_args()

    daily_path = args.daily_cleanliness.expanduser().resolve()
    rejected_path = args.rejected_bridge_summary.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    for path in (daily_path, rejected_path):
        if not path.exists():
            raise FileNotFoundError(path)

    print("[1/6] Load + validate frozen P2-0B-5.5b reconstruction")
    df = load_reconstruction(daily_path)

    print("[2/6] Load rejected P2-0C-1A temperature-difference bridge audit")
    rejected = load_rejected_bridge_summary(rejected_path)

    print("[3/6] Construct physical soiling state without altering observations")
    s_obs = pd.to_numeric(df["S_soil"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(s_obs)

    s_phys = np.full(len(df), np.nan, dtype=float)
    s_phys[finite] = np.clip(s_obs[finite], 0.0, 1.0)

    lower_projection = finite & (s_obs < 0.0)
    upper_projection = finite & (s_obs > 1.0)

    print("[4/6] Apply common-temperature PVWatts relative-power bridge")
    # Under common T_cell:
    # P_soil / P_clean = G_soil / G_clean = 1 - S_phys
    # Hence L_power_proxy = S_phys.
    l_proxy = s_phys.copy()

    valid = np.isfinite(l_proxy)
    coverage = float(valid.mean())

    df["S_soil_observed"] = s_obs
    df["S_soil_physical"] = s_phys
    df["S_lower_projection_applied"] = lower_projection
    df["S_upper_projection_applied"] = upper_projection
    df["L_power_proxy"] = l_proxy
    df["power_bridge_model"] = "COMMON_TEMPERATURE_PVWATTS_RATIO"
    df["measured_power_claimed"] = False
    df["bridge_valid"] = valid

    clean_mask = (
        df["modb_manual_cleaning_day"].fillna(False).astype(bool).to_numpy()
    )
    clean_l = l_proxy[clean_mask & valid]

    print("[5/6] Evaluate predeclared semantic/physical gates")
    negative_days = int((l_proxy[valid] < -1e-12).sum())
    gt1_days = int((l_proxy[valid] > 1.0 + 1e-12).sum())
    clean_zero = bool(
        len(clean_l) == EXPECTED_CLEAN_DAYS
        and np.allclose(clean_l, 0.0, rtol=0.0, atol=1e-12)
    )

    # On positive observational-soiling days, the mapping must be identity.
    positive_obs = finite & (s_obs > 0.0) & (s_obs <= 1.0)
    positive_identity = bool(
        np.allclose(
            l_proxy[positive_obs],
            s_obs[positive_obs],
            rtol=0.0,
            atol=1e-12,
        )
    )

    all_gates = bool(
        coverage >= MIN_COVERAGE
        and negative_days == 0
        and gt1_days == 0
        and clean_zero
        and positive_identity
    )

    rejected_bridge = rejected["power_bridge"]
    rejected_temp = rejected["temperature_audit"]

    summary = {
        "stage": "P2-0C-1B",
        "diagnostic_only": True,
        "power_loss_bridge_frozen": False,
        "paper1_support_audit_completed": False,
        "paper1_emulator_called": False,
        "rl_state_generated": False,
        "scientific_role": (
            "semantic alignment proxy from WAPP effective-irradiance soiling "
            "state to the Paper1/DeepSolarEye relative power-loss label domain"
        ),
        "primary_model": {
            "name": "COMMON_TEMPERATURE_PVWATTS_RATIO",
            "equation_power_model": (
                "Pdc=(G_eff/G_ref)*Pdc0*(1+gamma*(Tcell-Tref))"
            ),
            "counterfactual_assumption": (
                "clean and soiled states are compared at the same module temperature"
            ),
            "G_soil_relation": "G_soil=(1-S_phys)*G_clean",
            "derived_ratio": "P_soil/P_clean=1-S_phys",
            "derived_loss_proxy": "L_power_proxy=S_phys",
            "direct_semantic_identity_assumed_without_model": False,
            "TModB_used": False,
            "measured_malanville_power_claimed": False,
            "intended_use": "Paper1 perception-emulator input alignment only",
            "not_intended_use": "RL reward or absolute/field electrical-energy truth",
        },
        "observational_state": {
            "days": int(len(df)),
            "valid_observational_soiling_days": int(finite.sum()),
            "S_observed_distribution": qstats(s_obs),
            "observational_clipping_applied": False,
        },
        "physical_projection_for_bridge": {
            "lower_projection_days": int(lower_projection.sum()),
            "upper_projection_days": int(upper_projection.sum()),
            "S_physical_distribution": qstats(s_phys),
            "projection_changes_positive_observations": False,
        },
        "power_loss_proxy": {
            "valid_days": int(valid.sum()),
            "coverage": coverage,
            "L_power_proxy_distribution": qstats(l_proxy),
            "negative_days": negative_days,
            "gt1_days": gt1_days,
            "clean_days": int(len(clean_l)),
            "clean_day_distribution": qstats(clean_l),
            "positive_observational_days_preserved_exactly": positive_identity,
        },
        "rejected_1a_reference": {
            "model": "direct TModA/TModB differential-temperature correction",
            "clean_day_deltaT_bias_C": rejected_temp[
                "estimated_clean_day_deltaT_bias_C"
            ],
            "candidate_negative_days": rejected_bridge[
                "candidate_negative_days"
            ],
            "candidate_clean_day_distribution": rejected_bridge[
                "candidate_clean_day_distribution"
            ],
            "status": "REJECTED_AS_PRIMARY_BRIDGE",
        },
        "gates": {
            "coverage_gate": MIN_COVERAGE,
            "coverage_pass": bool(coverage >= MIN_COVERAGE),
            "negative_L_days_gate": 0,
            "negative_L_days_pass": bool(negative_days == 0),
            "L_gt1_days_gate": 0,
            "L_gt1_days_pass": bool(gt1_days == 0),
            "all_26_clean_days_L_zero_pass": clean_zero,
            "positive_S_to_L_identity_pass": positive_identity,
            "all_primary_gates_pass": all_gates,
        },
        "limitations": [
            "This is a first-order relative electrical power-loss proxy, not measured Malanville electrical power.",
            "Differential module-temperature effects are not point-corrected because the available TModA/TModB channels failed the P2-0C-1A sanity audit.",
            "Spatially nonuniform soiling and bypass-diode effects are not represented by the WAPP scalar soiling state.",
            "Paper1 support-domain and nearest-neighbor/extrapolation risk must still pass P2-0C-2 before the emulator is used.",
            "RL reward/energy-yield modeling must be built separately and must not reuse this semantic bridge as an absolute energy model.",
        ],
    }

    print("[6/6] Write audit outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        out_dir / "daily_common_temp_power_bridge.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_dir / "daily_common_temp_power_bridge.csv")
    print(out_dir / "audit_summary.json")
    print(
        "IMPORTANT: semantic bridge candidate only. Do NOT call the Paper1 "
        "emulator until P2-0C-2 support/extrapolation audit passes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
