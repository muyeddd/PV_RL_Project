#!/usr/bin/env python
"""P2-1D-2A-v1: Normalized Cleaning-Cost Scale Audit.

Frozen-input quantity-scale audit only. No cost selection or reward execution.
An explicit future invocation checks read-only Git provenance and input hashes,
then creates an immutable output directory. Importing does not run the audit.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


STAGE = "P2-1D-2A-v1"
TITLE = "Normalized Cleaning-Cost Scale Audit"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
OUTPUT = BASE / "p2_1d_2a_cleaning_cost_scale_audit_v1"
PREDECESSOR = "experiments/run_paper2_stage1d1a_daily_ghi_energy_value_audit_v1.py"
CURRENT_STAGE_SCRIPT = "experiments/run_paper2_stage1d2a_cleaning_cost_scale_audit_v1.py"
EXPECTED_HEAD = "fedfba7732a09b93c37d2f931ea5da72f0d0dcd0"
EXPECTED_BLOB = "1e9ec9182ef238d2c09e250c11ea48a3b3603fdc"
FROZEN = {
    "audit": (BASE / "p2_1d_1a_daily_ghi_energy_value_audit_v1/audit_summary.json",
              "c08330a968b96e7f38a5387ba1e8908069e9104b4e3e101e84861d15780cc4a3"),
    "energy": (BASE / "p2_1d_1a_daily_ghi_energy_value_audit_v1/daily_energy_value_proxy.csv",
               "443a023c42e9b0db9cf888c120eba53b5ad092779ea0875cd6de32826414c155"),
    "ledger": (BASE / "p2_1a_timeline_intervention_audit_v1/environment_master_ledger.csv",
               "7906680af03d94f4989efc16787d9f5431bfb53b87bd2eb0f8586c6a340fde04"),
}
ETA = 0.95
COST_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
UNIT = "normalized average-clean-day energy-value unit per CLEAN"
CALENDAR = pd.date_range("2021-08-09", "2023-08-08", freq="D")
GATES = {
    "G1": "predecessor_script_blob_pass",
    "G2": "predecessor_audit_hash_pass",
    "G3": "frozen_energy_proxy_hash_pass",
    "G4": "master_ledger_hash_pass",
    "G5": "calendar_alignment_pass",
    "G6": "frozen_energy_normalization_pass",
    "G7": "hidden_state_validity_pass",
    "G8": "729_scale_rows_pass",
    "G9": "G_inst_identity_pass",
    "G10": "cost_scale_anchor_pass",
    "G11": "candidate_grid_exact_pass",
    "G12": "no_policy_or_performance_tuning_pass",
    "G13": "normalized_economics_semantics_pass",
    "G14": "output_completeness_pass",
    "G15": "predecessor_head_ancestry_pass",
    "G16": "predecessor_audit_contract_pass",
    "G17": "current_stage_script_committed_unchanged_pass",
}
OUTPUT_FILES = ["protocol_manifest.json", "daily_cost_scale_components.csv",
                "cost_scale_summary.json", "candidate_lambda_grid.csv", "audit_summary.json"]


def git_read(*args: str) -> str:
    """Read-only commands, invoked only by main's explicit audit execution."""
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, timeout=30).stdout.strip()


def require(audit: dict, name: str, passed: bool, detail=None) -> None:
    audit[name] = bool(passed)
    audit["gate_details"][name] = detail
    if not passed:
        raise ValueError(f"Failed required gate: {name}; {detail}")


def date_strings(index) -> list[str]:
    return pd.DatetimeIndex(index).strftime("%Y-%m-%d").tolist()


def read_calendar(payload: bytes, required: set[str], name: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if (frame["date"].isna().any() or frame["date"].duplicated().any()
            or not frame["date"].eq(frame["date"].dt.normalize()).all()):
        raise ValueError(f"{name}: missing, duplicate, or non-calendar dates")
    frame = frame.set_index("date").sort_index()
    if len(frame) != 730 or not frame.index.equals(CALENDAR):
        raise ValueError(f"{name}: not the exact frozen 730-day calendar")
    return frame


def strict_bool(series: pd.Series) -> pd.Series:
    parsed = series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        raise ValueError(f"Ambiguous {series.name} at {date_strings(series.index[parsed.isna()])}")
    return parsed.astype(bool)


def statistics(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"count": 0}
    names = ["min", "P01", "P05", "P25", "median", "P75", "P90", "P95", "P99", "max"]
    quantiles = np.quantile(x, [0, .01, .05, .25, .5, .75, .9, .95, .99, 1], method="linear")
    return {"count": len(x), **dict(zip(names, map(float, quantiles))),
            "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else None}


def diagnostic_summary(frame: pd.DataFrame) -> dict:
    result = {column: statistics(frame[column])
              for column in ["G_inst", "L_power_proxy", "E_clean_GHI"]}
    result["correlations"] = {}
    for column in ["L_power_proxy", "E_clean_GHI"]:
        pair = frame[["G_inst", column]]
        pair = pair.loc[np.isfinite(pair).all(axis=1)]
        valid = len(pair) > 1 and (pair.nunique() > 1).all()
        result["correlations"][f"G_inst_vs_{column}"] = {
            "count": len(pair),
            "Pearson": float(pair.corr().iloc[0, 1]) if valid else None,
            "Spearman": float(pair.rank(method="average").corr().iloc[0, 1]) if valid else None,
            "role": "DIAGNOSTIC_ONLY"}
    return result


def protocol() -> dict:
    return {
        "stage": STAGE, "title": TITLE, "ETA": ETA,
        "eta_fitted": False, "eta_sensitivity_used": False,
        "expected_predecessor_git_HEAD": EXPECTED_HEAD,
        "predecessor_head_semantics": "EXPECTED_HEAD is the frozen predecessor checkpoint, not an equality constraint on the current stage HEAD.",
        "predecessor_head_requirement": "Commit exists and git merge-base --is-ancestor EXPECTED_HEAD HEAD returns 0",
        "predecessor_script": PREDECESSOR, "expected_predecessor_git_blob": EXPECTED_BLOB,
        "script_blob_verification": "Predecessor blob at EXPECTED_HEAD and working-tree Git clean-filtered blob must both equal EXPECTED_BLOB",
        "current_stage_script": CURRENT_STAGE_SCRIPT,
        "current_stage_script_requirement": "Formal execution requires the script committed at current HEAD and its HEAD blob exactly equal to the working-tree Git clean-filtered blob",
        "current_stage_provenance_role": "PROVENANCE_ONLY; does not enter cost scale calculations",
        "frozen_inputs": {key: {"path": str(path), "expected_sha256": digest}
                          for key, (path, digest) in FROZEN.items()},
        "calendar_start": "2021-08-09", "calendar_end": "2023-08-08",
        "calendar_days": 730, "expected_scale_rows": 729,
        "eligibility": "state_valid AND bridge_valid AND finite L_power_proxy AND finite E_clean_GHI",
        "hidden_state_source": "master ledger L_power_proxy only",
        "energy_source": "frozen E_clean_GHI only; no GHI recomputation or renormalization",
        "imputation": False, "excluded_L_imputation": False,
        "G_inst_formula": "ETA * E_clean_GHI * L_power_proxy",
        "G_inst_meaning": (
            "same-day normalized soiling-energy loss avoided by CLEAN relative to WAIT, "
            "before any future multi-day benefit"),
        "G_inst_claim_limits": "Not total cleaning benefit, economic profit, measured revenue, or actual plant energy gain.",
        "COST_SCALE_ANCHOR": "P95(G_inst), eligible rows only",
        "quantile_method": "linear", "std_ddof": 1,
        "anchor_meaning": "P95 is a robust scale anchor only, NOT an estimate of the real cleaning cost, NOT a selected optimal cost.",
        "COST_MULTIPLIERS": COST_MULTIPLIERS,
        "candidate_formula": "lambda_c = COST_SCALE_ANCHOR * multiplier",
        "lambda_c_unit": UNIT, "main_lambda_selected": False,
        "no_currency_claim": True, "no_euro_per_clean_claim": True,
        "no_CNY_per_clean_claim": True, "no_measured_economic_cost_claim": True,
        "normalized_economics_statement": "No currency claim. No €/clean. No CNY/clean. No measured economic cost claim.",
        "historical_cleaning_role": "BEHAVIOURAL_PLAUSIBILITY_REFERENCE",
        "historical_cleaning_used_for_anchor_grid_or_selection": False,
        "expected_historical_reference": "26 cleanings / 2 years, approximately 13/year; not a matching target",
        "policy_or_performance_tuning": False,
        "forbidden_selection_inputs": ["Point", "UA", "q50", "width", "k_star", "DEV-B Score",
                                       "PPO", "policy return", "reward optimization", "historical clean count",
                                       "future weather", "actual currency assumptions"],
        "future_reward_documentation_only": {
            "cost_t": "E_clean_GHI_t * L_post_t + lambda_c * action_t",
            "reward_t": "-cost_t"},
        "reward_implemented": False, "policy_implemented": False,
        "Gym_implemented": False, "PPO_implemented": False,
        "Point_UA_comparison": False,
        "E_clean_GHI_role": "REALIZED_REWARD_SETTLEMENT_ONLY",
        "forbidden_in_pre_action_observation": True,
        "no_future_GHI_provided_to_agent": True,
        "required_gates": GATES, "expected_output_files": OUTPUT_FILES,
    }


def build(audit: dict, manifest: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    head = git_read("rev-parse", "HEAD")
    manifest["current_git_head"] = head
    checkpoint = git_read("rev-parse", "--verify", f"{EXPECTED_HEAD}^{{commit}}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_HEAD, "HEAD"],
        cwd=ROOT, check=False, capture_output=True, text=True, timeout=30)
    require(audit, "predecessor_head_ancestry_pass",
            checkpoint == EXPECTED_HEAD and ancestry.returncode == 0,
            {"current_git_head": head, "frozen_predecessor_checkpoint": checkpoint,
             "ancestor_check_returncode": ancestry.returncode,
             "ancestor_check_stderr": ancestry.stderr.strip()})
    committed = git_read("rev-parse", f"{EXPECTED_HEAD}:{PREDECESSOR}")
    working = git_read("hash-object", f"--path={PREDECESSOR}", PREDECESSOR)
    require(audit, "predecessor_script_blob_pass", committed == working == EXPECTED_BLOB,
            {"committed_blob": committed, "working_tree_blob": working, "expected": EXPECTED_BLOB})
    stage_committed = git_read("rev-parse", "--verify", f"{head}:{CURRENT_STAGE_SCRIPT}")
    stage_working = git_read("hash-object", f"--path={CURRENT_STAGE_SCRIPT}", CURRENT_STAGE_SCRIPT)
    manifest["current_stage_script_sha256"] = hashlib.sha256(
        (ROOT / CURRENT_STAGE_SCRIPT).read_bytes()).hexdigest()
    manifest["current_stage_script_git_blob"] = stage_working
    manifest["current_stage_script_HEAD_blob"] = stage_committed
    require(audit, "current_stage_script_committed_unchanged_pass",
            stage_committed == stage_working,
            {"current_git_head": head, "HEAD_blob": stage_committed,
             "working_tree_filtered_blob": stage_working,
             "script_sha256": manifest["current_stage_script_sha256"]})
    payloads = {}
    hash_gates = {"audit": "predecessor_audit_hash_pass", "energy": "frozen_energy_proxy_hash_pass",
                  "ledger": "master_ledger_hash_pass"}
    for name, (path, expected) in FROZEN.items():
        # Hash and parse the same byte snapshot, avoiding a second input read.
        payloads[name] = path.read_bytes()
        actual = hashlib.sha256(payloads[name]).hexdigest()
        require(audit, hash_gates[name], actual == expected,
                {"path": str(path), "expected": expected, "actual": actual})
    predecessor = json.loads(payloads["audit"].decode("utf-8-sig"))
    contract_keys = ["stage_pass", "730_day_pass", "global_normalization_pass",
                     "information_role_no_leakage_pass"]
    require(audit, "predecessor_audit_contract_pass",
            all(predecessor.get(key) is True for key in contract_keys)
            and predecessor.get("failed_gates") == [],
            {key: predecessor.get(key) for key in contract_keys + ["failed_gates"]})
    energy = read_calendar(payloads["energy"], {"date", "E_clean_GHI"}, "energy")
    ledger = read_calendar(payloads["ledger"], {"date", "L_power_proxy", "state_valid", "bridge_valid"}, "ledger")
    require(audit, "calendar_alignment_pass", energy.index.equals(ledger.index), {"energy_rows": len(energy), "ledger_rows": len(ledger)})
    e = pd.to_numeric(energy["E_clean_GHI"], errors="coerce")
    finite_positive = np.isfinite(e) & e.gt(0)
    require(audit, "frozen_energy_normalization_pass",
            bool(finite_positive.all() and np.isclose(e.mean(), 1, rtol=1e-12, atol=1e-12)),
            {"mean": float(e.mean()) if np.isfinite(e.mean()) else None,
             "failed_dates": date_strings(e.index[~finite_positive]), "rtol": 1e-12, "atol": 1e-12})
    daily = pd.DataFrame(index=CALENDAR)
    daily.index.name = "date"
    daily["year_label"] = np.where(daily.index <= pd.Timestamp("2022-08-08"), "YEAR1", "YEAR2")
    daily["state_valid"] = strict_bool(ledger["state_valid"])
    daily["bridge_valid"] = strict_bool(ledger["bridge_valid"])
    daily["L_power_proxy"] = pd.to_numeric(ledger["L_power_proxy"], errors="coerce")
    daily["E_clean_GHI"] = e
    reasons = pd.Series("", index=daily.index)
    conditions = {
        "state_valid_false": ~daily["state_valid"],
        "bridge_valid_false": ~daily["bridge_valid"],
        "L_power_proxy_nonfinite": ~np.isfinite(daily["L_power_proxy"]),
        "E_clean_GHI_nonfinite": ~np.isfinite(daily["E_clean_GHI"]),
    }
    for reason, mask in conditions.items():
        reasons.loc[mask] = reasons.loc[mask] + reason + ";"
    daily["scale_eligible"] = reasons.eq("")
    daily["exclusion_reason"] = reasons.str.rstrip(";")
    eligible = daily["scale_eligible"]
    audit["excluded_dates_and_reasons"] = [
        {"date": date.strftime("%Y-%m-%d"), "reasons": row["exclusion_reason"].split(";")}
        for date, row in daily.loc[~eligible].iterrows()]
    special = daily.loc[pd.Timestamp("2022-06-29")]
    require(audit, "hidden_state_validity_pass",
            bool(not special["state_valid"] and not special["scale_eligible"]
                 and np.isfinite(special["E_clean_GHI"]) and special["E_clean_GHI"] > 0),
            {"date": "2022-06-29", "state_valid": bool(special["state_valid"]),
             "bridge_valid": bool(special["bridge_valid"]), "scale_eligible": bool(special["scale_eligible"]),
             "E_clean_GHI": float(special["E_clean_GHI"]), "L_interpolated": False})
    require(audit, "729_scale_rows_pass", int(eligible.sum()) == 729, {"actual": int(eligible.sum()), "expected": 729})
    daily["G_inst"] = np.nan
    daily.loc[eligible, "G_inst"] = ETA * daily.loc[eligible, "E_clean_GHI"] * daily.loc[eligible, "L_power_proxy"]
    gain = daily.loc[eligible, "G_inst"]
    identity = np.allclose(gain, ETA * daily.loc[eligible, "E_clean_GHI"] * daily.loc[eligible, "L_power_proxy"], rtol=1e-12, atol=1e-12)
    require(audit, "G_inst_identity_pass", bool(identity and np.isfinite(gain).all()
                                               and gain.ge(0).all() and daily.loc[~eligible, "G_inst"].isna().all()),
            {"nonfinite_or_negative_dates": date_strings(gain.index[~np.isfinite(gain) | gain.lt(0)]),
             "excluded_G_inst_all_missing": bool(daily.loc[~eligible, "G_inst"].isna().all())})
    anchor = float(np.quantile(gain.to_numpy(), .95, method="linear"))
    require(audit, "cost_scale_anchor_pass", bool(np.isfinite(anchor) and anchor > 0), {"P95_G_inst": anchor})
    grid = pd.DataFrame({"multiplier": COST_MULTIPLIERS})
    grid["lambda_c"] = anchor * grid["multiplier"]
    grid["role"] = ["ZERO_COST_SANITY_CONTROL"] + ["BEHAVIOUR_AUDIT_CANDIDATE"] * 7
    require(audit, "candidate_grid_exact_pass", bool(
        grid["multiplier"].tolist() == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        and grid["multiplier"].is_monotonic_increasing and len(grid) == 8
        and np.isfinite(grid["lambda_c"]).all()
        and np.array_equal(grid["lambda_c"].to_numpy(), anchor * np.asarray(COST_MULTIPLIERS))
        and grid["role"].tolist() == ["ZERO_COST_SANITY_CONTROL"] + ["BEHAVIOUR_AUDIT_CANDIDATE"] * 7))
    require(audit, "no_policy_or_performance_tuning_pass", bool(
        not manifest["main_lambda_selected"] and not manifest["policy_or_performance_tuning"]
        and not manifest["historical_cleaning_used_for_anchor_grid_or_selection"]
        and not manifest["reward_implemented"] and not manifest["policy_implemented"]
        and not manifest["Gym_implemented"] and not manifest["PPO_implemented"]
        and not manifest["Point_UA_comparison"]))
    require(audit, "normalized_economics_semantics_pass", bool(
        manifest["lambda_c_unit"] == UNIT and manifest["no_currency_claim"]
        and manifest["no_euro_per_clean_claim"] and manifest["no_CNY_per_clean_claim"]
        and manifest["no_measured_economic_cost_claim"]))
    # Historical counts are read only after anchor/grid construction, as a diagnostic.
    historical = {"role": "BEHAVIOURAL_PLAUSIBILITY_REFERENCE", "expected_count": 26,
                  "expected_approximate_per_year": 13, "used_for_selection": False, "status": "unavailable"}
    if "modb_manual_cleaning_day" in ledger:
        try:
            clean = strict_bool(ledger["modb_manual_cleaning_day"])
            historical.update(status="available", observed_count=int(clean.sum()),
                              observed_per_year=float(clean.sum() / 2), matches_expected=bool(clean.sum() == 26))
        except ValueError as exc:
            historical.update(status="ambiguous", reason=str(exc))
    scale_summary = {
        "stage": STAGE, "ETA": ETA, "COST_SCALE_ANCHOR": anchor,
        "anchor_definition": manifest["COST_SCALE_ANCHOR"], "anchor_meaning": manifest["anchor_meaning"],
        "lambda_c_unit": UNIT, "main_lambda_selected": False,
        "statistics_scope": "eligible rows only unless explicitly marked full calendar",
        "quantile_method": "linear", "std_ddof": 1,
        "ALL": diagnostic_summary(daily.loc[eligible]),
        "YEAR1": diagnostic_summary(daily.loc[eligible & daily["year_label"].eq("YEAR1")]),
        "YEAR2": diagnostic_summary(daily.loc[eligible & daily["year_label"].eq("YEAR2")]),
        "full_calendar_E_clean_GHI": statistics(e),
        "excluded_dates_and_reasons": audit["excluded_dates_and_reasons"],
        "historical_cleaning": historical,
    }
    return daily, scale_summary, grid


def json_text(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_json(path: Path, value: dict) -> None:
    content = json_text(value)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def main() -> int:
    # Refuse all existing paths before any input work; mkdir below is also exclusive.
    if OUTPUT.exists():
        raise FileExistsError(f"Immutable output directory already exists: {OUTPUT}")
    audit = {"stage": STAGE, "title": TITLE, **{name: False for name in GATES.values()},
             "gate_details": {}, "stage_pass": False}
    manifest = protocol()
    try:
        daily, scale_summary, grid = build(audit, manifest)
    except Exception as exc:
        # Fail closed before creating any outputs or candidates on invalid inputs.
        audit["execution_error"] = f"{type(exc).__name__}: {exc}"
        audit["failed_gates"] = [name for name in GATES.values() if not audit[name]]
        audit["unevaluated_gates"] = [name for name in GATES.values() if name not in audit["gate_details"]]
        print(json_text(audit))
        return 1
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_json(OUTPUT / "protocol_manifest.json", manifest)
    daily.to_csv(OUTPUT / "daily_cost_scale_components.csv", encoding="utf-8-sig", mode="x")
    write_json(OUTPUT / "cost_scale_summary.json", scale_summary)
    grid.to_csv(OUTPUT / "candidate_lambda_grid.csv", index=False, encoding="utf-8-sig", mode="x")
    # Verify persisted artifacts before preparing the final audit commit marker.
    written_daily = pd.read_csv(OUTPUT / "daily_cost_scale_components.csv", encoding="utf-8-sig")
    written_grid = pd.read_csv(OUTPUT / "candidate_lambda_grid.csv", encoding="utf-8-sig")
    complete = bool(len(written_daily) == 730 and len(written_grid) == 8
                    and set(["date", "year_label", "state_valid", "bridge_valid", "L_power_proxy",
                             "E_clean_GHI", "scale_eligible", "exclusion_reason", "G_inst"]).issubset(written_daily)
                    and written_grid.columns.tolist() == ["multiplier", "lambda_c", "role"]
                    and json.loads((OUTPUT / "protocol_manifest.json").read_text(encoding="utf-8")) == manifest
                    and json.loads((OUTPUT / "cost_scale_summary.json").read_text(encoding="utf-8")) == scale_summary)
    audit["output_completeness_pass"] = complete
    audit["gate_details"]["output_completeness_pass"] = {
        "verified_data_artifacts": OUTPUT_FILES[:-1], "final_audit_written_last": True}
    audit["stage_pass"] = all(audit[name] for name in GATES.values())
    audit["failed_gates"] = [name for name in GATES.values() if not audit[name]]
    write_json(OUTPUT / "audit_summary.json", audit)
    if not all((OUTPUT / name).is_file() and (OUTPUT / name).stat().st_size > 0 for name in OUTPUT_FILES):
        raise RuntimeError("Output completeness verification failed; immutable partial attempt retained")
    print(json_text(audit))
    return 0 if audit["stage_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
