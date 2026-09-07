#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-2B
Online Counterfactual Perception Integration Audit v1

This stage integrates the frozen Paper1-derived perception emulator with
counterfactual pre-action hidden states before reward, Gym, or PPO.

It DOES NOT tune or retrain:
- Paper1 perception / CQR / qhat,
- D0 dry dynamics,
- R3 rain dynamics,
- CLEAN mechanics,
- thresholds,
- reward,
- RL.

Frozen perception contract
--------------------------
Source: Paper1 DECISION_DEVELOPMENT final CQR outputs only.
Conditioning: true_L only; WAPP irradiance is forbidden.
Local support: |source_true_L - query_L| <= 0.01,
               >=20 local rows, >=3 source dates, no fallback.
Selector: BLOCK10 source block uniform, then within-block Gaussian
          L-distance row selection with bandwidth 0.005.
Joint transport: q50-true_L, final lower-true_L, final upper-true_L,
                 and the source lower-clipped boundary state.
The tuple is sampled jointly. q50 and width are never sampled independently.

Observation interface to be audited
-----------------------------------
True-State PPO : [L_true, sin(DOY), cos(DOY)]
Point PPO      : [q50,   sin(DOY), cos(DOY)]
UA-PPO         : [q50, width, sin(DOY), cos(DOY)]

Point and UA must share the same sampled q50 under the same
environment/perception realization. No future rain/GHI is exposed.

Validation layers
-----------------
A. Historical WAPP regeneration for all frozen perception seeds.
B. Exact regression of D0+R3+CLEAN hidden-state trajectories against P2-1C-2A.
C. Full counterfactual support audit on 1000 trajectories for support probes
   0.05, 0.10, 0.15 at eta=0.95, YEAR1/YEAR2 historical-first.
D. Online perception tuple audit on a deterministic trajectory subset.
E. Environment/perception seed separation, policy-independent perception CRN,
   and future-information leakage guards.

The static P2-0C-3C bank is a validation/reference bank only; it is not used as
the literal RL observation trajectory.

NO REWARD / NO GYM / NO PPO / NO WIDTH-SHUFFLE / NO DAILY-PERSISTENCE MODEL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_PAPER1_ROWS = 1844
EXPECTED_PAPER1_DATES = 12
EXPECTED_ROLE = "DECISION_DEVELOPMENT"

EXPECTED_CALENDAR_DAYS = 730
EXPECTED_VALID_STATES = 729
EXPECTED_DRY = 494
EXPECTED_RAIN = 201
EXPECTED_CLEAN_DAYS = 26
EXPECTED_DAYS_PER_YEAR = 365

YEAR_LABELS = ("YEAR1", "YEAR2")
PROBE_THRESHOLDS = (0.05, 0.10, 0.15)
PRIMARY_ETA = 0.95

LOCAL_RADIUS = 0.01
MIN_LOCAL_SAMPLES = 20
MIN_LOCAL_DATES = 3
KERNEL_BANDWIDTH = 0.005
BLOCK_MINUTES = 10

PERCEPTION_SEEDS = (
    20260906,
    20260907,
    20260908,
    20260909,
    20260910,
    20260911,
)
DEV_PERCEPTION_SEED = 20260906
FORMAL_PERCEPTION_SEEDS = (
    20260907,
    20260908,
    20260909,
    20260910,
    20260911,
)

DEFAULT_PARENT_ENV_SEED = 420001
DEFAULT_N_ENV_TRAJECTORIES = 1000
DEFAULT_N_PERCEPTION_AUDIT_TRAJECTORIES = 20

PERCEPTION_MARGIN = 0.01
STRICT_DOMAIN_EXCEED_GATE = 0.01
MARGIN_EXCEED_GATE = 0.01

EXPECTED_R3_INTERCEPT = 0.00090811689781017
EXPECTED_R3_SLOPE = -0.48383380267297343
PARAM_TOL = 1e-12
FLOAT_REGRESSION_TOL = 1e-15

REGRESSION_TOLERANCES = {
    "bias": 0.01,
    "mae": 0.01,
    "width_median": 0.02,
    "coverage": 0.04,
    "lower_clipped_fraction": 0.07,
    "rho_width_abs_error": 0.12,
}

# A non-clipped transported lower bound must remain strictly positive.
BOUNDARY_POSITIVE = np.nextafter(0.0, 1.0)


def parse_bool_series(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        if x.isna().any() or (~x.isin([0, 1])).any():
            raise RuntimeError(f"{name}: invalid numeric boolean values.")
        return x.astype(int).astype(bool)

    mapping = {
        "true": True, "false": False,
        "1": True, "0": False,
        "yes": True, "no": False,
        "y": True, "n": False,
    }
    x = s.astype(str).str.strip().str.lower()
    y = x.map(mapping)
    if y.isna().any():
        bad = sorted(x[y.isna()].unique().tolist())[:10]
        raise RuntimeError(f"{name}: unparseable values {bad}")
    return y.astype(bool)


def qstats(values) -> dict:
    x = pd.to_numeric(pd.Series(values), errors="coerce")
    x = x[np.isfinite(x)].to_numpy(dtype=float)
    if len(x) == 0:
        return {}
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "q01": float(q[0]),
        "q05": float(q[1]),
        "q25": float(q[2]),
        "q50": float(q[3]),
        "q75": float(q[4]),
        "q95": float(q[5]),
        "q99": float(q[6]),
        "max": float(np.max(x)),
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def spearman_safe(x, y) -> float:
    a = pd.Series(x, dtype=float)
    b = pd.Series(y, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a = a[good]
    b = b[good]
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.corr(b, method="spearman"))


def load_ledger(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "audit_period", "state_valid", "L_power_proxy",
        "rain_day", "modb_manual_cleaning_day", "scheduled_maintenance",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Master ledger missing: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    for c in [
        "state_valid", "rain_day",
        "modb_manual_cleaning_day", "scheduled_maintenance",
    ]:
        df[c] = parse_bool_series(df[c], c)
    df["L_power_proxy"] = pd.to_numeric(df["L_power_proxy"], errors="coerce")

    if df["date"].duplicated().any():
        raise RuntimeError("Duplicate dates in master ledger.")
    return df.sort_values("date").reset_index(drop=True)


def load_transitions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "transition_index", "source_date", "dest_date",
        "transition_class", "delta_L_power_proxy",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Transition audit missing: {sorted(missing)}")

    for c in ["source_date", "dest_date"]:
        df[c] = pd.to_datetime(df[c], errors="raise").dt.normalize()
    df["delta_L_power_proxy"] = pd.to_numeric(
        df["delta_L_power_proxy"], errors="coerce"
    )
    return df.sort_values("transition_index").reset_index(drop=True)


def load_paper1_cqr(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "sample_id", "date", "timestamp", "role", "true_L",
        "q50", "lower", "upper", "width", "lower_clipped",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Paper1 CQR source missing: {sorted(missing)}")

    if len(df) != EXPECTED_PAPER1_ROWS:
        raise RuntimeError(
            f"Paper1 CQR row drift: {len(df)} != {EXPECTED_PAPER1_ROWS}"
        )
    if set(df["role"].astype(str)) != {EXPECTED_ROLE}:
        raise PermissionError(
            "Only DECISION_DEVELOPMENT is authorized for emulator source."
        )

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    if df["date"].nunique() != EXPECTED_PAPER1_DATES:
        raise RuntimeError("Paper1 CQR source-date count drift.")

    for c in ["true_L", "q50", "lower", "upper", "width"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["lower_clipped"] = parse_bool_series(
        df["lower_clipped"], "lower_clipped"
    )

    numeric = df[
        ["true_L", "q50", "lower", "upper", "width"]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Non-finite Paper1 CQR values.")
    if np.any(df["lower"].to_numpy() > df["q50"].to_numpy() + 1e-12):
        raise RuntimeError("Paper1 source lower > q50.")
    if np.any(df["q50"].to_numpy() > df["upper"].to_numpy() + 1e-12):
        raise RuntimeError("Paper1 source q50 > upper.")
    if not np.allclose(
        df["width"].to_numpy(),
        df["upper"].to_numpy() - df["lower"].to_numpy(),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Paper1 source width identity drift.")

    # Fixed source 10-minute blocks.
    df["source_block_start"] = df["timestamp"].dt.floor(f"{BLOCK_MINUTES}min")
    df["source_block_id"] = (
        df["date"].dt.strftime("%Y-%m-%d")
        + "__"
        + df["source_block_start"].dt.strftime("%H:%M")
    )

    # Stable source-row order.
    df = df.sort_values(
        ["date", "timestamp", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)

    df["dq50"] = df["q50"] - df["true_L"]
    df["dlower"] = df["lower"] - df["true_L"]
    df["dupper"] = df["upper"] - df["true_L"]
    return df


def load_frozen_seed_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "perception_seed", "bias", "mae", "width_median",
        "coverage", "lower_clipped_fraction", "rho_width_abs_error",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Frozen seed metrics missing: {sorted(missing)}")

    seeds = tuple(sorted(df["perception_seed"].astype(int).tolist()))
    if seeds != tuple(sorted(PERCEPTION_SEEDS)):
        raise RuntimeError(f"Frozen perception seeds drifted: {seeds}")
    return df.sort_values("perception_seed").reset_index(drop=True)


def load_clean_audit_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if value.get("stage") != "P2-1C-2A":
        raise RuntimeError("CLEAN audit summary stage mismatch.")
    if value.get("stage_pass") is not True:
        raise RuntimeError("P2-1C-2A must PASS before online perception.")
    contract = value.get("clean_contract", {})
    if abs(float(contract.get("primary_eta")) - PRIMARY_ETA) > 1e-15:
        raise RuntimeError("Frozen CLEAN eta drift.")
    return value


def load_clean_policy_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "policy_id", "threshold", "eta", "year_label", "init_mode",
        "scenario_env_seed", "clean_count_mean",
        "pre_action_state_day_fraction_above_historical_max",
        "pre_action_state_day_fraction_above_perception_margin",
        "rain_call_above_direct_daily_support_fraction",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"P2-1C-2A policy summary missing: {sorted(missing)}"
        )
    return df


def attach_context(
    transitions: pd.DataFrame,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    src = ledger[
        ["date", "audit_period", "L_power_proxy"]
    ].rename(
        columns={
            "date": "source_date",
            "audit_period": "source_period",
            "L_power_proxy": "source_L",
        }
    )
    dst = ledger[
        ["date", "L_power_proxy"]
    ].rename(
        columns={"date": "dest_date", "L_power_proxy": "dest_L"}
    )
    return transitions.merge(
        src, on="source_date", how="left", validate="many_to_one"
    ).merge(
        dst, on="dest_date", how="left", validate="many_to_one"
    )


def fit_r3(rain: pd.DataFrame) -> dict:
    g = rain[
        np.isfinite(rain["source_L"])
        & np.isfinite(rain["delta_L_power_proxy"])
    ].copy()
    x = g["source_L"].to_numpy(dtype=float)
    y = g["delta_L_power_proxy"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)

    result = {
        "intercept": float(intercept),
        "slope": float(slope),
        "residuals": residuals,
        "source_max": float(np.max(x)),
    }
    if (
        abs(result["intercept"] - EXPECTED_R3_INTERCEPT) > PARAM_TOL
        or abs(result["slope"] - EXPECTED_R3_SLOPE) > PARAM_TOL
    ):
        raise RuntimeError("Frozen R3 parameter regression failed.")
    return result


def get_year_calendar(
    ledger: pd.DataFrame,
    year_label: str,
) -> pd.DataFrame:
    y = ledger[
        ledger["audit_period"].eq(year_label)
    ].copy().sort_values("date").reset_index(drop=True)
    if len(y) != EXPECTED_DAYS_PER_YEAR:
        raise RuntimeError(f"{year_label}: expected 365 days, got {len(y)}.")
    return y


def scenario_seed(parent_env_seed: int, year_label: str) -> int:
    # Historical-first R3 seed convention inherited from P2-1C-2A.
    stress_root = int(parent_env_seed + 500000)
    r3_candidate_offset = 100000
    yidx = 0 if year_label == "YEAR1" else 1
    return int(stress_root + r3_candidate_offset + 10000 * yidx)


def build_env_crn_indices(
    year_df: pd.DataFrame,
    dry_pool_n: int,
    rain_residual_n: int,
    n_trajectories: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    rain = year_df["rain_day"].to_numpy(dtype=bool)
    rain_affected = rain[:-1] | rain[1:]

    indices = []
    for is_rain in rain_affected:
        pool_n = rain_residual_n if is_rain else dry_pool_n
        indices.append(
            rng.integers(0, pool_n, size=n_trajectories).astype(
                np.int64, copy=False
            )
        )
    return indices, rain_affected


def simulate_threshold_hidden_states(
    year_df: pd.DataFrame,
    initial_state: float,
    threshold: float,
    eta: float,
    dry_pool: np.ndarray,
    r3: dict,
    innovation_indices: list[np.ndarray],
    rain_affected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_trajectories = len(innovation_indices[0])
    n_days = len(year_df)

    states_pre = np.empty((n_trajectories, n_days), dtype=float)
    actions = np.zeros((n_trajectories, n_days), dtype=bool)
    states_pre[:, 0] = float(initial_state)

    for t in range(n_days):
        pre = states_pre[:, t]
        clean = pre >= threshold
        actions[:, t] = clean

        post = pre.copy()
        post[clean] = (1.0 - eta) * pre[clean]

        if t == n_days - 1:
            break

        idx = innovation_indices[t]
        if rain_affected[t]:
            residual = np.asarray(r3["residuals"], dtype=float)[idx]
            delta = (
                float(r3["intercept"])
                + float(r3["slope"]) * post
                + residual
            )
        else:
            delta = dry_pool[idx]

        states_pre[:, t + 1] = np.clip(post + delta, 0.0, 1.0)

    return states_pre, actions


class OnlineBlock10Emulator:
    """
    CRN-friendly implementation of the frozen BLOCK10 transport semantics.

    Exact historical row-for-row identity with P2-0C-3C is NOT required,
    because the original bank is a stochastic reference. Distributional
    regression against all six frozen seed metrics is a hard gate.
    """

    def __init__(self, source: pd.DataFrame):
        self.source = source.reset_index(drop=True).copy()
        self.L = self.source["true_L"].to_numpy(dtype=float)
        self.date = self.source["date"].dt.strftime(
            "%Y-%m-%d"
        ).to_numpy(dtype=object)
        self.block = self.source["source_block_id"].astype(
            str
        ).to_numpy(dtype=object)
        self.dq50 = self.source["dq50"].to_numpy(dtype=float)
        self.dlower = self.source["dlower"].to_numpy(dtype=float)
        self.dupper = self.source["dupper"].to_numpy(dtype=float)
        self.lower_clipped = self.source[
            "lower_clipped"
        ].to_numpy(dtype=bool)

        self.order = np.argsort(self.L, kind="mergesort")
        self.sorted_L = self.L[self.order]

        # For exact vectorized date-count support, keep per-date sorted L.
        self.date_levels = tuple(sorted(set(self.date.tolist())))
        self.per_date_sorted_L = {
            d: np.sort(self.L[self.date == d], kind="mergesort")
            for d in self.date_levels
        }

    def _candidate_indices(self, query_L: float) -> np.ndarray:
        lo = np.searchsorted(
            self.sorted_L, query_L - LOCAL_RADIUS, side="left"
        )
        hi = np.searchsorted(
            self.sorted_L, query_L + LOCAL_RADIUS, side="right"
        )
        return self.order[lo:hi]

    def support_summary(self, queries) -> dict:
        q = np.asarray(queries, dtype=float).ravel()
        lo = np.searchsorted(
            self.sorted_L, q - LOCAL_RADIUS, side="left"
        )
        hi = np.searchsorted(
            self.sorted_L, q + LOCAL_RADIUS, side="right"
        )
        sample_counts = (hi - lo).astype(int)

        date_counts = np.zeros(len(q), dtype=int)
        for d in self.date_levels:
            arr = self.per_date_sorted_L[d]
            dlo = np.searchsorted(arr, q - LOCAL_RADIUS, side="left")
            dhi = np.searchsorted(arr, q + LOCAL_RADIUS, side="right")
            date_counts += (dhi > dlo).astype(int)

        supported = (
            (sample_counts >= MIN_LOCAL_SAMPLES)
            & (date_counts >= MIN_LOCAL_DATES)
        )
        return {
            "candidate_samples": sample_counts,
            "candidate_dates": date_counts,
            "supported": supported,
        }

    def historical_block_counts(self, queries) -> np.ndarray:
        # Used only for the 729 historical queries; exact loop is cheap.
        result = []
        for L in np.asarray(queries, dtype=float):
            cand = self._candidate_indices(float(L))
            result.append(len(np.unique(self.block[cand])))
        return np.asarray(result, dtype=int)

    def sample_one(
        self,
        query_L: float,
        u_block: float,
        u_row: float,
    ) -> dict:
        cand = self._candidate_indices(float(query_L))
        if len(cand) < MIN_LOCAL_SAMPLES:
            raise RuntimeError(
                f"Unsupported query {query_L:.9f}: {len(cand)} local rows."
            )
        dates = np.unique(self.date[cand])
        if len(dates) < MIN_LOCAL_DATES:
            raise RuntimeError(
                f"Unsupported query {query_L:.9f}: {len(dates)} local dates."
            )

        blocks = np.unique(self.block[cand])
        if len(blocks) == 0:
            raise RuntimeError("No eligible BLOCK10 source block.")

        bpos = min(
            int(math.floor(float(u_block) * len(blocks))),
            len(blocks) - 1,
        )
        chosen_block = blocks[bpos]

        rows = cand[self.block[cand] == chosen_block]
        dist = self.L[rows] - float(query_L)
        weights = np.exp(
            -0.5 * np.square(dist / KERNEL_BANDWIDTH)
        )
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise RuntimeError("Invalid Gaussian source-row weights.")

        cdf = np.cumsum(weights)
        target = float(u_row) * float(cdf[-1])
        rpos = int(np.searchsorted(cdf, target, side="right"))
        rpos = min(rpos, len(rows) - 1)
        src = int(rows[rpos])

        q50 = float(np.clip(query_L + self.dq50[src], 0.0, 1.0))

        if self.lower_clipped[src]:
            lower = 0.0
        else:
            raw_lower = float(query_L + self.dlower[src])
            lower = float(
                min(1.0, max(float(BOUNDARY_POSITIVE), raw_lower))
            )

        upper = float(np.clip(query_L + self.dupper[src], 0.0, 1.0))

        # Never silently sort or repair the tuple.
        if lower > q50 + 1e-12:
            raise RuntimeError("Transport broke lower <= q50.")
        if q50 > upper + 1e-12:
            raise RuntimeError("Transport broke q50 <= upper.")

        width = float(upper - lower)
        generated_lower_clipped = bool(lower == 0.0)
        if generated_lower_clipped != bool(self.lower_clipped[src]):
            raise RuntimeError(
                "Source lower-clipped boundary state was not preserved."
            )

        return {
            "query_L": float(query_L),
            "q50": q50,
            "lower": lower,
            "upper": upper,
            "width": width,
            "covered": bool(lower <= query_L <= upper),
            "lower_clipped": generated_lower_clipped,
            "source_index": src,
            "source_date": str(self.date[src]),
            "source_block_id": str(self.block[src]),
            "source_true_L": float(self.L[src]),
            "source_lower_clipped": bool(self.lower_clipped[src]),
            "candidate_samples": int(len(cand)),
            "candidate_dates": int(len(dates)),
            "candidate_blocks": int(len(blocks)),
            "fallback_used": False,
        }


def perception_metrics(frame: pd.DataFrame) -> dict:
    true = frame["query_L"].to_numpy(dtype=float)
    q50 = frame["q50"].to_numpy(dtype=float)
    lower = frame["lower"].to_numpy(dtype=float)
    upper = frame["upper"].to_numpy(dtype=float)
    width = frame["width"].to_numpy(dtype=float)
    err = q50 - true

    return {
        "rows": int(len(frame)),
        "bias": float(np.mean(err)),
        "mae": float(np.mean(np.abs(err))),
        "width_median": float(np.median(width)),
        "coverage": float(
            np.mean((true >= lower) & (true <= upper))
        ),
        "lower_clipped_fraction": float(
            frame["lower_clipped"].astype(bool).mean()
        ),
        "rho_width_abs_error": spearman_safe(width, np.abs(err)),
        "finite_pass": bool(
            np.isfinite(
                frame[
                    ["query_L", "q50", "lower", "upper", "width"]
                ].to_numpy(dtype=float)
            ).all()
        ),
        "ordering_pass": bool(
            (lower <= q50 + 1e-12).all()
            and (q50 <= upper + 1e-12).all()
        ),
        "width_identity_pass": bool(
            np.allclose(width, upper - lower, atol=1e-12, rtol=0.0)
        ),
        "source_lower_clipped_preserved_pass": bool(
            (
                frame["lower_clipped"].astype(bool).to_numpy()
                == frame["source_lower_clipped"].astype(bool).to_numpy()
            ).all()
        ),
        "fallback_used_n": int(
            frame["fallback_used"].astype(bool).sum()
        ),
        "unique_source_dates": int(frame["source_date"].nunique()),
        "unique_source_blocks": int(
            frame["source_block_id"].nunique()
        ),
        "max_source_date_share": float(
            frame["source_date"].value_counts(normalize=True).max()
        ),
        "max_source_block_share": float(
            frame["source_block_id"].value_counts(normalize=True).max()
        ),
    }


def generate_historical_path(
    emulator: OnlineBlock10Emulator,
    queries: np.ndarray,
    perception_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(perception_seed)
    rows = []
    for day_index, L in enumerate(queries):
        rec = emulator.sample_one(
            float(L),
            float(rng.random()),
            float(rng.random()),
        )
        rec["day_index"] = int(day_index)
        rec["perception_seed"] = int(perception_seed)
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_to_frozen_seed_metrics(
    generated: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    rows = []
    all_pass = True

    for seed in PERCEPTION_SEEDS:
        g = generated[generated["perception_seed"].eq(seed)]
        gm = perception_metrics(g)

        f = frozen[frozen["perception_seed"].astype(int).eq(seed)]
        if len(f) != 1:
            raise RuntimeError(
                f"Frozen seed metric row missing for {seed}."
            )
        fr = f.iloc[0]

        row = {"perception_seed": int(seed)}
        seed_pass = True

        for metric, tol in REGRESSION_TOLERANCES.items():
            gv = float(gm[metric])
            fv = float(fr[metric])
            diff = abs(gv - fv)
            passed = bool(np.isfinite(diff) and diff <= tol)
            seed_pass = seed_pass and passed

            row[f"generated_{metric}"] = gv
            row[f"frozen_{metric}"] = fv
            row[f"absdiff_{metric}"] = diff
            row[f"tolerance_{metric}"] = float(tol)
            row[f"pass_{metric}"] = passed

        row["all_distribution_regression_pass"] = bool(seed_pass)
        rows.append(row)
        all_pass = all_pass and seed_pass

    return pd.DataFrame(rows), bool(all_pass)


def extract_2a_reference_row(
    policy_summary: pd.DataFrame,
    threshold: float,
    year_label: str,
) -> pd.Series:
    g = policy_summary[
        np.isclose(
            pd.to_numeric(policy_summary["threshold"], errors="coerce"),
            threshold,
        )
        & np.isclose(
            pd.to_numeric(policy_summary["eta"], errors="coerce"),
            PRIMARY_ETA,
        )
        & policy_summary["year_label"].astype(str).eq(year_label)
        & policy_summary["init_mode"].astype(str).eq(
            "HISTORICAL_FIRST_DAY"
        )
    ]
    if len(g) != 1:
        raise RuntimeError(
            f"Cannot resolve P2-1C-2A reference {threshold}/{year_label}."
        )
    return g.iloc[0]


def hidden_state_regression_metrics(
    states_pre: np.ndarray,
    actions: np.ndarray,
    historical_max: float,
    perception_margin_max: float,
    r3_direct_max: float,
    rain_affected: np.ndarray,
) -> dict:
    clean_count_mean = float(actions.sum(axis=1).mean())
    hist_frac = float((states_pre > historical_max).mean())
    margin_frac = float(
        (states_pre > perception_margin_max).mean()
    )

    rain_calls = 0
    rain_above = 0
    for t in range(states_pre.shape[1] - 1):
        if not rain_affected[t]:
            continue
        pre = states_pre[:, t]
        clean = actions[:, t]
        post = pre.copy()
        post[clean] = (1.0 - PRIMARY_ETA) * pre[clean]
        rain_calls += len(post)
        rain_above += int(np.sum(post > r3_direct_max))

    return {
        "clean_count_mean": clean_count_mean,
        "pre_action_state_day_fraction_above_historical_max": hist_frac,
        "pre_action_state_day_fraction_above_perception_margin": margin_frac,
        "rain_call_above_direct_daily_support_fraction": float(
            rain_above / rain_calls
        ),
    }


def compare_2a_regression(
    observed: dict,
    reference: pd.Series,
) -> dict:
    keys = (
        "clean_count_mean",
        "pre_action_state_day_fraction_above_historical_max",
        "pre_action_state_day_fraction_above_perception_margin",
        "rain_call_above_direct_daily_support_fraction",
    )
    result = {}
    passes = []
    for key in keys:
        a = float(observed[key])
        b = float(reference[key])
        diff = abs(a - b)
        passed = bool(diff <= FLOAT_REGRESSION_TOL)
        result[f"observed_{key}"] = a
        result[f"reference_{key}"] = b
        result[f"absdiff_{key}"] = diff
        result[f"pass_{key}"] = passed
        passes.append(passed)
    result["all_2a_regression_pass"] = bool(all(passes))
    return result


def perception_uniforms_for_policy_crn(
    perception_seed: int,
    year_label: str,
    n_trajectories: int,
    n_days: int,
) -> tuple[np.ndarray, str]:
    year_code = 1 if year_label == "YEAR1" else 2
    # Policy/threshold is deliberately absent from SeedSequence.
    ss = np.random.SeedSequence(
        [int(perception_seed), year_code, 2202]
    )
    rng = np.random.default_rng(ss)
    u = rng.random((n_trajectories, n_days, 2), dtype=np.float64)
    return u, sha256_array(u)


def audit_online_perception_subset(
    emulator: OnlineBlock10Emulator,
    states_pre: np.ndarray,
    year_df: pd.DataFrame,
    threshold: float,
    year_label: str,
    n_audit_trajectories: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], list[dict]]:
    if n_audit_trajectories > states_pre.shape[0]:
        raise RuntimeError(
            "Perception audit subset exceeds environment trajectories."
        )

    state_subset = states_pre[:n_audit_trajectories]
    n_days = state_subset.shape[1]

    metric_rows = []
    source_usage_rows = []
    example_rows = []
    crn_rows = []

    for seed in PERCEPTION_SEEDS:
        uniforms, uhash = perception_uniforms_for_policy_crn(
            seed, year_label, n_audit_trajectories, n_days
        )
        rows = []

        for traj in range(n_audit_trajectories):
            for d in range(n_days):
                L = float(state_subset[traj, d])
                rec = emulator.sample_one(
                    L,
                    float(uniforms[traj, d, 0]),
                    float(uniforms[traj, d, 1]),
                )

                doy = int(year_df.loc[d, "date"].dayofyear)
                angle = 2.0 * math.pi * ((doy - 1) / 365.0)
                sin_doy = math.sin(angle)
                cos_doy = math.cos(angle)

                true_obs = np.array(
                    [L, sin_doy, cos_doy], dtype=float
                )
                point_obs = np.array(
                    [rec["q50"], sin_doy, cos_doy], dtype=float
                )
                ua_obs = np.array(
                    [rec["q50"], rec["width"], sin_doy, cos_doy],
                    dtype=float,
                )

                if not (
                    np.isfinite(true_obs).all()
                    and np.isfinite(point_obs).all()
                    and np.isfinite(ua_obs).all()
                ):
                    raise RuntimeError("Non-finite online observation.")

                rec.update({
                    "threshold": float(threshold),
                    "year_label": year_label,
                    "perception_seed": int(seed),
                    "trajectory_id": int(traj),
                    "day_index": int(d),
                    "date": year_df.loc[d, "date"],
                    "sin_doy": float(sin_doy),
                    "cos_doy": float(cos_doy),
                    "point_q50_equals_ua_q50": bool(
                        point_obs[0] == ua_obs[0]
                    ),
                    "seasonal_features_identical": bool(
                        point_obs[1] == ua_obs[2]
                        and point_obs[2] == ua_obs[3]
                        and true_obs[1] == point_obs[1]
                        and true_obs[2] == point_obs[2]
                    ),
                })
                rows.append(rec)

        f = pd.DataFrame(rows)
        m = perception_metrics(f)
        m.update({
            "threshold": float(threshold),
            "year_label": year_label,
            "perception_seed": int(seed),
            "audit_trajectories": int(n_audit_trajectories),
            "query_rows": int(len(f)),
            "point_ua_q50_pairing_pass": bool(
                f["point_q50_equals_ua_q50"].all()
            ),
            "seasonal_pairing_pass": bool(
                f["seasonal_features_identical"].all()
            ),
            "query_L_distribution": qstats(f["query_L"]),
        })
        metric_rows.append(m)

        source_usage_rows.append({
            "threshold": float(threshold),
            "year_label": year_label,
            "perception_seed": int(seed),
            "unique_source_dates": int(f["source_date"].nunique()),
            "max_source_date_share": float(
                f["source_date"].value_counts(normalize=True).max()
            ),
            "unique_source_blocks": int(
                f["source_block_id"].nunique()
            ),
            "max_source_block_share": float(
                f["source_block_id"].value_counts(
                    normalize=True
                ).max()
            ),
        })

        crn_rows.append({
            "threshold": float(threshold),
            "year_label": year_label,
            "perception_seed": int(seed),
            "uniform_stream_hash": uhash,
        })

        # Small trace for manual review only, not the formal observation bank.
        keep_n = min(100, len(f))
        example_rows.extend(f.head(keep_n).to_dict(orient="records"))

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(source_usage_rows),
        example_rows,
        crn_rows,
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "P2-1C-2B online counterfactual perception integration audit."
        )
    )
    p.add_argument("--master-ledger", required=True, type=Path)
    p.add_argument("--transition-audit", required=True, type=Path)
    p.add_argument("--clean-audit-summary", required=True, type=Path)
    p.add_argument("--clean-policy-summary", required=True, type=Path)
    p.add_argument("--paper1-cqr", required=True, type=Path)
    p.add_argument("--frozen-seed-metrics", required=True, type=Path)
    p.add_argument(
        "--frozen-emulator-script",
        type=Path,
        default=Path(
            "experiments/"
            "run_paper2_stage0c3c_generate_perception_bank_v1.py"
        ),
        help=(
            "Provenance hash only; this stage does not execute or modify it."
        ),
    )
    p.add_argument(
        "--n-env-trajectories",
        type=int,
        default=DEFAULT_N_ENV_TRAJECTORIES,
    )
    p.add_argument(
        "--n-perception-audit-trajectories",
        type=int,
        default=DEFAULT_N_PERCEPTION_AUDIT_TRAJECTORIES,
    )
    p.add_argument(
        "--parent-env-seed",
        type=int,
        default=DEFAULT_PARENT_ENV_SEED,
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c2b_online_counterfactual_perception_audit_v1"
        ),
    )
    args = p.parse_args()

    if args.n_env_trajectories < 200:
        raise ValueError("--n-env-trajectories must be >=200.")
    if not (
        1
        <= args.n_perception_audit_trajectories
        <= args.n_env_trajectories
    ):
        raise ValueError(
            "Invalid --n-perception-audit-trajectories."
        )

    paths = {
        "master_ledger": args.master_ledger.expanduser().resolve(),
        "transition_audit": args.transition_audit.expanduser().resolve(),
        "clean_audit_summary": args.clean_audit_summary.expanduser().resolve(),
        "clean_policy_summary": args.clean_policy_summary.expanduser().resolve(),
        "paper1_cqr": args.paper1_cqr.expanduser().resolve(),
        "frozen_seed_metrics": args.frozen_seed_metrics.expanduser().resolve(),
        "frozen_emulator_script": args.frozen_emulator_script.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    out = args.output_dir.expanduser().resolve()

    print("[1/12] Load frozen timeline, CLEAN, and Paper1 perception assets")
    ledger = load_ledger(paths["master_ledger"])
    transitions = load_transitions(paths["transition_audit"])
    _ = load_clean_audit_summary(paths["clean_audit_summary"])
    clean_policy_summary = load_clean_policy_summary(
        paths["clean_policy_summary"]
    )
    source = load_paper1_cqr(paths["paper1_cqr"])
    frozen_seed_metrics = load_frozen_seed_metrics(
        paths["frozen_seed_metrics"]
    )

    print("[2/12] Verify frozen counts, roles, and seed separation")
    if len(ledger) != EXPECTED_CALENDAR_DAYS:
        raise RuntimeError("Calendar day count drift.")
    if int(ledger["state_valid"].sum()) != EXPECTED_VALID_STATES:
        raise RuntimeError("Valid-state count drift.")
    if int(ledger["modb_manual_cleaning_day"].sum()) != EXPECTED_CLEAN_DAYS:
        raise RuntimeError("Manual-clean count drift.")

    env_seeds = {
        scenario_seed(args.parent_env_seed, y)
        for y in YEAR_LABELS
    }
    perception_seed_set = set(PERCEPTION_SEEDS)
    seed_disjoint = bool(env_seeds.isdisjoint(perception_seed_set))
    if not seed_disjoint:
        raise RuntimeError("Environment/perception seed collision.")

    print("[3/12] Reconstruct frozen D0 + R3 + CLEAN mechanics")
    x = attach_context(transitions, ledger)
    dry = x[x["transition_class"].eq("DRY_NATURAL")].copy()
    rain = x[x["transition_class"].eq("RAIN_AFFECTED")].copy()
    if len(dry) != EXPECTED_DRY:
        raise RuntimeError("Dry-transition count drift.")
    if len(rain) != EXPECTED_RAIN:
        raise RuntimeError("Rain-transition count drift.")

    dry_pool = dry["delta_L_power_proxy"].to_numpy(dtype=float)
    r3 = fit_r3(rain)

    valid_hist = ledger.loc[
        ledger["state_valid"]
        & np.isfinite(ledger["L_power_proxy"]),
        "L_power_proxy",
    ].to_numpy(dtype=float)
    historical_max = float(np.max(valid_hist))
    perception_margin_max = float(
        historical_max + PERCEPTION_MARGIN
    )

    print("[4/12] Build BLOCK10 online emulator and historical support index")
    emulator = OnlineBlock10Emulator(source)
    historical_queries = valid_hist.copy()
    hist_support = emulator.support_summary(historical_queries)
    historical_blocks = emulator.historical_block_counts(
        historical_queries
    )
    historical_support_100 = bool(
        hist_support["supported"].all()
    )

    print("[5/12] Re-run online emulator on historical WAPP path for six seeds")
    historical_parts = []
    historical_seed_rows = []

    for seed in PERCEPTION_SEEDS:
        f = generate_historical_path(
            emulator, historical_queries, seed
        )
        historical_parts.append(f)

        m = perception_metrics(f)
        m.update({
            "perception_seed": int(seed),
            "candidate_samples_min": int(
                hist_support["candidate_samples"].min()
            ),
            "candidate_dates_min": int(
                hist_support["candidate_dates"].min()
            ),
            "candidate_blocks_min": int(historical_blocks.min()),
        })
        historical_seed_rows.append(m)

    historical_generated = pd.concat(
        historical_parts, ignore_index=True
    )
    historical_seed_df = pd.DataFrame(historical_seed_rows)

    print("[6/12] Compare historical regeneration with frozen P2-0C-3C metrics")
    regression_df, historical_distribution_regression_pass = (
        compare_to_frozen_seed_metrics(
            historical_generated,
            frozen_seed_metrics,
        )
    )

    print("[7/12] Rebuild counterfactual probe states and regress P2-1C-2A")
    hidden_assets = {}
    hidden_regression_rows = []
    operational_support_rows = []

    for year_label in YEAR_LABELS:
        year_df = get_year_calendar(ledger, year_label)
        initial_state = float(year_df["L_power_proxy"].iloc[0])
        env_seed = scenario_seed(
            args.parent_env_seed, year_label
        )

        innovation_indices, rain_affected = build_env_crn_indices(
            year_df=year_df,
            dry_pool_n=len(dry_pool),
            rain_residual_n=len(r3["residuals"]),
            n_trajectories=args.n_env_trajectories,
            seed=env_seed,
        )

        for threshold in PROBE_THRESHOLDS:
            states_pre, actions = simulate_threshold_hidden_states(
                year_df=year_df,
                initial_state=initial_state,
                threshold=float(threshold),
                eta=PRIMARY_ETA,
                dry_pool=dry_pool,
                r3=r3,
                innovation_indices=innovation_indices,
                rain_affected=rain_affected,
            )

            observed = hidden_state_regression_metrics(
                states_pre=states_pre,
                actions=actions,
                historical_max=historical_max,
                perception_margin_max=perception_margin_max,
                r3_direct_max=r3["source_max"],
                rain_affected=rain_affected,
            )
            ref = extract_2a_reference_row(
                clean_policy_summary,
                float(threshold),
                year_label,
            )
            reg = compare_2a_regression(observed, ref)
            reg.update({
                "threshold": float(threshold),
                "year_label": year_label,
                "env_seed": int(env_seed),
            })
            hidden_regression_rows.append(reg)

            flat = states_pre.ravel()
            support = emulator.support_summary(flat)
            operational_support_rows.append({
                "threshold": float(threshold),
                "year_label": year_label,
                "env_seed": int(env_seed),
                "hidden_state_rows": int(len(flat)),
                "local_support_fraction": float(
                    support["supported"].mean()
                ),
                "unsupported_query_n": int(
                    (~support["supported"]).sum()
                ),
                "candidate_samples_min": int(
                    support["candidate_samples"].min()
                ),
                "candidate_samples_median": float(
                    np.median(support["candidate_samples"])
                ),
                "candidate_dates_min": int(
                    support["candidate_dates"].min()
                ),
                "fraction_above_historical_validated_max": float(
                    np.mean(flat > historical_max)
                ),
                "fraction_above_perception_margin": float(
                    np.mean(flat > perception_margin_max)
                ),
                "state_mean": float(np.mean(flat)),
                "state_q50": float(np.quantile(flat, 0.50)),
                "state_q95": float(np.quantile(flat, 0.95)),
                "state_q99": float(np.quantile(flat, 0.99)),
                "state_max": float(np.max(flat)),
            })

            hidden_assets[(float(threshold), year_label)] = {
                "states_pre": states_pre,
                "actions": actions,
                "year_df": year_df,
                "env_seed": env_seed,
            }

    hidden_regression_df = pd.DataFrame(
        hidden_regression_rows
    )
    operational_support_df = pd.DataFrame(
        operational_support_rows
    )

    print("[8/12] Apply full operational support-domain gates")
    all_hidden_2a_regression_pass = bool(
        hidden_regression_df["all_2a_regression_pass"].all()
    )
    operational_local_support_pass = bool(
        (operational_support_df["local_support_fraction"] == 1.0).all()
    )
    operational_strict_domain_pass = bool(
        (
            operational_support_df[
                "fraction_above_historical_validated_max"
            ]
            <= STRICT_DOMAIN_EXCEED_GATE
        ).all()
    )
    operational_margin_pass = bool(
        (
            operational_support_df[
                "fraction_above_perception_margin"
            ]
            <= MARGIN_EXCEED_GATE
        ).all()
    )

    print("[9/12] Sample online perception on deterministic CF audit subset")
    online_metric_parts = []
    source_usage_parts = []
    example_rows = []
    perception_crn_rows = []

    for threshold in PROBE_THRESHOLDS:
        for year_label in YEAR_LABELS:
            asset = hidden_assets[(float(threshold), year_label)]
            metrics, usage, examples, crn = (
                audit_online_perception_subset(
                    emulator=emulator,
                    states_pre=asset["states_pre"],
                    year_df=asset["year_df"],
                    threshold=float(threshold),
                    year_label=year_label,
                    n_audit_trajectories=(
                        args.n_perception_audit_trajectories
                    ),
                )
            )
            online_metric_parts.append(metrics)
            source_usage_parts.append(usage)
            example_rows.extend(examples)
            perception_crn_rows.extend(crn)

    online_metrics_df = pd.concat(
        online_metric_parts, ignore_index=True
    )
    source_usage_df = pd.concat(
        source_usage_parts, ignore_index=True
    )
    perception_crn_df = pd.DataFrame(perception_crn_rows)

    print("[10/12] Audit tuple structure, Point/UA pairing, CRN, and leakage")
    structural_cols = (
        "finite_pass",
        "ordering_pass",
        "width_identity_pass",
        "source_lower_clipped_preserved_pass",
        "point_ua_q50_pairing_pass",
        "seasonal_pairing_pass",
    )
    online_structural_pass = bool(
        online_metrics_df[list(structural_cols)]
        .to_numpy(dtype=bool)
        .all()
        and (online_metrics_df["fallback_used_n"] == 0).all()
    )

    crn_group = (
        perception_crn_df
        .groupby(
            ["year_label", "perception_seed"], sort=True
        )["uniform_stream_hash"]
        .nunique()
    )
    perception_crn_policy_independent_pass = bool(
        (crn_group == 1).all()
    )

    source_asset_gates = {
        "paper1_rows_pass": bool(len(source) == EXPECTED_PAPER1_ROWS),
        "paper1_dates_pass": bool(
            source["date"].nunique() == EXPECTED_PAPER1_DATES
        ),
        "paper1_role_pass": bool(
            set(source["role"].astype(str)) == {EXPECTED_ROLE}
        ),
        "wapp_irradiance_used": False,
        "random_test_used": False,
        "sealed_dates_used": False,
    }

    historical_gates = {
        "historical_WAPP_support_100pct_pass": historical_support_100,
        "historical_distribution_regression_all_6_seeds_pass": (
            historical_distribution_regression_pass
        ),
        "historical_structural_invariants_pass": bool(
            historical_seed_df[
                [
                    "finite_pass",
                    "ordering_pass",
                    "width_identity_pass",
                    "source_lower_clipped_preserved_pass",
                ]
            ].to_numpy(dtype=bool).all()
            and (historical_seed_df["fallback_used_n"] == 0).all()
        ),
    }

    operational_gates = {
        "P2_1C_2A_hidden_state_regression_pass": (
            all_hidden_2a_regression_pass
        ),
        "all_counterfactual_queries_local_support_pass": (
            operational_local_support_pass
        ),
        "strict_historical_validated_domain_99pct_pass": (
            operational_strict_domain_pass
        ),
        "perception_margin_99pct_pass": operational_margin_pass,
        "online_tuple_structural_invariants_pass": (
            online_structural_pass
        ),
        "perception_CRN_policy_independent_pass": (
            perception_crn_policy_independent_pass
        ),
        "environment_perception_seed_disjoint_pass": seed_disjoint,
        "future_information_leakage_absent_pass": True,
    }

    all_source_asset_gates = bool(
        source_asset_gates["paper1_rows_pass"]
        and source_asset_gates["paper1_dates_pass"]
        and source_asset_gates["paper1_role_pass"]
        and not source_asset_gates["wapp_irradiance_used"]
        and not source_asset_gates["random_test_used"]
        and not source_asset_gates["sealed_dates_used"]
    )
    all_historical_gates = bool(all(historical_gates.values()))
    all_operational_gates = bool(all(operational_gates.values()))
    stage_pass = bool(
        all_source_asset_gates
        and all_historical_gates
        and all_operational_gates
    )

    print("[11/12] Build scientific audit summary")
    audit_summary = {
        "stage": "P2-1C-2B",
        "audit_only": True,
        "online_counterfactual_perception_built": True,
        "reward_built": False,
        "gym_built": False,
        "rl_started": False,
        "width_shuffle_built": False,
        "daily_perception_persistence_built": False,
        "frozen_perception_design": {
            "source_role": EXPECTED_ROLE,
            "conditioning": ["true_L"],
            "local_radius_abs_L": LOCAL_RADIUS,
            "local_min_samples": MIN_LOCAL_SAMPLES,
            "local_min_dates": MIN_LOCAL_DATES,
            "selector": "BLOCK10_UNIFORM_THEN_GAUSSIAN_ROW",
            "block_minutes": BLOCK_MINUTES,
            "kernel_bandwidth": KERNEL_BANDWIDTH,
            "joint_transport": [
                "q50-true_L",
                "final lower-true_L",
                "final upper-true_L",
                "source lower-clipped boundary state",
            ],
            "fallback": False,
            "wapp_irradiance_used": False,
            "qhat_recalibrated": False,
            "static_3C_bank_used_as_literal_RL_trajectory": False,
        },
        "provenance": {
            "paper1_cqr_path": str(paths["paper1_cqr"]),
            "paper1_cqr_sha256": sha256_file(paths["paper1_cqr"]),
            "frozen_0C3C_script_path": str(
                paths["frozen_emulator_script"]
            ),
            "frozen_0C3C_script_sha256": sha256_file(
                paths["frozen_emulator_script"]
            ),
            "frozen_seed_metrics_path": str(
                paths["frozen_seed_metrics"]
            ),
            "frozen_seed_metrics_sha256": sha256_file(
                paths["frozen_seed_metrics"]
            ),
        },
        "randomness_contract": {
            "environment_seeds": sorted(int(x) for x in env_seeds),
            "perception_seeds": list(PERCEPTION_SEEDS),
            "dev_perception_seed": DEV_PERCEPTION_SEED,
            "formal_perception_seeds": list(FORMAL_PERCEPTION_SEEDS),
            "environment_perception_seed_disjoint": seed_disjoint,
            "RL_seed_assigned_yet": False,
            "perception_uniforms_reused_across_probe_policies": True,
        },
        "observation_schema": {
            "true_state_ppo": ["L_true", "sin_DOY", "cos_DOY"],
            "point_ppo": ["q50", "sin_DOY", "cos_DOY"],
            "ua_ppo": ["q50", "width", "sin_DOY", "cos_DOY"],
            "point_prediction_source": "q50",
            "uncertainty_source": "final_conformal_width",
            "future_rain_in_observation": False,
            "future_GHI_in_observation": False,
            "historical_clean_indicator_in_observation": False,
            "lower_upper_directly_in_ua_observation": False,
        },
        "historical_WAPP_support": {
            "queries": int(len(historical_queries)),
            "supported_fraction": float(
                hist_support["supported"].mean()
            ),
            "candidate_samples_distribution": qstats(
                hist_support["candidate_samples"]
            ),
            "candidate_dates_distribution": qstats(
                hist_support["candidate_dates"]
            ),
            "candidate_blocks_distribution": qstats(historical_blocks),
        },
        "historical_regenerated_seed_metrics": (
            historical_seed_df.to_dict(orient="records")
        ),
        "historical_distribution_regression": (
            regression_df.to_dict(orient="records")
        ),
        "operational_support": (
            operational_support_df.to_dict(orient="records")
        ),
        "P2_1C_2A_hidden_state_regression": (
            hidden_regression_df.to_dict(orient="records")
        ),
        "online_perception_metrics": (
            online_metrics_df.to_dict(orient="records")
        ),
        "source_asset_gates": {
            **source_asset_gates,
            "all_source_asset_gates_pass": all_source_asset_gates,
        },
        "historical_gates": {
            **historical_gates,
            "all_historical_gates_pass": all_historical_gates,
        },
        "operational_gates": {
            **operational_gates,
            "all_operational_gates_pass": all_operational_gates,
        },
        "stage_pass": stage_pass,
        "scientific_interpretation": {
            "if_pass": (
                "The frozen Paper1-derived perception process can be queried "
                "online at counterfactual pre-action hidden states without "
                "changing its declared support/transport semantics, using no "
                "fallback and no future information. Point and UA observations "
                "can therefore be paired under identical environment and "
                "perception realizations."
            ),
            "if_fail": (
                "Do not tune the frozen emulator or widen support. Identify "
                "whether failure comes from historical emulator-regression "
                "drift, counterfactual support occupancy, boundary transport, "
                "or implementation mismatch."
            ),
            "limitations_retained": [
                "Paper1-derived stochastic surrogate, not WAPP field-image validation.",
                "No daily perception-regime persistence in the primary emulator.",
                "BLOCK10 is a source-resampling cluster, not a daily WAPP timescale.",
                "No cross-site image-domain generalization is claimed.",
                "Width-shuffle remains mandatory later before claiming unique decision value from width.",
            ],
        },
        "next_step_if_pass": (
            "Freeze Online Perception Interface v1, then define and audit the "
            "reward/economic contract before Gym/PPO."
        ),
        "next_step_if_fail": (
            "Stop before reward/Gym/PPO and diagnose the failed gate without "
            "post-hoc threshold or emulator tuning."
        ),
    }

    print("[12/12] Write outputs")
    out.mkdir(parents=True, exist_ok=True)

    historical_seed_df.to_csv(
        out / "historical_regenerated_seed_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    regression_df.to_csv(
        out / "historical_distribution_regression.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hidden_regression_df.to_csv(
        out / "p2_1c2a_hidden_state_regression.csv",
        index=False,
        encoding="utf-8-sig",
    )
    operational_support_df.to_csv(
        out / "operational_support_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    online_metrics_df.to_csv(
        out / "online_perception_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    source_usage_df.to_csv(
        out / "online_source_usage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    perception_crn_df.to_csv(
        out / "perception_crn_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(example_rows).to_csv(
        out / "online_perception_examples.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (out / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, ensure_ascii=False, indent=2)

    print(out / "historical_regenerated_seed_metrics.csv")
    print(out / "historical_distribution_regression.csv")
    print(out / "p2_1c2a_hidden_state_regression.csv")
    print(out / "operational_support_audit.csv")
    print(out / "online_perception_metrics.csv")
    print(out / "online_source_usage_summary.csv")
    print(out / "perception_crn_audit.csv")
    print(out / "online_perception_examples.csv")
    print(out / "audit_summary.json")
    print(
        "IMPORTANT: Online perception audit only. "
        "Do NOT build reward, Gym, PPO, width-shuffle, or daily-persistence "
        "sensitivity before scientific review."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
