#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper2 / P2-1C-2B-R1
Frozen Sampler Implementation Equivalence Audit v1

PURPOSE
-------
Diagnose the P2-1C-2B historical-regression failure without changing any
scientific hyperparameter, seed, support radius, kernel bandwidth, CQR/qhat,
BLOCK10 rule, or tolerance.

This audit directly loads the FROZEN P2-0C-3C generator and checks:

1) The frozen generator still reproduces the frozen trajectory_bank.csv.
2) An independently written "corrected online core" reproduces the frozen
   generator row-for-row under the same seeds.
3) The exact lower-boundary semantics responsible for the failed
   lower_clipped_fraction regression:
       source_lower_clipped=True  -> generated lower=0
   BUT ALSO:
       source_lower_clipped=False can become generated lower=0 when
       L_query + (source_lower - source_true_L) < 0.
   The failed P2-1C-2B v1 incorrectly prohibited this second case.

This is an implementation-equivalence audit only.
NO environment changes, NO perception retuning, NO reward, NO Gym, NO PPO.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ALL_SEEDS = [
    20260906,
    20260907,
    20260908,
    20260909,
    20260910,
    20260911,
]
ATOL = 1e-12


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "frozen_p2_0c_3c_generator", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen generator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen_bank(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "date", "L_true", "q50", "lower", "upper", "width",
        "covered", "lower_clipped", "perception_seed",
        "source_sample_id", "source_date", "source_block10_id",
        "source_true_L", "source_lower_clipped",
        "candidate_samples", "candidate_dates", "candidate_blocks",
    }
    missing = required.difference(d.columns)
    if missing:
        raise RuntimeError(
            f"Frozen trajectory bank missing columns: {sorted(missing)}"
        )

    d["date"] = pd.to_datetime(d["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    d["source_date"] = pd.to_datetime(
        d["source_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    for c in [
        "L_true", "q50", "lower", "upper", "width", "source_true_L",
    ]:
        d[c] = pd.to_numeric(d[c], errors="raise")

    for c in ["covered", "lower_clipped", "source_lower_clipped"]:
        d[c] = d[c].astype(bool)

    d["perception_seed"] = d["perception_seed"].astype(int)

    expected = 729 * len(ALL_SEEDS)
    if len(d) != expected:
        raise RuntimeError(
            f"Frozen bank row count mismatch: {len(d)} != {expected}"
        )
    if sorted(d["perception_seed"].unique().tolist()) != ALL_SEEDS:
        raise RuntimeError("Frozen bank perception-seed set mismatch.")

    return d.reset_index(drop=True)


def corrected_choose_source(
    rng: np.random.Generator,
    p1: pd.DataFrame,
    query_L: float,
    radius: float,
    bandwidth: float,
) -> tuple[int, int, int, int]:
    """
    Independent concise implementation of the frozen selector:
    local rows -> uniform sorted BLOCK10 -> Gaussian L-distance row.
    Original Paper1 CSV row order is retained within each block.
    """
    source_L = p1["true_L"].to_numpy(dtype=float)
    distances = np.abs(source_L - float(query_L))
    idx = np.flatnonzero(distances <= radius)

    source_dates = p1["date"].to_numpy(dtype=str)
    source_blocks = p1["block10_id"].to_numpy(dtype=str)

    dates = np.unique(source_dates[idx])
    blocks = np.unique(source_blocks[idx])

    if len(idx) < 20 or len(dates) < 3 or len(blocks) < 1:
        raise RuntimeError(
            f"Unsupported query L={query_L:.9f}: "
            f"N={len(idx)}, dates={len(dates)}, blocks={len(blocks)}"
        )

    chosen_block = str(rng.choice(sorted(blocks.tolist())))
    block_rows = idx[source_blocks[idx] == chosen_block]

    dist = distances[block_rows]
    w = np.exp(-0.5 * np.square(dist / bandwidth))
    w = w / float(w.sum())

    source_index = int(rng.choice(block_rows, p=w))
    return source_index, len(idx), len(dates), len(blocks)


def corrected_transport(
    source_row: pd.Series,
    query_L: float,
) -> tuple[float, float, float]:
    source_L = float(source_row["true_L"])

    q50 = float(
        np.clip(
            float(query_L)
            + (float(source_row["q50"]) - source_L),
            0.0,
            1.0,
        )
    )
    upper = float(
        np.clip(
            float(query_L)
            + (float(source_row["upper"]) - source_L),
            0.0,
            1.0,
        )
    )

    if bool(source_row["lower_clipped"]):
        lower = 0.0
    else:
        # This is the key frozen semantic:
        # a non-source-clipped lower can become newly clipped after transport.
        lower = float(
            np.clip(
                float(query_L)
                + (float(source_row["lower"]) - source_L),
                0.0,
                1.0,
            )
        )

    if lower > q50 + ATOL or q50 > upper + ATOL:
        raise RuntimeError("Corrected transport interval ordering violated.")

    return q50, lower, upper


def corrected_generate(
    p1: pd.DataFrame,
    wapp: pd.DataFrame,
    seed: int,
    radius: float,
    bandwidth: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for i, r in wapp.iterrows():
        L = float(r["L_power_proxy"])

        source_index, n_samples, n_dates, n_blocks = (
            corrected_choose_source(
                rng=rng,
                p1=p1,
                query_L=L,
                radius=radius,
                bandwidth=bandwidth,
            )
        )

        s = p1.iloc[source_index]
        q50, lower, upper = corrected_transport(s, L)
        width = float(upper - lower)

        generated_lower_clipped = bool(
            np.isclose(lower, 0.0, atol=ATOL, rtol=0.0)
        )
        source_lower_clipped = bool(s["lower_clipped"])

        raw_transported_lower = float(
            L + (float(s["lower"]) - float(s["true_L"]))
        )
        newly_created_lower_clip = bool(
            (not source_lower_clipped)
            and raw_transported_lower <= ATOL
        )

        rows.append(
            {
                "date": str(r["date"]),
                "L_true": L,
                "q50": q50,
                "lower": lower,
                "upper": upper,
                "width": width,
                "covered": bool(lower <= L <= upper),
                "lower_clipped": generated_lower_clipped,
                "perception_seed": int(seed),
                "source_sample_id": s["sample_id"],
                "source_date": str(s["date"]),
                "source_block10_id": str(s["block10_id"]),
                "source_true_L": float(s["true_L"]),
                "source_lower_clipped": source_lower_clipped,
                "raw_transported_lower_before_clip": raw_transported_lower,
                "newly_created_lower_clip": newly_created_lower_clip,
                "candidate_samples": int(n_samples),
                "candidate_dates": int(n_dates),
                "candidate_blocks": int(n_blocks),
                "wapp_row": int(i),
            }
        )

    return pd.DataFrame(rows)


def compare_rows(
    a: pd.DataFrame,
    b: pd.DataFrame,
    label: str,
) -> dict:
    if len(a) != len(b):
        return {
            "label": label,
            "rows_equal": False,
            "reason": f"row count {len(a)} != {len(b)}",
            "all_exact_pass": False,
        }

    # String / categorical identities.
    exact_cols = [
        "date",
        "perception_seed",
        "source_sample_id",
        "source_date",
        "source_block10_id",
        "lower_clipped",
        "source_lower_clipped",
        "candidate_samples",
        "candidate_dates",
        "candidate_blocks",
    ]

    exact_results = {}
    for c in exact_cols:
        av = a[c].astype(str).to_numpy()
        bv = b[c].astype(str).to_numpy()
        exact_results[c] = bool(np.array_equal(av, bv))

    # Numeric identities.
    numeric_cols = [
        "L_true", "q50", "lower", "upper", "width", "source_true_L",
    ]
    numeric_results = {}
    max_abs_diff = {}

    for c in numeric_cols:
        av = pd.to_numeric(a[c], errors="raise").to_numpy(dtype=float)
        bv = pd.to_numeric(b[c], errors="raise").to_numpy(dtype=float)
        diff = np.abs(av - bv)
        numeric_results[c] = bool(
            np.allclose(av, bv, atol=ATOL, rtol=0.0)
        )
        max_abs_diff[c] = float(np.max(diff)) if len(diff) else 0.0

    all_exact = bool(
        all(exact_results.values()) and all(numeric_results.values())
    )

    return {
        "label": label,
        "rows": int(len(a)),
        "exact_column_pass": exact_results,
        "numeric_column_pass": numeric_results,
        "numeric_max_abs_diff": max_abs_diff,
        "all_exact_pass": all_exact,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "P2-1C-2B-R1 frozen sampler implementation equivalence audit."
        )
    )
    ap.add_argument(
        "--frozen-generator-script",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--paper1-dev-cqr",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--wapp-power-bridge",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--frozen-trajectory-bank",
        required=True,
        type=Path,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/paper2_uncertainty_rl_v1/"
            "p2_1c2b_r1_frozen_sampler_equivalence_audit_v1"
        ),
    )
    args = ap.parse_args()

    paths = {
        "generator": args.frozen_generator_script.expanduser().resolve(),
        "paper1": args.paper1_dev_cqr.expanduser().resolve(),
        "wapp": args.wapp_power_bridge.expanduser().resolve(),
        "bank": args.frozen_trajectory_bank.expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name}: {path}")

    out = args.output_dir.expanduser().resolve()

    print("[1/8] Import frozen P2-0C-3C generator")
    frozen = load_module(paths["generator"])

    print("[2/8] Load exact frozen Paper1 and WAPP assets")
    p1 = frozen.load_p1(paths["paper1"])
    wapp = frozen.load_wapp(paths["wapp"])
    frozen_bank = load_frozen_bank(paths["bank"])

    print("[3/8] Rebuild frozen support maps")
    support, maps = frozen.build_support(p1, wapp)
    if not bool(support["support_ok"].astype(bool).all()):
        raise RuntimeError("Frozen support unexpectedly failed.")

    print("[4/8] Reproduce frozen generator and compare with frozen bank")
    frozen_parts = []
    corrected_parts = []
    per_seed_rows = []

    for seed in ALL_SEEDS:
        role = "DEV" if seed == 20260906 else "FORMAL_EVAL"

        reference = frozen.generate(
            p1, wapp, support, maps, seed, role
        ).copy()
        reference["perception_seed"] = int(seed)

        corrected = corrected_generate(
            p1=p1,
            wapp=wapp,
            seed=seed,
            radius=float(frozen.RADIUS),
            bandwidth=float(frozen.BW),
        )

        bank_seed = frozen_bank.loc[
            frozen_bank["perception_seed"].eq(seed)
        ].reset_index(drop=True)

        reference = reference.reset_index(drop=True)
        corrected = corrected.reset_index(drop=True)

        c1 = compare_rows(
            reference,
            bank_seed,
            label="frozen_generator_vs_frozen_bank",
        )
        c2 = compare_rows(
            corrected,
            reference,
            label="corrected_online_core_vs_frozen_generator",
        )

        new_clip_fraction = float(
            corrected["newly_created_lower_clip"].astype(bool).mean()
        )
        source_clip_fraction = float(
            corrected["source_lower_clipped"].astype(bool).mean()
        )
        generated_clip_fraction = float(
            corrected["lower_clipped"].astype(bool).mean()
        )

        identity = float(
            generated_clip_fraction
            - source_clip_fraction
            - new_clip_fraction
        )

        per_seed_rows.append(
            {
                "perception_seed": int(seed),
                "frozen_generator_vs_bank_exact_pass": bool(
                    c1["all_exact_pass"]
                ),
                "corrected_core_vs_frozen_generator_exact_pass": bool(
                    c2["all_exact_pass"]
                ),
                "source_lower_clipped_fraction": source_clip_fraction,
                "new_transport_lower_clip_fraction": new_clip_fraction,
                "generated_lower_clipped_fraction": generated_clip_fraction,
                "boundary_fraction_identity_residual": identity,
            }
        )

        reference["audit_source"] = "FROZEN_GENERATOR"
        corrected["audit_source"] = "CORRECTED_ONLINE_CORE"
        frozen_parts.append(reference)
        corrected_parts.append(corrected)

    per_seed = pd.DataFrame(per_seed_rows)
    frozen_rebuilt = pd.concat(frozen_parts, ignore_index=True)
    corrected_rebuilt = pd.concat(corrected_parts, ignore_index=True)

    print("[5/8] Quantify the lower-boundary semantic discrepancy")
    boundary = corrected_rebuilt[
        [
            "perception_seed",
            "wapp_row",
            "date",
            "L_true",
            "source_sample_id",
            "source_true_L",
            "source_lower_clipped",
            "raw_transported_lower_before_clip",
            "newly_created_lower_clip",
            "lower_clipped",
        ]
    ].copy()

    newly_created = boundary[
        boundary["newly_created_lower_clip"].astype(bool)
    ].copy()

    print("[6/8] Apply implementation-equivalence gates")
    frozen_bank_regression_pass = bool(
        per_seed[
            "frozen_generator_vs_bank_exact_pass"
        ].all()
    )
    corrected_equivalence_pass = bool(
        per_seed[
            "corrected_core_vs_frozen_generator_exact_pass"
        ].all()
    )

    boundary_identity_pass = bool(
        np.all(
            np.abs(
                per_seed["boundary_fraction_identity_residual"].to_numpy(
                    dtype=float
                )
            )
            <= ATOL
        )
    )

    new_clip_exists = bool(
        (
            per_seed["new_transport_lower_clip_fraction"] > 0
        ).all()
    )

    all_pass = bool(
        frozen_bank_regression_pass
        and corrected_equivalence_pass
        and boundary_identity_pass
        and new_clip_exists
    )

    print("[7/8] Build audit summary")
    summary = {
        "stage": "P2-1C-2B-R1",
        "audit_only": True,
        "scientific_hyperparameters_changed": False,
        "perception_seeds_changed": False,
        "tolerances_changed": False,
        "root_cause": {
            "primary": (
                "P2-1C-2B v1 incorrectly forced a non-source-clipped "
                "transported lower bound to remain strictly positive. "
                "The frozen P2-0C-3C transport instead applies np.clip(...,0,1), "
                "so a non-source-clipped source can legitimately become "
                "lower-clipped after transport to a cleaner query state."
            ),
            "frozen_semantics": (
                "source_lower_clipped => generated lower=0, but the converse "
                "is not required."
            ),
            "secondary_alignment_repairs_for_v1_1": [
                "Preserve original Paper1 CSV row order.",
                "Use the exact frozen BLOCK10 string/group construction.",
                "Use rng.choice(sorted(blocks)) and rng.choice(rows,p=weights).",
                "For online CRN, use deterministic per-(seed,year,trajectory,day) "
                "substreams so policy-dependent candidate sets cannot desynchronize "
                "future perception randomness.",
            ],
        },
        "provenance": {
            "frozen_generator_sha256": sha256_file(paths["generator"]),
            "paper1_dev_cqr_sha256": sha256_file(paths["paper1"]),
            "wapp_power_bridge_sha256": sha256_file(paths["wapp"]),
            "frozen_trajectory_bank_sha256": sha256_file(paths["bank"]),
        },
        "per_seed": per_seed.to_dict(orient="records"),
        "new_transport_lower_clip_rows": int(len(newly_created)),
        "new_transport_lower_clip_fraction_overall": float(
            corrected_rebuilt[
                "newly_created_lower_clip"
            ].astype(bool).mean()
        ),
        "gates": {
            "frozen_generator_reproduces_frozen_bank_exactly": (
                frozen_bank_regression_pass
            ),
            "corrected_online_core_matches_frozen_generator_exactly": (
                corrected_equivalence_pass
            ),
            "boundary_fraction_identity_pass": boundary_identity_pass,
            "new_transport_lower_clip_exists_all_seeds": new_clip_exists,
            "all_gates_pass": all_pass,
        },
        "next_step_if_pass": (
            "Patch P2-1C-2B as v1.1 using the verified frozen sampler semantics, "
            "retain the original predeclared regression tolerances, rerun the "
            "historical + counterfactual online perception audit, and do not "
            "enter reward/Gym/PPO until v1.1 passes."
        ),
        "next_step_if_fail": (
            "Do not patch the full online interface yet. Inspect the exact "
            "row/source mismatch reported by this equivalence audit."
        ),
    }

    print("[8/8] Write outputs")
    out.mkdir(parents=True, exist_ok=True)

    per_seed.to_csv(
        out / "per_seed_equivalence.csv",
        index=False,
        encoding="utf-8-sig",
    )
    boundary.to_csv(
        out / "boundary_semantics_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    newly_created.to_csv(
        out / "new_transport_lower_clip_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (out / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out / "per_seed_equivalence.csv")
    print(out / "boundary_semantics_audit.csv")
    print(out / "new_transport_lower_clip_cases.csv")
    print(out / "audit_summary.json")
    print(
        "IMPORTANT: equivalence audit only. "
        "Do NOT change tolerance/model/support or start reward/Gym/PPO."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
