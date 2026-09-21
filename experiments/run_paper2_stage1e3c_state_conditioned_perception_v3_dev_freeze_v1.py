"""P2-1E-3C-v1 state-conditioned perception-v3 DEV fidelity freeze.

This is the terminal development attempt for the empirical perception-extension
branch. The method and gates are fixed before results:
  * source: Paper1 DECISION_DEVELOPMENT only;
  * complete leave-one-source-date-out (1844 targets / 12 dates);
  * v1-supported queries must remain exact;
  * adaptive-only queries use the single preregistered local-linear residual
    correction in paper2_state_conditioned_perception_v3.py;
  * FULL and ADAPTIVE_ONLY aggregates must BOTH pass the original P2-0C-3B3
    six distribution tolerances;
  * no RANDOM_TEST, SEALED_DATES, PPO or formal-RL access.

If DEV fails, the empirical perception-extension branch stops. There is no v4/v5
parameter search. If DEV passes, v3 is frozen and only then may a separate
one-shot RANDOM_TEST confirmation be implemented.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1e3c_state_conditioned_perception_v3_dev_freeze_v1.py"
MODULE = "experiments/paper2_state_conditioned_perception_v3.py"
V2_MODULE = "experiments/paper2_adaptive_radius_perception_v2.py"
PREDECESSOR_SOURCE = "experiments/run_paper2_stage1e3b_adaptive_radius_perception_v2_fidelity_freeze_v1.py"
LEGACY_SOURCE = "experiments/run_paper2_stage0c3b3_source_weighting_audit_v1.py"
FROZEN_SOURCE = "experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
LEGACY = BASE / "p2_0c_3b3_source_weighting_audit_v1"
PREVIOUS = BASE / "p2_1e_3b_adaptive_radius_perception_v2_fidelity_freeze_v1/formal"
STAGE = BASE / "p2_1e_3c_state_conditioned_perception_v3_dev_freeze_v1"
OUTPUT = STAGE / "formal"

CHECKPOINT = "973f2b177d56b240b85e73680336a625f868d972"
PREVIOUS_AUDIT_SHA256 = "82d79b3b8d85a4f6f5cef82388a31303adaed03883db64a6f7c65834a164d48a"

# Git blobs pinned at the P2-1E-3B source checkpoint.
PINNED_BLOBS = {
    V2_MODULE: "b2266cc78185c71fb0aa61a08e5f4fe4976f96f5",
    PREDECESSOR_SOURCE: "0cd35843038bb43898069b5e2f60da0be229b2cd",
    LEGACY_SOURCE: "2a094decbfbc4883d22abcb3e41e4501ab09e956",
    FROZEN_SOURCE: "6f940272ce1a262f325af310c9e72ed0adc401eb",
}

EXPECTED_PREVIOUS_FAILED_GATES = [
    "B FULL mae",
    "B FULL coverage",
    "B FULL rho_width_abs_error",
    "B ADAPTIVE_ONLY bias",
    "B ADAPTIVE_ONLY mae",
    "B ADAPTIVE_ONLY coverage",
    "B ADAPTIVE_ONLY lower_clipped_fraction",
    "B ADAPTIVE_ONLY rho_width_abs_error",
]

SCOPE = (
    "Paper1 DECISION_DEVELOPMENT internal leave-one-source-date-out development "
    "validation only; no RANDOM_TEST, SEALED_DATES, protected Paper1 role, "
    "external generalization, PPO or formal-RL claim"
)
PASS_DECLARATION = (
    "STATE-CONDITIONED RESIDUAL-TRANSPORT PERCEPTION V3 FROZEN AFTER COMPLETE "
    "DECISION_DEVELOPMENT SOURCE-DATE VALIDATION; V1-SUPPORTED QUERIES REMAIN "
    "EXACT; FULL AND ADAPTIVE-ONLY POPULATIONS PASS THE ORIGINAL P2-0C-3B3 "
    "DISTRIBUTION TOLERANCES; NO RANDOM_TEST, SEALED_DATES, PPO OR FORMAL RL "
    "PERFORMANCE WAS ACCESSED"
)
FAIL_DECLARATION = (
    "STATE-CONDITIONED RESIDUAL-TRANSPORT V3 DEV FIDELITY FAIL; EMPIRICAL "
    "PERCEPTION-EXTENSION BRANCH STOPS; POINT/UA PPO REMAINS BLOCKED; NO V4/V5 "
    "PARAMETER SEARCH, RETRY OR THRESHOLD RELAXATION"
)


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1048576), b""):
            h.update(data)
    return h.hexdigest()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_bytes(name, data):
    path = OUTPUT / name
    with path.open("xb") as f:
        f.write(data)
    require(path.read_bytes() == data, "Exact output readback: " + name)


def write_json(name, value):
    data = (
        json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    write_bytes(name, data)


def write_csv(name, rows):
    require(bool(rows), "Nonempty CSV output: " + name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(clean(rows))
    write_bytes(name, stream.getvalue().encode())


def exact_number(a, b):
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or a.hex() == b.hex()


def previous_provenance(head):
    """Consume the completed 3B failure as immutable scientific predecessor."""
    require(sha(PREVIOUS / "audit_summary.json") == PREVIOUS_AUDIT_SHA256,
            "Exact P2-1E-3B audit-summary SHA256")
    audit = read_json(PREVIOUS / "audit_summary.json")
    require(
        audit["HEAD"] == CHECKPOINT
        and audit["stage"] == "P2-1E-3B-v1"
        and audit["stage_pass"] is False
        and audit["scientific_status"] == "ADAPTIVE_RADIUS_PERCEPTION_V2_FIDELITY_FAIL"
        and audit["declaration"] == "POINT/UA PPO REMAINS BLOCKED; NO RETRY OR PARAMETER CHANGE"
        and audit["failed_gates"] == EXPECTED_PREVIOUS_FAILED_GATES
        and audit["PPO_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["protected_Paper1_roles_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0,
        "Accepted P2-1E-3B scientific failure exact",
    )
    for name, expected in audit["output_sha256"].items():
        require(sha(PREVIOUS / name) == expected, "P2-1E-3B output hash: " + name)

    metrics = read_json(PREVIOUS / "adaptive_v2_fidelity_metrics.json")
    require(
        metrics["N_targets"] == 1844
        and metrics["N_source_dates"] == 12
        and metrics["legacy_supported_n"] == 1231
        and metrics["adaptive_only_n"] == 613
        and metrics["full_and_adaptive_only_pass"] is False,
        "P2-1E-3B population/failure counts exact",
    )
    identity = read_json(PREVIOUS / "v1_backward_identity_summary.json")
    require(
        identity["pass"] is True
        and identity["exact_identity_fraction"] == 1.0
        and identity["historical_macro"]["all_pass"] is True,
        "P2-1E-3B legacy identity predecessor PASS",
    )

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Frozen predecessor Git blob: " + name,
        )

    provenance = read_json(PREVIOUS / "v2_source_provenance.json")
    for raw_name, expected in provenance["sha256"].items():
        p = Path(raw_name)
        path = p if p.is_absolute() else ROOT / p
        require(path.exists() and sha(path) == expected,
                "Predecessor scientific input unchanged: " + raw_name)
    return audit, metrics, identity, provenance


def verify_static_contract(v3):
    """Reject hidden tuning, new RNG, candidate changes or legacy-path changes."""
    cls = v3.StateConditionedResidualTransportV3
    require(
        issubclass(cls, v3.v2.AdaptiveRadiusOnlineBlock10EmulatorV2),
        "V3 inherits accepted adaptive geometry implementation",
    )
    require(
        "_candidate_indices" not in cls.__dict__
        and "radius_requirement" not in cls.__dict__,
        "V3 does not change adaptive candidate geometry",
    )

    tree = ast.parse((ROOT / MODULE).read_text(encoding="utf-8"))
    random_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in {"default_rng", "RandomState"}
    ]
    require(not random_calls, "V3 module initializes no RNG")

    sample = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "sample_one"
    )
    names = {n.id for n in ast.walk(sample) if isinstance(n, ast.Name)}
    require("rng" in names and "KERNEL_BANDWIDTH" not in names,
            "sample_one reuses supplied RNG and adds no bandwidth")
    require(
        not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in {"polyfit", "lstsq"}
            for n in ast.walk(tree)
        ),
        "No hidden polynomial/model selection implementation",
    )

    source = (ROOT / MODULE).read_text(encoding="utf-8")
    forbidden = [
        "RANDOM_TEST",
        "SEALED_DATES",
        "GridSearch",
        "RandomizedSearch",
        "Ridge(",
        "Lasso(",
    ]
    require(not any(token in source for token in forbidden),
            "No protected-role or tuning implementation in v3 module")


def seed_formula(authority):
    """Mechanically recover the original P2-0C-3B3 BLOCK10 RNG formula."""
    tree = ast.parse((ROOT / LEGACY_SOURCE).read_text(encoding="utf-8"))
    nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "default_rng"
    ]
    require(len(nodes) == 1, "One frozen 3B3 RNG initializer")
    expression = nodes[0].args[0]
    require(
        {n.id for n in ast.walk(expression) if isinstance(n, ast.Name)}
        == {"BASE_SEED", "selector_idx", "fold_idx", "rep"},
        "Original RNG namespace exact",
    )
    program = compile(ast.Expression(expression), "<frozen seed>", "eval")
    selector_idx = authority.SELECTORS.index("BLOCK10")

    def value(fold, rep):
        return eval(
            program,
            {"__builtins__": {}},
            {
                "BASE_SEED": authority.BASE_SEED,
                "selector_idx": selector_idx,
                "fold_idx": fold,
                "rep": rep,
            },
        )

    return value, ast.unparse(expression)


def mean_finite(values):
    import numpy as np
    x = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(x)) if x else float("nan")


def reference_metrics(authority, frame):
    return authority.metric_dict(
        frame["true_L"], frame["q50"], frame["lower"], frame["upper"]
    )


def metrics(authority, records):
    return authority.metric_dict(
        [r["query_L"] for r in records],
        [r["q50"] for r in records],
        [r["lower"] for r in records],
        [r["upper"] for r in records],
    )


def date_row(authority, actual, rep_metrics, **metadata):
    generated = {
        k: mean_finite([r[k] for r in rep_metrics]) for k in authority.GATES
    }
    return {
        **metadata,
        **{"actual_" + k: float(actual[k]) for k in authority.GATES},
        **{"generated_" + k: generated[k] for k in authority.GATES},
        **{
            "absdiff_" + k: abs(generated[k] - actual[k])
            for k in authority.GATES
        },
    }


def macro_gates(authority, rows):
    import pandas as pd

    def macro(prefix):
        return {
            k: float(
                pd.to_numeric(
                    pd.Series([r[prefix + k] for r in rows], dtype=float),
                    errors="coerce",
                )
                .dropna()
                .mean()
            )
            for k in authority.GATES
        }

    return authority.compare_gates(macro("actual_"), macro("generated_"))


def radius_band(radius, base):
    if radius == base:
        return "r == base"
    if radius <= 0.015:
        return "base < r <= 0.015"
    if radius <= 0.020:
        return "0.015 < r <= 0.020"
    if radius <= 0.030:
        return "0.020 < r <= 0.030"
    if radius <= 0.040:
        return "0.030 < r <= 0.040"
    return "r > 0.040"


def assert_finite_sample(sample):
    require(
        all(math.isfinite(float(sample[k])) for k in ("q50", "lower", "upper", "width"))
        and sample["lower"] <= sample["q50"] <= sample["upper"]
        and sample["width"] == sample["upper"] - sample["lower"]
        and sample["fallback_used"] is False,
        "Finite ordered perception sample; no fallback",
    )


def development_population(authority, v2, v3, source, seed):
    """Complete 1844-row/12-date LODO with fixed FULL and ADAPTIVE_ONLY gates."""
    import numpy as np

    dates = sorted(source["date"].unique())
    target_inventory = []
    by_date = []
    all_generated = []
    identity_total = 0
    identity_exact = 0
    source_identity_total = 0
    source_identity_exact = 0
    rng_identity_total = 0
    rng_identity_exact = 0

    for fold, heldout in enumerate(dates):
        pool = source[~source["date"].eq(heldout)].copy().reset_index(drop=True)
        target = source[source["date"].eq(heldout)].copy().reset_index(drop=True)
        require(
            len(target) > 0 and not pool["date"].eq(heldout).any(),
            "LODO source/target disjoint",
        )

        old_model = v2.AdaptiveRadiusOnlineBlock10EmulatorV2(pool)
        new_model = v3.StateConditionedResidualTransportV3(pool)

        metadata = []
        for _, row in target.iterrows():
            q = float(row["true_L"])
            requirement = new_model.radius_requirement(q)
            legacy = requirement["required_radius"] == v3.v2.frozen.LOCAL_RADIUS
            trend = new_model.trend_diagnostics(q)
            item = {
                "heldout_date": str(heldout),
                "target_sample_id": str(row["sample_id"]),
                "query_L": q,
                "stratum": "LEGACY_SUPPORTED" if legacy else "ADAPTIVE_ONLY",
                "radius_band": radius_band(
                    requirement["required_radius"], v3.v2.frozen.LOCAL_RADIUS
                ),
                **requirement,
                **{
                    k: trend[k]
                    for k in (
                        "sampling_probability_ess",
                        "slope_dq50",
                        "slope_dlower",
                        "slope_dupper",
                    )
                },
            }
            target_inventory.append(item)
            metadata.append(item)

        masks = {
            "FULL": np.ones(len(target), dtype=bool),
            "LEGACY_SUPPORTED": np.asarray(
                [m["stratum"] == "LEGACY_SUPPORTED" for m in metadata]
            ),
            "ADAPTIVE_ONLY": np.asarray(
                [m["stratum"] == "ADAPTIVE_ONLY" for m in metadata]
            ),
        }
        rep_metrics = {k: [] for k in masks}

        for rep in range(authority.MC_REPS):
            old_rng = np.random.default_rng(seed(fold, rep))
            new_rng = np.random.default_rng(seed(fold, rep))
            old_model = v2.AdaptiveRadiusOnlineBlock10EmulatorV2(pool)
            new_model = v3.StateConditionedResidualTransportV3(pool)
            samples = []

            for (_, row), info in zip(target.iterrows(), metadata):
                q = float(row["true_L"])
                try:
                    old = old_model.sample_one(q, old_rng)
                    new = new_model.sample_one(q, new_rng)
                except Exception as exc:
                    raise RuntimeError(
                        f"V3 sample failure heldout={heldout} rep={rep} "
                        f"sample_id={row['sample_id']} L={q:.17g}: {exc}"
                    ) from exc

                assert_finite_sample(new)
                source_identity_total += 1
                same_source = (
                    int(old["source_index"]) == int(new["source_index"])
                    and str(old["source_sample_id"]) == str(new["source_sample_id"])
                    and str(old["source_date"]) == str(new["source_date"])
                    and str(old["source_block_id"]) == str(new["source_block_id"])
                )
                source_identity_exact += int(same_source)
                require(same_source, "V3 must preserve v2 source selection exactly")

                rng_identity_total += 1
                same_rng = old_rng.bit_generator.state == new_rng.bit_generator.state
                rng_identity_exact += int(same_rng)
                require(same_rng, "V3 must preserve v2 RNG consumption exactly")

                if info["stratum"] == "LEGACY_SUPPORTED":
                    identity_total += 1
                    exact = old == new
                    identity_exact += int(exact)
                    require(exact, "Legacy-supported v3 output must be exact v2/v1")
                else:
                    require(
                        new.get("v3_applied") is True,
                        "Adaptive-only query must use v3 correction",
                    )

                record = {
                    **new,
                    "heldout_date": str(heldout),
                    "rep": rep,
                    "target_sample_id": str(row["sample_id"]),
                    "stratum": info["stratum"],
                    "required_radius": info["required_radius"],
                    "radius_band": info["radius_band"],
                }
                samples.append(record)
                all_generated.append(record)

            for name, mask in masks.items():
                if mask.any():
                    rep_metrics[name].append(
                        metrics(
                            authority,
                            [r for r, ok in zip(samples, mask) if ok],
                        )
                    )

        for name, mask in masks.items():
            subset = target.loc[mask]
            if len(subset):
                by_date.append(
                    date_row(
                        authority,
                        reference_metrics(authority, subset),
                        rep_metrics[name],
                        population="DEV",
                        stratum=name,
                        heldout_date=str(heldout),
                        N_targets=int(len(subset)),
                        empty_stratum=False,
                    )
                )
            else:
                by_date.append(
                    {
                        "population": "DEV",
                        "stratum": name,
                        "heldout_date": str(heldout),
                        "N_targets": 0,
                        "empty_stratum": True,
                    }
                )

    require(
        len(target_inventory) == authority.EXPECTED_N == 1844
        and len(dates) == authority.EXPECTED_DATES == 12
        and len({r["target_sample_id"] for r in target_inventory}) == 1844
        and len(all_generated) == 1844 * authority.MC_REPS,
        "Complete DEV population/repetitions with no target deletion",
    )

    counts = {
        name: sum(r["stratum"] == name for r in target_inventory)
        for name in ("LEGACY_SUPPORTED", "ADAPTIVE_ONLY")
    }
    require(
        counts == {"LEGACY_SUPPORTED": 1231, "ADAPTIVE_ONLY": 613},
        "Predeclared 3B DEV strata reproduced exactly",
    )
    require(identity_total == counts["LEGACY_SUPPORTED"] * authority.MC_REPS,
            "All legacy draws audited")
    require(identity_exact == identity_total, "100% legacy exact identity")
    require(source_identity_exact == source_identity_total, "100% source identity")
    require(rng_identity_exact == rng_identity_total, "100% RNG identity")

    gates = {
        name: macro_gates(
            authority,
            [
                r
                for r in by_date
                if r["stratum"] == name and not r["empty_stratum"]
            ],
        )
        for name in ("FULL", "ADAPTIVE_ONLY")
    }

    by_band = []
    labels = [
        "r == base",
        "base < r <= 0.015",
        "0.015 < r <= 0.020",
        "0.020 < r <= 0.030",
        "0.030 < r <= 0.040",
        "r > 0.040",
    ]
    for label in labels:
        targets = [r for r in target_inventory if r["radius_band"] == label]
        samples = [r for r in all_generated if r["radius_band"] == label]
        row = {
            "radius_band": label,
            "N_queries": len(targets),
            "N_draws": len(samples),
            "target_dates_represented": len({r["heldout_date"] for r in targets}),
            "source_dates_represented": len({r["source_date"] for r in samples}),
            "descriptive_only": True,
        }
        if samples:
            radii = [float(r["required_radius"]) for r in targets]
            row["mean_required_radius"] = float(np.mean(radii))
            row["median_required_radius"] = float(np.median(radii))
            row.update(metrics(authority, samples))
        by_band.append(row)

    summary = {
        "N_targets": len(target_inventory),
        "N_source_dates": len(dates),
        "legacy_supported_n": counts["LEGACY_SUPPORTED"],
        "adaptive_only_n": counts["ADAPTIVE_ONLY"],
        "identity": {
            "legacy_draws": identity_total,
            "legacy_exact_draws": identity_exact,
            "legacy_exact_fraction": identity_exact / identity_total,
            "all_draws_source_identity_fraction":
                source_identity_exact / source_identity_total,
            "all_draws_rng_identity_fraction":
                rng_identity_exact / rng_identity_total,
        },
        "gates": gates,
        "full_and_adaptive_only_pass": all(g["all_pass"] for g in gates.values()),
        "hard_gate_weighting": (
            "Original P2-0C-3B3: mean finite replicate metrics within heldout "
            "date, then equal-weight finite date macro"
        ),
        "per_date_hard_gates": False,
        "scope": SCOPE,
    }
    return target_inventory, by_date, by_band, summary


def witness_regression(v3, source):
    import numpy as np

    previous = read_json(PREVIOUS / "witness_v2_regression.json")
    q = float(previous["radius"]["representative_L"])
    old_model = v3.v2.AdaptiveRadiusOnlineBlock10EmulatorV2(source)
    new_model = v3.StateConditionedResidualTransportV3(source)
    requirement = new_model.radius_requirement(q)
    require(
        requirement == {
            k: previous["radius"][k] for k in requirement
        },
        "Witness radius exact vs accepted v2 predecessor",
    )

    a = np.random.default_rng(previous["diagnostic_seed"])
    b = np.random.default_rng(previous["diagnostic_seed"])
    old = old_model.sample_one(q, a)
    new = new_model.sample_one(q, b)
    assert_finite_sample(new)
    require(
        int(old["source_index"]) == int(new["source_index"])
        and str(old["source_sample_id"]) == str(new["source_sample_id"])
        and a.bit_generator.state == b.bit_generator.state,
        "Witness source/RNG identity",
    )
    require(new.get("v3_applied") is True, "Witness uses v3 adaptive correction")
    for sample in (old, new):
        sample["source_sample_id"] = str(sample["source_sample_id"])
    return {
        "pass": True,
        "query_L": q,
        "radius": requirement,
        "diagnostic_seed": int(previous["diagnostic_seed"]),
        "v2_sample": old,
        "v3_sample": new,
        "role": "Integration proof only; no PPO/performance claim",
    }


def final_integrity(head, sources, blobs):
    require(
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == "",
        "HEAD and working tree unchanged",
    )
    require(all(sha(ROOT / n) == h for n, h in sources.items()),
            "Published sources unchanged")
    require(
        all(
            git("rev-parse", f"{head}:{n}")
            == git("rev-parse", ":" + n)
            == git("hash-object", "--path=" + n, n)
            == b
            for n, b in blobs.items()
        ),
        "Published source blobs unchanged",
    )


def publish_audit(passed, head, failed_gates, expected=None):
    existing = {p.name for p in OUTPUT.iterdir()}
    require(
        "audit_summary.json" not in existing
        and "output_hashes.json" not in existing,
        "No repeated publication",
    )
    if expected is not None:
        require(existing == expected, "Complete pre-audit output inventory")
    hashes = {name: sha(OUTPUT / name) for name in sorted(existing)}
    write_json(
        "output_hashes.json",
        {
            "hashes": hashes,
            "exclusions": {
                "output_hashes.json": "Pinned by final audit",
                "audit_summary.json": "Published last; no self-hash",
            },
        },
    )
    require(
        all(sha(OUTPUT / n) == h for n, h in hashes.items()),
        "Output hashes exact after readback",
    )
    hashes["output_hashes.json"] = sha(OUTPUT / "output_hashes.json")
    write_json(
        "audit_summary.json",
        {
            "stage": "P2-1E-3C-v1",
            "stage_pass": bool(passed),
            "HEAD": head,
            "scientific_status": (
                "STATE_CONDITIONED_RESIDUAL_V3_DEV_FIDELITY_PASS_FROZEN"
                if passed
                else "STATE_CONDITIONED_RESIDUAL_V3_DEV_FIDELITY_FAIL"
            ),
            "declaration": PASS_DECLARATION if passed else FAIL_DECLARATION,
            "failed_gates": failed_gates,
            "scope": SCOPE,
            "PPO_training_runs": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "formal_RL_access_count": 0,
            "protected_Paper1_roles_access_count": 0,
            "output_sha256": hashes,
            "audit_summary_published_last": True,
        },
    )


def run_formal():
    require(not STAGE.exists(), "Immutable stage: no overwrite, resume or retry")
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "New v3 sources must be committed unchanged",
    )
    head = git("rev-parse", "HEAD")
    previous_audit, previous_metrics, previous_identity, predecessor_prov = (
        previous_provenance(head)
    )

    sources = {}
    blobs = {}
    for name in (SELF, MODULE):
        sources[name] = sha(ROOT / name)
        blobs[name] = git("rev-parse", f"{head}:{name}")
    final_integrity(head, sources, blobs)

    import paper2_state_conditioned_perception_v3 as v3
    import paper2_adaptive_radius_perception_v2 as v2
    import run_paper2_stage0c3b3_source_weighting_audit_v1 as authority

    verify_static_contract(v3)
    require(
        v3.v2 is v2,
        "V3 uses the exact committed v2 candidate-geometry module",
    )
    f = v3.v2.frozen

    legacy = read_json(LEGACY / "audit_summary.json")
    require(
        f.LOCAL_RADIUS
        == authority.LOCAL_RADIUS
        == legacy["fixed_transport"]["local_radius_abs_L"]
        and f.MIN_LOCAL_SAMPLES == authority.LOCAL_MIN_SAMPLES == 20
        and f.MIN_LOCAL_DATES == authority.LOCAL_MIN_DATES == 3
        and f.KERNEL_BANDWIDTH
        == authority.KERNEL_BANDWIDTH
        == legacy["fixed_transport"]["kernel_bandwidth"]
        and f.BLOCK_MINUTES == authority.BLOCK_MINUTES == 10,
        "Frozen support/BLOCK10/bandwidth constants exact",
    )

    gate_rows = [
        r
        for r in read_csv(LEGACY / "selector_gate_comparison.csv")
        if r["selector"] == "BLOCK10"
    ]
    require(
        {r["metric"]: float(r["tolerance"]) for r in gate_rows}
        == authority.GATES
        and all(r["pass"] == "True" for r in gate_rows),
        "Original P2-0C-3B3 tolerance authority exact",
    )

    source_entries = [
        (raw, h)
        for raw, h in predecessor_prov["sha256"].items()
        if h == f.EXPECTED_HASHES["paper1_cqr"]
    ]
    require(len(source_entries) == 1, "Unique pinned DECISION_DEVELOPMENT source")
    raw_path, expected_source_hash = source_entries[0]
    p = Path(raw_path)
    source_path = p if p.is_absolute() else ROOT / p
    require(
        sha(source_path) == expected_source_hash,
        "Paper1 DECISION_DEVELOPMENT source exact",
    )
    source = f.load_paper1_cqr(source_path)
    require(
        len(source) == authority.EXPECTED_N == 1844
        and source["date"].nunique() == authority.EXPECTED_DATES == 12
        and set(source["role"]) == {authority.EXPECTED_ROLE},
        "Authorized source role/N/dates exact",
    )

    seed, seed_expression = seed_formula(authority)

    STAGE.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        write_json(
            "v3_source_provenance.json",
            {
                "HEAD": head,
                "predecessor_HEAD": CHECKPOINT,
                "predecessor_audit_sha256": PREVIOUS_AUDIT_SHA256,
                "module_sha256": sources,
                "module_git_blobs": blobs,
                "paper1_source": {
                    "path": str(source_path),
                    "sha256": expected_source_hash,
                    "role": authority.EXPECTED_ROLE,
                    "rows": len(source),
                    "dates": int(source["date"].nunique()),
                },
            },
        )
        write_json(
            "v3_protocol_manifest.json",
            {
                "stage": "P2-1E-3C-v1",
                "method": "Sampling-law-weighted local-linear residual transport",
                "terminal_development_attempt": True,
                "if_DEV_fail": (
                    "STOP empirical perception-extension branch; no v4/v5 "
                    "parameter search or tolerance relaxation"
                ),
                "if_DEV_pass": (
                    "Freeze v3 before any separate one-shot RANDOM_TEST confirmation"
                ),
                "source_role": "DECISION_DEVELOPMENT only",
                "population": "Complete 1844 rows / 12 dates LODO",
                "hard_gate_populations": ["FULL", "ADAPTIVE_ONLY"],
                "legacy_identity_hard_gate": "100% exact outputs on legacy-supported draws",
                "source_selection_hard_gate": "100% exact source row vs v2 on all draws",
                "rng_consumption_hard_gate": "100% exact RNG state vs v2 on all draws",
                "tolerance_authority": LEGACY_SOURCE,
                "tolerances": authority.GATES,
                "MC_REPS": authority.MC_REPS,
                "BASE_SEED": authority.BASE_SEED,
                "seed_expression": seed_expression,
                "selector": "BLOCK10 unchanged",
                "base_radius": f.LOCAL_RADIUS,
                "support_thresholds": {
                    "min_rows": f.MIN_LOCAL_SAMPLES,
                    "min_dates": f.MIN_LOCAL_DATES,
                },
                "kernel_bandwidth": f.KERNEL_BANDWIDTH,
                "block_minutes": f.BLOCK_MINUTES,
                "adaptive_radius": "minimum binary64 radius from accepted v2/3A geometry",
                "new_operation_only": (
                    "adaptive-only sampling-law-weighted unregularized local-linear "
                    "trend removal for dq50/dlower/dupper"
                ),
                "no_new_hyperparameters": True,
                "no_radius_cap": True,
                "no_ridge_or_slope_clipping": True,
                "no_fallback": True,
                "RANDOM_TEST_access": "PROHIBITED in 3C",
                "SEALED_DATES_access": "PROHIBITED in 3C",
                "PPO_training_runs": 0,
                "scope": SCOPE,
            },
        )

        inventory, by_date, by_band, result = development_population(
            authority, v2, v3, source, seed
        )
        witness = witness_regression(v3, source)

        write_json("v3_dev_fidelity_metrics.json", result)
        write_csv("v3_dev_fidelity_by_date.csv", by_date)
        write_csv("v3_dev_fidelity_by_radius_band.csv", by_band)
        write_csv("v3_target_inventory.csv", inventory)
        write_json("witness_v3_regression.json", witness)

        failed = []
        for stratum, gate in result["gates"].items():
            for metric in authority.GATES:
                if not gate[metric]["pass"]:
                    failed.append(f"DEV {stratum} {metric}")
        if result["identity"]["legacy_exact_fraction"] != 1.0:
            failed.append("DEV LEGACY exact identity")
        if result["identity"]["all_draws_source_identity_fraction"] != 1.0:
            failed.append("DEV all-draw source identity")
        if result["identity"]["all_draws_rng_identity_fraction"] != 1.0:
            failed.append("DEV all-draw RNG identity")

        require(
            sha(source_path) == expected_source_hash,
            "Paper1 DECISION_DEVELOPMENT source unchanged after validation",
        )
        require(
            sha(PREVIOUS / "audit_summary.json") == PREVIOUS_AUDIT_SHA256,
            "P2-1E-3B predecessor evidence unchanged after validation",
        )
        final_integrity(head, sources, blobs)
        expected = {
            "v3_source_provenance.json",
            "v3_protocol_manifest.json",
            "v3_dev_fidelity_metrics.json",
            "v3_dev_fidelity_by_date.csv",
            "v3_dev_fidelity_by_radius_band.csv",
            "v3_target_inventory.csv",
            "witness_v3_regression.json",
        }
        publish_audit(not failed, head, failed, expected)
    except Exception as error:
        if (
            not (OUTPUT / "audit_summary.json").exists()
            and not (OUTPUT / "output_hashes.json").exists()
        ):
            write_json(
                "execution_failure.json",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "action": (
                        "STOP; EMPIRICAL PERCEPTION-EXTENSION BRANCH REMAINS "
                        "BLOCKED; NO RETRY/TUNING/V4/V5"
                    ),
                },
            )
            publish_audit(False, head, [str(error)])
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    run_formal()


if __name__ == "__main__":
    main()
