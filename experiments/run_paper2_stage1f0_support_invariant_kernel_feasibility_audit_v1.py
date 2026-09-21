
"""P2-1F-0-v1: frozen-perception support-invariant kernel feasibility audit.

This stage does NOT train PPO and does NOT alter/extend perception.

Question
--------
Does the maximal connected frozen-v1 perception-supported component containing
L=0 admit a robust controlled-invariant subset under the already-frozen
WAIT/CLEAN mechanics and D0/R3 natural dynamics, WITHOUT using rain/future
weather information?

Main safety domain
------------------
The starting domain is the exact P2-1E-2 origin-supported component. Because its
right endpoint is half-open in the accepted binary64 support map, the usable
upper bound is the greatest representable float strictly below that endpoint.

Robust transition semantics
---------------------------
For every state x and action, safety is checked against BOTH frozen transition
families:
  * D0 dry: clip(post + dry_innovation, 0, 1)
  * R3 rain: clip(post + intercept + slope*post + residual, 0, 1)
using ALL frozen innovations/residuals, not a quantile and not a date-aware
weather oracle. CLEAN uses the frozen multiplier (1-eta).

A scalar binary64 fixed-point iteration computes the greatest origin interval
[0,U*] that is robustly controlled invariant. At each U, exact frozen transition
functions are called inside a monotone binary64 search to find the largest state
for which WAIT and CLEAN respectively keep every possible next state in [0,U].

This is a LATENT-STATE FEASIBILITY CERTIFICATE ONLY. It proves whether a
support-invariant control architecture is possible in the frozen simulator. It
does NOT yet authorize a deployment shield, does NOT let Point/UA policies see
L_true, and does NOT claim that a CQR interval supplies a deterministic state
bound. Observation-compatible shielding must be audited separately before PPO.

Import is inert.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1f0_support_invariant_kernel_feasibility_audit_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
CLOSURE_SOURCE = "experiments/run_paper2_stage1e2_rl_perception_support_closure_audit_v1.py"
NATURAL_SOURCE = "experiments/run_paper2_stage1c1r_b_rain_repair_candidate_validation_v1.py"
PERCEPTION_SOURCE = "experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
CLOSURE = BASE / "p2_1e_2_rl_perception_support_closure_audit_v1/formal"
PREVIOUS_3C = BASE / "p2_1e_3c_state_conditioned_perception_v3_dev_freeze_v1/formal"
STAGE = BASE / "p2_1f_0_support_invariant_kernel_feasibility_audit_v1"
OUTPUT = STAGE / "formal"

CHECKPOINT = "f23ef4c871f144a427973eae2d7b38a17f16cbc7"
PINNED_BLOBS = {
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    CLOSURE_SOURCE: "8838f844e69d7420cca8257d4feaf5ac69547d59",
    NATURAL_SOURCE: "c1430b7b4ccac776433294489984b07af0c82c5f",
    PERCEPTION_SOURCE: "6f940272ce1a262f325af310c9e72ed0adc401eb",
}

MAX_FIXED_POINT_ITERATIONS = 2048


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


def write_bytes(name, data):
    path = OUTPUT / name
    with path.open("xb") as f:
        f.write(data)
    require(path.read_bytes() == data, "Exact output readback: " + name)


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    return value


def write_json(name, value):
    write_bytes(
        name,
        (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode(),
    )


def write_csv(name, rows):
    require(bool(rows), "Nonempty CSV: " + name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(clean(rows))
    write_bytes(name, stream.getvalue().encode())


def float_bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def bits_float(value):
    return struct.unpack(">d", struct.pack(">Q", int(value)))[0]


def parse_bool(value):
    value = str(value).strip().lower()
    require(value in ("true", "false"), "Boolean CSV field")
    return value == "true"


def verify_predecessors(head):
    require((CLOSURE / "audit_summary.json").exists(), "P2-1E-2 audit exists")
    closure_audit = read_json(CLOSURE / "audit_summary.json")
    require(
        closure_audit["stage"] == "P2-1E-2-v1"
        and closure_audit["stage_pass"] is True
        and closure_audit["support_closure_pass"] is False
        and closure_audit["scientific_status"] == "RL_PERCEPTION_SUPPORT_CLOSURE_FAIL_CONFIRMED"
        and closure_audit["witness_reproduction_pass"] is True
        and closure_audit["formal_trajectory_access_count"] == 0
        and closure_audit["formal_perception_seed_access_count"] == 0
        and closure_audit["stochastic_perception_draws"] == 0,
        "Accepted P2-1E-2 closure evidence",
    )
    for name, expected in closure_audit["output_sha256"].items():
        require(sha(CLOSURE / name) == expected, "P2-1E-2 output hash: " + name)

    require((PREVIOUS_3C / "audit_summary.json").exists(), "P2-1E-3C audit exists")
    require((PREVIOUS_3C / "execution_failure.json").exists(), "P2-1E-3C failure exists")
    v3 = read_json(PREVIOUS_3C / "audit_summary.json")
    failure = read_json(PREVIOUS_3C / "execution_failure.json")
    require(
        v3["stage"] == "P2-1E-3C-v1"
        and v3["stage_pass"] is False
        and v3["scientific_status"] == "STATE_CONDITIONED_RESIDUAL_V3_DEV_FIDELITY_FAIL"
        and v3["PPO_training_runs"] == 0
        and v3["RANDOM_TEST_access_count"] == 0
        and v3["SEALED_DATES_access_count"] == 0
        and "V3 transport broke q50 <= upper" in failure["message"],
        "Accepted terminal empirical-extension failure",
    )

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Frozen source blob unchanged: " + name,
        )
    return closure_audit, v3


def accepted_origin_component(closure_module, assets):
    summary = read_json(CLOSURE / "support_closure_summary.json")
    rows = read_csv(CLOSURE / "perception_support_intervals.csv")
    require(rows, "Accepted support map nonempty")

    first = rows[0]
    require(
        float(first["interval_left"]) == 0.0
        and parse_bool(first["supported"]) is True
        and parse_bool(first["left_closed"]) is True,
        "Origin belongs to first supported component",
    )
    boundary = float(first["interval_right"])
    right_closed = parse_bool(first["right_closed"])
    require(
        boundary == float(summary["highest_fully_supported_L_from_origin"]),
        "Origin component boundary exact vs accepted summary",
    )

    usable_upper = boundary if right_closed else math.nextafter(boundary, -math.inf)
    require(0.0 <= usable_upper < 1.0, "Finite usable origin upper bound")

    at_upper = closure_module.support_probe(assets.emulator, usable_upper)
    require(at_upper["supported"] is True, "Greatest usable origin state supported")
    if not right_closed:
        require(
            math.nextafter(usable_upper, math.inf) == boundary,
            "Usable endpoint is immediate predecessor of half-open boundary",
        )
        at_boundary = closure_module.support_probe(assets.emulator, boundary)
        require(at_boundary["supported"] is False, "Accepted half-open support boundary reproduced")
    else:
        at_boundary = at_upper

    return {
        "interval_left": 0.0,
        "accepted_interval_right": boundary,
        "accepted_right_closed": right_closed,
        "usable_binary64_upper": usable_upper,
        "usable_upper_hex": usable_upper.hex(),
        "support_probe_at_usable_upper": at_upper,
        "support_probe_at_boundary": at_boundary,
    }


def rain_model(assets):
    return {
        "candidate": assets.reward_protocol["environment_parameters"]["rain"],
        "intercept": float(assets.r3["intercept"]),
        "slope": float(assets.r3["slope"]),
        "residuals": assets.r3["residuals"],
    }


def action_extrema(assets, state, clean_multiplier):
    x = float(state)
    post = float(clean_multiplier) * x
    require(math.isfinite(post) and 0.0 <= post <= 1.0, "Finite post-action state")

    dry = np.asarray(assets.natural.dry_next_samples(post, assets.dry), dtype=float)
    rain = np.asarray(assets.natural.rain_next_samples(post, rain_model(assets)), dtype=float)
    require(
        len(dry) == len(assets.dry)
        and len(rain) == len(assets.r3["residuals"])
        and np.isfinite(dry).all()
        and np.isfinite(rain).all()
        and ((dry >= 0.0) & (dry <= 1.0)).all()
        and ((rain >= 0.0) & (rain <= 1.0)).all(),
        "Frozen transition outputs finite/bounded for all disturbances",
    )
    return {
        "post_state": post,
        "dry_min": float(dry.min()),
        "dry_max": float(dry.max()),
        "rain_min": float(rain.min()),
        "rain_max": float(rain.max()),
        "all_min": float(min(dry.min(), rain.min())),
        "all_max": float(max(dry.max(), rain.max())),
    }


def action_safe(assets, state, clean_multiplier, target_upper):
    ex = action_extrema(assets, state, clean_multiplier)
    return bool(ex["all_min"] >= 0.0 and ex["all_max"] <= float(target_upper)), ex


def largest_safe_binary64(assets, target_upper, clean_multiplier):
    upper = float(target_upper)
    require(math.isfinite(upper) and 0.0 <= upper < 1.0, "Binary search domain")

    zero_safe, zero_ex = action_safe(assets, 0.0, clean_multiplier, upper)
    if not zero_safe:
        return None, {"zero_extrema": zero_ex, "upper_extrema": None}

    upper_safe, upper_ex = action_safe(assets, upper, clean_multiplier, upper)
    if upper_safe:
        return upper, {"zero_extrema": zero_ex, "upper_extrema": upper_ex}

    low = float_bits(0.0)
    high = float_bits(upper)
    require(low < high, "Nontrivial safe-search range")
    while low + 1 < high:
        mid = (low + high) // 2
        value = bits_float(mid)
        ok, _ = action_safe(assets, value, clean_multiplier, upper)
        if ok:
            low = mid
        else:
            high = mid
    result = bits_float(low)
    result_ok, result_ex = action_safe(assets, result, clean_multiplier, upper)
    next_value = bits_float(high)
    next_ok, next_ex = action_safe(assets, next_value, clean_multiplier, upper)
    require(result_ok and not next_ok, "Exact last-safe binary64 certificate")
    return result, {
        "zero_extrema": zero_ex,
        "last_safe_extrema": result_ex,
        "first_unsafe_state": next_value,
        "first_unsafe_extrema": next_ex,
    }


def invariant_fixed_point(assets, origin_upper):
    clean_multiplier = float(1.0 - assets.eta)
    require(0.0 < clean_multiplier <= 1.0, "Frozen CLEAN multiplier")

    current = float(origin_upper)
    rows = []
    for iteration in range(MAX_FIXED_POINT_ITERATIONS):
        wait_max, wait_cert = largest_safe_binary64(assets, current, 1.0)
        clean_max, clean_cert = largest_safe_binary64(
            assets, current, clean_multiplier
        )
        candidates = [v for v in (wait_max, clean_max) if v is not None]
        next_upper = 0.0 if not candidates else min(current, max(candidates))

        rows.append(
            {
                "iteration": iteration,
                "target_upper": current,
                "wait_safe_max": "" if wait_max is None else wait_max,
                "clean_safe_max": "" if clean_max is None else clean_max,
                "controlled_predecessor_upper": next_upper,
                "fixed": next_upper == current,
            }
        )
        if next_upper == current:
            return {
                "kernel_upper": current,
                "kernel_upper_hex": current.hex(),
                "iterations": iteration + 1,
                "wait_safe_max": wait_max,
                "clean_safe_max": clean_max,
                "wait_certificate": wait_cert,
                "clean_certificate": clean_cert,
                "clean_multiplier": clean_multiplier,
                "rows": rows,
            }

        require(
            0.0 <= next_upper < current,
            "Invariant iteration must monotonically shrink",
        )
        current = next_upper

    raise RuntimeError("Invariant fixed point did not converge within deterministic limit")


def initial_states(assets):
    result = {}
    for year in ("YEAR1", "YEAR2"):
        calendar = assets.mechanics.get_year_calendar(assets.ledger, year)
        value = float(calendar["L_power_proxy"].iloc[0])
        require(
            bool(calendar["state_valid"].iloc[0])
            and math.isfinite(value)
            and 0.0 <= value <= 1.0,
            "Frozen historical initial state",
        )
        result[year] = value
    return result


def final_integrity(head, sources, blobs):
    require(
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == "",
        "HEAD/tree unchanged",
    )
    require(all(sha(ROOT / n) == h for n, h in sources.items()), "New source unchanged")
    require(
        all(
            git("rev-parse", f"{head}:{n}")
            == git("rev-parse", ":" + n)
            == git("hash-object", "--path=" + n, n)
            == b
            for n, b in blobs.items()
        ),
        "Published source blob unchanged",
    )


def publish_audit(head, feasibility, failed_gates, hashes):
    write_json(
        "audit_summary.json",
        {
            "stage": "P2-1F-0-v1",
            "stage_pass": True,
            "stage_pass_meaning": "FEASIBILITY AUDIT EXECUTED SUCCESSFULLY",
            "support_invariant_feasibility_pass": feasibility,
            "scientific_status": (
                "SUPPORT_INVARIANT_KERNEL_FEASIBILITY_PASS"
                if feasibility
                else "SUPPORT_INVARIANT_KERNEL_FEASIBILITY_FAIL"
            ),
            "declaration": (
                "A NONEMPTY ROBUST ORIGIN SUPPORT-INVARIANT LATENT-STATE KERNEL "
                "CONTAINING BOTH FROZEN INITIAL STATES EXISTS; OBSERVATION-COMPATIBLE "
                "SHIELDING REMAINS UNPROVEN AND PPO REMAINS BLOCKED"
                if feasibility
                else
                "NO ACCEPTABLE ROBUST ORIGIN SUPPORT-INVARIANT LATENT-STATE KERNEL "
                "WAS CERTIFIED; DO NOT START SHIELDED POINT/UA PPO"
            ),
            "failed_gates": failed_gates,
            "HEAD": head,
            "PPO_training_runs": 0,
            "formal_RL_access_count": 0,
            "formal_perception_seed_access_count": 0,
            "stochastic_perception_draws": 0,
            "RANDOM_TEST_access_count": 0,
            "SEALED_DATES_access_count": 0,
            "output_sha256": hashes,
            "audit_summary_published_last": True,
        },
    )


def run_formal():
    require(not STAGE.exists(), "Immutable stage; no overwrite/resume/retry")
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean tree required",
    )
    head = git("rev-parse", "HEAD")
    closure_audit, v3_audit = verify_predecessors(head)

    sources = {SELF: sha(ROOT / SELF)}
    blobs = {SELF: git("rev-parse", f"{head}:{SELF}")}
    require(
        blobs[SELF]
        == git("rev-parse", ":" + SELF)
        == git("hash-object", "--path=" + SELF, SELF),
        "Published audit source committed unchanged",
    )
    final_integrity(head, sources, blobs)

    import paper2_gym_pomdp_env_v1 as core
    import run_paper2_stage1e2_rl_perception_support_closure_audit_v1 as closure_module

    assets = core.Paper2EnvAssets()
    assets.ensure_perception()

    source_paths = [ROOT / NATURAL_SOURCE, ROOT / PERCEPTION_SOURCE]
    monotonicity = closure_module.monotonicity_audit(assets, source_paths)

    origin = accepted_origin_component(closure_module, assets)
    kernel = invariant_fixed_point(assets, origin["usable_binary64_upper"])
    initials = initial_states(assets)

    kernel_upper = float(kernel["kernel_upper"])
    feasibility_gates = {
        "nonempty_kernel": kernel_upper > 0.0,
        "kernel_within_accepted_origin_support":
            kernel_upper <= float(origin["usable_binary64_upper"]),
        "YEAR1_initial_inside_kernel": initials["YEAR1"] <= kernel_upper,
        "YEAR2_initial_inside_kernel": initials["YEAR2"] <= kernel_upper,
        "at_least_one_robust_action_everywhere":
            max(
                -1.0 if kernel["wait_safe_max"] is None else float(kernel["wait_safe_max"]),
                -1.0 if kernel["clean_safe_max"] is None else float(kernel["clean_safe_max"]),
            ) >= kernel_upper,
        "transition_class_blind": True,
        "monotonicity_reaudited": monotonicity["passed"] is True,
    }
    failed = [name for name, passed in feasibility_gates.items() if not passed]
    feasibility = not failed

    wait_max = None if kernel["wait_safe_max"] is None else float(kernel["wait_safe_max"])
    clean_max = None if kernel["clean_safe_max"] is None else float(kernel["clean_safe_max"])
    both_action_upper = (
        min(wait_max, clean_max)
        if wait_max is not None and clean_max is not None
        else None
    )

    static_freedom = {
        "kernel_length": kernel_upper,
        "wait_safe_max": wait_max,
        "clean_safe_max": clean_max,
        "both_actions_safe_upper": both_action_upper,
        "both_actions_safe_measure_fraction":
            None if kernel_upper == 0.0 or both_action_upper is None
            else float(both_action_upper / kernel_upper),
        "clean_only_measure_fraction":
            None if kernel_upper == 0.0 or wait_max is None or clean_max is None
            else float(
                max(0.0, min(kernel_upper, clean_max) - min(kernel_upper, wait_max))
                / kernel_upper
            ),
        "interpretation":
            "STATIC STATE-SPACE MEASURE ONLY; NOT A POLICY OCCUPANCY OR INTERVENTION RATE",
    }

    STAGE.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        write_json(
            "protocol_manifest.json",
            {
                "stage": "P2-1F-0-v1",
                "role": "LATENT-STATE SUPPORT-INVARIANT FEASIBILITY ONLY",
                "starting_domain":
                    "Maximal connected accepted frozen-v1 support component containing L=0",
                "transition_uncertainty":
                    "ALL frozen D0 innovations AND ALL frozen R3 residuals at every check",
                "weather_information":
                    "NONE; dry and rain families are both required safe regardless of calendar date",
                "actions": {
                    "WAIT": "post=L",
                    "CLEAN": "post=(1-eta)*L with frozen eta",
                },
                "kernel":
                    "Greatest origin interval fixed point under robust controlled predecessor",
                "numeric_search":
                    "Monotone binary64 last-safe search using actual frozen transition functions; no grid",
                "policy_information":
                    "NONE; no policy/model constructed, no L_true exposed to Point/UA",
                "observation_compatible_shield_authorized": False,
                "reason_not_authorized":
                    "CQR interval coverage is probabilistic, not yet a deterministic bound on latent L",
                "next_stage_if_pass":
                    "Observation-compatible predictive shield audit using frozen perception outputs; true L audit-only",
                "PPO_training_runs": 0,
                "formal_RL_access_count": 0,
                "formal_perception_seed_access_count": 0,
                "RANDOM_TEST_access_count": 0,
                "SEALED_DATES_access_count": 0,
                "predecessor_1E2_status": closure_audit["scientific_status"],
                "predecessor_3C_status": v3_audit["scientific_status"],
            },
        )
        write_json("origin_support_component.json", origin)
        write_csv("kernel_fixed_point_iterations.csv", kernel["rows"])
        write_json(
            "invariant_kernel_summary.json",
            {
                "support_invariant_feasibility_pass": feasibility,
                "feasibility_gates": feasibility_gates,
                "origin_support": origin,
                "kernel_upper": kernel_upper,
                "kernel_upper_hex": kernel["kernel_upper_hex"],
                "fixed_point_iterations": kernel["iterations"],
                "frozen_initial_states": initials,
                "wait_safe_max": wait_max,
                "clean_safe_max": clean_max,
                "wait_certificate": kernel["wait_certificate"],
                "clean_certificate": kernel["clean_certificate"],
                "static_action_freedom": static_freedom,
                "monotonicity": monotonicity,
                "claim_boundary":
                    "Latent-state simulator feasibility only; not yet a deployable or observation-only safety guarantee",
            },
        )
        write_json(
            "source_artifact_provenance.json",
            {
                "HEAD": head,
                "checkpoint": CHECKPOINT,
                "git_blobs": {**PINNED_BLOBS, SELF: blobs[SELF]},
                "source_sha256": {
                    SELF: sources[SELF],
                    CORE: sha(ROOT / CORE),
                    CLOSURE_SOURCE: sha(ROOT / CLOSURE_SOURCE),
                    NATURAL_SOURCE: sha(ROOT / NATURAL_SOURCE),
                    PERCEPTION_SOURCE: sha(ROOT / PERCEPTION_SOURCE),
                },
                "closure_output_sha256": closure_audit["output_sha256"],
                "core_assets_provenance": assets.provenance,
            },
        )

        final_integrity(head, sources, blobs)
        existing = {p.name for p in OUTPUT.iterdir()}
        expected = {
            "protocol_manifest.json",
            "origin_support_component.json",
            "kernel_fixed_point_iterations.csv",
            "invariant_kernel_summary.json",
            "source_artifact_provenance.json",
        }
        require(existing == expected, "Exact pre-audit output inventory")

        hashes = {name: sha(OUTPUT / name) for name in sorted(existing)}
        write_json(
            "output_hashes.json",
            {
                "hashes": hashes,
                "exclusions": {
                    "output_hashes.json": "Pinned by audit summary",
                    "audit_summary.json": "Published last",
                },
            },
        )
        hashes["output_hashes.json"] = sha(OUTPUT / "output_hashes.json")
        publish_audit(head, feasibility, failed, hashes)
        return read_json(OUTPUT / "audit_summary.json")
    except Exception as error:
        if not (OUTPUT / "audit_summary.json").exists():
            write_json(
                "execution_failure.json",
                {
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "action": "STOP; NO PPO; NO PERCEPTION EXTENSION; PRESERVE EVIDENCE",
                },
            )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("formal",))
    parser.parse_args()
    print(json.dumps(run_formal(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
