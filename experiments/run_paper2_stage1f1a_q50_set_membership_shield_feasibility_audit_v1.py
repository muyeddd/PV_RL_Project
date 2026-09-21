
"""P2-1F-1A-v1: q50-only set-membership predictive-shield feasibility audit.

No PPO is trained. No formal held-out trajectory or formal perception seed is
accessed. Frozen perception, dynamics, reward and the P2-1F-0 kernel are read
only.

Primary integration population
------------------------------
Every TRAINING and DEVELOPMENT episode specification (both years, every frozen
trajectory column) is run with an EXTERNAL proposed-WAIT trace. The q50-only
shield may execute CLEAN. This is a stress/integration audit, not a performance
baseline.

Hard gates
----------
* full 365-day completion for every audited episode;
* zero PerceptionSupportError / unsupported observation;
* nonempty q50-model-consistent belief at every decision;
* true latent state (audit-only) lies in the maintained belief at every decision;
* every belief lies inside the frozen P2-1F-0 kernel;
* every NONTERMINAL executed action is robustly kernel-safe against BOTH full
  frozen D0 and R3 disturbance supports;
* minimal intervention semantics on NONTERMINAL decisions: proposed WAIT passes
  iff certified safe, otherwise CLEAN; no discretionary override;
* day 364 is a reward-only terminal decision with no t->t+1 transition, so no
  next-state safety intervention is permitted there;
* at least one free WAIT and at least one shield intervention are exercised;
* Point/UA paired regression uses the same q50-only shield and yields identical
  shield decisions/hidden dynamics/rewards; UA width never enters shield logic;
* no PPO, formal roots/seeds, RANDOM_TEST or SEALED_DATES access.

Intervention rate and belief width are DESCRIPTIVE, not pass/fail thresholds.
If this stage passes, a separate reusable Gym wrapper may be built for PPO.
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

import numpy as np

import paper2_q50_set_membership_shield_v1 as shield


ROOT = Path(__file__).resolve().parents[1]
SELF = "experiments/run_paper2_stage1f1a_q50_set_membership_shield_feasibility_audit_v1.py"
MODULE = "experiments/paper2_q50_set_membership_shield_v1.py"
KERNEL_SOURCE = "experiments/run_paper2_stage1f0_support_invariant_kernel_feasibility_audit_v1.py"
CORE = "experiments/paper2_gym_pomdp_env_v1.py"
PERCEPTION_SOURCE = "experiments/run_paper2_stage1c2b_online_counterfactual_perception_audit_v1_1.py"
PERCEPTION_RUNNER = "experiments/paper2_perception_ppo_runner_v1.py"
TRUE_RUNNER = "experiments/paper2_true_state_ppo_runner_v1.py"

BASE = ROOT / "outputs/paper2_uncertainty_rl_v1"
KERNEL = BASE / "p2_1f_0_support_invariant_kernel_feasibility_audit_v1/formal"
STAGE = BASE / "p2_1f_1a_q50_set_membership_shield_feasibility_audit_v1"
OUTPUT = STAGE / "formal"

CHECKPOINT = "88a7310121166e18de10fa34abb41fe2b76dd014"
PINNED_BLOBS = {
    KERNEL_SOURCE: "d5fa596b865861ebcaa897d188e9f2cb993cbc9b",
    CORE: "5233c7d45cb24db3fd55c56ba658b4ae0b9cafb4",
    PERCEPTION_SOURCE: "6f940272ce1a262f325af310c9e72ed0adc401eb",
    PERCEPTION_RUNNER: "3de9802a5bc6f0f3c528cd0d0ddeeef95cff0c3d",
    TRUE_RUNNER: "637869eaef97cf52ccbb52a63db9984576c4ad3c",
}

PAIRING_TIDS = (0, 149, 299)
ACTION_PATTERN_TIDS = (0, 149, 299)
PRIMARY_PROPOSED_ACTION = 0  # WAIT


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=30
    ).stdout.strip()


def read_json(path):
    return json.loads(Path(path).read_bytes())


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
        (json.dumps(clean(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    )


def write_csv(name, rows):
    require(bool(rows), "Nonempty CSV: " + name)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(clean(rows))
    write_bytes(name, stream.getvalue().encode())


def verify_kernel_predecessor(head):
    audit = read_json(KERNEL / "audit_summary.json")
    summary = read_json(KERNEL / "invariant_kernel_summary.json")
    require(
        audit["HEAD"] == CHECKPOINT
        and audit["stage"] == "P2-1F-0-v1"
        and audit["stage_pass"] is True
        and audit["support_invariant_feasibility_pass"] is True
        and audit["scientific_status"] == "SUPPORT_INVARIANT_KERNEL_FEASIBILITY_PASS"
        and audit["PPO_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0
        and summary["support_invariant_feasibility_pass"] is True,
        "Accepted P2-1F-0 kernel evidence",
    )
    for name, expected in audit["output_sha256"].items():
        require(sha(KERNEL / name) == expected, "P2-1F-0 output hash: " + name)
    require(
        summary["kernel_upper"] == 0.21684390976200002
        and summary["wait_safe_max"] == 0.18001929645470102
        and summary["clean_safe_max"] == 0.21684390976200002
        and summary["static_action_freedom"]["both_actions_safe_measure_fraction"]
            == 0.8301791673664418
        and summary["static_action_freedom"]["clean_only_measure_fraction"]
            == 0.16982083263355818,
        "Exact frozen P2-1F-0 kernel/action geometry",
    )

    git("merge-base", "--is-ancestor", CHECKPOINT, head)
    for name, blob in PINNED_BLOBS.items():
        require(
            git("rev-parse", f"{CHECKPOINT}:{name}") == blob
            and git("rev-parse", f"{head}:{name}") == blob,
            "Frozen source blob unchanged: " + name,
        )
    return audit, summary


def static_contract():
    tree = ast.parse((ROOT / MODULE).read_text(encoding="utf-8"))
    classes = {
        n.name: n for n in tree.body if isinstance(n, ast.ClassDef)
    }
    target = classes["Q50SetMembershipShield"]
    methods = {
        n.name: n for n in target.body if isinstance(n, ast.FunctionDef)
    }
    required = {
        "observe": ("self", "q50_float32"),
        "filter_action": ("self", "proposed_action"),
        "predict_after_action": ("self", "executed_action"),
    }
    for name, args in required.items():
        actual = tuple(a.arg for a in methods[name].args.args)
        require(actual == args, "Shield public signature: " + name)

    # Current latent, interval width and weather flag cannot be public shield inputs.
    forbidden_args = {"latent", "L_true", "true_L", "width", "rain", "rain_flag", "date"}
    for name in required:
        args = {a.arg for a in methods[name].args.args}
        require(not (args & forbidden_args), "No hidden/UA-only/future input: " + name)

    module_text = (ROOT / MODULE).read_text(encoding="utf-8")
    require(
        "paper2_state_conditioned_perception_v3" not in module_text
        and "paper2_adaptive_radius_perception_v2" not in module_text,
        "No rejected perception-extension dependency",
    )
    random_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr in {"default_rng", "RandomState"}
    ]
    require(not random_nodes, "Shield introduces no stochastic RNG")
    return {
        "q50_only_public_observation_input": True,
        "UA_width_not_used": True,
        "current_latent_not_used": True,
        "future_weather_not_used": True,
        "no_rejected_perception_extension": True,
        "no_new_rng": True,
    }


def build_shield(assets, kernel_summary):
    f = assets.perception
    require(
        f.LOCAL_RADIUS == 0.01
        and f.MIN_LOCAL_SAMPLES == 20
        and f.MIN_LOCAL_DATES == 3
        and f.KERNEL_BANDWIDTH == 0.005,
        "Frozen perception geometry constants",
    )
    inverse = shield.Q50ObservationInverse(
        assets.emulator,
        f.LOCAL_RADIUS,
        kernel_summary["kernel_upper"],
    )
    transition = shield.FrozenRobustTransitionEnvelope(
        assets.dry,
        assets.r3["intercept"],
        assets.r3["slope"],
        assets.r3["residuals"],
        1.0 - assets.eta,
        kernel_summary["kernel_upper"],
    )
    return shield.Q50SetMembershipShield(inverse, transition)


def transition_regression(assets, kernel_summary):
    """Analytic envelope must agree with frozen functions at deterministic states."""
    import run_paper2_stage1f0_support_invariant_kernel_feasibility_audit_v1 as k0

    s = build_shield(assets, kernel_summary)
    states = sorted(set(
        [0.0, kernel_summary["wait_safe_max"], kernel_summary["kernel_upper"]]
        + [float(x) for x in assets.emulator.L if 0.0 <= float(x) <= kernel_summary["kernel_upper"]]
    ))
    max_abs = 0.0
    checks = 0
    for state in states:
        for action, multiplier in ((0, 1.0), (1, 1.0 - assets.eta)):
            analytic = s.transition.next_bounds(state, state, action)
            frozen = k0.action_extrema(assets, state, multiplier)
            max_abs = max(
                max_abs,
                abs(analytic.lower - frozen["all_min"]),
                abs(analytic.upper - frozen["all_max"]),
                abs(analytic.dry_lower - frozen["dry_min"]),
                abs(analytic.dry_upper - frozen["dry_max"]),
                abs(analytic.rain_lower - frozen["rain_min"]),
                abs(analytic.rain_upper - frozen["rain_max"]),
            )
            checks += 1
    require(max_abs <= 1e-15, "Analytic/frozen transition envelope regression")
    return {"checks": checks, "max_abs_diff": max_abs, "tolerance": 1e-15}


def make_row(runner, role, year, root, tid):
    part = runner.partition(role)
    return {
        "year": year,
        "environment_root": int(root),
        "trajectory_id": int(tid),
        "population_size": len(part.trajectory_ids),
        "perception_seed": runner.get_perception_contract().development_seed,
    }


def run_episode(env, safety, proposed_action, trace_rows=None):
    """Run one complete episode.

    ``proposed_action`` may be a constant integer or a pure day->action callable.
    Safety filtering applies only to the 364 decisions that have a natural
    t->t+1 transition. The day-364 action still settles reward but cannot violate
    next-state perception support because the episode terminates.

    The shield decision is completed BEFORE private ``_audit_snapshot`` is read.
    Hence current true L cannot influence the shield path even accidentally.
    """
    obs, info = env.reset()
    safety.reset()
    max_belief_width = 0.0
    sum_belief_width = 0.0
    interventions = free_waits = clean_exec = proposed_clean_count = 0
    max_true_L = 0.0
    min_margin = math.inf
    decisions = safety_filtered_decisions = 0

    for day in range(365):
        proposed = (
            int(proposed_action(day))
            if callable(proposed_action)
            else int(proposed_action)
        )
        require(proposed in (0, 1), "External proposed action must be WAIT/CLEAN")

        # POLICY/SHIELD-visible path only.
        q = np.float32(obs[0])
        belief = safety.observe(q)
        terminal = day == 364
        if terminal:
            executed_action = proposed
            intervention = False
            proposed_safe = None
            clean_safe = None
            executed_next_upper = None
        else:
            decision = safety.filter_action(proposed)
            require(decision.clean_safe, "CLEAN recovery always certified")
            require(
                (decision.executed_action == decision.proposed_action)
                == decision.proposed_safe,
                "Minimal intervention: pass iff proposed action certified",
            )
            require(
                decision.executed_next_upper <= safety.kernel_upper,
                "Executed action robustly kernel-safe",
            )
            executed_action = decision.executed_action
            intervention = decision.intervention
            proposed_safe = decision.proposed_safe
            clean_safe = decision.clean_safe
            executed_next_upper = decision.executed_next_upper
            safety_filtered_decisions += 1
            min_margin = min(
                min_margin,
                safety.kernel_upper - decision.executed_next_upper,
            )

        # PRIVATE AUDIT begins only after the decision above is frozen.
        snap = env._audit_snapshot()
        require(snap["day_index"] == day, "Exact decision day")
        true_L = float(snap["L_pre"])
        require(
            belief["belief_lower"] <= true_L <= belief["belief_upper"],
            "Audit-only true state contained in q50/history belief",
        )
        require(
            0.0 <= belief["belief_lower"] <= belief["belief_upper"] <= safety.kernel_upper,
            "Belief remains inside frozen kernel",
        )

        width = belief["belief_width"]
        max_belief_width = max(max_belief_width, width)
        sum_belief_width += width
        interventions += int(intervention)
        free_waits += int(proposed == 0 and executed_action == 0)
        proposed_clean_count += int(proposed == 1)
        clean_exec += int(executed_action == 1)
        max_true_L = max(max_true_L, true_L)
        decisions += 1

        if trace_rows is not None:
            trace_rows.append({
                "day_index": day,
                "date": snap["date"],
                "q50_float32": float(q),
                "true_L_audit_only": true_L,
                "belief_lower": belief["belief_lower"],
                "belief_upper": belief["belief_upper"],
                "belief_width": width,
                "compatible_source_rows": belief["compatible_source_rows"],
                "proposed_action": proposed,
                "safety_constraint_applicable": not terminal,
                "proposed_safe": proposed_safe,
                "executed_action": executed_action,
                "intervention": intervention,
                "clean_safe": clean_safe,
                "executed_next_upper_certificate": executed_next_upper,
                "kernel_upper": safety.kernel_upper,
            })

        if not terminal:
            safety.predict_after_action(executed_action)

        obs, reward, terminated, truncated, info = env.step(executed_action)
        require(
            np.isfinite(reward) and not truncated and terminated == terminal,
            "Finite reward/exact frozen horizon",
        )

    require(
        decisions == 365 and safety_filtered_decisions == 364,
        "365 reward decisions / 364 safety-constrained transitions",
    )
    require(math.isfinite(min_margin), "At least one nonterminal safety margin")
    return {
        "steps": decisions,
        "safety_filtered_decisions": safety_filtered_decisions,
        "interventions": interventions,
        "free_waits": free_waits,
        "proposed_clean_count": proposed_clean_count,
        "executed_clean_count": clean_exec,
        "intervention_fraction": interventions / safety_filtered_decisions,
        "free_wait_fraction": free_waits / decisions,
        "mean_belief_width": sum_belief_width / decisions,
        "max_belief_width": max_belief_width,
        "max_true_L_audit_only": max_true_L,
        "min_executed_safety_margin": min_margin,
    }


def primary_population(runner, assets, kernel_summary):
    ledger = runner.AccessLedger()
    rows = []
    witness_trace = None
    total_steps = total_interventions = total_free_waits = 0

    for role in ("TRAINING", "DEVELOPMENT"):
        part = runner.partition(role)
        for year, root in zip(part.years, part.roots):
            for tid in part.trajectory_ids:
                row = make_row(runner, role, year, root, tid)
                env = runner.construct_perception_core(
                    assets, ledger, role, row, "Point"
                )
                trace = []
                try:
                    stats = run_episode(
                        env,
                        build_shield(assets, kernel_summary),
                        PRIMARY_PROPOSED_ACTION,
                        trace,
                    )
                finally:
                    env.close()

                if witness_trace is None and stats["interventions"] > 0:
                    witness_trace = [
                        {
                            "role": role,
                            "year": year,
                            "trajectory_id": int(tid),
                            **r,
                        }
                        for r in trace
                    ]
                rows.append({
                    "role": role,
                    "year": year,
                    "environment_root": int(root),
                    "trajectory_id": int(tid),
                    **stats,
                })
                total_steps += stats["steps"]
                total_interventions += stats["interventions"]
                total_free_waits += stats["free_waits"]

    expected_episodes = sum(
        len(runner.partition(role).years) * len(runner.partition(role).trajectory_ids)
        for role in ("TRAINING", "DEVELOPMENT")
    )
    require(len(rows) == expected_episodes, "Complete TRAINING+DEVELOPMENT episode population")
    require(witness_trace is not None, "At least one shield intervention exercised")
    require(total_free_waits > 0, "At least one free WAIT exercised")
    require(
        ledger.heldout_ppo_trajectory_access_count == 0
        and ledger.formal_perception_seed_access_count == 0
        and ledger.denied_accesses == 0,
        "No formal root/seed access",
    )
    return rows, witness_trace, ledger



def action_pattern_regression(runner, assets, kernel_summary):
    """Exercise proposed-CLEAN and mixed-action branches without PPO."""
    rows = []
    patterns = {
        "ALWAYS_CLEAN": lambda day: 1,
        "ALTERNATING_WAIT_CLEAN": lambda day: day % 2,
    }
    part = runner.partition("DEVELOPMENT")
    roots = dict(zip(part.years, part.roots))
    for year in part.years:
        for tid in ACTION_PATTERN_TIDS:
            for pattern_name, pattern in patterns.items():
                row = make_row(
                    runner, "DEVELOPMENT", year, roots[year], int(tid)
                )
                ledger = runner.AccessLedger()
                env = runner.construct_perception_core(
                    assets, ledger, "DEVELOPMENT", row, "Point"
                )
                try:
                    stats = run_episode(
                        env,
                        build_shield(assets, kernel_summary),
                        pattern,
                    )
                finally:
                    env.close()
                require(
                    ledger.heldout_ppo_trajectory_access_count == 0
                    and ledger.formal_perception_seed_access_count == 0
                    and ledger.denied_accesses == 0,
                    "Action-pattern regression has no formal access",
                )
                if pattern_name == "ALWAYS_CLEAN":
                    require(
                        stats["proposed_clean_count"] == 365
                        and stats["interventions"] == 0
                        and stats["executed_clean_count"] == 365,
                        "Proposed CLEAN always passes unchanged",
                    )
                else:
                    require(
                        stats["proposed_clean_count"] > 0
                        and stats["free_waits"] > 0,
                        "Mixed proposed-action branches exercised",
                    )
                rows.append({
                    "year": year,
                    "trajectory_id": int(tid),
                    "pattern": pattern_name,
                    **stats,
                })
    require(len(rows) == 12, "Complete deterministic action-pattern audit")
    return rows


def run_pair(runner, assets, kernel_summary, year, tid):
    part = runner.partition("DEVELOPMENT")
    root = dict(zip(part.years, part.roots))[year]
    row = make_row(runner, "DEVELOPMENT", year, root, tid)
    ledger = runner.AccessLedger()

    point = runner.construct_perception_core(assets, ledger, "DEVELOPMENT", row, "Point")
    ua = runner.construct_perception_core(assets, ledger, "DEVELOPMENT", row, "UA")
    sp, su = build_shield(assets, kernel_summary), build_shield(assets, kernel_summary)

    try:
        po, pi = point.reset()
        uo, ui = ua.reset()
        sp.reset(); su.reset()
        interventions = 0
        for day in range(365):
            # Shield-visible path first: only common q50.
            require(
                np.array_equal(po, uo[[0, 2, 3]]),
                "Point/UA shared q50/season exact before shield",
            )
            require(np.isfinite(uo[1]), "UA width finite but not shield input")
            pb, ub = sp.observe(np.float32(po[0])), su.observe(np.float32(uo[0]))
            require(pb == ub, "Point/UA q50-only belief exact")

            terminal = day == 364
            if terminal:
                point_action = ua_action = 0
            else:
                pd, ud = sp.filter_action(0), su.filter_action(0)
                require(pd == ud, "Point/UA shield decision exact")
                interventions += int(pd.intervention)
                point_action, ua_action = pd.executed_action, ud.executed_action
                require(
                    sp.predict_after_action(point_action)
                    == su.predict_after_action(ua_action),
                    "Point/UA prediction exact",
                )

            # Private hidden-state pairing audit only after actions are fixed.
            psnap, usnap = point._audit_snapshot(), ua._audit_snapshot()
            require(
                psnap["L_pre"] == usnap["L_pre"]
                and psnap["date"] == usnap["date"]
                and psnap["day_index"] == usnap["day_index"] == day,
                "Point/UA hidden state/date exact",
            )

            pr = point.step(point_action)
            ur = ua.step(ua_action)
            require(
                pr[1:4] == ur[1:4]
                and pr[2] == terminal
                and not pr[3],
                "Point/UA reward/termination exact",
            )
            po, uo = pr[0], ur[0]

        return {
            "year": year,
            "trajectory_id": int(tid),
            "steps": 365,
            "safety_filtered_decisions": 364,
            "interventions": interventions,
            "point_ua_exact": True,
            "width_used_by_shield": False,
            "terminal_action_shielded": False,
        }, ledger
    finally:
        point.close()
        ua.close()


def pairing_population(runner, assets, kernel_summary):
    rows = []
    ledgers = []
    for year in runner.partition("DEVELOPMENT").years:
        for tid in PAIRING_TIDS:
            row, ledger = run_pair(runner, assets, kernel_summary, year, tid)
            rows.append(row)
            ledgers.append(ledger)
    require(all(r["point_ua_exact"] for r in rows), "All Point/UA pairings exact")
    require(
        all(
            l.heldout_ppo_trajectory_access_count == 0
            and l.formal_perception_seed_access_count == 0
            and l.denied_accesses == 0
            for l in ledgers
        ),
        "Pairing no formal access",
    )
    return rows


def final_integrity(head, sources, blobs):
    require(
        git("rev-parse", "HEAD") == head
        and git("status", "--porcelain", "--untracked-files=all") == "",
        "HEAD/tree unchanged",
    )
    require(all(sha(ROOT / p) == h for p, h in sources.items()), "New source hashes unchanged")
    require(
        all(
            git("rev-parse", f"{head}:{p}")
            == git("rev-parse", ":" + p)
            == git("hash-object", "--path=" + p, p)
            == blob
            for p, blob in blobs.items()
        ),
        "New source blobs unchanged",
    )


def run_formal():
    require(not STAGE.exists(), "Immutable stage; no overwrite/resume/retry")
    require(
        git("status", "--porcelain", "--untracked-files=all") == "",
        "Committed unchanged clean tree required",
    )
    head = git("rev-parse", "HEAD")
    kernel_audit, kernel_summary = verify_kernel_predecessor(head)

    sources = {p: sha(ROOT / p) for p in (SELF, MODULE)}
    blobs = {p: git("rev-parse", f"{head}:{p}") for p in (SELF, MODULE)}
    final_integrity(head, sources, blobs)

    import paper2_perception_ppo_runner_v1 as runner

    static = static_contract()
    assets = runner.load_assets()
    assets.ensure_perception()
    transition_check = transition_regression(assets, kernel_summary)

    STAGE.mkdir(parents=True, exist_ok=False)
    OUTPUT.mkdir(exist_ok=False)
    try:
        primary_rows, witness_trace, ledger = primary_population(
            runner, assets, kernel_summary
        )
        pattern_rows = action_pattern_regression(
            runner, assets, kernel_summary
        )
        pairing_rows = pairing_population(runner, assets, kernel_summary)

        total_steps = sum(r["steps"] for r in primary_rows)
        total_safety_filtered = sum(
            r["safety_filtered_decisions"] for r in primary_rows
        )
        total_interventions = sum(r["interventions"] for r in primary_rows)
        total_free_waits = sum(r["free_waits"] for r in primary_rows)
        primary_summary = {
            "episodes": len(primary_rows),
            "steps": total_steps,
            "proposed_policy": "WAIT_EVERY_DAY",
            "safety_filtered_decisions": total_safety_filtered,
            "interventions": total_interventions,
            "intervention_fraction": total_interventions / total_safety_filtered,
            "free_waits": total_free_waits,
            "free_wait_fraction": total_free_waits / total_steps,
            "max_episode_intervention_fraction":
                max(r["intervention_fraction"] for r in primary_rows),
            "max_belief_width":
                max(r["max_belief_width"] for r in primary_rows),
            "mean_episode_mean_belief_width":
                float(np.mean([r["mean_belief_width"] for r in primary_rows])),
            "min_executed_safety_margin":
                min(r["min_executed_safety_margin"] for r in primary_rows),
            "descriptive_only": {
                "intervention_fraction": True,
                "belief_width": True,
                "reason":
                    "No post-result tuning threshold is allowed; structural safety gates are primary",
            },
        }

        gates = {
            "complete_training_and_development_population":
                len(primary_rows) == 1200 and total_steps == 1200 * 365,
            "all_episodes_complete":
                all(
                    r["steps"] == 365
                    and r["safety_filtered_decisions"] == 364
                    for r in primary_rows
                ),
            "free_WAIT_branch_exercised": total_free_waits > 0,
            "intervention_branch_exercised": total_interventions > 0,
            "no_formal_root_or_seed_access":
                ledger.heldout_ppo_trajectory_access_count == 0
                and ledger.formal_perception_seed_access_count == 0
                and ledger.denied_accesses == 0,
            "proposed_CLEAN_and_mixed_action_branches":
                len(pattern_rows) == 12
                and all(r["steps"] == 365 for r in pattern_rows)
                and all(
                    r["interventions"] == 0
                    for r in pattern_rows
                    if r["pattern"] == "ALWAYS_CLEAN"
                ),
            "point_ua_pairing_exact":
                all(r["point_ua_exact"] for r in pairing_rows),
            "q50_only_static_contract":
                all(static.values()),
            "analytic_transition_regression":
                transition_check["max_abs_diff"] <= transition_check["tolerance"],
        }
        failed = [k for k, v in gates.items() if not v]
        feasibility = not failed

        write_json(
            "protocol_manifest.json",
            {
                "stage": "P2-1F-1A-v1",
                "role": "OBSERVATION-COMPATIBLE SHIELD FEASIBILITY; NO PPO",
                "kernel_upper": kernel_summary["kernel_upper"],
                "wait_safe_max_latent_reference": kernel_summary["wait_safe_max"],
                "policy_visible_Point": ["q50", "sin_DOY", "cos_DOY"],
                "policy_visible_UA": ["q50", "width", "sin_DOY", "cos_DOY"],
                "shield_visible": ["q50_float32_history", "executed_action_history"],
                "shield_forbidden": [
                    "current_true_L", "UA_width", "rain_flag",
                    "future_weather", "formal_roots", "formal_perception_seeds"
                ],
                "observer":
                    "Conservative interval-hull set membership from frozen q50 transport + frozen dynamics",
                "float32_semantics":
                    "Conservative inverse of the exact policy-visible float32 q50 rounding bin",
                "source_candidate_semantics":
                    "Exact binary64 candidate-entry/exit geometry under frozen radius=0.01",
                "transition_uncertainty":
                    "Both full frozen D0 and R3 disturbance supports at every nonterminal prediction",
                "terminal_day_semantics":
                    "Day 364 has reward settlement but no t->t+1 transition; proposed action passes unchanged",
                "primary_population":
                    "ALL TRAINING + DEVELOPMENT specs; external proposed WAIT every day",
                "branch_regression":
                    "DEVELOPMENT YEAR1/YEAR2 tids 0/149/299 under ALWAYS_CLEAN and ALTERNATING_WAIT_CLEAN",
                "hard_intervention_rate_threshold": None,
                "reason":
                    "Intervention rate is descriptive; no result-driven safety threshold/model tuning",
                "PPO_training_runs": 0,
                "RANDOM_TEST_access_count": 0,
                "SEALED_DATES_access_count": 0,
                "formal_RL_access_count": 0,
            },
        )
        write_json("static_contract.json", static)
        write_json("transition_envelope_regression.json", transition_check)
        write_csv("primary_episode_summary.csv", primary_rows)
        write_csv("first_intervention_witness_trace.csv", witness_trace)
        write_csv("action_pattern_regression.csv", pattern_rows)
        write_csv("point_ua_pairing_summary.csv", pairing_rows)
        write_json("primary_population_summary.json", primary_summary)
        write_json(
            "shield_feasibility_summary.json",
            {
                "support_compatible_shield_feasibility_pass": feasibility,
                "gates": gates,
                "failed_gates": failed,
                "primary": primary_summary,
                "pairings": pairing_rows,
                "claim_boundary":
                    "Simulator/model-consistent q50-history shield feasibility only; no real-world deterministic safety claim",
                "next_if_pass":
                    "Freeze reusable shielded Gym adapter before Point-PPO smoke",
            },
        )

        final_integrity(head, sources, blobs)
        expected = {
            "protocol_manifest.json",
            "static_contract.json",
            "transition_envelope_regression.json",
            "primary_episode_summary.csv",
            "first_intervention_witness_trace.csv",
            "action_pattern_regression.csv",
            "point_ua_pairing_summary.csv",
            "primary_population_summary.json",
            "shield_feasibility_summary.json",
        }
        require({p.name for p in OUTPUT.iterdir()} == expected, "Exact pre-audit outputs")
        hashes = {n: sha(OUTPUT / n) for n in sorted(expected)}
        write_json(
            "output_hashes.json",
            {"hashes": hashes,
             "exclusions": {
                 "output_hashes.json": "Pinned by final audit",
                 "audit_summary.json": "Published last"
             }}
        )
        hashes["output_hashes.json"] = sha(OUTPUT / "output_hashes.json")
        write_json(
            "audit_summary.json",
            {
                "stage": "P2-1F-1A-v1",
                "stage_pass": True,
                "stage_pass_meaning": "FEASIBILITY AUDIT EXECUTED SUCCESSFULLY",
                "support_compatible_shield_feasibility_pass": feasibility,
                "scientific_status": (
                    "Q50_SET_MEMBERSHIP_SHIELD_FEASIBILITY_PASS"
                    if feasibility
                    else "Q50_SET_MEMBERSHIP_SHIELD_FEASIBILITY_FAIL"
                ),
                "declaration": (
                    "Q50-ONLY HISTORY-CONSISTENT PREDICTIVE SHIELD COMPLETES ALL "
                    "TRAINING/DEVELOPMENT WAIT-STRESS EPISODES WITHOUT LEAVING THE "
                    "FROZEN PERCEPTION-VALIDITY KERNEL; POINT/UA SHIELD LOGIC IS "
                    "IDENTICAL; PPO REMAINS UNRUN"
                    if feasibility else
                    "Q50-ONLY OBSERVATION-COMPATIBLE SHIELD FEASIBILITY FAILED; "
                    "DO NOT START POINT/UA PPO OR TUNE PERCEPTION/SAFETY THRESHOLDS"
                ),
                "failed_gates": failed,
                "HEAD": head,
                "PPO_training_runs": 0,
                "formal_RL_access_count": 0,
                "formal_perception_seed_access_count": 0,
                "RANDOM_TEST_access_count": 0,
                "SEALED_DATES_access_count": 0,
                "output_sha256": hashes,
                "audit_summary_published_last": True,
            },
        )
        return read_json(OUTPUT / "audit_summary.json")
    except Exception as exc:
        if not (OUTPUT / "audit_summary.json").exists():
            write_json(
                "execution_failure.json",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "action":
                        "STOP; NO PPO; NO PERCEPTION EXTENSION; NO SAFETY-THRESHOLD TUNING",
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
