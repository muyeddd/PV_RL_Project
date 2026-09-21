
"""Shielded Point/UA PPO compatibility runner.

This module composes the frozen perception-aware PPO runner with the audited
q50-only set-membership predictive shield.

Contract:
- Point/UA observations stay unchanged.
- Shield input is only float32 q50 history plus executed-action history.
- No current latent L, UA width, rain flag, future weather, reward shaping,
  PPO hyperparameter, dynamics, or cleaning-mechanics change.
- PPO action is a proposal; on nonterminal days the shield maps it to the
  executed action before the frozen core settles reward/physics.
- Day 364 is reward-only and passes the proposal unchanged.
- Import is inert.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np

import paper2_perception_ppo_runner_v1 as base
import paper2_q50_set_membership_shield_v1 as shieldlib


get_protocol = base.get_protocol
candidate = base.candidate
partition = base.partition
derive_frozen_rl_seeds = base.derive_frozen_rl_seeds
child_seeds = base.child_seeds
build_training_schedule = base.build_training_schedule
verify_model_contract = base.verify_model_contract
parameters_finite = base.parameters_finite
train_final_checkpoint = base.train_final_checkpoint
reload_final_checkpoint = base.reload_final_checkpoint
file_sha256 = base.file_sha256
model_manifest = base.model_manifest
ModelBundle = base.ModelBundle
load_assets = base.load_assets
require = base.require
get_perception_contract = base.get_perception_contract
inherited_config_id = base.inherited_config_id
AccessLedger = base.AccessLedger
audit_perception_observation = base.audit_perception_observation
core = base.core
true_runner = base.true_runner
DIAGNOSTIC_SMOKE_BUDGET = base.DIAGNOSTIC_SMOKE_BUDGET

ROOT = Path(__file__).resolve().parents[1]
KERNEL_AUTHORITY = ROOT / "outputs/paper2_uncertainty_rl_v1/p2_1f_0_support_invariant_kernel_feasibility_audit_v1/formal"
SHIELD_AUTHORITY = ROOT / "outputs/paper2_uncertainty_rl_v1/p2_1f_1a_q50_set_membership_shield_feasibility_audit_v1/formal"

EXPECTED_KERNEL = {
    "kernel_upper": 0.21684390976200002,
    "wait_safe_max": 0.18001929645470102,
    "clean_safe_max": 0.21684390976200002,
}
EXPECTED_FEASIBILITY = {
    "episodes": 1200,
    "steps": 438000,
    "safety_filtered_decisions": 436800,
    "interventions": 82727,
    "free_waits": 355273,
    "intervention_fraction": 0.18939331501831502,
    "free_wait_fraction": 0.8111255707762557,
}


@dataclass(frozen=True)
class ShieldAuthority:
    kernel_summary: dict
    feasibility_audit: dict
    feasibility_summary: dict


@lru_cache(maxsize=1)
def read_shield_authority():
    kernel_audit = json.loads((KERNEL_AUTHORITY / "audit_summary.json").read_bytes())
    kernel = json.loads((KERNEL_AUTHORITY / "invariant_kernel_summary.json").read_bytes())
    require(
        kernel_audit["stage"] == "P2-1F-0-v1"
        and kernel_audit["stage_pass"] is True
        and kernel_audit["support_invariant_feasibility_pass"] is True
        and kernel_audit["scientific_status"] == "SUPPORT_INVARIANT_KERNEL_FEASIBILITY_PASS",
        "Accepted P2-1F-0 kernel authority",
    )
    for key, expected in EXPECTED_KERNEL.items():
        require(kernel[key] == expected, "Frozen kernel regression: " + key)
    for name, digest in kernel_audit["output_sha256"].items():
        require(file_sha256(KERNEL_AUTHORITY / name) == digest, "P2-1F-0 output hash: " + name)

    audit = json.loads((SHIELD_AUTHORITY / "audit_summary.json").read_bytes())
    summary = json.loads((SHIELD_AUTHORITY / "shield_feasibility_summary.json").read_bytes())
    primary = json.loads((SHIELD_AUTHORITY / "primary_population_summary.json").read_bytes())
    require(
        audit["stage"] == "P2-1F-1A-v1"
        and audit["stage_pass"] is True
        and audit["support_compatible_shield_feasibility_pass"] is True
        and audit["scientific_status"] == "Q50_SET_MEMBERSHIP_SHIELD_FEASIBILITY_PASS"
        and audit["failed_gates"] == []
        and audit["PPO_training_runs"] == 0
        and audit["formal_RL_access_count"] == 0
        and audit["formal_perception_seed_access_count"] == 0
        and audit["RANDOM_TEST_access_count"] == 0
        and audit["SEALED_DATES_access_count"] == 0,
        "Accepted P2-1F-1A shield authority",
    )
    require(
        summary["support_compatible_shield_feasibility_pass"] is True
        and summary["failed_gates"] == []
        and all(summary["gates"].values()),
        "All P2-1F-1A structural gates passed",
    )
    for key, expected in EXPECTED_FEASIBILITY.items():
        require(primary[key] == expected, "Frozen shield feasibility regression: " + key)
    for name, digest in audit["output_sha256"].items():
        require(file_sha256(SHIELD_AUTHORITY / name) == digest, "P2-1F-1A output hash: " + name)
    return ShieldAuthority(kernel, audit, summary)


def build_q50_shield(assets):
    authority = read_shield_authority()
    assets.ensure_perception()
    frozen = assets.perception
    require(
        frozen.LOCAL_RADIUS == 0.01
        and frozen.MIN_LOCAL_SAMPLES == 20
        and frozen.MIN_LOCAL_DATES == 3
        and frozen.KERNEL_BANDWIDTH == 0.005,
        "Frozen v1 perception geometry",
    )
    kernel_upper = authority.kernel_summary["kernel_upper"]
    inverse = shieldlib.Q50ObservationInverse(
        assets.emulator, frozen.LOCAL_RADIUS, kernel_upper
    )
    transition = shieldlib.FrozenRobustTransitionEnvelope(
        assets.dry,
        assets.r3["intercept"],
        assets.r3["slope"],
        assets.r3["residuals"],
        1.0 - assets.eta,
        kernel_upper,
    )
    return shieldlib.Q50SetMembershipShield(inverse, transition)


class Q50ShieldedPerceptionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, inner, assets, observation_mode):
        super().__init__()
        require(observation_mode in ("Point", "UA"), "Point/UA shield only")
        require(
            inner.observation_mode == observation_mode
            and inner.observation_space.shape
            == (dict(get_protocol().observation_dimensions)[observation_mode],)
            and inner.action_space.n == len(get_protocol().actions)
            and inner.action_space.start == 0,
            "Frozen inner observation/action contract",
        )
        self.inner = inner
        self.assets = assets
        self.observation_mode = observation_mode
        self.observation_space = inner.observation_space
        self.action_space = inner.action_space
        self.shield = build_q50_shield(assets)

        self._episode_live = False
        self._decision_index = 0
        self._last_shield_decision = None
        self._episode_records = []

        self.total_shield_interventions = 0
        self.total_safety_filtered_decisions = 0
        self.total_proposed_clean = 0
        self.total_executed_clean = 0
        self.total_free_wait = 0
        self.total_terminal_passthrough = 0

    @property
    def episode_records(self):
        return self._episode_records

    @property
    def last_shield_decision(self):
        return None if self._last_shield_decision is None else dict(self._last_shield_decision)

    def _episode_identity(self):
        core_env = self.inner.current if hasattr(self.inner, "current") else self.inner
        spec = core_env.episode_spec
        return {
            "year": spec.year,
            "environment_root": int(spec.environment_root),
            "trajectory_id": int(spec.trajectory_id),
            "population_size": int(spec.population_size),
            "perception_seed": int(spec.perception_seed),
        }

    def reset(self, *, seed=None, options=None):
        require(not self._episode_live, "Cannot reset live shielded episode")
        super().reset(seed=seed)
        obs, info = self.inner.reset(seed=seed, options=options)
        self.shield.reset()
        belief = self.shield.observe(np.float32(obs[0]))
        require(
            0.0 <= belief["belief_lower"] <= belief["belief_upper"] <= self.shield.kernel_upper,
            "Reset q50 belief inside frozen kernel",
        )
        self._decision_index = 0
        self._episode_live = True
        self._last_shield_decision = None
        self._episode_records.append(
            {
                **self._episode_identity(),
                "steps": 0,
                "safety_filtered_decisions": 0,
                "proposed_clean_count": 0,
                "executed_clean_count": 0,
                "interventions": 0,
                "free_wait_count": 0,
                "terminal_passthrough_count": 0,
                "completed": False,
            }
        )
        return obs, info

    def step(self, proposed_action):
        require(self._episode_live, "Live shielded episode required")
        if (
            not isinstance(proposed_action, (int, np.integer))
            or isinstance(proposed_action, (bool, np.bool_))
            or int(proposed_action) not in (0, 1)
        ):
            raise ValueError("Action must be integer 0 (WAIT) or 1 (CLEAN)")
        proposed = int(proposed_action)
        terminal_decision = self._decision_index == get_protocol().decision_reward_steps - 1

        if terminal_decision:
            executed = proposed
            intervention = False
            proposed_safe = None
            clean_safe = None
            proposed_next_upper = None
            executed_next_upper = None
        else:
            decision = self.shield.filter_action(proposed)
            executed = int(decision.executed_action)
            intervention = bool(decision.intervention)
            proposed_safe = bool(decision.proposed_safe)
            clean_safe = bool(decision.clean_safe)
            proposed_next_upper = float(decision.proposed_next_upper)
            executed_next_upper = float(decision.executed_next_upper)
            require(
                (executed == proposed) == proposed_safe
                and clean_safe
                and executed_next_upper <= self.shield.kernel_upper,
                "Minimal robust shield semantics",
            )
            self.shield.predict_after_action(executed)

        obs, reward, terminated, truncated, info = self.inner.step(executed)

        if not terminal_decision:
            belief = self.shield.observe(np.float32(obs[0]))
            require(
                0.0 <= belief["belief_lower"] <= belief["belief_upper"] <= self.shield.kernel_upper,
                "Next q50 belief remains in frozen kernel",
            )

        row = self._episode_records[-1]
        row["steps"] += 1
        row["proposed_clean_count"] += int(proposed == 1)
        row["executed_clean_count"] += int(executed == 1)
        row["interventions"] += int(intervention)
        row["free_wait_count"] += int(proposed == 0 and executed == 0)
        row["terminal_passthrough_count"] += int(terminal_decision)
        if not terminal_decision:
            row["safety_filtered_decisions"] += 1

        self.total_proposed_clean += int(proposed == 1)
        self.total_executed_clean += int(executed == 1)
        self.total_shield_interventions += int(intervention)
        self.total_free_wait += int(proposed == 0 and executed == 0)
        self.total_terminal_passthrough += int(terminal_decision)
        self.total_safety_filtered_decisions += int(not terminal_decision)

        self._last_shield_decision = {
            "day_index": int(self._decision_index),
            "proposed_action": proposed,
            "executed_action": executed,
            "intervention": intervention,
            "safety_constraint_applicable": not terminal_decision,
            "proposed_safe": proposed_safe,
            "clean_safe": clean_safe,
            "proposed_next_upper": proposed_next_upper,
            "executed_next_upper": executed_next_upper,
        }

        self._decision_index += 1
        require(
            not truncated
            and terminated == (self._decision_index == get_protocol().decision_reward_steps),
            "Exact frozen 365-day horizon",
        )
        if terminated:
            require(
                row["steps"] == get_protocol().decision_reward_steps
                and row["safety_filtered_decisions"] == get_protocol().natural_transitions
                and row["terminal_passthrough_count"] == 1,
                "Complete shielded episode accounting",
            )
            row["completed"] = True
            self._episode_live = False
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _shield_audit_snapshot(self):
        return {
            "observation_mode": self.observation_mode,
            "decision_index": int(self._decision_index),
            "episode_live": bool(self._episode_live),
            "last_shield_decision": self.last_shield_decision,
            "total_shield_interventions": int(self.total_shield_interventions),
            "total_safety_filtered_decisions": int(self.total_safety_filtered_decisions),
            "total_proposed_clean": int(self.total_proposed_clean),
            "total_executed_clean": int(self.total_executed_clean),
            "total_free_wait": int(self.total_free_wait),
            "total_terminal_passthrough": int(self.total_terminal_passthrough),
            "episode_records": [dict(r) for r in self._episode_records],
        }

    def close(self):
        self.inner.close()


class ScheduledShieldedPerceptionEnv(Q50ShieldedPerceptionEnv):
    def __init__(self, assets, schedule, ledger, observation_mode):
        self._scheduled_inner = base.ScheduledPerceptionEnv(
            assets, schedule, ledger, observation_mode
        )
        super().__init__(self._scheduled_inner, assets, observation_mode)

    @property
    def schedule(self):
        return self._scheduled_inner.schedule

    @property
    def ledger(self):
        return self._scheduled_inner.ledger

    @property
    def assignments(self):
        return self._scheduled_inner.assignments

    @property
    def started(self):
        return self._scheduled_inner.started

    @property
    def done(self):
        return self._scheduled_inner.done

    @property
    def failed(self):
        return self._scheduled_inner.failed

    @property
    def total_steps(self):
        return self._scheduled_inner.total_steps

    @property
    def observations_checked(self):
        return self._scheduled_inner.observations_checked

    @property
    def episodes_completed(self):
        return self._scheduled_inner.episodes_completed

    @property
    def perception_query_count(self):
        return self._scheduled_inner.perception_query_count


def build_shielded_scheduled_perception_env(
    assets, rl_seed, ledger, config_id, observation_mode
):
    require(config_id == inherited_config_id(), "Frozen config only")
    return ScheduledShieldedPerceptionEnv(
        assets,
        build_training_schedule(rl_seed, config_id),
        ledger,
        observation_mode,
    )


def build_shielded_perception_eval_env(
    assets, year, trajectory_id, ledger, observation_mode
):
    inner = base.build_perception_eval_env(
        assets, year, trajectory_id, ledger, observation_mode
    )
    return Q50ShieldedPerceptionEnv(inner, assets, observation_mode)


def build_shielded_perception_ppo_model(env, config_id, rl_seed):
    import torch
    from stable_baselines3 import PPO

    p, c = get_protocol(), candidate(config_id)
    require(
        type(env) is ScheduledShieldedPerceptionEnv
        and env.schedule.parent_seed == rl_seed
        and not env.started
        and not env.failed
        and config_id == inherited_config_id(),
        "Fresh matching shielded perception environment",
    )
    common, architecture = dict(p.common), dict(p.architecture)
    require(common.pop("algorithm") == "PPO" and common.pop("n_envs") == 1, "Frozen PPO/single env")
    require(common.pop("learning_rate_schedule") == common.pop("clip_schedule") == "NONE", "Constant schedules")
    policy = common.pop("policy")
    require(not p.observation_normalization and not p.reward_normalization, "Normalization prohibited")
    require(architecture["activation"] == "Tanh", "Frozen activation")
    learner = child_seeds(rl_seed)["learner"]
    torch.set_num_threads(p.torch_num_threads)
    random.seed(learner)
    np.random.seed(learner)
    torch.manual_seed(learner)
    model = PPO(policy=policy, env=env, learning_rate=c.learning_rate, n_steps=c.n_steps,
                policy_kwargs={"net_arch": {"pi": list(architecture["actor"]), "vf": list(architecture["critic"])},
                               "activation_fn": torch.nn.Tanh, "ortho_init": architecture["ortho_init"]},
                seed=learner, verbose=0, **common)
    require(model.device.type == p.device and model.n_envs == dict(p.common)["n_envs"], "CPU/single env")
    verify_model_contract(model, c.name, rl_seed)
    return ModelBundle(model=model, env=env, config_id=c.name, parent_seed=rl_seed)


def evaluate_shielded_perception_deterministic(
    model, assets, specs, ledger, observation_mode
):
    p = get_protocol()
    rows = []
    for year, tid in specs:
        env = build_shielded_perception_eval_env(
            assets, year, tid, ledger, observation_mode
        )
        try:
            obs, info = env.reset()
            audit_perception_observation(env.inner, obs, info)
            rewards, costs, states = [], [], []
            proposed_actions, executed_actions = [], []
            for index in range(p.decision_reward_steps):
                action_array, _ = model.predict(
                    obs, deterministic=p.evaluation.deterministic
                )
                array = np.asarray(action_array)
                require(
                    array.size == 1 and np.issubdtype(array.dtype, np.integer),
                    "Integer scalar model proposal",
                )
                proposed = int(array.item())
                require(env.action_space.contains(proposed), "Valid PPO proposal")
                obs, reward, terminated, truncated, info = env.step(proposed)
                snap = audit_perception_observation(
                    env.inner, obs, info, terminal=terminated
                )
                parts = snap["last_reward_components"]
                decision = env.last_shield_decision
                require(
                    decision is not None
                    and decision["proposed_action"] == proposed
                    and decision["executed_action"] == parts["action"],
                    "Proposal/executed-action audit identity",
                )
                rewards.append(float(reward))
                costs.append(float(parts["total_cost"]))
                states.append(float(parts["L_pre"]))
                proposed_actions.append(proposed)
                executed_actions.append(int(parts["action"]))
                require(
                    not truncated
                    and terminated == (index == p.decision_reward_steps - 1),
                    "Exact horizon",
                )

            episode_return, total_cost = sum(rewards), sum(costs)
            record = env._shield_audit_snapshot()["episode_records"][-1]
            require(
                episode_return == snap["episode_return"] == info["episode_return"]
                and abs(episode_return + total_cost) <= 1e-12,
                "Frozen return/cost sign",
            )
            require(
                sum(executed_actions)
                == snap["clean_count"]
                == info["clean_count"]
                == record["executed_clean_count"],
                "Executed CLEAN count identity",
            )
            require(
                sum(proposed_actions) == record["proposed_clean_count"]
                and record["steps"] == p.decision_reward_steps
                and record["safety_filtered_decisions"] == p.natural_transitions
                and record["terminal_passthrough_count"] == 1,
                "Proposal/shield accounting",
            )
            rows.append(
                {
                    "year": year,
                    "trajectory_id": tid,
                    "environment_root": env.inner.episode_spec.environment_root,
                    "step_count": len(proposed_actions),
                    "natural_transition_count": snap["transition_count"],
                    "J_total": total_cost,
                    "episode_return": episode_return,
                    "reward_sign_regression_diff": abs(episode_return + total_cost),
                    "N_clean": snap["clean_count"],
                    "N_clean_proposed": sum(proposed_actions),
                    "shield_interventions": record["interventions"],
                    "shield_intervention_fraction":
                        record["interventions"] / p.natural_transitions,
                    "free_wait_count": record["free_wait_count"],
                    "mean_L_pre": float(np.mean(states)),
                    "max_L_pre": max(states),
                    "proposed_action_min": min(proposed_actions),
                    "proposed_action_max": max(proposed_actions),
                    "executed_action_min": min(executed_actions),
                    "executed_action_max": max(executed_actions),
                    "all_observations_finite": True,
                    "all_rewards_finite": True,
                    "all_states_finite": True,
                    "deterministic_action": p.evaluation.deterministic,
                    "perception_query_count": snap["perception_query_count"],
                    "indices_sha256": snap["indices_sha256"],
                    "full_bank_sha256": snap["full_bank_sha256"],
                }
            )
        finally:
            env.close()
    return rows
