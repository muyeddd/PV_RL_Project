"""Point/UA compatibility bridge; no assets, trajectories or outputs at import.

Frozen True-State runner intentionally contains an exact ScheduledTrueStateEnv
guard, True-State-only construct_core and a no-perception audit. It is P2-2C
provenance evidence and is not generalized in place. Only mode-specific adapter,
observation audit, evaluation shell and constructor compatibility live here.
PPO training, optimizer hook, save/reload and contract verification stay frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np

import paper2_true_state_ppo_runner_v1 as true_runner


get_protocol = true_runner.get_protocol
candidate = true_runner.candidate
partition = true_runner.partition
derive_frozen_rl_seeds = true_runner.derive_frozen_rl_seeds
child_seeds = true_runner.child_seeds
build_training_schedule = true_runner.build_training_schedule
verify_model_contract = true_runner.verify_model_contract
parameters_finite = true_runner.parameters_finite
train_final_checkpoint = true_runner.train_final_checkpoint
reload_final_checkpoint = true_runner.reload_final_checkpoint
file_sha256 = true_runner.file_sha256
model_manifest = true_runner.model_manifest
ModelBundle = true_runner.ModelBundle
load_assets = true_runner.load_assets
require = true_runner.require
core = true_runner.core
DIAGNOSTIC_SMOKE_BUDGET = true_runner.DIAGNOSTIC_SMOKE_BUDGET
ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "outputs/paper2_uncertainty_rl_v1/p2_1e_1_gym_pomdp_environment_audit_v1_1/formal"
AUTHORITY_HASHES = {
    "audit_summary.json": "e1e65e746a965aa115f17c8a6a170c62061dd9ff0d56e8f1a64f4a16aa67dc88",
    "protocol_manifest.json": "4e0f1ee5e9c4fcc1387e10dcb223a2b57db920ace994c9b15cba8e1aaffda646",
    "output_hashes.json": "cbad9c06de056fa3d5874095ac296b31712a42109fb5a743c1ac6d9a65fb16af",
}
SELECTED_ARTIFACT = ROOT / "outputs/paper2_uncertainty_rl_v1/p2_2c_2_true_state_ppo_development_selection_v1/formal/selected_config.json"
SELECTED_SHA256 = "3cc4bdc4393936f736e659fbbe0ded37adb39ebb0871ccf1b5269fdefe4e9a67"


@dataclass(frozen=True)
class PerceptionContract:
    development_seed: int
    public_info_keys: tuple[str, ...]


def read_perception_authority():
    docs = {}
    for name, digest in AUTHORITY_HASHES.items():
        path = AUTHORITY / name
        require(file_sha256(path) == digest, f"Frozen P2-1E-1 authority: {name}")
        docs[name] = json.loads(path.read_bytes())
    audit, manifest = docs["audit_summary.json"], docs["protocol_manifest.json"]
    require(audit.get("stage_pass") is True and audit.get("failed_gates") == []
            and audit.get("scientific_status") == "GYM_POMDP_ENVIRONMENT_PASS_FROZEN"
            and all(v is True for v in audit["gates"].values()), "Accepted perception/environment contract")
    require(docs["output_hashes.json"]["hashes"]["protocol_manifest.json"] == AUTHORITY_HASHES["protocol_manifest.json"],
            "Perception authority publication identity")
    require(manifest["development_only"] is True and manifest["frozen_perception_support"]["fallback"] is False,
            "Development-only no-fallback contract")
    part = partition("DEVELOPMENT")
    require(manifest["environment_roots"] == dict(zip(part.years, part.roots)), "Frozen development roots agree")
    return manifest


@lru_cache(maxsize=1)
def get_perception_contract():
    manifest = read_perception_authority()
    registered = get_protocol().seeds.perception_seed
    # The accepted True-State protocol intentionally says None. Only this new
    # bridge fills that absence from the pinned Point/UA environment authority.
    seed = manifest["perception_seed"] if registered is None else registered
    require(type(seed) is int and seed == manifest["perception_seed"] == 20260906, "DEV perception seed authority")
    return PerceptionContract(seed, tuple(manifest["public_info_allowlist"]))


@lru_cache(maxsize=1)
def inherited_config_id():
    require(file_sha256(SELECTED_ARTIFACT) == SELECTED_SHA256, "Frozen selected config artifact")
    selected = json.loads(SELECTED_ARTIFACT.read_bytes())
    require(selected["declaration"] == "NO HYPERPARAMETER RESELECTION AFTER P2-2C-2", "No reselection")
    name = selected["selected_config"]
    require(name == "CONFIG_A", "Expected selected config regression")
    candidate(name)
    return name


@dataclass
class AccessLedger:
    """Authorize role and DEV seed BEFORE core construction; deny everything else."""
    records: list = field(default_factory=list)
    denied_accesses: int = 0

    def authorize(self, role, row, observation_mode):
        if role not in ("TRAINING", "DEVELOPMENT") or observation_mode not in ("Point", "UA"):
            self.denied_accesses += 1
            raise RuntimeError("Only TRAINING/DEVELOPMENT Point/UA construction allowed")
        part = partition(role)
        valid = (row["year"], row["environment_root"]) in tuple(zip(part.years, part.roots))
        valid = valid and type(row["trajectory_id"]) is int and row["trajectory_id"] in part.trajectory_ids
        valid = valid and row["population_size"] == len(part.trajectory_ids)
        valid = valid and type(row["perception_seed"]) is int and row["perception_seed"] == get_perception_contract().development_seed
        if not valid:
            self.denied_accesses += 1
            raise RuntimeError("Frozen role/year/root/tid/population/DEV perception seed required")
        self.records.append({**row, "role": role, "observation_mode": observation_mode})

    @property
    def heldout_ppo_trajectory_access_count(self):
        allowed = set(partition("TRAINING").roots) | set(partition("DEVELOPMENT").roots)
        return sum(r["environment_root"] not in allowed for r in self.records)

    @property
    def formal_perception_seed_access_count(self):
        return sum(r["perception_seed"] != get_perception_contract().development_seed for r in self.records)


def construct_perception_core(assets, ledger, role, row, observation_mode):
    ledger.authorize(role, row, observation_mode)
    spec = core.EpisodeSpec(year=row["year"], environment_root=row["environment_root"],
                            trajectory_id=row["trajectory_id"], population_size=row["population_size"],
                            perception_seed=row["perception_seed"])
    env = core.Paper2CleaningEnv(assets, spec, observation_mode=observation_mode)
    require(env.observation_space.shape == (dict(get_protocol().observation_dimensions)[observation_mode],), "Frozen observation shape")
    require(env.action_space.n == len(get_protocol().actions) and env.action_space.start == 0, "Frozen actions")
    return env


def audit_perception_observation(env, obs, info, *, terminal=False):
    """Private audit only: read cached q50/width, never query/sample a second time.

    Snapshot has no q50 field. Its exact (seed, year, tid, day, latent) key
    identifies the frozen core's existing cache record. No observation is
    altered, and neither this snapshot nor the cache is exposed to policy.
    reset queries once; each nonterminal step once; terminal zero never queries.
    Therefore query_count == transition_count + 1, including terminal state.
    """
    p, contract = get_protocol(), get_perception_contract()
    require(env.observation_mode in ("Point", "UA"), "Perception mode")
    require(obs.shape == env.observation_space.shape and obs.dtype == env.observation_space.dtype == np.dtype("float32")
            and np.isfinite(obs).all() and env.observation_space.contains(obs), "Finite exact observation space/dtype")
    require(set(info) <= set(contract.public_info_keys) and info["observation_mode"] == env.observation_mode,
            "Public info boundary: no hidden state, width, energy or future weather")
    snap = env._audit_snapshot()
    require(snap["perception_seed"] == contract.development_seed and not snap["needs_reset"], "DEV seed and successful observation")
    require(np.isfinite(snap["L_pre"]) and 0 <= snap["L_pre"] <= 1, "Finite private state")
    require(snap["perception_query_count"] == snap["transition_count"] + 1 > 0, "Core observation-query identity")
    require(bool(snap["terminated"]) == bool(terminal), "Terminal identity")
    if terminal:
        require(np.array_equal(obs, np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)), "Frozen zero sentinel")
    else:
        key = (snap["perception_seed"], snap["year"], snap["trajectory_id"], snap["day_index"], snap["L_pre"])
        require(key in env._assets._perception_cache, "Decision observation has existing frozen perception record")
        q50, width = env._assets._perception_cache[key]
        require(np.isfinite([q50, width]).all(), "Finite cached perception")
        doy = date.fromisoformat(snap["date"]).timetuple().tm_yday
        phase = 2 * np.pi * (doy - 1) / p.decision_reward_steps
        content = [q50] if env.observation_mode == "Point" else [q50, width]
        expected = np.asarray(content + [np.sin(phase), np.cos(phase)], dtype=env.observation_space.dtype)
        require(np.array_equal(obs, expected), "Exact q50/optional-width/season semantics")
    return snap


class ScheduledPerceptionEnv(gym.Env):
    """Schedule/delegate/audit only. Support failure poisons this adapter; no retry."""
    metadata = {"render_modes": []}

    def __init__(self, assets, schedule, ledger, observation_mode):
        super().__init__()
        require(observation_mode in ("Point", "UA"), "Shared perception modes only")
        self.assets, self.schedule, self.ledger = assets, schedule, ledger
        self.observation_mode = observation_mode
        self.pending = self._next_assignment()
        self.current = construct_perception_core(assets, ledger, "TRAINING", self.pending, observation_mode)
        self.observation_space, self.action_space = self.current.observation_space, self.current.action_space
        self.assignments = []
        self.started = self.done = self.failed = False
        self.total_steps = self.observations_checked = self.episodes_completed = self.perception_query_count = 0

    def _next_assignment(self):
        # Only perception seed is supplied; latent trajectory/order are unchanged.
        return {**self.schedule.next_assignment(), "perception_seed": get_perception_contract().development_seed}

    def reset(self, *, seed=None, options=None):
        require(not self.failed and (not self.started or self.done), "No retry/discard of a live or failed episode")
        require(options is None or options == {}, "No EpisodeSpec override")
        super().reset(seed=seed)
        if self.started:
            self.current.close()
            self.pending = self._next_assignment()
            self.current = construct_perception_core(self.assets, self.ledger, "TRAINING", self.pending, self.observation_mode)
            require(self.current.observation_space == self.observation_space and self.current.action_space == self.action_space, "Invariant spaces")
        try:
            obs, info = self.current.reset(seed=seed, options=options)
        except core.PerceptionSupportError:
            self.failed = True
            raise
        snap = audit_perception_observation(self.current, obs, info)
        self.observations_checked += 1
        self.perception_query_count += snap["perception_query_count"]
        self.assignments.append({**self.pending, "steps": 0, "completed": False, "natural_transition_count": 0,
                                 "perception_query_count": snap["perception_query_count"]})
        self.started, self.done = True, False
        return obs, info

    def step(self, action):
        require(self.started and not self.done and not self.failed, "Live nonfailed episode required")
        try:
            result = self.current.step(action)
        except core.PerceptionSupportError:
            self.failed = True
            raise
        obs, reward, terminated, truncated, info = result
        snap = audit_perception_observation(self.current, obs, info, terminal=terminated)
        parts = snap["last_reward_components"]
        require(np.isfinite([reward, parts["L_pre"], parts["L_post"], parts["total_cost"]]).all(), "Finite training reward/settlement/state")
        if parts["L_next"] is not None:
            require(np.isfinite(parts["L_next"]), "Finite next state")
        row = self.assignments[-1]
        self.perception_query_count += snap["perception_query_count"] - row["perception_query_count"]
        row["perception_query_count"] = snap["perception_query_count"]
        row["steps"] += 1
        row["natural_transition_count"] = snap["transition_count"]
        self.total_steps += 1
        self.observations_checked += 1
        self.done = bool(terminated or truncated)
        require(not truncated and terminated == (row["steps"] == get_protocol().decision_reward_steps), "Frozen training horizon")
        if self.done:
            require(snap["transition_count"] == get_protocol().natural_transitions
                    and snap["perception_query_count"] == get_protocol().decision_reward_steps, "Complete-episode query/transition identity")
            row["completed"] = True
            self.episodes_completed += 1
        return result

    def close(self):
        self.current.close()


def build_scheduled_perception_env(assets, rl_seed, ledger, config_id, observation_mode):
    require(config_id == inherited_config_id(), "Frozen config only")
    return ScheduledPerceptionEnv(assets, build_training_schedule(rl_seed, config_id), ledger, observation_mode)


def build_perception_eval_env(assets, year, trajectory_id, ledger, observation_mode):
    # No arbitrary root or perception seed argument; DEVELOPMENT only.
    row = {**true_runner.development_assignment(year, trajectory_id),
           "perception_seed": get_perception_contract().development_seed}
    return construct_perception_core(assets, ledger, "DEVELOPMENT", row, observation_mode)


def build_perception_ppo_model(env, config_id, rl_seed):
    """Only exact-type compatibility is new; frozen constructor body follows.

    The smoke audit compares this body structurally with the frozen builder,
    including PPO keyword arguments, seeding, threads and implicit optimizer
    defaults. No new PPO hyperparameter or training rule is introduced.
    """
    import torch
    from stable_baselines3 import PPO

    p, c = get_protocol(), candidate(config_id)
    require(type(env) is ScheduledPerceptionEnv and env.schedule.parent_seed == rl_seed
            and not env.started and not env.failed and config_id == inherited_config_id(), "Fresh matching perception environment")
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


def evaluate_perception_deterministic(model, assets, specs, ledger, observation_mode):
    """Mechanically mirrored evaluation shell; settlement remains frozen core.

    Same deterministic prediction, cost/return/cleaning accumulation, terminal
    handling, metric names and reward-sign regression as the True-State runner.
    Only environment creation and observation/perception auditing differ.
    """
    p = get_protocol()
    rows = []
    for year, tid in specs:
        env = build_perception_eval_env(assets, year, tid, ledger, observation_mode)
        try:
            obs, info = env.reset()
            audit_perception_observation(env, obs, info)
            rewards, costs, states, actions = [], [], [], []
            for index in range(p.decision_reward_steps):
                action_array, _ = model.predict(obs, deterministic=p.evaluation.deterministic)
                array = np.asarray(action_array)
                require(array.size == 1 and np.issubdtype(array.dtype, np.integer), "Integer scalar model action")
                action = int(array.item())
                require(env.action_space.contains(action), "Valid PPO action")
                obs, reward, terminated, truncated, info = env.step(action)
                snap = audit_perception_observation(env, obs, info, terminal=terminated)
                parts = snap["last_reward_components"]
                require(np.isfinite([reward, parts["total_cost"], parts["L_pre"], parts["L_post"]]).all(), "Finite eval state/reward/cost")
                if parts["L_next"] is not None:
                    require(np.isfinite(parts["L_next"]), "Finite next state")
                rewards.append(float(reward))
                costs.append(float(parts["total_cost"]))
                states.append(float(parts["L_pre"]))
                actions.append(action)
                require(not truncated and terminated == (index == p.decision_reward_steps - 1), "Exact horizon")
            episode_return, total_cost = sum(rewards), sum(costs)
            require(episode_return == snap["episode_return"] == info["episode_return"], "Return accumulation")
            require(sum(actions) == snap["clean_count"] == info["clean_count"], "Cleaning count")
            require(abs(episode_return + total_cost) <= 1e-12, "Frozen reward sign")
            require(snap["transition_count"] == p.natural_transitions, "Natural transitions")
            require(snap["perception_query_count"] == p.decision_reward_steps, "One perception query per decision observation")
            rows.append({"year": year, "trajectory_id": tid, "environment_root": env.episode_spec.environment_root,
                         "step_count": len(actions), "natural_transition_count": snap["transition_count"],
                         "J_total": total_cost, "episode_return": episode_return,
                         "reward_sign_regression_diff": abs(episode_return + total_cost),
                         "N_clean": snap["clean_count"], "mean_L_pre": float(np.mean(states)), "max_L_pre": max(states),
                         "action_min": min(actions), "action_max": max(actions),
                         "all_observations_finite": True, "all_rewards_finite": True, "all_states_finite": True,
                         "deterministic_action": p.evaluation.deterministic, "perception_query_count": snap["perception_query_count"],
                         "indices_sha256": snap["indices_sha256"], "full_bank_sha256": snap["full_bank_sha256"]})
        finally:
            env.close()
    return rows
