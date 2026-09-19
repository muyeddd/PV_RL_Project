"""True-State infrastructure; no assets, environments, models or RNGs at import.

Scientific values come exclusively from the frozen P2-2C-0 module. The only
additional budget is explicitly diagnostic. Formal evaluation is deliberately
unavailable here until a later stage supplies its own frozen access gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
import hashlib
from pathlib import Path
import random
import time

import gymnasium as gym
import numpy as np

import paper2_ppo_protocol_v1 as frozen
import paper2_gym_pomdp_env_v1 as core


DIAGNOSTIC_SMOKE_BUDGET = 4096


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


@lru_cache(maxsize=1)
def get_protocol():
    # Lazy cache of an entirely frozen dataclass: no import-time validation/work.
    p = frozen.Protocol()
    frozen.validate_protocol(p)
    return p


def partition(role):
    return next(p for p in get_protocol().partitions if p.role == role)


def candidate(config_id):
    name = config_id if config_id.startswith("CONFIG_") else "CONFIG_" + config_id
    matches = [c for c in get_protocol().candidates if c.name == name]
    require(len(matches) == 1, "Only frozen candidate IDs accepted")
    return matches[0]


def derive_frozen_rl_seeds(rl_seed):
    return frozen.derive_rl_subseeds(rl_seed)


def child_seeds(rl_seed):
    return {c["role"]: c["integer_seed"] for c in derive_frozen_rl_seeds(rl_seed)["children"]}


class TrainingSchedule:
    """Pure metadata scheduler; no core EpisodeSpec or trajectory construction."""

    def __init__(self, rl_seed):
        self.parent_seed = rl_seed
        self.canonical = frozen.canonical_training_specs()
        self.rng = np.random.default_rng(child_seeds(rl_seed)["training_schedule"])
        self.order = ()
        self.ordinal = 0

    def next_assignment(self):
        size = len(self.canonical)
        cycle, position = divmod(self.ordinal, size)
        if position == 0:
            self.order = self.rng.permutation(size)
        year, root, tid, population, perception = self.canonical[int(self.order[position])]
        row = {"episode_ordinal": self.ordinal, "cycle_index": cycle,
               "position_in_cycle": position, "year": year, "trajectory_id": tid,
               "environment_root": root, "population_size": population,
               "perception_seed": perception}
        self.ordinal += 1
        return row


def build_training_schedule(rl_seed, config_id=None):
    # Accept config for the A/B independence audit, but never pass it into RNG logic.
    if config_id is not None:
        candidate(config_id)
    return TrainingSchedule(rl_seed)


def assignment_key(row):
    return row["year"], row["environment_root"], row["trajectory_id"]


@dataclass
class AccessLedger:
    """Record actual constructor access, and reject formal BEFORE EpisodeSpec."""
    records: list = field(default_factory=list)
    denied_accesses: int = 0

    def authorize(self, role, row):
        if role not in ("TRAINING", "DEVELOPMENT"):
            self.denied_accesses += 1
            raise RuntimeError("Formal trajectory construction unavailable in this runner")
        part = partition(role)
        valid = (row["year"], row["environment_root"]) in tuple(zip(part.years, part.roots))
        valid = valid and type(row["trajectory_id"]) is int and row["trajectory_id"] in part.trajectory_ids
        valid = valid and row["population_size"] == len(part.trajectory_ids) and row["perception_seed"] is None
        if not valid:
            self.denied_accesses += 1
            raise RuntimeError("Role/year/root/tid/population/perception contract violation")
        self.records.append({"role": role, **row})

    @property
    def heldout_ppo_trajectory_access_count(self):
        roots = set(partition("FORMAL HELD-OUT").roots)
        return sum(r["environment_root"] in roots for r in self.records)


def load_assets():
    """Explicit frozen loader; True-State does not call ensure_perception()."""
    return core.Paper2EnvAssets()


def construct_core(assets, ledger, role, row):
    ledger.authorize(role, row)
    spec = core.EpisodeSpec(year=row["year"], environment_root=row["environment_root"],
                            trajectory_id=row["trajectory_id"], population_size=row["population_size"],
                            perception_seed=get_protocol().seeds.perception_seed)
    env = core.Paper2CleaningEnv(assets, spec, observation_mode="True-State")
    require(env.observation_space.shape == (dict(get_protocol().observation_dimensions)["True-State"],), "Observation shape")
    require(isinstance(env.action_space, gym.spaces.Discrete)
            and env.action_space.n == len(get_protocol().actions) and env.action_space.start == 0, "Action space")
    return env


def audit_observation(env, obs, *, terminal=False):
    """Private state is read only for audit; never fed to model.predict/learn."""
    require(obs.shape == env.observation_space.shape and obs.dtype == env.observation_space.dtype,
            "Frozen observation shape/dtype")
    require(np.isfinite(obs).all() and env.observation_space.contains(obs), "Finite bounded observation")
    snap = env._audit_snapshot()
    require(snap["perception_seed"] is None and snap["perception_query_count"] == 0, "No perception")
    require(np.isfinite(snap["L_pre"]) and 0 <= snap["L_pre"] <= 1, "Finite bounded state")
    if terminal:
        require(np.array_equal(obs, np.zeros(env.observation_space.shape, dtype=env.observation_space.dtype)), "Frozen terminal sentinel")
    else:
        doy = date.fromisoformat(snap["date"]).timetuple().tm_yday
        phase = 2 * np.pi * (doy - 1) / get_protocol().decision_reward_steps
        expected = np.asarray([snap["L_pre"], np.sin(phase), np.cos(phase)], dtype=env.observation_space.dtype)
        require(np.array_equal(obs, expected), "L_true/sin_DOY/cos_DOY semantic regression")
    return snap


class ScheduledTrueStateEnv(gym.Env):
    """Only schedule, delegate and audit. First core is pending until first reset.

    The core has constructor-bound EpisodeSpec and disallows reconfiguration via
    reset options. A replacement core is created only after episode termination.
    No physics, reward, observation or termination logic is implemented here.
    """
    metadata = {"render_modes": []}

    def __init__(self, assets, schedule, ledger):
        super().__init__()
        self.assets, self.schedule, self.ledger = assets, schedule, ledger
        self.pending = schedule.next_assignment()
        self.current = construct_core(assets, ledger, "TRAINING", self.pending)
        self.observation_space = self.current.observation_space
        self.action_space = self.current.action_space
        self.assignments = []
        self.started = False
        self.done = False
        self.total_steps = 0
        self.observations_checked = 0
        self.episodes_completed = 0
        self.perception_query_count = 0

    def reset(self, *, seed=None, options=None):
        require(options is None or options == {}, "No episode override through reset options")
        require(not self.started or self.done, "Cannot discard a live training episode")
        super().reset(seed=seed)
        if self.started:
            self.current.close()
            self.pending = self.schedule.next_assignment()
            self.current = construct_core(self.assets, self.ledger, "TRAINING", self.pending)
            require(self.current.observation_space == self.observation_space and self.current.action_space == self.action_space,
                    "Spaces invariant across scheduled episodes")
        obs, info = self.current.reset(seed=seed, options=options)
        audit_observation(self.current, obs)
        self.observations_checked += 1
        self.assignments.append({**self.pending, "steps": 0, "completed": False,
                                 "natural_transition_count": 0, "perception_query_count": 0})
        self.started, self.done = True, False
        return obs, info

    def step(self, action):
        require(self.started and not self.done, "Reset required")
        result = self.current.step(action)
        obs, reward, terminated, truncated, info = result
        snap = audit_observation(self.current, obs, terminal=terminated)
        require(np.isfinite(reward), "Finite training reward")
        parts = snap["last_reward_components"]
        require(np.isfinite([parts["L_pre"], parts["L_post"], parts["total_cost"]]).all(), "Finite training settlement/state")
        if parts["L_next"] is not None:
            require(np.isfinite(parts["L_next"]), "Finite training next state")
        self.total_steps += 1
        self.observations_checked += 1
        row = self.assignments[-1]
        row["steps"] += 1
        row["natural_transition_count"] = snap["transition_count"]
        row["perception_query_count"] = snap["perception_query_count"]
        self.done = bool(terminated or truncated)
        if self.done:
            require(terminated and not truncated and row["steps"] == get_protocol().decision_reward_steps
                    and snap["transition_count"] == get_protocol().natural_transitions, "Frozen horizon")
            row["completed"] = True
            self.episodes_completed += 1
            self.perception_query_count += snap["perception_query_count"]
        return result

    def close(self):
        self.current.close()


def build_scheduled_true_state_env(assets, rl_seed, ledger, config_id=None):
    return ScheduledTrueStateEnv(assets, build_training_schedule(rl_seed, config_id), ledger)


def development_assignment(year, trajectory_id):
    part = partition("DEVELOPMENT")
    roots = dict(zip(part.years, part.roots))
    require(year in roots and type(trajectory_id) is int and trajectory_id in part.trajectory_ids, "Development spec")
    return {"year": year, "environment_root": roots[year], "trajectory_id": trajectory_id,
            "population_size": len(part.trajectory_ids), "perception_seed": get_protocol().seeds.perception_seed}


def build_true_state_eval_env(assets, year, trajectory_id, ledger):
    # No root argument: callers cannot smuggle a formal root into evaluation.
    return construct_core(assets, ledger, "DEVELOPMENT", development_assignment(year, trajectory_id))


@dataclass
class ModelBundle:
    model: object
    env: ScheduledTrueStateEnv
    config_id: str
    parent_seed: int
    training_finished: bool = False


def build_ppo_model(env, config_id, rl_seed):
    import torch
    from stable_baselines3 import PPO

    p, c = get_protocol(), candidate(config_id)
    require(type(env) is ScheduledTrueStateEnv and env.schedule.parent_seed == rl_seed
            and not env.started, "Fresh matching scheduled training environment required")
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
    return ModelBundle(model, env, c.name, rl_seed)


def verify_model_contract(model, config_id, rl_seed):
    import torch
    p, c = get_protocol(), candidate(config_id)
    common, arch = dict(p.common), dict(p.architecture)
    for key in ("batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range_vf", "normalize_advantage",
                "ent_coef", "vf_coef", "max_grad_norm", "use_sde", "sde_sample_freq", "target_kl"):
        require(getattr(model, key) == common[key], f"Model parameter mismatch: {key}")
    require(model.learning_rate == c.learning_rate and model.n_steps == c.n_steps, "Candidate parameters")
    # SB3 internally wraps constants as callables; check constant values, not callable identity.
    require(all(model.lr_schedule(progress) == c.learning_rate and model.clip_range(progress) == common["clip_range"]
                for progress in (0.0, 0.5, 1.0)), "Constant LR/clip")
    require(model.policy.net_arch == {"pi": list(arch["actor"]), "vf": list(arch["critic"])}
            and model.policy.activation_fn is torch.nn.Tanh and model.policy.ortho_init == arch["ortho_init"], "Network")
    require(model.seed == child_seeds(rl_seed)["learner"] and model.device.type == p.device
            and model.n_envs == common["n_envs"], "Learner seed/device/n_envs")
    return True


def parameters_finite(model):
    import torch
    return all(bool(torch.isfinite(value).all()) for value in model.policy.parameters())


def train_final_checkpoint(bundle, model_path, *, purpose):
    """Single fresh run; fixed budget; no callback or checkpoint selection inputs.

    A numeric-finiteness hook raises on invalid parameters after every optimizer
    step. It never chooses a model or stops based on evaluation/performance.
    """
    p = get_protocol()
    require(purpose in ("smoke", "development", "final"), "Explicit training purpose")
    if purpose == "smoke":
        require(bundle.config_id == p.candidates[0].name and bundle.parent_seed == p.seeds.development[0], "Smoke config/seed")
        budget = DIAGNOSTIC_SMOKE_BUDGET
    elif purpose == "development":
        require(bundle.parent_seed in p.seeds.development, "Development learner seed")
        budget = p.development_budget
    else:
        require(bundle.parent_seed in p.seeds.final, "Final learner seed")
        budget = p.final_budget
    model, env = bundle.model, bundle.env
    require(not bundle.training_finished and model.num_timesteps == 0 and not env.started,
            "Only fresh final-checkpoint training; no continuation or reset of prior training")
    require(budget % model.n_steps == 0, "Exact rollout budget; no SB3 overshoot")
    path = Path(model_path)
    require(path.suffix == ".zip" and path.parent.is_dir() and not path.exists(), "Exclusive .zip destination")
    require(parameters_finite(model), "Finite initial model")
    checks = {"optimizer_steps_checked": 0}

    def check_update(optimizer, args, kwargs):
        require(parameters_finite(model), "Nonfinite model during training")
        checks["optimizer_steps_checked"] += 1

    hook = model.policy.optimizer.register_step_post_hook(check_update)
    started = time.perf_counter()
    try:
        model.learn(total_timesteps=budget, reset_num_timesteps=True, progress_bar=False)
    finally:
        hook.remove()
    elapsed = time.perf_counter() - started
    require(model.num_timesteps == budget == env.total_steps and parameters_finite(model), "Exact completed training")
    require(checks["optimizer_steps_checked"] > 0, "Optimizer path exercised")
    bundle.training_finished = True
    # Exclusive file handle avoids silent overwrite by SB3's path-based saver.
    with path.open("xb") as stream:
        model.save(stream)
    return {"model_path": str(path), "model_sha256": file_sha256(path), "total_timesteps": budget,
            "wall_clock_seconds": elapsed, "model_parameter_finite": True, **checks,
            "rollout_updates": budget // model.n_steps, "scientific_budget_used": purpose != "smoke",
            "final_checkpoint_only": True}


def reload_final_checkpoint(model_path, bundle):
    from stable_baselines3 import PPO
    # Fresh PPO object without an attached environment: loading cannot reset a trajectory.
    loaded = PPO.load(str(model_path), device=get_protocol().device)
    require(loaded.observation_space == bundle.env.observation_space
            and loaded.action_space == bundle.env.action_space, "Reload spaces")
    require(loaded.num_timesteps == bundle.model.num_timesteps and parameters_finite(loaded), "Reload timesteps/parameters")
    for name, value in bundle.model.policy.state_dict().items():
        require(np.array_equal(value.detach().cpu().numpy(), loaded.policy.state_dict()[name].detach().cpu().numpy()), "Reload parameter equality")
    verify_model_contract(loaded, bundle.config_id, bundle.parent_seed)
    return loaded


def evaluate_deterministic(model, assets, specs, ledger):
    """Development-only causal evaluation; model receives only core observation.

    J_total is independently accumulated from private frozen settlement costs,
    not defined as minus return, making the reward-sign regression meaningful.
    """
    p = get_protocol()
    rows = []
    for year, tid in specs:
        env = build_true_state_eval_env(assets, year, tid, ledger)
        try:
            obs, info = env.reset()
            audit_observation(env, obs)
            rewards, costs, states, actions = [], [], [], []
            for index in range(p.decision_reward_steps):
                action_array, _ = model.predict(obs, deterministic=p.evaluation.deterministic)
                array = np.asarray(action_array)
                require(array.size == 1 and np.issubdtype(array.dtype, np.integer), "Integer scalar model action")
                action = int(array.item())
                require(env.action_space.contains(action), "Valid PPO action")
                obs, reward, terminated, truncated, info = env.step(action)
                snap = audit_observation(env, obs, terminal=terminated)
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


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_manifest(bundle, training):
    p, c = get_protocol(), candidate(bundle.config_id)
    seeds, common, arch = child_seeds(bundle.parent_seed), dict(p.common), dict(p.architecture)
    return {**common, "model_file": Path(training["model_path"]).name, "model_sha256": training["model_sha256"],
            "config_id": c.name, "network": {"pi": list(arch["actor"]), "vf": list(arch["critic"])},
            "activation": arch["activation"], "ortho_init": arch["ortho_init"],
            "learning_rate": c.learning_rate, "n_steps": c.n_steps, "total_timesteps": training["total_timesteps"],
            "rl_parent_seed": bundle.parent_seed, "learner_seed": seeds["learner"], "schedule_seed": seeds["training_schedule"],
            "final_checkpoint_only": True, "smoke_only": not training["scientific_budget_used"],
            "not_for_scientific_selection": not training["scientific_budget_used"]}
