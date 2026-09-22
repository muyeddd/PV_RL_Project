
"""Action-masked Point/UA PPO runner built on the frozen q50 safety shield.

Scientific contract
-------------------
This module changes only the ACTION-SELECTION SEMANTICS required after the
P2-2D-3R / P2-2D-4AR / P2-2D-4B diagnostics:

    q50-history -> frozen safety mask -> MaskablePPO action
                -> unchanged frozen q50 shield (runtime fail-safe)
                -> unchanged reward/physics

The safety set is NOT changed.  The mask is a read-only view of the SAME
frozen Q50SetMembershipShield belief and FrozenRobustTransitionEnvelope used by
the already-audited post-hoc shield.

Nonterminal decisions:
    WAIT safe  -> [WAIT=True, CLEAN=True]
    WAIT unsafe -> [WAIT=False, CLEAN=True]

Terminal day 364 has no t->t+1 safety transition, therefore both actions are
valid, matching the frozen terminal semantics.

The underlying shield remains active as a fail-safe.  A masked nonterminal
action that is subsequently changed by the shield is a hard integration error.

Point and UA use exactly the same q50-only mask.  UA interval width is never an
input to the mask.

Import is inert.  sb3-contrib is imported only when model construction/loading
is explicitly requested.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as metadata
import inspect
from pathlib import Path
import random

import gymnasium as gym
import numpy as np

import paper2_perception_ppo_runner_v1 as perception
import paper2_shielded_perception_ppo_runner_v1 as shielded


get_protocol = shielded.get_protocol
candidate = shielded.candidate
partition = shielded.partition
child_seeds = shielded.child_seeds
build_training_schedule = shielded.build_training_schedule
verify_model_contract = shielded.verify_model_contract
parameters_finite = shielded.parameters_finite
file_sha256 = shielded.file_sha256
load_assets = shielded.load_assets
require = shielded.require
get_perception_contract = shielded.get_perception_contract
inherited_config_id = shielded.inherited_config_id
AccessLedger = shielded.AccessLedger
audit_perception_observation = shielded.audit_perception_observation
read_shield_authority = shielded.read_shield_authority
DIAGNOSTIC_SMOKE_BUDGET = shielded.DIAGNOSTIC_SMOKE_BUDGET

EXPECTED_MASKED_DEPENDENCIES = {
    "stable-baselines3": "2.7.1",
    "sb3-contrib": "2.7.1",
}


def masked_dependency_versions():
    versions = {}
    for package, expected in EXPECTED_MASKED_DEPENDENCIES.items():
        try:
            value = metadata.version(package)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required masked-PPO dependency missing: {package}=={expected}"
            ) from exc
        require(
            value == expected,
            f"Masked-PPO dependency mismatch: {package}={value}, expected {expected}",
        )
        versions[package] = value
    return versions


def verify_maskable_api_contract():
    """Mechanical runtime API check; no model/environment is created."""
    masked_dependency_versions()
    from sb3_contrib import MaskablePPO

    learn = inspect.signature(MaskablePPO.learn)
    predict = inspect.signature(MaskablePPO.predict)
    require(
        "use_masking" in learn.parameters
        and learn.parameters["use_masking"].default is True,
        "MaskablePPO.learn must default to use_masking=True",
    )
    require(
        "action_masks" in predict.parameters,
        "MaskablePPO.predict must expose action_masks",
    )
    return {
        "MaskablePPO_learn_default_use_masking": True,
        "MaskablePPO_predict_accepts_action_masks": True,
        "dependency_versions": masked_dependency_versions(),
    }


class Q50MaskedShieldedPerceptionEnv(shielded.Q50ShieldedPerceptionEnv):
    """Frozen shield plus a read-only action mask derived from the same belief."""

    metadata = {"render_modes": []}

    def __init__(self, inner, assets, observation_mode):
        super().__init__(inner, assets, observation_mode)
        self.total_masked_safety_decisions = 0
        self.total_forced_clean = 0
        self.total_free_choice_decisions = 0
        self.total_voluntary_clean = 0
        self.total_mask_free_wait = 0
        self.total_terminal_clean = 0
        self.total_terminal_wait = 0

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        row = self._episode_records[-1]
        row.update(
            {
                "masked_safety_decisions": 0,
                "forced_clean_count": 0,
                "free_choice_decisions": 0,
                "voluntary_clean_count": 0,
                "mask_free_wait_count": 0,
                "terminal_clean_count": 0,
                "terminal_wait_count": 0,
            }
        )
        # A valid mask must already exist at the first decision.
        mask = self._compute_action_mask()
        require(mask.shape == (2,) and mask.dtype == np.bool_, "Reset mask schema")
        return obs, info

    def _compute_action_mask(self):
        require(self._episode_live, "Live episode required for action mask")
        terminal_decision = (
            self._decision_index
            == get_protocol().decision_reward_steps - 1
        )
        if terminal_decision:
            return np.asarray([True, True], dtype=np.bool_)

        lo = self.shield.belief_lower
        hi = self.shield.belief_upper
        require(
            lo is not None
            and hi is not None
            and 0.0 <= lo <= hi <= self.shield.kernel_upper,
            "Observed q50 belief required for mask",
        )
        wait_safe, wait_bound = self.shield.transition.action_safe(
            float(lo), float(hi), 0
        )
        clean_safe, clean_bound = self.shield.transition.action_safe(
            float(lo), float(hi), 1
        )
        require(
            clean_safe
            and clean_bound.upper <= self.shield.kernel_upper,
            "Frozen mask must retain robustly safe CLEAN",
        )
        require(
            bool(wait_safe)
            == bool(wait_bound.upper <= self.shield.kernel_upper),
            "WAIT mask matches frozen kernel test",
        )
        return np.asarray([bool(wait_safe), True], dtype=np.bool_)

    def action_masks(self):
        """SB3-Contrib convention: True means the discrete action is valid."""
        return self._compute_action_mask().copy()

    def step(self, proposed_action):
        if (
            not isinstance(proposed_action, (int, np.integer))
            or isinstance(proposed_action, (bool, np.bool_))
            or int(proposed_action) not in (0, 1)
        ):
            raise ValueError("Action must be integer 0 (WAIT) or 1 (CLEAN)")
        proposed = int(proposed_action)

        terminal_decision = (
            self._decision_index
            == get_protocol().decision_reward_steps - 1
        )
        mask = self._compute_action_mask()
        require(
            bool(mask[proposed]),
            "Masked runtime received an invalid action",
        )

        result = super().step(proposed)
        decision = self.last_shield_decision
        require(decision is not None, "Shield decision audit required")

        row = self._episode_records[-1]
        if terminal_decision:
            require(
                mask.tolist() == [True, True]
                and decision["safety_constraint_applicable"] is False
                and decision["intervention"] is False
                and decision["executed_action"] == proposed,
                "Frozen terminal mask/pass-through semantics",
            )
            if proposed == 1:
                self.total_terminal_clean += 1
                row["terminal_clean_count"] += 1
            else:
                self.total_terminal_wait += 1
                row["terminal_wait_count"] += 1
        else:
            require(
                decision["safety_constraint_applicable"] is True
                and decision["proposed_safe"] is True
                and decision["clean_safe"] is True
                and decision["intervention"] is False
                and decision["executed_action"] == proposed,
                "Mask and frozen shield must agree exactly",
            )
            self.total_masked_safety_decisions += 1
            row["masked_safety_decisions"] += 1

            if not bool(mask[0]):
                require(
                    mask.tolist() == [False, True] and proposed == 1,
                    "Forced-CLEAN mask semantics",
                )
                self.total_forced_clean += 1
                row["forced_clean_count"] += 1
            else:
                require(
                    mask.tolist() == [True, True],
                    "Free-choice mask semantics",
                )
                self.total_free_choice_decisions += 1
                row["free_choice_decisions"] += 1
                if proposed == 1:
                    self.total_voluntary_clean += 1
                    row["voluntary_clean_count"] += 1
                else:
                    self.total_mask_free_wait += 1
                    row["mask_free_wait_count"] += 1

        return result

    def _mask_audit_snapshot(self):
        shield = self._shield_audit_snapshot()
        return {
            **shield,
            "total_masked_safety_decisions":
                int(self.total_masked_safety_decisions),
            "total_forced_clean": int(self.total_forced_clean),
            "total_free_choice_decisions":
                int(self.total_free_choice_decisions),
            "total_voluntary_clean": int(self.total_voluntary_clean),
            "total_mask_free_wait": int(self.total_mask_free_wait),
            "total_terminal_clean": int(self.total_terminal_clean),
            "total_terminal_wait": int(self.total_terminal_wait),
            "mask_shield_intervention_invariant":
                int(self.total_shield_interventions) == 0,
        }


class ScheduledMaskedPerceptionEnv(Q50MaskedShieldedPerceptionEnv):
    def __init__(self, assets, schedule, ledger, observation_mode):
        self._scheduled_inner = perception.ScheduledPerceptionEnv(
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


def build_masked_scheduled_perception_env(
    assets, rl_seed, ledger, config_id, observation_mode
):
    require(
        config_id == inherited_config_id(),
        "Frozen CONFIG_A only",
    )
    return ScheduledMaskedPerceptionEnv(
        assets,
        build_training_schedule(rl_seed, config_id),
        ledger,
        observation_mode,
    )


def build_masked_perception_eval_env(
    assets, year, trajectory_id, ledger, observation_mode
):
    inner = perception.build_perception_eval_env(
        assets, year, trajectory_id, ledger, observation_mode
    )
    return Q50MaskedShieldedPerceptionEnv(inner, assets, observation_mode)


def verify_masked_model_contract(model, config_id, rl_seed):
    verify_maskable_api_contract()
    from sb3_contrib import MaskablePPO

    require(
        isinstance(model, MaskablePPO),
        "Runtime algorithm must be sb3-contrib MaskablePPO",
    )
    verify_model_contract(model, config_id, rl_seed)
    return True


def build_masked_perception_ppo_model(
    env, config_id, rl_seed
):
    import torch
    from sb3_contrib import MaskablePPO

    verify_maskable_api_contract()
    p, c = get_protocol(), candidate(config_id)
    require(
        type(env) is ScheduledMaskedPerceptionEnv
        and env.schedule.parent_seed == rl_seed
        and not env.started
        and not env.failed
        and config_id == inherited_config_id(),
        "Fresh matching masked perception environment",
    )

    # Both mask patterns must be syntactically representable before learning.
    require(
        env.action_space.n == 2
        and env.action_space.start == 0,
        "Discrete WAIT/CLEAN action space",
    )

    common = dict(p.common)
    architecture = dict(p.architecture)
    require(
        common.pop("algorithm") == "PPO"
        and common.pop("n_envs") == 1,
        "Frozen PPO family/single env",
    )
    require(
        common.pop("learning_rate_schedule")
        == common.pop("clip_schedule")
        == "NONE",
        "Frozen constant schedules",
    )
    policy = common.pop("policy")
    require(
        policy == "MlpPolicy"
        and not p.observation_normalization
        and not p.reward_normalization,
        "Frozen policy/normalization contract",
    )
    require(
        architecture["activation"] == "Tanh",
        "Frozen activation",
    )

    learner = child_seeds(rl_seed)["learner"]
    torch.set_num_threads(p.torch_num_threads)
    random.seed(learner)
    np.random.seed(learner)
    torch.manual_seed(learner)

    model = MaskablePPO(
        policy=policy,
        env=env,
        learning_rate=c.learning_rate,
        n_steps=c.n_steps,
        policy_kwargs={
            "net_arch": {
                "pi": list(architecture["actor"]),
                "vf": list(architecture["critic"]),
            },
            "activation_fn": torch.nn.Tanh,
            "ortho_init": architecture["ortho_init"],
        },
        seed=learner,
        verbose=0,
        **common,
    )
    require(
        model.device.type == p.device
        and model.n_envs == dict(p.common)["n_envs"],
        "Frozen CPU/single-env runtime",
    )
    verify_masked_model_contract(model, c.name, rl_seed)
    return shielded.ModelBundle(
        model=model,
        env=env,
        config_id=c.name,
        parent_seed=rl_seed,
    )


def train_masked_final_checkpoint(bundle, model_path, *, purpose):
    """Reuse the frozen trainer; MaskablePPO.learn defaults to use_masking=True."""
    verify_masked_model_contract(
        bundle.model, bundle.config_id, bundle.parent_seed
    )
    api = verify_maskable_api_contract()
    require(
        api["MaskablePPO_learn_default_use_masking"] is True,
        "Training must use action masking",
    )
    return shielded.train_final_checkpoint(
        bundle, model_path, purpose=purpose
    )


def reload_masked_final_checkpoint(model_path, bundle):
    from sb3_contrib import MaskablePPO

    verify_maskable_api_contract()
    loaded = MaskablePPO.load(
        str(model_path),
        device=get_protocol().device,
    )
    require(
        loaded.observation_space == bundle.env.observation_space
        and loaded.action_space == bundle.env.action_space,
        "Reload spaces",
    )
    require(
        loaded.num_timesteps == bundle.model.num_timesteps
        and parameters_finite(loaded),
        "Reload timesteps/parameters",
    )
    for name, value in bundle.model.policy.state_dict().items():
        require(
            np.array_equal(
                value.detach().cpu().numpy(),
                loaded.policy.state_dict()[name].detach().cpu().numpy(),
            ),
            "Reload parameter equality",
        )
    verify_masked_model_contract(
        loaded, bundle.config_id, bundle.parent_seed
    )
    return loaded


def masked_model_manifest(bundle, training):
    manifest = shielded.model_manifest(bundle, training)
    manifest.update(
        {
            "runtime_algorithm": "MaskablePPO",
            "mask_source":
                "frozen q50-history set-membership safety shield belief",
            "mask_semantics": {
                "nonterminal_wait_safe": [True, True],
                "nonterminal_wait_unsafe": [False, True],
                "terminal_day_364": [True, True],
            },
            "shield_remains_runtime_failsafe": True,
            "masked_dependencies": masked_dependency_versions(),
        }
    )
    return manifest


def evaluate_masked_perception_deterministic(
    model, assets, specs, ledger, observation_mode
):
    p = get_protocol()

    verify_maskable_api_contract()
    rows = []
    for year, tid in specs:
        env = build_masked_perception_eval_env(
            assets, year, tid, ledger, observation_mode
        )
        try:
            obs, info = env.reset()
            audit_perception_observation(env.inner, obs, info)
            rewards, costs, states = [], [], []
            actions, masks = [], []

            for index in range(p.decision_reward_steps):
                action_mask = env.action_masks()
                require(
                    action_mask.shape == (2,)
                    and action_mask.dtype == np.bool_
                    and bool(action_mask[1]),
                    "Valid q50 safety action mask",
                )
                action_array, _ = model.predict(
                    obs,
                    deterministic=p.evaluation.deterministic,
                    action_masks=action_mask,
                )
                array = np.asarray(action_array)
                require(
                    array.size == 1
                    and np.issubdtype(array.dtype, np.integer),
                    "Integer scalar masked action",
                )
                action = int(array.item())
                require(
                    env.action_space.contains(action)
                    and bool(action_mask[action]),
                    "MaskablePPO action must be valid",
                )

                obs, reward, terminated, truncated, info = env.step(action)
                snap = audit_perception_observation(
                    env.inner, obs, info, terminal=terminated
                )
                parts = snap["last_reward_components"]
                decision = env.last_shield_decision
                require(
                    decision is not None
                    and decision["proposed_action"] == action
                    and decision["executed_action"] == parts["action"],
                    "Masked action/executed action audit identity",
                )
                if index < p.natural_transitions:
                    require(
                        decision["intervention"] is False
                        and decision["proposed_safe"] is True
                        and decision["executed_action"] == action,
                        "No post-hoc shield intervention after masking",
                    )

                rewards.append(float(reward))
                costs.append(float(parts["total_cost"]))
                states.append(float(parts["L_pre"]))
                actions.append(action)
                masks.append(tuple(bool(v) for v in action_mask))
                require(
                    not truncated
                    and terminated
                    == (index == p.decision_reward_steps - 1),
                    "Exact frozen horizon",
                )

            episode_return = sum(rewards)
            total_cost = sum(costs)
            record = env._mask_audit_snapshot()["episode_records"][-1]
            mask_audit = env._mask_audit_snapshot()

            require(
                episode_return
                == snap["episode_return"]
                == info["episode_return"]
                and abs(episode_return + total_cost) <= p.selection.atol,
                "Frozen return/cost sign",
            )
            require(
                sum(actions)
                == snap["clean_count"]
                == info["clean_count"]
                == record["executed_clean_count"],
                "Masked cleaning-count identity",
            )
            require(
                mask_audit["total_shield_interventions"] == 0
                and record["interventions"] == 0,
                "Zero shield intervention invariant",
            )
            require(
                record["masked_safety_decisions"]
                == p.natural_transitions
                == record["forced_clean_count"]
                + record["free_choice_decisions"],
                "Masked nonterminal decision partition",
            )
            require(
                record["free_choice_decisions"]
                == record["voluntary_clean_count"]
                + record["mask_free_wait_count"],
                "Free-choice action partition",
            )
            require(
                record["executed_clean_count"]
                == record["forced_clean_count"]
                + record["voluntary_clean_count"]
                + record["terminal_clean_count"],
                "Executed CLEAN partition",
            )

            free = record["free_choice_decisions"]
            rows.append(
                {
                    "year": year,
                    "trajectory_id": tid,
                    "environment_root":
                        env.inner.episode_spec.environment_root,
                    "step_count": len(actions),
                    "natural_transition_count":
                        snap["transition_count"],
                    "J_total": total_cost,
                    "episode_return": episode_return,
                    "reward_sign_regression_diff":
                        abs(episode_return + total_cost),
                    "N_clean": snap["clean_count"],
                    "N_forced_clean": record["forced_clean_count"],
                    "N_voluntary_clean":
                        record["voluntary_clean_count"],
                    "N_free_wait": record["mask_free_wait_count"],
                    "N_free_choice": free,
                    "N_terminal_clean":
                        record["terminal_clean_count"],
                    "N_shield_interventions":
                        record["interventions"],
                    "forced_clean_fraction":
                        record["forced_clean_count"]
                        / p.natural_transitions,
                    "free_choice_fraction":
                        free / p.natural_transitions,
                    "voluntary_clean_fraction_given_free":
                        (
                            record["voluntary_clean_count"] / free
                            if free else 0.0
                        ),
                    "mean_L_pre": float(np.mean(states)),
                    "max_L_pre": max(states),
                    "action_min": min(actions),
                    "action_max": max(actions),
                    "forced_mask_count":
                        sum(mask == (False, True) for mask in masks),
                    "free_mask_count":
                        sum(mask == (True, True) for mask in masks[:-1]),
                    "terminal_mask":
                        list(masks[-1]),
                    "all_observations_finite": True,
                    "all_rewards_finite": True,
                    "all_states_finite": True,
                    "deterministic_action":
                        p.evaluation.deterministic,
                    "perception_query_count":
                        snap["perception_query_count"],
                    "indices_sha256": snap["indices_sha256"],
                    "full_bank_sha256": snap["full_bank_sha256"],
                }
            )
        finally:
            env.close()
    return rows
