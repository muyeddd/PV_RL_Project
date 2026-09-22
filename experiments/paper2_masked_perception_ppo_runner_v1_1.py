
"""API-compatible q50-masked Point/UA PPO runner recovery adapter.

This module is a minimal recovery layer over paper2_masked_perception_ppo_runner_v1.

Why it exists
-------------
P2-2D-5A failed before model construction because sb3-contrib==2.7.1
MaskablePPO.__init__ does not accept the frozen SB3 PPO kwargs:
    use_sde=False
    sde_sample_freq=-1

Those two frozen values mean "no gSDE". MaskablePPO for discrete masked actions
does not expose gSDE constructor arguments at all. This adapter therefore:

1. mechanically verifies the frozen protocol still says exactly
       use_sde is False
       sde_sample_freq == -1
2. mechanically verifies MaskablePPO.__init__ does NOT accept those arguments;
3. removes only those two unsupported no-gSDE kwargs before construction;
4. verifies every other frozen PPO/network/seed/device parameter exactly;
5. changes nothing about mask, shield, reward, physics, perception, schedules,
   budgets, observations, or seeds.

Import is inert. No environment, model, RNG, training, or file write occurs at
import.
"""
from __future__ import annotations

import inspect
import random

import numpy as np

import paper2_masked_perception_ppo_runner_v1 as prior


# Re-export the already-audited masked environment/mask/shield surface unchanged.
get_protocol = prior.get_protocol
candidate = prior.candidate
partition = prior.partition
child_seeds = prior.child_seeds
build_training_schedule = prior.build_training_schedule
parameters_finite = prior.parameters_finite
file_sha256 = prior.file_sha256
load_assets = prior.load_assets
require = prior.require
get_perception_contract = prior.get_perception_contract
inherited_config_id = prior.inherited_config_id
AccessLedger = prior.AccessLedger
audit_perception_observation = prior.audit_perception_observation
read_shield_authority = prior.read_shield_authority
DIAGNOSTIC_SMOKE_BUDGET = prior.DIAGNOSTIC_SMOKE_BUDGET

Q50MaskedShieldedPerceptionEnv = prior.Q50MaskedShieldedPerceptionEnv
ScheduledMaskedPerceptionEnv = prior.ScheduledMaskedPerceptionEnv
build_masked_scheduled_perception_env = (
    prior.build_masked_scheduled_perception_env
)
build_masked_perception_eval_env = prior.build_masked_perception_eval_env
evaluate_masked_perception_deterministic = (
    prior.evaluate_masked_perception_deterministic
)

ModelBundle = prior.shielded.ModelBundle

UNSUPPORTED_NO_GSDE_KWARGS = ("use_sde", "sde_sample_freq")
EXPECTED_NO_GSDE_VALUES = {
    "use_sde": False,
    "sde_sample_freq": -1,
}


def masked_dependency_versions():
    return prior.masked_dependency_versions()


def verify_maskable_api_contract():
    """Verify the exact 2.7.1 masked API and the recovery premise."""
    base = prior.verify_maskable_api_contract()
    from sb3_contrib import MaskablePPO

    init = inspect.signature(MaskablePPO.__init__)
    require(
        all(name not in init.parameters for name in UNSUPPORTED_NO_GSDE_KWARGS),
        "Recovery premise: MaskablePPO.__init__ must not expose gSDE kwargs",
    )

    common = dict(get_protocol().common)
    require(
        common["use_sde"] is EXPECTED_NO_GSDE_VALUES["use_sde"]
        and common["sde_sample_freq"]
        == EXPECTED_NO_GSDE_VALUES["sde_sample_freq"],
        "Frozen protocol must retain exact no-gSDE semantics",
    )

    return {
        **base,
        "MaskablePPO_init_rejects_use_sde": True,
        "MaskablePPO_init_rejects_sde_sample_freq": True,
        "frozen_use_sde": common["use_sde"],
        "frozen_sde_sample_freq": common["sde_sample_freq"],
        "adapter_rule":
            "assert frozen no-gSDE values then omit only unsupported kwargs",
    }


def _split_maskable_common():
    """Return MaskablePPO kwargs after one mechanical compatibility omission."""
    p = get_protocol()
    common = dict(p.common)

    require(
        common.pop("algorithm") == "PPO"
        and common.pop("n_envs") == 1,
        "Frozen PPO family/single-env semantics",
    )
    require(
        common.pop("learning_rate_schedule")
        == common.pop("clip_schedule")
        == "NONE",
        "Frozen constant schedules",
    )
    policy = common.pop("policy")
    require(policy == "MlpPolicy", "Frozen MlpPolicy")

    frozen_use_sde = common.pop("use_sde")
    frozen_sde_sample_freq = common.pop("sde_sample_freq")
    require(
        frozen_use_sde is False
        and frozen_sde_sample_freq == -1,
        "Only exact frozen no-gSDE kwargs may be omitted",
    )

    # Guard against accidental future omission or mutation.
    expected_remaining = {
        key for key, _ in p.common
    } - {
        "algorithm",
        "n_envs",
        "learning_rate_schedule",
        "clip_schedule",
        "policy",
        "use_sde",
        "sde_sample_freq",
    }
    require(
        set(common) == expected_remaining,
        "Exactly two unsupported no-gSDE kwargs omitted; nothing else",
    )
    return policy, common


def verify_masked_model_contract(model, config_id, rl_seed):
    """Frozen PPO contract adapted only for MaskablePPO's unsupported gSDE API."""
    import torch
    from sb3_contrib import MaskablePPO

    verify_maskable_api_contract()
    p, c = get_protocol(), candidate(config_id)
    common = dict(p.common)
    arch = dict(p.architecture)

    require(
        isinstance(model, MaskablePPO),
        "Runtime algorithm must be sb3-contrib MaskablePPO",
    )

    # Supported PPO semantics must remain byte-for-byte numerically identical.
    for key in (
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range_vf",
        "normalize_advantage",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
        "target_kl",
    ):
        require(
            getattr(model, key) == common[key],
            "Model parameter mismatch: " + key,
        )

    require(
        common["use_sde"] is False
        and common["sde_sample_freq"] == -1,
        "Frozen no-gSDE semantics retained",
    )
    if hasattr(model, "use_sde"):
        require(model.use_sde is False, "Internal MaskablePPO gSDE must remain disabled")
    if hasattr(model, "sde_sample_freq"):
        require(
            model.sde_sample_freq == -1,
            "Internal MaskablePPO sde_sample_freq must retain disabled semantics",
        )

    require(
        model.learning_rate == c.learning_rate
        and model.n_steps == c.n_steps,
        "Frozen candidate parameters",
    )
    require(
        all(
            model.lr_schedule(progress) == c.learning_rate
            and model.clip_range(progress) == common["clip_range"]
            for progress in (0.0, 0.5, 1.0)
        ),
        "Frozen constant LR/clip",
    )
    require(
        model.policy.net_arch
        == {
            "pi": list(arch["actor"]),
            "vf": list(arch["critic"]),
        }
        and model.policy.activation_fn is torch.nn.Tanh
        and model.policy.ortho_init == arch["ortho_init"],
        "Frozen network",
    )
    require(
        model.seed == child_seeds(rl_seed)["learner"]
        and model.device.type == p.device
        and model.n_envs == common["n_envs"],
        "Frozen learner seed/device/n_envs",
    )
    return True


def build_masked_perception_ppo_model(env, config_id, rl_seed):
    """Construct MaskablePPO with only unsupported no-gSDE kwargs omitted."""
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
    require(
        env.action_space.n == 2
        and env.action_space.start == 0,
        "Discrete WAIT/CLEAN action space",
    )
    require(
        not p.observation_normalization
        and not p.reward_normalization,
        "Frozen normalization contract",
    )

    policy, compatible_common = _split_maskable_common()
    architecture = dict(p.architecture)
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
        **compatible_common,
    )
    verify_masked_model_contract(model, c.name, rl_seed)
    return ModelBundle(
        model=model,
        env=env,
        config_id=c.name,
        parent_seed=rl_seed,
    )


def train_masked_final_checkpoint(bundle, model_path, *, purpose):
    """Reuse frozen training/save logic; only constructor API was adapted."""
    verify_masked_model_contract(
        bundle.model, bundle.config_id, bundle.parent_seed
    )
    api = verify_maskable_api_contract()
    require(
        api["MaskablePPO_learn_default_use_masking"] is True,
        "MaskablePPO training must use action masking",
    )
    return prior.shielded.train_final_checkpoint(
        bundle, model_path, purpose=purpose
    )


def reload_masked_final_checkpoint(model_path, bundle):
    """Reload MaskablePPO and verify the adapted frozen contract exactly."""
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
    manifest = prior.masked_model_manifest(bundle, training)
    manifest.update(
        {
            "api_compatibility_recovery": "P2-2D-5AR-v1",
            "unsupported_kwargs_omitted":
                list(UNSUPPORTED_NO_GSDE_KWARGS),
            "frozen_no_gSDE_values":
                dict(EXPECTED_NO_GSDE_VALUES),
            "scientific_semantics_changed": False,
        }
    )
    return manifest
