"""P2-2C-0-v1 immutable pre-registration; importing performs no experiment or RNG work.

Any core protocol change after Formal Freeze requires a new protocol version.
This module describes future training; it does not implement training or perception.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class RootPartition:
    role: str
    roots: tuple[int, int]
    years: tuple[str, str] = ("YEAR1", "YEAR2")
    trajectory_ids: tuple[int, ...] = tuple(range(300))
    population_size: int = 600


@dataclass(frozen=True)
class PPOCandidate:
    name: str
    learning_rate: float
    n_steps: int


@dataclass(frozen=True)
class SeedRegistry:
    development: tuple[int, ...] = (510001, 510002)
    final: tuple[int, ...] = (520001, 520002, 520003, 520004, 520005)
    perception_seed: None = None
    namespace: str = "RL only; disjoint from environment roots and perception seeds"
    derivation: str = "np.random.SeedSequence(rl_seed).spawn(2)"
    child_order: tuple[str, str] = ("learner", "training_schedule")
    integer_conversion: str = "int(child.generate_state(1, dtype=np.uint32)[0])"
    learner_consumers: tuple[str, ...] = ("random", "numpy", "torch", "SB3")


@dataclass(frozen=True)
class SelectionContract:
    population: str = "8 runs: CONFIG_A-D x development RL seeds; TRAINING roots only"
    evaluation: str = "final checkpoint of each run; full 600 DEVELOPMENT specs; deterministic=True"
    primary_metric: str = "mean of the two seed-level development mean_J_total values; minimize"
    atol: float = 1e-12
    rtol: float = 0.0
    tie_breaks: tuple[str, ...] = ("lower worst_seed_dev_mean_J_total", "CONFIG_A", "CONFIG_B", "CONFIG_C", "CONFIG_D")
    tie_algorithm: str = "retain scores within atol of minimum; retain worst-seed scores within atol of their minimum; candidate order"
    checkpoint: str = "FINAL TRAINING CHECKPOINT ONLY"
    development_checkpoint: int = 491520
    final_checkpoint: int = 983040
    early_stopping: bool = False
    best_model_selection: bool = False
    diagnostics: str = "intermediate checkpoints and training curves diagnostic only; no evaluation-based rollback"
    prohibited_selection: tuple[str, ...] = ("held-out", "Oracle distance", "cleaning count", "single year", "one lucky seed", "P95 only", "CVaR only", "visual preference", "training-root performance")


@dataclass(frozen=True)
class EvaluationContract:
    deterministic: bool = True
    episodes_per_model: int = 600
    metrics: tuple[str, ...] = ("mean_J_total", "std_J_total", "median_J_total", "P95_J_total", "CVaR95_J_total", "mean_N_clean")
    threshold_baseline: str = "TRUE_THRESHOLD_0.05"
    threshold_mean: float = 7.926837776021219
    tolerance_multiplier: float = 1.03
    gate_a_max: float = 8.164642909301856
    periodic_baseline: str = "PERIODIC_14"
    periodic_mean: float = 10.126875738879782
    gate_a: str = "mean of all 5 final seeds' DEVELOPMENT mean_J_total <= gate_a_max"
    gate_b: str = "5/5 final seeds' DEVELOPMENT mean_J_total < periodic_mean"
    gate_failure: str = "STOP BEFORE POINT/UA; diagnose implementation/convergence; bug fixes require audit; protocol changes require new version"
    gate_meaning: str = "near strong causal threshold and better than periodic; Oracle excluded; no required Oracle or significant threshold superiority"
    formal_prerequisites: tuple[str, ...] = ("selected config frozen", "all 5 final training runs complete", "final model SHA256 frozen", "development sanity gate passed", "training/development artifacts Git checkpointed")
    formal_access: str = "each fixed model once on all 600 FORMAL HELD-OUT specs; deterministic=True"
    prior_access: str = "formal roots used by frozen non-RL baseline and clairvoyant Oracle; no PPO performance observed before freeze"
    formal_prohibitions: tuple[str, ...] = ("hyperparameter tuning", "budget selection", "checkpoint selection", "architecture selection", "performance debugging", "early stopping", "entropy tuning", "reward tuning", "seed replacement", "bad-seed deletion", "retraining to overwrite formal claim")
    after_formal: str = "retain and report original result; any further optimization requires new protocol AND new confirmatory roots; old roots are not untouched"
    replication_unit: str = "RL TRAINING SEED"
    replication_n: int = 5
    primary_report: tuple[str, ...] = ("each seed mean_J_total over 600 held-out trajectories", "5-seed mean", "5-seed std", "5-seed median", "5-seed range")
    secondary_report: str = "within-fixed-policy paired environment distributions allowed; distinguish environment randomness from training randomness; never n=3000 independent algorithm replications"


@dataclass(frozen=True)
class Protocol:
    stage: str = "P2-2C-0-v1"
    title: str = "True-State PPO Development and Formal Evaluation Protocol Freeze"
    version: str = "v1"
    mutation_rule: str = "core protocol changes after Formal Freeze require explicit new protocol version; never silently edit v1"
    role: str = "CAUSAL_FULL_STATE_RL_BENCHMARK"
    observations: tuple = (("True-State", ("L_true", "sin_DOY", "cos_DOY")), ("Point", ("q50", "sin_DOY", "cos_DOY")), ("UA", ("q50", "width", "sin_DOY", "cos_DOY")))
    observation_dimensions: tuple = (("True-State", 3), ("Point", 3), ("UA", 4))
    forbidden_true_state_information: tuple = ("future GHI", "current/future rain indicator", "future innovation", "historical clean indicator", "Oracle information", "q50", "width")
    actions: tuple = ((0, "WAIT"), (1, "CLEAN"))
    eta: float = 0.95
    lambda_c: float = 0.19925428989934618
    reward_reference: str = "frozen P2-1D: p2_1d_3_final_reward_contract_audit_v1/formal/reward_contract_manifest.json; core paper2_gym_pomdp_env_v1.py"
    reward_scale: float = 1.0
    reward_prohibitions: tuple = ("reward shaping", "bonus reward", "Oracle-distance reward", "imitation reward (including threshold imitation)", "action penalty outside frozen cleaning cost", "reward normalization", "performance-tuned reward scaling")
    daily_order: tuple = ("L_pre", "observation", "action", "L_post", "same-day reward settlement", "natural transition", "next state")
    decision_reward_steps: int = 365
    natural_transitions: int = 364
    partitions: tuple[RootPartition, ...] = (RootPartition("TRAINING", (1320001, 1330001)), RootPartition("DEVELOPMENT", (1020001, 1030001)), RootPartition("FORMAL HELD-OUT", (1220001, 1230001)))
    partition_uses: tuple = (("TRAINING", "gradient training only"), ("DEVELOPMENT", "hyperparameter selection; implementation sanity; performance gate; NEVER gradient training"), ("FORMAL HELD-OUT", "FINAL PPO EVALUATION ONLY"))
    seeds: SeedRegistry = SeedRegistry()
    schedule: tuple = (("canonical_order", "YEAR1 tids 0..299 then YEAR2 tids 0..299"), ("cycle", "each of 600 specs exactly once without replacement"), ("rng", "np.random.default_rng(schedule child integer seed); permutation(600); same persistent RNG across cycles"), ("fairness", "same RL seed implies identical episode order across configs and True-State/Point/UA"), ("forbidden", "replacement sampling; config-dependent schedule; any curriculum; performance-dependent sampling; held-out insertion"), ("budget_end", "stop at exact interaction budget even within episode/cycle; no padding"))
    architecture: tuple = (("policy", "MlpPolicy"), ("actor", (64, 64)), ("critic", (64, 64)), ("activation", "Tanh"), ("ortho_init", True))
    common: tuple = (("algorithm", "PPO"), ("policy", "MlpPolicy"), ("n_envs", 1), ("batch_size", 256), ("n_epochs", 10), ("gamma", 1.0), ("gae_lambda", 0.95), ("clip_range", 0.2), ("clip_range_vf", None), ("normalize_advantage", True), ("ent_coef", 0.0), ("vf_coef", 0.5), ("max_grad_norm", 0.5), ("use_sde", False), ("sde_sample_freq", -1), ("target_kl", None), ("learning_rate_schedule", "NONE"), ("clip_schedule", "NONE"), ("device", "cpu"))
    gamma_rationale: str = "gamma=1.0 aligns PPO with finite 365-day UNDISCOUNTED annual total cost; not SB3 default gamma=0.99"
    observation_normalization: bool = False
    reward_normalization: bool = False
    normalization_rationale: str = "frozen observation scales controlled; avoid train/test normalization-statistics leakage; no VecNormalize"
    device: str = "cpu"
    torch_num_threads: int = 1
    determinism: str = "seed random/numpy/torch/SB3 with learner child; no automatic GPU; repeatable same machine/environment/seed; no cross-hardware bitwise guarantee"
    candidates: tuple[PPOCandidate, ...] = (PPOCandidate("CONFIG_A", 3e-4, 2048), PPOCandidate("CONFIG_B", 1e-4, 2048), PPOCandidate("CONFIG_C", 3e-4, 4096), PPOCandidate("CONFIG_D", 1e-4, 4096))
    candidate_rule: str = "only learning_rate and n_steps differ; no additional entropy/gamma/network/batch/activation/clip/optimizer grid without new version"
    development_budget: int = 491520
    final_budget: int = 983040
    budget_rule: str = "exact environment interactions per run; no wall-clock stopping, early stopping or adaptive training budget; final 5 seeds trained on TRAINING only"
    selection: SelectionContract = SelectionContract()
    evaluation: EvaluationContract = EvaluationContract()
    inheritance: tuple = ("training roots", "development roots", "formal roots", "RL seed sets", "schedule algorithm", "network architecture", "selected learning rate", "selected n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda", "clip_range", "entropy", "value coefficient", "max grad norm", "training budgets", "final-checkpoint-only rule", "deterministic evaluation rule")
    inheritance_rule: str = "True-State selected config C* is common Point/UA backbone; no PPO hyperparameter reselection or larger network; only observation information/input dimension differs; no perception implemented in P2-2C-0"
    width_ablation: str = "mandatory after UA main results: within-q50-bin width shuffle ablation; destroy width/sample-error pairing while preserving q50 regime and width marginal distribution as far as possible; not run in P2-2C-0"
    dependencies: tuple = (("Python", "3.11.15"), ("gymnasium", "1.2.3"), ("stable_baselines3", "2.7.1"), ("torch", "2.10.0+cu130"), ("numpy", "2.3.5"), ("pandas", "2.3.3"))


def protocol_payload(protocol: Protocol = Protocol()) -> dict:
    """Detached JSON-compatible metadata; no shared mutable containers."""
    return json.loads(json.dumps(asdict(protocol), allow_nan=False))


def derive_rl_subseeds(rl_seed: int) -> dict:
    """Local SeedSequence only; does not seed or mutate any global generator."""
    import numpy as np
    if type(rl_seed) is not int or rl_seed not in SeedRegistry().development + SeedRegistry().final:
        raise ValueError("Unregistered RL seed")
    children = np.random.SeedSequence(rl_seed).spawn(2)
    return {"parent_seed": rl_seed, "conversion": SeedRegistry().integer_conversion,
            "children": [{"role": role, "spawn_key": list(child.spawn_key),
                          "integer_seed": int(child.generate_state(1, dtype=np.uint32)[0])}
                         for role, child in zip(SeedRegistry().child_order, children)]}


def canonical_training_specs() -> tuple:
    """Metadata tuples (year, root, trajectory_id, population_size, perception_seed)."""
    partition = Protocol().partitions[0]
    return tuple((year, root, tid, 300, None)
                 for year, root in zip(partition.years, partition.roots)
                 for tid in partition.trajectory_ids)


def validate_protocol(protocol: Protocol = Protocol()) -> None:
    """Reject v1 overrides; separate audit supplies independent literal checks."""
    if protocol_payload(protocol) != protocol_payload(Protocol()):
        raise ValueError("v1 is immutable; create a new protocol version")
    roots = [set(p.roots) for p in protocol.partitions]
    if any(roots[i] & roots[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping environment roles")
    seeds = protocol.seeds.development + protocol.seeds.final
    if len(set(seeds)) != 7 or set(seeds) & set.union(*roots):
        raise ValueError("Seed namespace collision")
    if any(b % c.n_steps for b in (protocol.development_budget, protocol.final_budget) for c in protocol.candidates):
        raise ValueError("Inexact interaction budget")
