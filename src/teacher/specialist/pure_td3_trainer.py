"""Reward-only deterministic TD3 training for one fixed aircraft."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.aircraft.sampler import PlantRecord
from src.envs.roll_rate_commands import (
    RandomCommandDistribution,
    RollRateCommand,
    RollRateCommandSequence,
    sample_mixed_duration_training_episode,
    sample_random_training_command,
    sample_random_training_sequence,
)
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.specialist.trainer import (
    SpecialistActorPolicy,
    SpecialistTrainingConfig,
    build_specialist_env,
    evaluate_specialist,
    specialist_quality_gate,
)
from src.teacher.td3.actor import DeterministicActor
from src.teacher.td3.teacher import PrivilegedTD3
from src.teacher.transitions import (
    continuing_task_contract,
    continuing_task_transition_flags,
)
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class PureRewardTD3Config:
    total_steps: int = 30_000
    warmup_steps: int = 3_000
    batch_size: int = 256
    replay_capacity: int = 50_000
    updates_per_step: int = 1
    exploration_std_initial: float = 0.20
    exploration_std_final: float = 0.02
    exploration_decay_steps: int = 27_000
    network_width: int = 128
    residual_blocks: int = 2
    gamma: float = 0.9995
    target_tau: float = 0.005
    target_policy_noise: float = 0.10
    target_noise_clip: float = 0.25
    policy_delay: int = 2
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    critic_warmup_updates: int = 500
    q_normalization_scale: float = 1.0
    maximum_q_coefficient: float = 1.0
    reward_multiplier: float = 1.0
    gradient_norm_limit: float = 10.0
    progress_interval_steps: int = 10_000
    evaluation_interval_steps: int = 0
    randomize_training_commands: bool = True
    random_command_sequence: bool = False
    random_sequence_segment_duration_range_s: tuple[float, float] = (2.0, 5.0)
    long_dwell_step_probability: float = 0.0
    long_dwell_duration_range_s: tuple[float, float] = (15.0, 30.0)
    random_command_distribution: RandomCommandDistribution = RandomCommandDistribution()
    seed: int = 20260828
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.updates_per_step,
            self.exploration_std_initial,
            self.exploration_decay_steps,
            self.network_width,
            self.residual_blocks,
            self.target_policy_noise,
            self.target_noise_clip,
            self.policy_delay,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.q_normalization_scale,
            self.maximum_q_coefficient,
            self.reward_multiplier,
            self.gradient_norm_limit,
            self.progress_interval_steps,
        )
        if min(positive) <= 0:
            raise ValueError(
                "invalid pure-reward TD3 dimensions or optimization settings"
            )
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("TD3 warmup_steps must be in [0, total_steps)")
        if self.critic_warmup_updates < 0:
            raise ValueError("TD3 critic_warmup_updates cannot be negative")
        if self.replay_capacity < self.batch_size:
            raise ValueError("TD3 replay capacity must fit one batch")
        if not 0 <= self.exploration_std_final <= self.exploration_std_initial:
            raise ValueError("TD3 exploration noise must decay to a nonnegative value")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("invalid TD3 discount or target update rate")
        if self.evaluation_interval_steps < 0:
            raise ValueError("TD3 evaluation interval cannot be negative")
        segment_min_s, segment_max_s = self.random_sequence_segment_duration_range_s
        if segment_min_s <= 0 or segment_min_s > segment_max_s:
            raise ValueError("TD3 random sequence segment duration range is invalid")
        dwell_min_s, dwell_max_s = self.long_dwell_duration_range_s
        if dwell_min_s <= 0 or dwell_min_s > dwell_max_s:
            raise ValueError("TD3 long-dwell duration range is invalid")
        if not 0.0 <= self.long_dwell_step_probability <= 1.0:
            raise ValueError("TD3 long-dwell step probability must be in [0, 1]")
        if self.random_command_sequence and not self.randomize_training_commands:
            raise ValueError("random command sequences require randomized commands")
        if self.long_dwell_step_probability and not self.random_command_sequence:
            raise ValueError("long-dwell mixing requires random command sequences")

    def exploration_std(self, step: int) -> float:
        progress = np.clip(
            (step - self.warmup_steps) / self.exploration_decay_steps,
            0.0,
            1.0,
        )
        return float(
            self.exploration_std_initial
            + progress * (self.exploration_std_final - self.exploration_std_initial)
        )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _plant_payload(record: PlantRecord) -> dict[str, object]:
    return {
        "plant_id": record.plant_id,
        "split": record.split,
        "quality_region": record.quality_region,
        "aircraft_class": record.aircraft_class,
        "flight_phase": record.flight_phase,
        "parameters": asdict(record.parameters),
    }


def _validation_point(
    evaluation: dict[str, object],
    *,
    step: int,
    updates: int,
    actor_updates: int,
    episodes: int,
) -> dict[str, object]:
    rows = evaluation["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("TD3 validation requires non-empty evaluation rows")
    teacher_metrics = [row["teacher"] for row in rows]
    episode_costs = np.asarray(
        [metrics["episode_cost"] for metrics in teacher_metrics], dtype=float
    )
    return {
        "step": step,
        "updates": updates,
        "actor_updates": actor_updates,
        "episodes": episodes,
        "mean_episode_cost": float(np.mean(episode_costs)),
        "median_episode_cost": float(np.median(episode_costs)),
        "mean_tracking_rmse_deg_s": evaluation["mean_teacher_tracking_rmse_deg_s"],
        "maximum_peak_error_deg_s": evaluation["maximum_teacher_peak_error_deg_s"],
        "mean_requested_force_total_variation_n": evaluation[
            "mean_teacher_requested_force_total_variation_n"
        ],
        "mean_force_saturation_fraction": evaluation[
            "mean_teacher_force_saturation_fraction"
        ],
        "tracking_improvement_rate": evaluation["tracking_improvement_rate"],
        "harm_rate": evaluation["harm_rate"],
    }


def train_pure_reward_td3(
    record: PlantRecord,
    output_dir: str | Path,
    environment_config: SpecialistTrainingConfig,
    td3_config: PureRewardTD3Config = PureRewardTD3Config(),
    *,
    library_path: str | Path | None = None,
) -> dict[str, object]:
    """Train a direct-force deterministic policy using reward and no PID signal."""

    if environment_config.history_steps != 0:
        raise ValueError(
            "pure-reward TD3 uses controller state and action memory, with no raw history window"
        )
    if not environment_config.include_actor_actuator_state:
        raise ValueError(
            "pure-reward TD3 Actor must observe commanded and applied force"
        )
    required_action_steps = (
        int(np.ceil(record.parameters.tau_p / environment_config.policy_dt_s)) + 1
    )
    if environment_config.requested_action_history_steps < required_action_steps:
        raise ValueError(
            "pure-reward TD3 requested-action history does not cover the plant delay: "
            f"need at least {required_action_steps} policy steps"
        )
    if environment_config.critic_include_episode_progress:
        raise ValueError(
            "pure-reward TD3 Critic must exclude artificial episode progress"
        )
    if (
        td3_config.random_command_sequence
        and environment_config.critic_include_command_context
    ):
        raise ValueError(
            "random command sequences require the Critic to exclude future command context"
        )
    if (
        td3_config.long_dwell_step_probability
        and td3_config.long_dwell_duration_range_s[1]
        > environment_config.episode_duration_s
    ):
        raise ValueError("long-dwell duration range must fit inside the episode")
    device = torch.device(td3_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(td3_config.seed)
    rng = np.random.default_rng(td3_config.seed)
    command_rng = np.random.default_rng(td3_config.seed + 1_000_003)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()

    deployment_config = replace(
        environment_config,
        total_steps=td3_config.total_steps,
        warmup_steps=td3_config.warmup_steps,
        batch_size=td3_config.batch_size,
        replay_capacity=td3_config.replay_capacity,
        network_width=td3_config.network_width,
        residual_blocks=td3_config.residual_blocks,
        gamma=td3_config.gamma,
        target_tau=td3_config.target_tau,
        learning_rate=td3_config.actor_learning_rate,
        enforce_odd_policy=True,
        odd_policy_projection_stage="training",
        seed=td3_config.seed,
        device=td3_config.device,
    )
    environment = build_specialist_env(record, deployment_config)
    actor_observation, info = environment.reset(seed=td3_config.seed)
    critic_observation = np.asarray(info["critic_state"], dtype=np.float32)
    actor_observation_contract = environment.actor_observation_contract()
    critic_observation_contract = environment.critic_observation_contract()
    if actor_observation_contract["uses_raw_history_window"]:
        raise RuntimeError("pure-reward TD3 Actor contract cannot use raw history")

    actor = DeterministicActor(
        actor_observation.size,
        1,
        width=td3_config.network_width,
        residual_blocks=td3_config.residual_blocks,
        enforce_odd_symmetry=True,
    )
    initial_actor = {
        name: value.detach().cpu().clone() for name, value in actor.state_dict().items()
    }
    learner = PrivilegedTD3(
        actor_observation.size,
        critic_observation.size,
        1,
        gamma=td3_config.gamma,
        tau=td3_config.target_tau,
        target_policy_noise=td3_config.target_policy_noise,
        target_noise_clip=td3_config.target_noise_clip,
        policy_delay=td3_config.policy_delay,
        actor_learning_rate=td3_config.actor_learning_rate,
        critic_learning_rate=td3_config.critic_learning_rate,
        gradient_norm_limit=td3_config.gradient_norm_limit,
        reward_scale=td3_config.reward_multiplier,
        behavior_regularization_weight=0.0,
        q_normalization_scale=td3_config.q_normalization_scale,
        maximum_q_coefficient=td3_config.maximum_q_coefficient,
        actor=actor,
        critic_width=td3_config.network_width,
        critic_residual_blocks=td3_config.residual_blocks,
        device=device,
    )
    replay = TwoStreamReplayBuffer(
        td3_config.replay_capacity,
        actor_observation.size,
        critic_observation.size,
        1,
        seed=td3_config.seed,
    )

    command_order = np.empty(0, dtype=int)
    command_position = 0
    sampled_durations_s: list[float] = []
    sampled_segment_durations_s: list[float] = []
    sampled_amplitudes_deg_s: list[float] = []
    sampled_frequencies_hz: list[float] = []
    sampled_segment_counts: Counter[str] = Counter()
    sampled_profile_type_counts: Counter[str] = Counter()

    def record_sampled_profile(profile: RollRateCommand) -> None:
        sampled_durations_s.append(profile.duration_s)
        profile_type = "single_random_command"
        if isinstance(profile, RollRateCommandSequence):
            profile_type = "short_random_sequence"
        elif profile.command_id.startswith("random-long-dwell-"):
            profile_type = "long_dwell_step"
        sampled_profile_type_counts[profile_type] += 1
        segments = (
            profile.segments
            if isinstance(profile, RollRateCommandSequence)
            else (profile,)
        )
        for segment in segments:
            sampled_segment_counts[segment.kind] += 1
            sampled_segment_durations_s.append(segment.duration_s)
            if segment.kind == "multisine":
                sampled_amplitudes_deg_s.append(
                    float(
                        sum(
                            abs(amplitude)
                            for amplitude, _ in segment.multisine_components
                        )
                    )
                )
                sampled_frequencies_hz.extend(
                    frequency for _, frequency in segment.multisine_components
                )
            else:
                sampled_amplitudes_deg_s.append(abs(segment.amplitude_deg_s))
                if segment.frequency_hz is not None:
                    sampled_frequencies_hz.append(segment.frequency_hz)

    def reset_training_episode(
        episode_index: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal command_order, command_position
        if td3_config.randomize_training_commands:
            if td3_config.random_command_sequence:
                if td3_config.long_dwell_step_probability:
                    profile = sample_mixed_duration_training_episode(
                        command_rng,
                        episode_index,
                        policy_dt_s=environment_config.policy_dt_s,
                        duration_s=environment_config.episode_duration_s,
                        long_dwell_step_probability=(
                            td3_config.long_dwell_step_probability
                        ),
                        short_segment_duration_range_s=(
                            td3_config.random_sequence_segment_duration_range_s
                        ),
                        long_dwell_duration_range_s=(
                            td3_config.long_dwell_duration_range_s
                        ),
                        config=td3_config.random_command_distribution,
                    )
                else:
                    profile = sample_random_training_sequence(
                        command_rng,
                        episode_index,
                        policy_dt_s=environment_config.policy_dt_s,
                        duration_s=environment_config.episode_duration_s,
                        segment_duration_range_s=(
                            td3_config.random_sequence_segment_duration_range_s
                        ),
                        config=td3_config.random_command_distribution,
                    )
            else:
                profile = sample_random_training_command(
                    command_rng,
                    episode_index,
                    policy_dt_s=environment_config.policy_dt_s,
                    config=td3_config.random_command_distribution,
                )
            record_sampled_profile(profile)
            return environment.reset(
                seed=td3_config.seed + episode_index,
                options={"command_profile": profile},
            )
        if command_position >= len(command_order):
            command_order = command_rng.permutation(len(environment.command_profiles))
            command_position = 0
        command_index = int(command_order[command_position])
        command_position += 1
        return environment.reset(
            seed=td3_config.seed + episode_index,
            options={"command_index": command_index},
        )

    actor_observation, info = reset_training_episode(0)
    critic_observation = np.asarray(info["critic_state"], dtype=np.float32)
    episodes = 0
    updates = 0
    actor_updates = 0
    command_counts: Counter[str] = Counter()
    losses: dict[str, float] = {}
    validation_history: list[dict[str, object]] = []
    validation_path = destination / "learning_curve.json"
    validation_checkpoint_dir = destination / "validation_checkpoints"
    if td3_config.evaluation_interval_steps:
        validation_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def persist_validation(
        evaluation_result: dict[str, object],
        completed_steps: int,
        *,
        save_actor: bool,
    ) -> dict[str, object]:
        point = _validation_point(
            evaluation_result,
            step=completed_steps,
            updates=updates,
            actor_updates=actor_updates,
            episodes=episodes,
        )
        validation_history.append(point)
        _write_json(
            validation_path,
            {
                "schema_version": "pure_reward_td3_learning_curve_v1",
                "plant_id": record.plant_id,
                "fixed_validation_episode_duration_s": (
                    deployment_config.episode_duration_s
                ),
                "points": validation_history,
            },
        )
        if save_actor:
            _atomic_torch_save(
                {
                    "schema_version": "pure_reward_td3_validation_actor_v1",
                    "algorithm": "pure_reward_td3",
                    "plant": _plant_payload(record),
                    "config": asdict(deployment_config),
                    "td3_config": asdict(td3_config),
                    "step": completed_steps,
                    "validation": point,
                    "actor_observation_dim": int(actor_observation.size),
                    "actor_observation_contract": actor_observation_contract,
                    "actor_architecture": DeterministicActor.architecture_name,
                    "actor": {
                        name: value.detach().cpu()
                        for name, value in learner.actor.state_dict().items()
                    },
                },
                validation_checkpoint_dir / f"actor_step_{completed_steps:08d}.pt",
            )
        return point

    def run_periodic_validation(completed_steps: int) -> dict[str, object]:
        validation_policy = SpecialistActorPolicy(learner.actor, learner.device)
        validation_result = evaluate_specialist(
            validation_policy,
            record,
            deployment_config,
            controller_label="teacher",
            controller_display_label="Reward-only TD3",
        )
        return persist_validation(
            validation_result,
            completed_steps,
            save_actor=True,
        )

    latest_validation: dict[str, object] | None = None
    if td3_config.evaluation_interval_steps:
        latest_validation = run_periodic_validation(0)
    training_started = time.perf_counter()
    for step in range(td3_config.total_steps):
        if step < td3_config.warmup_steps:
            action = rng.uniform(-1.0, 1.0, size=(1,))
            exploration_std = None
        else:
            action = learner.act(
                torch.as_tensor(actor_observation).unsqueeze(0)
            ).numpy()[0]
            exploration_std = td3_config.exploration_std(step)
            action = action + rng.normal(0.0, exploration_std, size=action.shape)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        next_actor_observation, reward, terminated, truncated, next_info = (
            environment.step(action)
        )
        episode_done, bellman_terminal = continuing_task_transition_flags(
            terminated, truncated
        )
        next_critic_observation = np.asarray(
            next_info["critic_state"], dtype=np.float32
        )
        replay.add(
            actor_observation,
            critic_observation,
            action,
            reward,
            next_actor_observation,
            next_critic_observation,
            bellman_terminal,
        )
        actor_observation = next_actor_observation
        critic_observation = next_critic_observation
        if step >= td3_config.warmup_steps and len(replay) >= td3_config.batch_size:
            for _ in range(td3_config.updates_per_step):
                losses = learner.update(
                    replay.sample(td3_config.batch_size, learner.device),
                    allow_actor_update=(updates >= td3_config.critic_warmup_updates),
                )
                updates += 1
                actor_updates += int(losses["actor_updated"])
        if episode_done:
            command_counts[str(next_info["command_kind"])] += 1
            episodes += 1
            actor_observation, info = reset_training_episode(episodes)
            critic_observation = np.asarray(info["critic_state"], dtype=np.float32)
        completed_steps = step + 1
        if (
            td3_config.evaluation_interval_steps
            and completed_steps % td3_config.evaluation_interval_steps == 0
            and completed_steps < td3_config.total_steps
        ):
            latest_validation = run_periodic_validation(completed_steps)
        if completed_steps % td3_config.progress_interval_steps == 0:
            _write_json(
                destination / "progress.json",
                {
                    "status": "training",
                    "algorithm": "pure_reward_td3",
                    "plant_id": record.plant_id,
                    "step": completed_steps,
                    "updates": updates,
                    "actor_updates": actor_updates,
                    "episodes": episodes,
                    "exploration_std": exploration_std,
                    "elapsed_s": time.perf_counter() - training_started,
                    "last_losses": losses,
                    "latest_validation": latest_validation,
                },
            )
    training_elapsed_s = time.perf_counter() - training_started

    policy = SpecialistActorPolicy(learner.actor, learner.device)
    evaluation = evaluate_specialist(
        policy,
        record,
        deployment_config,
        output_dir=destination,
        controller_label="teacher",
        controller_display_label="Reward-only TD3",
    )
    latest_validation = persist_validation(
        evaluation,
        td3_config.total_steps,
        save_actor=bool(td3_config.evaluation_interval_steps),
    )
    best_validation = min(
        validation_history,
        key=lambda point: float(point["mean_episode_cost"]),
    )
    quality_gate = specialist_quality_gate(evaluation, deployment_config)
    actor_delta_squared = 0.0
    for name, value in learner.actor.state_dict().items():
        actor_delta_squared += float(
            (value.detach().cpu() - initial_actor[name]).square().sum()
        )
    actor_delta_l2 = float(np.sqrt(actor_delta_squared))
    quality_gate["checks"]["online_actor_updates"] = actor_updates > 0
    quality_gate["checks"]["actor_changed_by_rl"] = actor_delta_l2 > 1e-12
    quality_gate["passed"] = bool(
        quality_gate["passed"]
        and quality_gate["checks"]["online_actor_updates"]
        and quality_gate["checks"]["actor_changed_by_rl"]
    )
    quality_gate["observed"]["online_actor_updates"] = actor_updates
    quality_gate["observed"]["rl_actor_parameter_delta_l2"] = actor_delta_l2

    source = git_source_revision()
    library = None
    if library_path is not None:
        library_source = Path(library_path)
        library = {
            "path": str(library_source.resolve()),
            "sha256": sha256_file(library_source),
        }
    training_contract = {
        "algorithm": "twin_delayed_deep_deterministic_policy_gradient",
        "supervision": "environment_reward_only",
        "uses_pid_demonstrations": False,
        "uses_behavior_cloning": False,
        "uses_embedded_control_prior": False,
        "uses_pid_regularization": False,
        "uses_entropy_regularization": False,
        "actor_action": "normalized_direct_full_F_as",
    }
    time_limit_contract = continuing_task_contract(
        critic_includes_episode_progress=(
            deployment_config.critic_include_episode_progress
        )
    )
    command_sampling_contract = {
        "scheduler": (
            "mixed_long_dwell_step_and_random_sequence_v1"
            if td3_config.long_dwell_step_probability
            else (
                "continuous_random_command_sequence_v1"
                if td3_config.random_command_sequence
                else (
                    "continuous_random_command_distribution_v1"
                    if td3_config.randomize_training_commands
                    else "seeded_permutation_without_replacement"
                )
            )
        ),
        "randomized": td3_config.randomize_training_commands,
        "continuous_within_episode": td3_config.random_command_sequence,
        "long_dwell_step_probability": td3_config.long_dwell_step_probability,
        "long_dwell_duration_range_s": list(
            td3_config.long_dwell_duration_range_s
        ),
        "distribution": asdict(td3_config.random_command_distribution),
        "sampled_profile_count": len(sampled_durations_s),
        "sampled_profile_type_counts": dict(sampled_profile_type_counts),
        "sampled_duration_s": {
            "minimum": min(sampled_durations_s, default=None),
            "mean": (
                float(np.mean(sampled_durations_s)) if sampled_durations_s else None
            ),
            "maximum": max(sampled_durations_s, default=None),
        },
        "sampled_segment_count": len(sampled_segment_durations_s),
        "sampled_segment_kind_counts": dict(sampled_segment_counts),
        "sampled_segment_duration_s": {
            "minimum": min(sampled_segment_durations_s, default=None),
            "mean": (
                float(np.mean(sampled_segment_durations_s))
                if sampled_segment_durations_s
                else None
            ),
            "maximum": max(sampled_segment_durations_s, default=None),
        },
        "sampled_absolute_amplitude_deg_s": {
            "minimum": min(sampled_amplitudes_deg_s, default=None),
            "mean": (
                float(np.mean(sampled_amplitudes_deg_s))
                if sampled_amplitudes_deg_s
                else None
            ),
            "maximum": max(sampled_amplitudes_deg_s, default=None),
        },
        "sampled_frequency_hz": {
            "minimum": min(sampled_frequencies_hz, default=None),
            "mean": (
                float(np.mean(sampled_frequencies_hz))
                if sampled_frequencies_hz
                else None
            ),
            "maximum": max(sampled_frequencies_hz, default=None),
        },
    }
    odd_policy_contract = {
        "enabled": True,
        "projection_stage": "td3_training_and_inference",
        "applied_during_td3_training": True,
        "applied_to_deterministic_teacher": True,
        "applied_to_distillation_labels": True,
    }
    actor_payload = {
        "schema_version": "specialist_actor_v1",
        "algorithm": "pure_reward_td3",
        "actor_architecture": DeterministicActor.architecture_name,
        "plant": _plant_payload(record),
        "config": asdict(deployment_config),
        "td3_config": asdict(td3_config),
        "actor_observation_dim": int(actor_observation.size),
        "actor_observation_contract": actor_observation_contract,
        "critic_observation_contract": critic_observation_contract,
        "training_contract": training_contract,
        "continuing_task_contract": time_limit_contract,
        "command_sampling_contract": command_sampling_contract,
        "odd_policy_contract": odd_policy_contract,
        "actor": {
            name: value.detach().cpu()
            for name, value in learner.actor.state_dict().items()
        },
        "source": source,
        "library": library,
    }
    actor_path = destination / "teacher_actor.pt"
    checkpoint_path = destination / "training_checkpoint.pt"
    best_validation_actor_path = (
        validation_checkpoint_dir
        / f"actor_step_{int(best_validation['step']):08d}.pt"
        if td3_config.evaluation_interval_steps
        else actor_path
    )
    _atomic_torch_save(actor_payload, actor_path)
    _atomic_torch_save(
        {
            **actor_payload,
            "schema_version": "pure_reward_td3_checkpoint_v1",
            "critic": learner.critic.state_dict(),
            "target_actor": learner.target_actor.state_dict(),
            "target_critic": learner.target_critic.state_dict(),
            "actor_optimizer": learner.actor_optimizer.state_dict(),
            "critic_optimizer": learner.critic_optimizer.state_dict(),
            "updates": updates,
            "actor_updates": actor_updates,
        },
        checkpoint_path,
    )
    report: dict[str, object] = {
        "status": "complete",
        "algorithm": "pure_reward_td3",
        "actor_architecture": DeterministicActor.architecture_name,
        "plant_id": record.plant_id,
        "plant": actor_payload["plant"],
        "config": asdict(deployment_config),
        "environment_config": asdict(deployment_config),
        "td3_config": asdict(td3_config),
        "source": source,
        "library": library,
        "actor_receives_theta": False,
        "actor_uses_raw_history": False,
        "actor_uses_requested_action_history": True,
        "privileged_critic_training_only": True,
        "action_definition": "direct_full_F_as",
        "training_contract": training_contract,
        "continuing_task_contract": time_limit_contract,
        "command_sampling_contract": command_sampling_contract,
        "actor_observation_dim": int(actor_observation.size),
        "actor_observation_contract": actor_observation_contract,
        "critic_observation_dim": int(critic_observation.size),
        "critic_observation_contract": critic_observation_contract,
        "odd_policy_contract": odd_policy_contract,
        "parameter_counts": learner.parameter_counts(),
        "steps": td3_config.total_steps,
        "updates": updates,
        "actor_updates": actor_updates,
        "rl_actor_parameter_delta_l2": actor_delta_l2,
        "episodes": episodes,
        "training_command_counts": dict(command_counts),
        "training_segment_counts": dict(sampled_segment_counts),
        "training_command_scheduler": command_sampling_contract["scheduler"],
        "training_elapsed_s": training_elapsed_s,
        "total_elapsed_s": time.perf_counter() - run_started,
        "last_losses": losses,
        "learning_curve": validation_history,
        "best_validation": best_validation,
        "evaluation": evaluation,
        "quality_gate": quality_gate,
        "accepted_for_distillation": quality_gate["passed"],
        "actor_checkpoint": str(actor_path),
        "training_checkpoint": str(checkpoint_path),
        "best_validation_actor_checkpoint": str(best_validation_actor_path),
        "artifacts": {
            "final_evaluation": str(destination / "evaluation.json"),
            "response_plot": str(destination / "response_comparison.png"),
            "all_commands_plot": str(destination / "all_evaluation_commands.png"),
            "learning_curve": str(validation_path),
            "validation_checkpoints": (
                str(validation_checkpoint_dir)
                if td3_config.evaluation_interval_steps
                else None
            ),
            "best_validation_actor_checkpoint": str(best_validation_actor_path),
        },
    }
    _write_json(destination / "report.json", report)
    _write_json(
        destination / "progress.json",
        {
            "status": "complete",
            "algorithm": "pure_reward_td3",
            "plant_id": record.plant_id,
            "step": td3_config.total_steps,
            "updates": updates,
            "actor_updates": actor_updates,
            "latest_validation": latest_validation,
            "best_validation": best_validation,
            "accepted_for_distillation": quality_gate["passed"],
            "actor_checkpoint": str(actor_path),
            "report": str(destination / "report.json"),
        },
    )
    return report
