"""PID-guided deterministic TD3 training for one fixed aircraft."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.aircraft.sampler import PlantRecord
from src.controllers.pid import PIDGains, RollRatePIDPolicy
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.specialist.trainer import (
    SpecialistActorPolicy,
    SpecialistTrainingConfig,
    build_specialist_env,
    evaluate_specialist,
    specialist_quality_gate,
)
from src.teacher.td3.actor import PIDResidualActor
from src.teacher.td3.teacher import PrivilegedTD3
from src.teacher.transitions import (
    continuing_task_contract,
    continuing_task_transition_flags,
)
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class PIDGuidedTD3Config:
    total_steps: int = 5_000
    batch_size: int = 256
    replay_capacity: int = 50_000
    behavior_clone_epochs: int = 100
    behavior_clone_batch_size: int = 256
    update_interval_steps: int = 4
    updates_per_interval: int = 1
    exploration_std: float = 0.01
    network_width: int = 128
    residual_blocks: int = 2
    gamma: float = 0.995
    target_tau: float = 0.005
    target_policy_noise: float = 0.05
    target_noise_clip: float = 0.25
    policy_delay: int = 2
    behavior_clone_learning_rate: float = 3e-4
    actor_learning_rate: float = 1e-5
    critic_learning_rate: float = 3e-4
    reward_multiplier: float = 1.0
    behavior_regularization_weight: float = 100.0
    q_normalization_scale: float = 0.05
    maximum_q_coefficient: float = 1.0
    critic_warmup_updates: int = 500
    actor_trust_region_l2: float = 0.002
    actor_architecture: str = "pid_residual"
    residual_action_limit: float = 0.05
    require_initialization_quality_gate: bool = True
    gradient_norm_limit: float = 10.0
    progress_interval_steps: int = 1_000
    seed: int = 20260828
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.behavior_clone_epochs,
            self.behavior_clone_batch_size,
            self.update_interval_steps,
            self.updates_per_interval,
            self.network_width,
            self.residual_blocks,
            self.target_policy_noise,
            self.target_noise_clip,
            self.policy_delay,
            self.behavior_clone_learning_rate,
            self.actor_learning_rate,
            self.critic_learning_rate,
            self.reward_multiplier,
            self.behavior_regularization_weight,
            self.q_normalization_scale,
            self.maximum_q_coefficient,
            self.actor_trust_region_l2,
            self.residual_action_limit,
            self.gradient_norm_limit,
            self.progress_interval_steps,
        )
        if min(positive) <= 0 or self.replay_capacity < self.batch_size:
            raise ValueError(
                "invalid PID-guided TD3 dimensions or optimization settings"
            )
        if self.exploration_std < 0:
            raise ValueError("TD3 exploration standard deviation cannot be negative")
        if self.critic_warmup_updates < 0:
            raise ValueError("TD3 Critic warmup updates cannot be negative")
        if self.actor_architecture not in {
            "pid_residual",
            "behavior_cloned_mlp",
        }:
            raise ValueError("unsupported PID-guided TD3 Actor architecture")
        if self.residual_action_limit > 1:
            raise ValueError("residual_action_limit cannot exceed one")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("invalid TD3 discount or target update rate")


def _pid_policy(
    gains: PIDGains,
    environment,
) -> RollRatePIDPolicy:
    return RollRatePIDPolicy(
        gains,
        policy_dt_s=environment.policy_dt_s,
        command_scale_rad_s=environment.command_scale_rad_s,
        integral_error_scale_rad=environment.integral_error_scale_rad,
        roll_acceleration_scale_rad_s2=environment.roll_acceleration_scale_rad_s2,
        force_limit_n=environment.force_limit_n,
    )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _pid_actions_for_observations(
    observations: torch.Tensor,
    gains: PIDGains,
    environment,
) -> torch.Tensor:
    """Evaluate the PID oracle on the same states used by the Actor loss."""

    if observations.ndim != 2 or observations.shape[1] < 7:
        raise ValueError(
            "PID behavior targets require batched seven-signal observations"
        )
    error = observations[:, 3:4] * environment.command_scale_rad_s
    integral_error = observations[:, 4:5] * environment.integral_error_scale_rad
    roll_acceleration = (
        observations[:, 5:6] * environment.roll_acceleration_scale_rad_s2
    )
    force = (
        gains.proportional * error
        + gains.integral * integral_error
        - gains.derivative * roll_acceleration
    )
    return (force / environment.force_limit_n).clamp(-1.0, 1.0)


def _pid_prior_coefficients(
    observation_dim: int,
    gains: PIDGains,
    environment,
) -> torch.Tensor:
    if observation_dim != 7:
        raise ValueError("embedded PID prior requires the seven-signal Actor contract")
    coefficients = torch.zeros((1, observation_dim), dtype=torch.float32)
    coefficients[0, 3] = (
        gains.proportional * environment.command_scale_rad_s / environment.force_limit_n
    )
    coefficients[0, 4] = (
        gains.integral
        * environment.integral_error_scale_rad
        / environment.force_limit_n
    )
    coefficients[0, 5] = (
        -gains.derivative
        * environment.roll_acceleration_scale_rad_s2
        / environment.force_limit_n
    )
    return coefficients


def _atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _collect_pid_demonstrations(
    record: PlantRecord,
    environment_config: SpecialistTrainingConfig,
    gains: PIDGains,
    replay: TwoStreamReplayBuffer,
) -> tuple[np.ndarray, np.ndarray, int]:
    actor_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    transitions = 0
    profiles = build_specialist_env(record, environment_config).command_profiles
    for profile_index, profile in enumerate(profiles):
        environment = build_specialist_env(record, environment_config, (profile,))
        policy = _pid_policy(gains, environment)
        policy.reset()
        actor_observation, info = environment.reset(
            seed=environment_config.seed + profile_index
        )
        critic_observation = np.asarray(info["critic_state"], dtype=np.float32)
        while True:
            action = policy.predict(actor_observation)
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
            actor_observations.append(actor_observation.copy())
            actions.append(action.copy())
            transitions += 1
            if episode_done:
                break
            actor_observation = next_actor_observation
            critic_observation = next_critic_observation
    return (
        np.asarray(actor_observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        transitions,
    )


def train_pid_guided_td3(
    record: PlantRecord,
    pid_gains: PIDGains,
    output_dir: str | Path,
    environment_config: SpecialistTrainingConfig,
    td3_config: PIDGuidedTD3Config = PIDGuidedTD3Config(),
    *,
    library_path: str | Path | None = None,
) -> dict[str, object]:
    """Clone one tuned PID and refine its direct full-force policy with TD3."""

    if environment_config.history_steps != 0:
        raise ValueError(
            "PID-guided TD3 uses the seven controller signals and no raw history window"
        )
    device = torch.device(td3_config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(td3_config.seed)
    rng = np.random.default_rng(td3_config.seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()

    deployment_config = replace(
        environment_config,
        total_steps=td3_config.total_steps,
        warmup_steps=0,
        batch_size=td3_config.batch_size,
        replay_capacity=td3_config.replay_capacity,
        network_width=td3_config.network_width,
        residual_blocks=td3_config.residual_blocks,
        gamma=td3_config.gamma,
        target_tau=td3_config.target_tau,
        learning_rate=td3_config.actor_learning_rate,
        device=td3_config.device,
    )
    environment = build_specialist_env(record, deployment_config)
    actor_observation, info = environment.reset(seed=td3_config.seed)
    critic_observation = np.asarray(info["critic_state"], dtype=np.float32)
    actor_observation_contract = environment.actor_observation_contract()
    critic_observation_contract = environment.critic_observation_contract()
    if (
        actor_observation.size != 7
        or actor_observation_contract["uses_raw_history_window"]
    ):
        raise RuntimeError(
            "formal TD3 Teacher Actor contract must be seven-dimensional"
        )
    structured_actor = None
    if td3_config.actor_architecture == "pid_residual":
        structured_actor = PIDResidualActor(
            actor_observation.size,
            1,
            _pid_prior_coefficients(actor_observation.size, pid_gains, environment),
            width=td3_config.network_width,
            residual_blocks=td3_config.residual_blocks,
            residual_action_limit=td3_config.residual_action_limit,
            enforce_odd_symmetry=deployment_config.enforce_odd_policy,
        )
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
        behavior_regularization_weight=(td3_config.behavior_regularization_weight),
        q_normalization_scale=td3_config.q_normalization_scale,
        maximum_q_coefficient=td3_config.maximum_q_coefficient,
        actor_width=td3_config.network_width,
        actor_residual_blocks=td3_config.residual_blocks,
        actor_enforce_odd_symmetry=False,
        actor=structured_actor,
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
    demonstration_started = time.perf_counter()
    demo_observations, demo_actions, demonstration_steps = _collect_pid_demonstrations(
        record, deployment_config, pid_gains, replay
    )
    demonstration_elapsed_s = time.perf_counter() - demonstration_started
    initialization_started = time.perf_counter()
    if td3_config.actor_architecture == "pid_residual":
        with torch.no_grad():
            initialized_actions = learner.act(
                torch.as_tensor(demo_observations)
            ).numpy()
        initialization_mse = float(
            np.mean(np.square(initialized_actions - demo_actions))
        )
        clone_report = {
            "method": "exact_embedded_pid_linear_prior",
            "final_mse": initialization_mse,
            "samples": float(len(demo_observations)),
            "epochs": 0.0,
            "optimizer_steps": 0.0,
        }
    else:
        clone_report = learner.behavior_clone(
            torch.as_tensor(demo_observations),
            torch.as_tensor(demo_actions),
            epochs=td3_config.behavior_clone_epochs,
            batch_size=td3_config.behavior_clone_batch_size,
            learning_rate=td3_config.behavior_clone_learning_rate,
            seed=td3_config.seed,
        )
        if deployment_config.enforce_odd_policy:
            learner.actor.enforce_odd_symmetry = True
            learner.target_actor.enforce_odd_symmetry = True
    initialization_elapsed_s = time.perf_counter() - initialization_started
    learner.set_actor_trust_region(td3_config.actor_trust_region_l2)
    clone_actor = {
        name: value.detach().cpu().clone()
        for name, value in learner.actor.state_dict().items()
    }
    cloned_policy = SpecialistActorPolicy(learner.actor, learner.device)
    initialization_evaluation = evaluate_specialist(
        cloned_policy,
        record,
        deployment_config,
        controller_label="teacher",
    )
    initialization_quality_gate = specialist_quality_gate(
        initialization_evaluation, deployment_config
    )
    initialization_payload = {
        "status": "complete",
        "algorithm_phase": (
            "embedded_pid_control_prior_initialization"
            if td3_config.actor_architecture == "pid_residual"
            else "pid_behavior_cloning_initialization"
        ),
        "evaluation": initialization_evaluation,
        "quality_gate": initialization_quality_gate,
    }
    _write_json(
        destination / "initialization_evaluation.json",
        initialization_payload,
    )
    # Retain the old artifact name for readers of earlier diagnostic runs.
    _write_json(
        destination / "behavior_clone_evaluation.json",
        initialization_payload,
    )
    if (
        deployment_config.enforce_quality_gate
        and td3_config.require_initialization_quality_gate
        and not initialization_quality_gate["passed"]
    ):
        _write_json(
            destination / "progress.json",
            {
                "status": "initialization_quality_gate_failed",
                "algorithm": "pid_guided_td3",
                "plant_id": record.plant_id,
                "initialization_quality_gate": initialization_quality_gate,
            },
        )
        raise RuntimeError(
            "Teacher initialization failed its closed-loop quality gate: "
            f"{initialization_quality_gate['observed']}"
        )

    command_rng = np.random.default_rng(td3_config.seed + 1_000_003)
    command_order = np.empty(0, dtype=int)
    command_position = 0

    def reset_training_episode(
        episode_index: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal command_order, command_position
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
    trust_region_projection_count = 0
    command_counts: Counter[str] = Counter()
    losses: dict[str, float] = {}
    started = time.perf_counter()
    for step in range(td3_config.total_steps):
        action = learner.act(torch.as_tensor(actor_observation).unsqueeze(0)).numpy()[0]
        if td3_config.exploration_std:
            action = action + rng.normal(
                0.0, td3_config.exploration_std, size=action.shape
            )
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
        if (step + 1) % td3_config.update_interval_steps == 0:
            for _ in range(td3_config.updates_per_interval):
                batch = replay.sample(td3_config.batch_size, learner.device)
                batch["behavior_actor_obs"] = batch["actor_obs"]
                batch["behavior_action"] = _pid_actions_for_observations(
                    batch["actor_obs"], pid_gains, environment
                )
                losses = learner.update(
                    batch,
                    allow_actor_update=(updates >= td3_config.critic_warmup_updates),
                )
                updates += 1
                actor_updates += int(losses["actor_updated"])
                trust_region_projection_count += int(losses["trust_region_projected"])
        if episode_done:
            command_counts[str(next_info["command_id"])] += 1
            episodes += 1
            actor_observation, info = reset_training_episode(episodes)
            critic_observation = np.asarray(info["critic_state"], dtype=np.float32)
        completed_steps = step + 1
        if completed_steps % td3_config.progress_interval_steps == 0:
            _write_json(
                destination / "progress.json",
                {
                    "status": "training",
                    "algorithm": "pid_guided_td3",
                    "plant_id": record.plant_id,
                    "step": completed_steps,
                    "updates": updates,
                    "actor_updates": actor_updates,
                    "episodes": episodes,
                    "elapsed_s": time.perf_counter() - started,
                    "last_losses": losses,
                },
            )
    training_elapsed_s = time.perf_counter() - started

    evaluation_started = time.perf_counter()
    policy = SpecialistActorPolicy(learner.actor, learner.device)
    evaluation = evaluate_specialist(
        policy,
        record,
        deployment_config,
        output_dir=destination,
        controller_label="teacher",
        controller_display_label="TD3 Teacher",
    )
    evaluation_elapsed_s = time.perf_counter() - evaluation_started
    actor_delta_squared = 0.0
    for name, value in learner.actor.state_dict().items():
        delta = value.detach().cpu() - clone_actor[name]
        actor_delta_squared += float(delta.square().sum())
    actor_delta_l2 = float(np.sqrt(actor_delta_squared))
    quality_gate = specialist_quality_gate(evaluation, deployment_config)
    quality_gate["checks"]["online_actor_updates"] = actor_updates > 0
    quality_gate["checks"]["actor_changed_by_rl"] = actor_delta_l2 > 1e-12
    quality_gate["checks"]["actor_trust_region"] = (
        actor_delta_l2 <= td3_config.actor_trust_region_l2 + 1e-6
    )
    quality_gate["passed"] = bool(
        quality_gate["passed"]
        and quality_gate["checks"]["online_actor_updates"]
        and quality_gate["checks"]["actor_changed_by_rl"]
        and quality_gate["checks"]["actor_trust_region"]
    )
    quality_gate["observed"]["online_actor_updates"] = actor_updates
    quality_gate["observed"]["rl_actor_parameter_delta_l2"] = actor_delta_l2
    quality_gate["thresholds"]["minimum_online_actor_updates"] = 1
    quality_gate["thresholds"]["maximum_actor_parameter_delta_l2"] = (
        td3_config.actor_trust_region_l2
    )
    source = git_source_revision()
    library = None
    if library_path is not None:
        library_source = Path(library_path)
        library = {
            "path": str(library_source.resolve()),
            "sha256": sha256_file(library_source),
        }
    odd_policy_contract = {
        "enabled": deployment_config.enforce_odd_policy,
        "projection_stage": (
            "embedded_prior_td3_and_inference"
            if td3_config.actor_architecture == "pid_residual"
            else "post_clone_td3_and_inference"
        ),
        "applied_during_initialization": (
            deployment_config.enforce_odd_policy
            and td3_config.actor_architecture == "pid_residual"
        ),
        "applied_during_behavior_cloning": False,
        "applied_during_td3_training": deployment_config.enforce_odd_policy,
        "applied_to_deterministic_teacher": deployment_config.enforce_odd_policy,
        "applied_to_distillation_labels": deployment_config.enforce_odd_policy,
    }
    time_limit_contract = continuing_task_contract(
        critic_includes_episode_progress=(
            deployment_config.critic_include_episode_progress
        )
    )
    actor_payload = {
        "schema_version": "specialist_actor_v1",
        "algorithm": "pid_guided_td3",
        "actor_architecture": (
            PIDResidualActor.architecture_name
            if td3_config.actor_architecture == "pid_residual"
            else "squashed_gaussian_mlp_v1"
        ),
        "plant": {
            "plant_id": record.plant_id,
            "split": record.split,
            "quality_region": record.quality_region,
            "aircraft_class": record.aircraft_class,
            "flight_phase": record.flight_phase,
            "parameters": asdict(record.parameters),
        },
        "config": asdict(deployment_config),
        "td3_config": asdict(td3_config),
        "actor_observation_dim": int(actor_observation.size),
        "actor_observation_contract": actor_observation_contract,
        "critic_observation_contract": critic_observation_contract,
        "odd_policy_contract": odd_policy_contract,
        "continuing_task_contract": time_limit_contract,
        "actor": {
            name: value.detach().cpu()
            for name, value in learner.actor.state_dict().items()
        },
        "source": source,
        "library": library,
    }
    actor_path = destination / "teacher_actor.pt"
    _atomic_torch_save(actor_payload, actor_path)
    _atomic_torch_save(
        {
            **actor_payload,
            "schema_version": "pid_guided_td3_checkpoint_v1",
            "td3_config": asdict(td3_config),
            "pid_gains": asdict(pid_gains),
            "critic": learner.critic.state_dict(),
            "target_actor": learner.target_actor.state_dict(),
            "target_critic": learner.target_critic.state_dict(),
            "actor_optimizer": learner.actor_optimizer.state_dict(),
            "critic_optimizer": learner.critic_optimizer.state_dict(),
            "updates": updates,
            "actor_updates": actor_updates,
        },
        destination / "training_checkpoint.pt",
    )
    total_elapsed_s = time.perf_counter() - run_started
    report: dict[str, object] = {
        "status": "complete",
        "algorithm": "pid_guided_td3",
        "actor_architecture": actor_payload["actor_architecture"],
        "plant_id": record.plant_id,
        "plant": actor_payload["plant"],
        "pid_gains": asdict(pid_gains),
        "config": asdict(deployment_config),
        "environment_config": asdict(deployment_config),
        "td3_config": asdict(td3_config),
        "source": source,
        "library": library,
        "actor_receives_theta": False,
        "actor_uses_raw_history": False,
        "pid_used_at_deployment": (td3_config.actor_architecture == "pid_residual"),
        "pid_controller_object_used_at_deployment": False,
        "embedded_linear_control_prior": (
            td3_config.actor_architecture == "pid_residual"
        ),
        "control_prior_source": (
            "matching_aircraft_tuned_pid_gains"
            if td3_config.actor_architecture == "pid_residual"
            else None
        ),
        "privileged_critic_training_only": True,
        "action_definition": "direct_full_F_as",
        "training_contract": {
            "initialization": (
                "exact_embedded_pid_linear_prior"
                if td3_config.actor_architecture == "pid_residual"
                else "behavior_cloning_from_matching_aircraft_pid"
            ),
            "online_algorithm": "state_matched_td3_bc",
            "online_behavior_target": "matching_pid_on_replay_actor_state",
            "policy_constraint": (
                "bounded_action_residual_plus_l2_parameter_trust_region"
                if td3_config.actor_architecture == "pid_residual"
                else "l2_trust_region_around_verified_clone"
            ),
            "deployment_controller": (
                "neural_residual_actor_with_embedded_linear_control_prior"
                if td3_config.actor_architecture == "pid_residual"
                else "pure_neural_actor"
            ),
        },
        "actor_observation_dim": int(actor_observation.size),
        "actor_observation_contract": actor_observation_contract,
        "critic_observation_dim": int(critic_observation.size),
        "critic_observation_contract": critic_observation_contract,
        "odd_policy_contract": odd_policy_contract,
        "continuing_task_contract": time_limit_contract,
        "parameter_counts": learner.parameter_counts(),
        "demonstration_steps": demonstration_steps,
        "behavior_cloning": clone_report,
        "initialization": clone_report,
        "initialization_evaluation": initialization_evaluation,
        "initialization_quality_gate": initialization_quality_gate,
        "behavior_clone_evaluation": initialization_evaluation,
        "behavior_clone_quality_gate": initialization_quality_gate,
        "online_steps": td3_config.total_steps,
        "online_updates": updates,
        "online_actor_updates": actor_updates,
        "rl_actor_parameter_delta_l2": actor_delta_l2,
        "trust_region_projection_count": trust_region_projection_count,
        "online_episodes": episodes,
        "training_command_counts": dict(command_counts),
        "training_command_scheduler": "seeded_permutation_without_replacement",
        "training_elapsed_s": total_elapsed_s,
        "online_training_elapsed_s": training_elapsed_s,
        "phase_elapsed_s": {
            "pid_demonstration_collection": demonstration_elapsed_s,
            "actor_initialization": initialization_elapsed_s,
            "online_td3": training_elapsed_s,
            "final_closed_loop_evaluation": evaluation_elapsed_s,
        },
        "last_losses": losses,
        "evaluation": evaluation,
        "quality_gate": quality_gate,
        "accepted_for_distillation": quality_gate["passed"],
        "actor_checkpoint": str(actor_path),
        "training_checkpoint": str(destination / "training_checkpoint.pt"),
        "artifacts": {
            "initialization_evaluation": str(
                destination / "initialization_evaluation.json"
            ),
            "behavior_clone_evaluation": str(
                destination / "behavior_clone_evaluation.json"
            ),
            "final_evaluation": str(destination / "evaluation.json"),
            "response_plot": str(destination / "response_comparison.png"),
            "all_commands_plot": str(destination / "all_evaluation_commands.png"),
        },
    }
    _write_json(destination / "report.json", report)
    _write_json(
        destination / "progress.json",
        {
            "status": "complete",
            "algorithm": "pid_guided_td3",
            "plant_id": record.plant_id,
            "step": td3_config.total_steps,
            "updates": updates,
            "actor_updates": actor_updates,
            "accepted_for_distillation": quality_gate["passed"],
            "actor_checkpoint": str(actor_path),
            "report": str(destination / "report.json"),
        },
    )
    return report
