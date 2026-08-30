"""Independent fixed-aircraft SAC Teacher training and evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Protocol

import numpy as np
import torch

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import PlantRecord
from src.envs.reference_model import SecondOrderReferenceConfig
from src.envs.roll_rate_commands import (
    RollRateCommand,
    RollRateCommandProfile,
    specialist_evaluation_commands,
    specialist_extended_commands,
    specialist_step_commands,
)
from src.envs.specialist_tracking_env import (
    CommandForceBaseline,
    SpecialistRollRateEnv,
    TrackingRewardWeights,
)
from src.experiments.exploratory_sac import load_persisted_records
from src.teacher.sac.actor import SquashedGaussianActor
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.sac.teacher import PrivilegedSAC
from src.teacher.td3.actor import DeterministicActor, PIDResidualActor
from src.teacher.transitions import (
    continuing_task_contract,
    continuing_task_transition_flags,
)
from src.utils.plotting import (
    save_specialist_evaluation_grid,
    save_specialist_response_plot,
)
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class SpecialistTrainingConfig:
    total_steps: int = 100_000
    warmup_steps: int = 10_000
    batch_size: int = 256
    replay_capacity: int = 200_000
    episode_duration_s: float = 5.0
    command_mode: str = "step"
    history_steps: int = 0
    requested_action_history_steps: int = 0
    include_actor_actuator_state: bool = False
    include_reference_derivative: bool = False
    critic_include_episode_progress: bool = True
    critic_include_command_context: bool = True
    plant_dt_s: float = 0.001
    policy_dt_s: float = 0.020
    command_scale_deg_s: float = 30.0
    force_limit_n: float = 22.0
    force_rate_limit_n_s: float = 88.0
    actuator_time_constant_s: float = 0.0
    reference_natural_frequency_rad_s: float = 2.0
    reference_damping_ratio: float = 0.7
    reference_delay_mode: str = "match_plant_transport_delay"
    tracking_error_weight: float = 1.0
    force_energy_weight: float = 0.02
    force_delta_weight: float = 0.02
    reward_scale: float = 50.0
    enforce_odd_policy: bool = True
    odd_policy_projection_stage: str = "inference"
    network_width: int = 128
    residual_blocks: int = 2
    gamma: float = 0.995
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    initial_alpha: float = 0.005
    gradient_norm_limit: float = 10.0
    updates_per_step: int = 1
    enforce_quality_gate: bool = True
    quality_gate_minimum_improvement_rate: float = 1.0
    quality_gate_maximum_mean_rmse_deg_s: float = 1.0
    quality_gate_maximum_peak_error_deg_s: float = 3.0
    quality_gate_maximum_mean_requested_force_variation_n: float = 50.0
    quality_gate_maximum_saturation_fraction: float = 0.01
    seed: int = 20260828
    device: str = "cpu"
    progress_interval_steps: int = 10_000

    def __post_init__(self) -> None:
        positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.episode_duration_s,
            self.plant_dt_s,
            self.policy_dt_s,
            self.command_scale_deg_s,
            self.force_limit_n,
            self.force_rate_limit_n_s,
            self.reward_scale,
            self.reference_natural_frequency_rad_s,
            self.reference_damping_ratio,
            self.network_width,
            self.residual_blocks,
            self.learning_rate,
            self.initial_alpha,
            self.gradient_norm_limit,
            self.updates_per_step,
            self.quality_gate_maximum_mean_rmse_deg_s,
            self.quality_gate_maximum_peak_error_deg_s,
            self.quality_gate_maximum_mean_requested_force_variation_n,
            self.progress_interval_steps,
        )
        if min(positive) <= 0:
            raise ValueError(
                "specialist training dimensions and scales must be positive"
            )
        if self.history_steps < 0:
            raise ValueError("history_steps cannot be negative")
        if self.requested_action_history_steps < 0:
            raise ValueError("requested_action_history_steps cannot be negative")
        if not 0 <= self.quality_gate_minimum_improvement_rate <= 1:
            raise ValueError("quality-gate improvement rate must be in [0, 1]")
        if not 0 <= self.quality_gate_maximum_saturation_fraction <= 1:
            raise ValueError("quality-gate saturation fraction must be in [0, 1]")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be non-negative and below total_steps")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must fit one batch")
        if self.command_mode not in {"step", "extended"}:
            raise ValueError("command_mode must be 'step' or 'extended'")
        if self.odd_policy_projection_stage not in {"training", "inference"}:
            raise ValueError(
                "odd_policy_projection_stage must be 'training' or 'inference'"
            )
        if self.reference_delay_mode not in {"match_plant_transport_delay", "none"}:
            raise ValueError("unsupported specialist reference delay mode")
        ratio = self.policy_dt_s / self.plant_dt_s
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("policy_dt_s must be an integer multiple of plant_dt_s")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("invalid specialist SAC discount or target tau")


class PredictivePolicy(Protocol):
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray: ...


class SpecialistActorPolicy:
    def __init__(self, actor: torch.nn.Module, device: str | torch.device) -> None:
        self.actor = actor
        self.device = torch.device(device)

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(tensor, deterministic=deterministic)
        values = action.cpu().numpy()
        return values[0] if single else values


def _training_commands(
    config: SpecialistTrainingConfig,
) -> tuple[RollRateCommandProfile, ...]:
    if config.command_mode == "step":
        return specialist_step_commands(config.episode_duration_s)
    return specialist_extended_commands(config.episode_duration_s)


def build_specialist_env(
    record: PlantRecord,
    config: SpecialistTrainingConfig,
    command_profiles: tuple[RollRateCommand, ...] | None = None,
) -> SpecialistRollRateEnv:
    return SpecialistRollRateEnv(
        record,
        command_profiles=command_profiles or _training_commands(config),
        plant_dt_s=config.plant_dt_s,
        policy_dt_s=config.policy_dt_s,
        history_steps=config.history_steps,
        requested_action_history_steps=config.requested_action_history_steps,
        include_actor_actuator_state=config.include_actor_actuator_state,
        include_reference_derivative=config.include_reference_derivative,
        critic_include_episode_progress=config.critic_include_episode_progress,
        critic_include_command_context=config.critic_include_command_context,
        command_scale_deg_s=config.command_scale_deg_s,
        force_limit_n=config.force_limit_n,
        force_rate_limit_n_s=config.force_rate_limit_n_s,
        actuator_time_constant_s=config.actuator_time_constant_s,
        reference_config=SecondOrderReferenceConfig(
            config.reference_natural_frequency_rad_s,
            config.reference_damping_ratio,
        ),
        reference_delay_s=(
            record.parameters.tau_p
            if config.reference_delay_mode == "match_plant_transport_delay"
            else 0.0
        ),
        reward_weights=TrackingRewardWeights(
            config.tracking_error_weight,
            config.force_energy_weight,
            config.force_delta_weight,
        ),
        reward_scale=config.reward_scale,
    )


def rollout_policy(
    policy: PredictivePolicy,
    env: SpecialistRollRateEnv,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    reset = getattr(policy, "reset", None)
    if callable(reset):
        reset()
    observation, _ = env.reset(seed=seed)
    while True:
        action = np.asarray(
            policy.predict(observation, deterministic=True), dtype=np.float32
        )
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return env.trajectory()


def tracking_metrics(
    trace: dict[str, np.ndarray], force_limit_n: float
) -> dict[str, float]:
    error = np.asarray(trace["p_rad_s"]) - np.asarray(trace["p_reference_rad_s"])
    force = np.asarray(trace["f_as_n"])
    commanded = np.asarray(trace["commanded_f_as_n"])
    requested = np.asarray(trace["requested_f_as_n"])
    force_rate_limit_gap = requested - commanded
    return {
        "tracking_rmse_rad_s": float(np.sqrt(np.mean(np.square(error)))),
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(np.square(error))))),
        "tracking_peak_error_deg_s": float(np.rad2deg(np.max(np.abs(error)))),
        "episode_cost": float(-np.sum(trace["reward"])),
        "tracking_error_cost": float(np.sum(trace["tracking_error_cost"])),
        "force_energy_cost": float(np.sum(trace["force_energy_cost"])),
        "requested_force_delta_cost": float(np.sum(trace["force_delta_cost"])),
        "force_rms_n": float(np.sqrt(np.mean(np.square(force)))),
        "force_total_variation_n": float(np.sum(np.abs(np.diff(commanded)))),
        "requested_force_total_variation_n": float(np.sum(np.abs(np.diff(requested)))),
        "force_rate_limit_active_fraction": float(
            np.mean(np.abs(force_rate_limit_gap) > 1e-6)
        ),
        "mean_abs_force_rate_limit_gap_n": float(np.mean(np.abs(force_rate_limit_gap))),
        "maximum_abs_force_rate_limit_gap_n": float(
            np.max(np.abs(force_rate_limit_gap))
        ),
        "force_saturation_fraction": float(
            np.mean(np.abs(commanded) >= force_limit_n - 1e-9)
        ),
    }


def evaluate_specialist(
    policy: PredictivePolicy,
    record: PlantRecord,
    config: SpecialistTrainingConfig,
    *,
    output_dir: str | Path | None = None,
    controller_label: str = "teacher",
    controller_display_label: str | None = None,
) -> dict[str, object]:
    if not controller_label or controller_label == "raw":
        raise ValueError("controller_label must be non-empty and cannot be 'raw'")
    rows: list[dict[str, object]] = []
    plot_payload: tuple[dict[str, np.ndarray], dict[str, np.ndarray], str] | None = None
    evaluation_traces: list[
        tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = []
    profiles = specialist_evaluation_commands(config.episode_duration_s)
    for index, profile in enumerate(profiles):
        raw_env = build_specialist_env(record, config, (profile,))
        controller_env = build_specialist_env(record, config, (profile,))
        raw_trace = rollout_policy(
            CommandForceBaseline(), raw_env, seed=config.seed + index
        )
        controller_trace = rollout_policy(
            policy, controller_env, seed=config.seed + index
        )
        raw_metrics = tracking_metrics(raw_trace, config.force_limit_n)
        controller_metrics = tracking_metrics(controller_trace, config.force_limit_n)
        evaluation_traces.append((profile.command_id, raw_trace, controller_trace))
        rows.append(
            {
                "plant_id": record.plant_id,
                "command_id": profile.command_id,
                "command_kind": profile.kind,
                "raw": raw_metrics,
                controller_label: controller_metrics,
                "tracking_rmse_change_rad_s": (
                    controller_metrics["tracking_rmse_rad_s"]
                    - raw_metrics["tracking_rmse_rad_s"]
                ),
                "episode_cost_change": controller_metrics["episode_cost"]
                - raw_metrics["episode_cost"],
            }
        )
        if (
            plot_payload is None
            and profile.kind == "step"
            and profile.amplitude_deg_s > 0
        ):
            plot_payload = raw_trace, controller_trace, profile.command_id

    rmse_changes = np.asarray(
        [row["tracking_rmse_change_rad_s"] for row in rows], dtype=float
    )
    cost_changes = np.asarray([row["episode_cost_change"] for row in rows], dtype=float)
    controller_rmse_deg_s = np.asarray(
        [row[controller_label]["tracking_rmse_deg_s"] for row in rows], dtype=float
    )
    controller_peak_deg_s = np.asarray(
        [row[controller_label]["tracking_peak_error_deg_s"] for row in rows],
        dtype=float,
    )
    requested_variation_n = np.asarray(
        [row[controller_label]["requested_force_total_variation_n"] for row in rows],
        dtype=float,
    )
    saturation = np.asarray(
        [row[controller_label]["force_saturation_fraction"] for row in rows],
        dtype=float,
    )
    summary: dict[str, object] = {
        "pairs": len(rows),
        "tracking_improvement_rate": float(np.mean(rmse_changes < 0.0)),
        "median_tracking_rmse_change_rad_s": float(np.median(rmse_changes)),
        "harm_rate": float(np.mean(cost_changes > 0.0)),
        "median_episode_cost_change": float(np.median(cost_changes)),
        f"mean_{controller_label}_tracking_rmse_deg_s": float(
            np.mean(controller_rmse_deg_s)
        ),
        f"maximum_{controller_label}_peak_error_deg_s": float(
            np.max(controller_peak_deg_s)
        ),
        f"mean_{controller_label}_force_rms_n": float(
            np.mean([row[controller_label]["force_rms_n"] for row in rows])
        ),
        f"mean_{controller_label}_force_total_variation_n": float(
            np.mean([row[controller_label]["force_total_variation_n"] for row in rows])
        ),
        f"mean_{controller_label}_requested_force_total_variation_n": float(
            np.mean(requested_variation_n)
        ),
        f"maximum_{controller_label}_requested_force_total_variation_n": float(
            np.max(requested_variation_n)
        ),
        f"mean_{controller_label}_force_saturation_fraction": float(
            np.mean(saturation)
        ),
        "rows": rows,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "evaluation.json", summary)
        display_label = controller_display_label or (
            "SAC" if controller_label == "teacher" else controller_label
        )
        if plot_payload is not None:
            raw_trace, controller_trace, command_id = plot_payload
            save_specialist_response_plot(
                raw_trace,
                controller_trace,
                destination / "response_comparison.png",
                title=f"{record.plant_id}: {command_id}",
                controller_label=display_label,
            )
        save_specialist_evaluation_grid(
            evaluation_traces,
            destination / "all_evaluation_commands.png",
            title=f"{record.plant_id}: all held-out commands",
            controller_label=display_label,
        )
    return summary


def specialist_quality_gate(
    evaluation: dict[str, object],
    config: SpecialistTrainingConfig,
    *,
    controller_label: str = "teacher",
) -> dict[str, object]:
    observed = {
        "tracking_improvement_rate": float(evaluation["tracking_improvement_rate"]),
        "mean_tracking_rmse_deg_s": float(
            evaluation[f"mean_{controller_label}_tracking_rmse_deg_s"]
        ),
        "maximum_peak_error_deg_s": float(
            evaluation[f"maximum_{controller_label}_peak_error_deg_s"]
        ),
        "mean_requested_force_total_variation_n": float(
            evaluation[f"mean_{controller_label}_requested_force_total_variation_n"]
        ),
        "mean_force_saturation_fraction": float(
            evaluation[f"mean_{controller_label}_force_saturation_fraction"]
        ),
    }
    checks = {
        "tracking_improvement_rate": observed["tracking_improvement_rate"]
        >= config.quality_gate_minimum_improvement_rate,
        "mean_tracking_rmse": observed["mean_tracking_rmse_deg_s"]
        <= config.quality_gate_maximum_mean_rmse_deg_s,
        "peak_tracking_error": observed["maximum_peak_error_deg_s"]
        <= config.quality_gate_maximum_peak_error_deg_s,
        "requested_force_variation": observed["mean_requested_force_total_variation_n"]
        <= config.quality_gate_maximum_mean_requested_force_variation_n,
        "force_saturation": observed["mean_force_saturation_fraction"]
        <= config.quality_gate_maximum_saturation_fraction,
    }
    return {
        "enforced": config.enforce_quality_gate,
        "passed": not config.enforce_quality_gate or all(checks.values()),
        "checks": checks,
        "observed": observed,
        "thresholds": {
            "minimum_tracking_improvement_rate": config.quality_gate_minimum_improvement_rate,
            "maximum_mean_tracking_rmse_deg_s": config.quality_gate_maximum_mean_rmse_deg_s,
            "maximum_peak_error_deg_s": config.quality_gate_maximum_peak_error_deg_s,
            "maximum_mean_requested_force_total_variation_n": config.quality_gate_maximum_mean_requested_force_variation_n,
            "maximum_saturation_fraction": config.quality_gate_maximum_saturation_fraction,
        },
    }


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


def _record_payload(record: PlantRecord) -> dict[str, object]:
    return {
        "plant_id": record.plant_id,
        "split": record.split,
        "quality_region": record.quality_region,
        "aircraft_class": record.aircraft_class,
        "flight_phase": record.flight_phase,
        "parameters": asdict(record.parameters),
    }


def record_from_payload(payload: dict[str, object]) -> PlantRecord:
    parameters = payload["parameters"]
    if not isinstance(parameters, dict):
        raise ValueError("specialist checkpoint parameters must be a mapping")
    return PlantRecord(
        plant_id=str(payload["plant_id"]),
        split=str(payload["split"]),
        quality_region=str(payload["quality_region"]),
        aircraft_class=str(payload["aircraft_class"]),
        flight_phase=str(payload["flight_phase"]),
        parameters=PChannelParameters(**parameters),
    )


def load_specialist_actor(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[
    SpecialistActorPolicy, PlantRecord, SpecialistTrainingConfig, dict[str, object]
]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") not in {
        "specialist_actor_v1",
        "pure_reward_td3_validation_actor_v1",
    }:
        raise ValueError("unsupported specialist actor checkpoint schema")
    config_payload = dict(payload["config"])
    config_payload.setdefault("reference_delay_mode", "none")
    config_payload.setdefault("enforce_odd_policy", False)
    config_payload.setdefault("odd_policy_projection_stage", "training")
    config = SpecialistTrainingConfig(**config_payload)
    observation_dim = int(payload["actor_observation_dim"])
    actor_architecture = str(
        payload.get("actor_architecture", "squashed_gaussian_mlp_v1")
    )
    if actor_architecture == PIDResidualActor.architecture_name:
        td3_config = payload.get("td3_config")
        if not isinstance(td3_config, dict):
            raise ValueError("PID-residual checkpoint is missing TD3 configuration")
        actor = PIDResidualActor(
            observation_dim,
            1,
            torch.zeros((1, observation_dim)),
            width=config.network_width,
            residual_blocks=config.residual_blocks,
            residual_action_limit=float(td3_config["residual_action_limit"]),
            enforce_odd_symmetry=config.enforce_odd_policy,
        ).to(device)
    elif actor_architecture == DeterministicActor.architecture_name:
        actor = DeterministicActor(
            observation_dim,
            1,
            width=config.network_width,
            residual_blocks=config.residual_blocks,
            enforce_odd_symmetry=config.enforce_odd_policy,
        ).to(device)
    elif actor_architecture == "squashed_gaussian_mlp_v1":
        actor = SquashedGaussianActor(
            observation_dim,
            1,
            width=config.network_width,
            residual_blocks=config.residual_blocks,
            enforce_odd_symmetry=config.enforce_odd_policy,
        ).to(device)
    else:
        raise ValueError(
            f"unsupported specialist Actor architecture: {actor_architecture}"
        )
    actor.load_state_dict(payload["actor"])
    actor.eval()
    record = record_from_payload(payload["plant"])
    return SpecialistActorPolicy(actor, device), record, config, payload


def train_specialist(
    record: PlantRecord,
    output_dir: str | Path,
    config: SpecialistTrainingConfig = SpecialistTrainingConfig(),
    *,
    library_path: str | Path | None = None,
) -> dict[str, object]:
    """Train exactly one fixed-aircraft Teacher and write independent artifacts."""

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    command_rng = np.random.default_rng(config.seed + 1_000_003)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    env = build_specialist_env(record, config)
    command_order = np.empty(0, dtype=int)
    command_position = 0

    def reset_training_episode(
        episode_index: int,
    ) -> tuple[np.ndarray, dict[str, object]]:
        nonlocal command_order, command_position
        if command_position >= len(command_order):
            command_order = command_rng.permutation(len(env.command_profiles))
            command_position = 0
        command_index = int(command_order[command_position])
        command_position += 1
        return env.reset(
            seed=config.seed + episode_index,
            options={"command_index": command_index},
        )

    actor_observation, reset_info = reset_training_episode(0)
    critic_observation = np.asarray(reset_info["critic_state"], dtype=np.float32)
    actor_observation_contract = env.actor_observation_contract()
    critic_observation_contract = env.critic_observation_contract()
    learner = PrivilegedSAC(
        actor_observation.size,
        critic_observation.size,
        1,
        gamma=config.gamma,
        tau=config.target_tau,
        initial_alpha=config.initial_alpha,
        learning_rate=config.learning_rate,
        gradient_norm_limit=config.gradient_norm_limit,
        actor_width=config.network_width,
        actor_residual_blocks=config.residual_blocks,
        actor_enforce_odd_symmetry=(
            config.enforce_odd_policy
            and config.odd_policy_projection_stage == "training"
        ),
        critic_width=config.network_width,
        critic_residual_blocks=config.residual_blocks,
        device=device,
    )
    replay = TwoStreamReplayBuffer(
        config.replay_capacity,
        actor_observation.size,
        critic_observation.size,
        1,
        seed=config.seed,
    )
    command_counts: Counter[str] = Counter()
    losses: dict[str, float] = {}
    updates = 0
    episodes = 0
    started = time.perf_counter()

    for step in range(config.total_steps):
        if step < config.warmup_steps:
            action = rng.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
        else:
            action = (
                learner.act(torch.as_tensor(actor_observation).unsqueeze(0))
                .numpy()[0]
                .astype(np.float32)
            )
        next_actor_observation, reward, terminated, truncated, next_info = env.step(
            action
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
        if step >= config.warmup_steps and len(replay) >= config.batch_size:
            for _ in range(config.updates_per_step):
                losses = learner.update(
                    replay.sample(config.batch_size, learner.device)
                )
                updates += 1
        if episode_done:
            command_counts[str(next_info["command_id"])] += 1
            episodes += 1
            actor_observation, reset_info = reset_training_episode(episodes)
            critic_observation = np.asarray(
                reset_info["critic_state"], dtype=np.float32
            )
        completed_steps = step + 1
        if completed_steps % config.progress_interval_steps == 0:
            progress = {
                "status": "training",
                "plant_id": record.plant_id,
                "step": completed_steps,
                "updates": updates,
                "episodes": episodes,
                "elapsed_s": time.perf_counter() - started,
                "last_losses": losses,
            }
            _write_json(destination / "progress.json", progress)

    training_elapsed_s = time.perf_counter() - started
    if config.enforce_odd_policy:
        learner.actor.enforce_odd_symmetry = True
    odd_policy_contract = {
        "enabled": config.enforce_odd_policy,
        "projection_stage": config.odd_policy_projection_stage,
        "applied_during_sac_training": bool(
            config.enforce_odd_policy
            and config.odd_policy_projection_stage == "training"
        ),
        "applied_to_deterministic_teacher": config.enforce_odd_policy,
        "applied_to_distillation_labels": config.enforce_odd_policy,
    }
    training_contract = {
        "algorithm": "soft_actor_critic",
        "supervision": "environment_reward_only",
        "uses_pid_demonstrations": False,
        "uses_behavior_cloning": False,
        "uses_embedded_control_prior": False,
        "uses_pid_regularization": False,
        "actor_action": "normalized_direct_full_F_as",
    }
    time_limit_contract = continuing_task_contract(
        critic_includes_episode_progress=config.critic_include_episode_progress
    )
    source = git_source_revision()
    library = None
    if library_path is not None:
        library_source = Path(library_path)
        library = {
            "path": str(library_source.resolve()),
            "sha256": sha256_file(library_source),
        }
    actor_payload = {
        "schema_version": "specialist_actor_v1",
        "plant": _record_payload(record),
        "config": asdict(config),
        "actor_observation_dim": int(actor_observation.size),
        "actor_observation_contract": actor_observation_contract,
        "critic_observation_contract": critic_observation_contract,
        "odd_policy_contract": odd_policy_contract,
        "training_contract": training_contract,
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
    checkpoint = {
        **actor_payload,
        "schema_version": "specialist_training_checkpoint_v1",
        "critic_observation_dim": int(critic_observation.size),
        "critic": learner.critic.state_dict(),
        "target_critic": learner.target_critic.state_dict(),
        "log_alpha": learner.log_alpha.detach().cpu(),
        "actor_optimizer": learner.actor_optim.state_dict(),
        "critic_optimizer": learner.critic_optim.state_dict(),
        "alpha_optimizer": learner.alpha_optim.state_dict(),
        "steps": config.total_steps,
        "updates": updates,
    }
    checkpoint_path = destination / "training_checkpoint.pt"
    _atomic_torch_save(checkpoint, checkpoint_path)
    policy = SpecialistActorPolicy(learner.actor, learner.device)
    evaluation = evaluate_specialist(policy, record, config, output_dir=destination)
    quality_gate = specialist_quality_gate(evaluation, config)
    report: dict[str, object] = {
        "status": "complete",
        "plant": _record_payload(record),
        "config": asdict(config),
        "source": source,
        "library": library,
        "actor_receives_theta": False,
        "action_definition": "direct_full_F_as",
        "actor_observation_dim": int(actor_observation.size),
        "actor_observation_contract": actor_observation_contract,
        "critic_observation_contract": critic_observation_contract,
        "odd_policy_contract": odd_policy_contract,
        "training_contract": training_contract,
        "continuing_task_contract": time_limit_contract,
        "critic_observation_dim": int(critic_observation.size),
        "parameter_counts": learner.parameter_counts(),
        "steps": config.total_steps,
        "updates": updates,
        "episodes": episodes,
        "training_command_counts": dict(command_counts),
        "training_command_scheduler": "seeded_permutation_without_replacement",
        "training_elapsed_s": training_elapsed_s,
        "last_losses": losses,
        "actor_checkpoint": str(actor_path),
        "training_checkpoint": str(checkpoint_path),
        "evaluation": evaluation,
        "quality_gate": quality_gate,
        "accepted_for_distillation": quality_gate["passed"],
    }
    _write_json(destination / "report.json", report)
    _write_json(
        destination / "progress.json",
        {
            "status": "complete",
            "plant_id": record.plant_id,
            "step": config.total_steps,
            "updates": updates,
            "actor_checkpoint": str(actor_path),
            "report": str(destination / "report.json"),
        },
    )
    return report


def train_specialist_from_library(
    library_path: str | Path,
    plant_id: str,
    output_dir: str | Path,
    config: SpecialistTrainingConfig = SpecialistTrainingConfig(),
) -> dict[str, object]:
    record = load_persisted_records(library_path, [plant_id])[0]
    return train_specialist(record, output_dir, config, library_path=library_path)
