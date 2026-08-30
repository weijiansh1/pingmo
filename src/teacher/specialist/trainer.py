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
from src.utils.plotting import save_specialist_response_plot
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class SpecialistTrainingConfig:
    total_steps: int = 100_000
    warmup_steps: int = 10_000
    batch_size: int = 256
    replay_capacity: int = 200_000
    episode_duration_s: float = 5.0
    command_mode: str = "step"
    history_steps: int = 250
    dt_s: float = 0.001
    command_scale_deg_s: float = 30.0
    force_limit_n: float = 22.0
    force_rate_limit_n_s: float = 88.0
    actuator_time_constant_s: float = 0.0
    reference_natural_frequency_rad_s: float = 2.0
    reference_damping_ratio: float = 0.7
    tracking_error_weight: float = 1.0
    force_energy_weight: float = 0.02
    force_delta_weight: float = 0.05
    network_width: int = 128
    residual_blocks: int = 2
    gamma: float = 0.9995
    target_tau: float = 0.005
    learning_rate: float = 3e-4
    initial_alpha: float = 0.1
    gradient_norm_limit: float = 10.0
    updates_per_step: int = 1
    seed: int = 20260828
    device: str = "cpu"
    progress_interval_steps: int = 10_000

    def __post_init__(self) -> None:
        positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.episode_duration_s,
            self.history_steps,
            self.dt_s,
            self.command_scale_deg_s,
            self.force_limit_n,
            self.force_rate_limit_n_s,
            self.reference_natural_frequency_rad_s,
            self.reference_damping_ratio,
            self.network_width,
            self.residual_blocks,
            self.learning_rate,
            self.initial_alpha,
            self.gradient_norm_limit,
            self.updates_per_step,
            self.progress_interval_steps,
        )
        if min(positive) <= 0:
            raise ValueError("specialist training dimensions and scales must be positive")
        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError("warmup_steps must be non-negative and below total_steps")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay capacity must fit one batch")
        if self.command_mode not in {"step", "extended"}:
            raise ValueError("command_mode must be 'step' or 'extended'")
        if not 0 < self.gamma <= 1 or not 0 < self.target_tau <= 1:
            raise ValueError("invalid specialist SAC discount or target tau")


class PredictivePolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray: ...


class SpecialistActorPolicy:
    def __init__(self, actor: SquashedGaussianActor, device: str | torch.device) -> None:
        self.actor = actor
        self.device = torch.device(device)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor.sample(tensor, deterministic=deterministic)
        values = action.cpu().numpy()
        return values[0] if single else values


def _training_commands(config: SpecialistTrainingConfig) -> tuple[RollRateCommandProfile, ...]:
    if config.command_mode == "step":
        return specialist_step_commands(config.episode_duration_s)
    return specialist_extended_commands(config.episode_duration_s)


def build_specialist_env(
    record: PlantRecord,
    config: SpecialistTrainingConfig,
    command_profiles: tuple[RollRateCommandProfile, ...] | None = None,
) -> SpecialistRollRateEnv:
    return SpecialistRollRateEnv(
        record,
        command_profiles=command_profiles or _training_commands(config),
        dt_s=config.dt_s,
        history_steps=config.history_steps,
        command_scale_deg_s=config.command_scale_deg_s,
        force_limit_n=config.force_limit_n,
        force_rate_limit_n_s=config.force_rate_limit_n_s,
        actuator_time_constant_s=config.actuator_time_constant_s,
        reference_config=SecondOrderReferenceConfig(
            config.reference_natural_frequency_rad_s,
            config.reference_damping_ratio,
        ),
        reward_weights=TrackingRewardWeights(
            config.tracking_error_weight,
            config.force_energy_weight,
            config.force_delta_weight,
        ),
    )


def rollout_policy(
    policy: PredictivePolicy,
    env: SpecialistRollRateEnv,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    observation, _ = env.reset(seed=seed)
    while True:
        action = np.asarray(policy.predict(observation, deterministic=True), dtype=np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return env.trajectory()


def tracking_metrics(trace: dict[str, np.ndarray], force_limit_n: float) -> dict[str, float]:
    error = np.asarray(trace["p_rad_s"]) - np.asarray(trace["p_reference_rad_s"])
    force = np.asarray(trace["f_as_n"])
    commanded = np.asarray(trace["commanded_f_as_n"])
    return {
        "tracking_rmse_rad_s": float(np.sqrt(np.mean(np.square(error)))),
        "tracking_rmse_deg_s": float(np.rad2deg(np.sqrt(np.mean(np.square(error))))),
        "tracking_peak_error_deg_s": float(np.rad2deg(np.max(np.abs(error)))),
        "episode_cost": float(-np.sum(trace["reward"])),
        "force_rms_n": float(np.sqrt(np.mean(np.square(force)))),
        "force_total_variation_n": float(np.sum(np.abs(np.diff(commanded)))),
        "force_saturation_fraction": float(np.mean(np.abs(commanded) >= force_limit_n - 1e-9)),
    }


def evaluate_specialist(
    policy: PredictivePolicy,
    record: PlantRecord,
    config: SpecialistTrainingConfig,
    *,
    output_dir: str | Path | None = None,
    controller_label: str = "teacher",
) -> dict[str, object]:
    if not controller_label or controller_label == "raw":
        raise ValueError("controller_label must be non-empty and cannot be 'raw'")
    rows: list[dict[str, object]] = []
    plot_payload: tuple[dict[str, np.ndarray], dict[str, np.ndarray], str] | None = None
    profiles = specialist_evaluation_commands(config.episode_duration_s)
    for index, profile in enumerate(profiles):
        raw_env = build_specialist_env(record, config, (profile,))
        controller_env = build_specialist_env(record, config, (profile,))
        raw_trace = rollout_policy(CommandForceBaseline(), raw_env, seed=config.seed + index)
        controller_trace = rollout_policy(policy, controller_env, seed=config.seed + index)
        raw_metrics = tracking_metrics(raw_trace, config.force_limit_n)
        controller_metrics = tracking_metrics(controller_trace, config.force_limit_n)
        rows.append(
            {
                "plant_id": record.plant_id,
                "command_id": profile.command_id,
                "command_kind": profile.kind,
                "raw": raw_metrics,
                controller_label: controller_metrics,
                "tracking_rmse_change_rad_s": (
                    controller_metrics["tracking_rmse_rad_s"] - raw_metrics["tracking_rmse_rad_s"]
                ),
                "episode_cost_change": controller_metrics["episode_cost"] - raw_metrics["episode_cost"],
            }
        )
        if plot_payload is None and profile.kind == "step" and profile.amplitude_deg_s > 0:
            plot_payload = raw_trace, controller_trace, profile.command_id

    rmse_changes = np.asarray([row["tracking_rmse_change_rad_s"] for row in rows], dtype=float)
    cost_changes = np.asarray([row["episode_cost_change"] for row in rows], dtype=float)
    summary: dict[str, object] = {
        "pairs": len(rows),
        "tracking_improvement_rate": float(np.mean(rmse_changes < 0.0)),
        "median_tracking_rmse_change_rad_s": float(np.median(rmse_changes)),
        "harm_rate": float(np.mean(cost_changes > 0.0)),
        "median_episode_cost_change": float(np.median(cost_changes)),
        f"mean_{controller_label}_force_rms_n": float(
            np.mean([row[controller_label]["force_rms_n"] for row in rows])
        ),
        f"mean_{controller_label}_force_total_variation_n": float(
            np.mean([row[controller_label]["force_total_variation_n"] for row in rows])
        ),
        "rows": rows,
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        _write_json(destination / "evaluation.json", summary)
        if plot_payload is not None:
            raw_trace, controller_trace, command_id = plot_payload
            save_specialist_response_plot(
                raw_trace,
                controller_trace,
                destination / "response_comparison.png",
                title=f"{record.plant_id}: {command_id}",
                controller_label=controller_label,
            )
    return summary


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
) -> tuple[SpecialistActorPolicy, PlantRecord, SpecialistTrainingConfig, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "specialist_actor_v1":
        raise ValueError("unsupported specialist actor checkpoint schema")
    config = SpecialistTrainingConfig(**payload["config"])
    actor = SquashedGaussianActor(
        int(payload["actor_observation_dim"]),
        1,
        width=config.network_width,
        residual_blocks=config.residual_blocks,
    ).to(device)
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
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    env = build_specialist_env(record, config)
    actor_observation, reset_info = env.reset(seed=config.seed)
    critic_observation = np.asarray(reset_info["critic_state"], dtype=np.float32)
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
            action = learner.act(torch.as_tensor(actor_observation).unsqueeze(0)).numpy()[0].astype(np.float32)
        next_actor_observation, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_critic_observation = np.asarray(next_info["critic_state"], dtype=np.float32)
        replay.add(
            actor_observation,
            critic_observation,
            action,
            reward,
            next_actor_observation,
            next_critic_observation,
            done,
        )
        actor_observation = next_actor_observation
        critic_observation = next_critic_observation
        if step >= config.warmup_steps and len(replay) >= config.batch_size:
            for _ in range(config.updates_per_step):
                losses = learner.update(replay.sample(config.batch_size, learner.device))
                updates += 1
        if done:
            command_counts[str(next_info["command_id"])] += 1
            episodes += 1
            actor_observation, reset_info = env.reset(seed=config.seed + episodes)
            critic_observation = np.asarray(reset_info["critic_state"], dtype=np.float32)
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
    source = git_source_revision()
    library = None
    if library_path is not None:
        library_source = Path(library_path)
        library = {"path": str(library_source.resolve()), "sha256": sha256_file(library_source)}
    actor_payload = {
        "schema_version": "specialist_actor_v1",
        "plant": _record_payload(record),
        "config": asdict(config),
        "actor_observation_dim": int(actor_observation.size),
        "actor": {name: value.detach().cpu() for name, value in learner.actor.state_dict().items()},
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
    report: dict[str, object] = {
        "status": "complete",
        "plant": _record_payload(record),
        "config": asdict(config),
        "source": source,
        "library": library,
        "actor_receives_theta": False,
        "action_definition": "direct_full_F_as",
        "actor_observation_dim": int(actor_observation.size),
        "critic_observation_dim": int(critic_observation.size),
        "parameter_counts": learner.parameter_counts(),
        "steps": config.total_steps,
        "updates": updates,
        "episodes": episodes,
        "training_command_counts": dict(command_counts),
        "training_elapsed_s": training_elapsed_s,
        "last_losses": losses,
        "actor_checkpoint": str(actor_path),
        "training_checkpoint": str(checkpoint_path),
        "evaluation": evaluation,
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
