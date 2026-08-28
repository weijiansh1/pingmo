"""Bounded multi-aircraft Teacher training with held-out matched evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.aircraft.sampler import PlantRecord
from src.envs.commands import CommandProfile, evaluation_command_suite
from src.envs.p_channel_env import RollQualityEnv
from src.experiments.exploratory_sac import load_persisted_records
from src.experiments.policy_evaluation import ZeroPolicy, evaluate_policy_pairs
from src.teacher.moe.teacher import MoETeacher
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.sac.teacher import PrivilegedSAC


@dataclass(frozen=True, slots=True)
class ShortTrialConfig:
    teacher_kind: str = "mlp"
    total_steps: int = 6_000
    warmup_steps: int = 1_000
    batch_size: int = 64
    update_every_steps: int = 2
    replay_capacity: int = 20_000
    episode_duration_s: float = 1.25
    evaluation_plants: int = 3
    seed: int = 20260828
    device: str = "cuda"
    progress_interval_steps: int = 500
    network_width: int = 896
    residual_blocks: int = 14
    moe_shared_blocks: int = 10
    moe_expert_blocks: int = 2
    moe_bottleneck_width: int = 448

    def __post_init__(self) -> None:
        if self.teacher_kind not in {"mlp", "moe"}:
            raise ValueError("teacher_kind must be 'mlp' or 'moe'")
        if self.total_steps <= self.warmup_steps:
            raise ValueError("total_steps must exceed warmup_steps")
        if min(
            self.batch_size,
            self.update_every_steps,
            self.replay_capacity,
            self.evaluation_plants,
            self.progress_interval_steps,
            self.network_width,
            self.residual_blocks,
        ) <= 0:
            raise ValueError("trial counts and network dimensions must be positive")
        if self.replay_capacity < self.batch_size or self.episode_duration_s < 1.10:
            raise ValueError("replay must fit one batch and episodes must cover the 1 s sensitivity metric")


def short_training_command_suite(duration_s: float = 1.25) -> tuple[CommandProfile, ...]:
    """Short commands that still leave enough time to measure the 1 s step response."""

    if duration_s < 1.10:
        raise ValueError("short training commands must last at least 1.10 s")
    onset = 0.05
    return (
        CommandProfile("short-step-pos-0.50", "step", amplitude=0.50, onset_s=onset, duration_s=duration_s),
        CommandProfile("short-step-neg-0.50", "step", amplitude=-0.50, onset_s=onset, duration_s=duration_s),
        CommandProfile("short-step-pos-1.00", "step", amplitude=1.00, onset_s=onset, duration_s=duration_s),
        CommandProfile("short-step-neg-1.00", "step", amplitude=-1.00, onset_s=onset, duration_s=duration_s),
        CommandProfile("short-pulse-pos", "pulse", amplitude=0.75, onset_s=onset, segment_duration_s=0.35, duration_s=duration_s),
        CommandProfile("short-pulse-neg", "pulse", amplitude=-0.75, onset_s=onset, segment_duration_s=0.35, duration_s=duration_s),
        CommandProfile("short-doublet-pos", "doublet", amplitude=0.75, onset_s=onset, segment_duration_s=0.25, duration_s=duration_s),
        CommandProfile("short-doublet-neg", "doublet", amplitude=-0.75, onset_s=onset, segment_duration_s=0.25, duration_s=duration_s),
        CommandProfile("short-square-1hz", "square", amplitude=0.75, onset_s=onset, frequency_hz=1.0, duration_s=duration_s),
        CommandProfile("short-sine-0.75hz", "sine", amplitude=0.75, onset_s=onset, frequency_hz=0.75, duration_s=duration_s),
        CommandProfile("short-sine-1.50hz", "sine", amplitude=0.75, onset_s=onset, frequency_hz=1.50, duration_s=duration_s),
        CommandProfile("short-chirp", "chirp", amplitude=0.625, onset_s=onset, frequency_hz=0.50, final_frequency_hz=3.0, duration_s=duration_s),
        CommandProfile("short-staircase", "staircase", onset_s=onset, levels=(0.5, 1.0, 0.0, -0.5, -1.0, 0.0), segment_duration_s=0.20, duration_s=duration_s),
        CommandProfile("short-piecewise", "piecewise", amplitude=0.875, onset_s=onset, segment_duration_s=0.20, duration_s=duration_s, seed=211),
    )


def _records_for_splits(library_path: str | Path, splits: set[str]) -> list[PlantRecord]:
    rows = [json.loads(line) for line in Path(library_path).read_text(encoding="utf-8").splitlines()]
    plant_ids = [str(row["plant_id"]) for row in rows if row.get("split") in splits]
    if not plant_ids:
        raise ValueError(f"no plants found for splits {sorted(splits)}")
    return load_persisted_records(library_path, plant_ids)


def _balanced_evaluation_records(records: list[PlantRecord], count: int) -> list[PlantRecord]:
    groups: dict[str, list[PlantRecord]] = {}
    for record in records:
        groups.setdefault(record.quality_region, []).append(record)
    selected: list[PlantRecord] = []
    ordered_groups = [groups[name] for name in sorted(groups)]
    cursor = 0
    while len(selected) < count:
        made_progress = False
        for group in ordered_groups:
            if cursor < len(group) and len(selected) < count:
                selected.append(group[cursor])
                made_progress = True
        if not made_progress:
            break
        cursor += 1
    if len(selected) != count:
        raise ValueError(f"requested {count} evaluation plants but selected {len(selected)}")
    return selected


def _held_out_profiles() -> tuple[CommandProfile, ...]:
    selected_ids = {
        "eval-step-pos-0.75",
        "eval-step-neg-0.75",
        "eval-doublet-0.75",
        "eval-sine-1.25hz",
        "eval-chirp-0.15-2.50hz",
    }
    return tuple(profile for profile in evaluation_command_suite() if profile.command_id in selected_ids)


class _TeacherPolicy:
    def __init__(self, learner: PrivilegedSAC | MoETeacher) -> None:
        self.learner = learner

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        return self.learner.act(tensor, deterministic=deterministic).numpy()[0]


def _build_learner(config: ShortTrialConfig, actor_dim: int, critic_dim: int) -> PrivilegedSAC | MoETeacher:
    if config.teacher_kind == "mlp":
        return PrivilegedSAC(
            actor_dim,
            critic_dim,
            1,
            actor_width=config.network_width,
            actor_residual_blocks=config.residual_blocks,
            critic_width=config.network_width,
            critic_residual_blocks=config.residual_blocks,
            device=config.device,
        )
    return MoETeacher(
        actor_dim,
        critic_dim,
        1,
        experts=4,
        actor_width=config.network_width,
        shared_residual_blocks=config.moe_shared_blocks,
        expert_residual_blocks=config.moe_expert_blocks,
        expert_bottleneck_width=config.moe_bottleneck_width,
        critic_width=config.network_width,
        critic_residual_blocks=config.residual_blocks,
        device=config.device,
    )


def _metric_delta(rows: list[dict[str, object]], name: str) -> float | None:
    values = [
        float(row[f"controlled_{name}"]) - float(row[f"raw_{name}"])
        for row in rows
        if row.get(f"controlled_{name}") is not None and row.get(f"raw_{name}") is not None
    ]
    return None if not values else float(np.median(values))


def summarize_trial_evaluation(
    raw_rows: list[dict[str, object]],
    controlled_rows: list[dict[str, object]],
) -> dict[str, float | int | None]:
    raw_by_pair = {(row["plant_id"], row["command_id"]): row for row in raw_rows}
    cost_changes = []
    for row in controlled_rows:
        raw = raw_by_pair[(row["plant_id"], row["command_id"])]
        cost_changes.append(-float(row["episode_reward"]) + float(raw["episode_reward"]))
    action_rms = [float(row["controlled_action_rms_n"]) for row in controlled_rows]
    action_tv = [float(row["controlled_action_total_variation_n"]) for row in controlled_rows]
    saturation = [float(row["controlled_action_saturation_fraction"]) for row in controlled_rows]
    return {
        "evaluation_pairs": len(controlled_rows),
        "harm_rate": float(np.mean(np.asarray(cost_changes) > 0.0)),
        "median_episode_cost_change": float(np.median(cost_changes)),
        "median_onset_delay_change_s": _metric_delta(controlled_rows, "response_onset_delay_s"),
        "median_sensitivity_1s_change_deg_per_n": _metric_delta(controlled_rows, "sensitivity_1s_deg_per_n"),
        "median_oscillation_ratio_change": _metric_delta(controlled_rows, "oscillation_ratio_proxy"),
        "median_post_release_roll_rms_change_rad_s": _metric_delta(controlled_rows, "post_release_roll_rms_rad_s"),
        "mean_action_rms_n": float(np.mean(action_rms)),
        "mean_action_total_variation_n": float(np.mean(action_tv)),
        "mean_action_saturation_fraction": float(np.mean(saturation)),
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_short_teacher_trial(
    library_path: str | Path,
    output_dir: str | Path,
    config: ShortTrialConfig,
) -> dict[str, object]:
    """Train a bounded production Teacher and compare it with Raw on held-out pairs."""

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    np_rng = np.random.default_rng(config.seed)
    train_records = _records_for_splits(library_path, {"train_core", "train_boundary"})
    validation_records = _balanced_evaluation_records(
        _records_for_splits(library_path, {"validation"}),
        config.evaluation_plants,
    )
    profiles = short_training_command_suite(config.episode_duration_s)
    horizon_steps = int(round(config.episode_duration_s / 0.001))
    env = RollQualityEnv(train_records, horizon_steps=horizon_steps, command_profiles=profiles)
    actor_obs, info = env.reset(seed=config.seed)
    critic_obs = np.asarray(info["critic_state"], dtype=np.float32)
    learner = _build_learner(config, actor_obs.size, critic_obs.size)
    replay = TwoStreamReplayBuffer(
        config.replay_capacity,
        actor_obs.size,
        critic_obs.size,
        1,
        seed=config.seed,
    )
    episode_reward = 0.0
    completed_episodes: list[dict[str, object]] = []
    losses: dict[str, float] = {}
    updates = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(config.total_steps):
        if step < config.warmup_steps:
            action = np_rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
        else:
            action = learner.act(torch.as_tensor(actor_obs).unsqueeze(0)).numpy()[0].astype(np.float32)
        next_actor_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_critic_obs = np.asarray(next_info["critic_state"], dtype=np.float32)
        replay.add(actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done)
        actor_obs, critic_obs = next_actor_obs, next_critic_obs
        episode_reward += float(reward)
        if step >= config.warmup_steps and step % config.update_every_steps == 0 and len(replay) >= config.batch_size:
            losses = learner.update(replay.sample(config.batch_size, learner.device))
            updates += 1
        if done:
            completed_episodes.append({
                "plant_id": next_info["plant_id"],
                "command_id": next_info["command_id"],
                "episode_reward": episode_reward,
            })
            actor_obs, reset_info = env.reset()
            critic_obs = np.asarray(reset_info["critic_state"], dtype=np.float32)
            episode_reward = 0.0
        if (step + 1) % config.progress_interval_steps == 0:
            progress = {
                "step": step + 1,
                "updates": updates,
                "completed_episodes": len(completed_episodes),
                "elapsed_s": time.perf_counter() - started,
                "last_losses": losses,
            }
            _write_json(destination / "progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    training_elapsed = time.perf_counter() - started
    checkpoint = {
        "teacher_kind": config.teacher_kind,
        "actor": learner.actor.state_dict(),
        "critic": learner.critic.state_dict(),
        "target_critic": learner.target_critic.state_dict(),
        "log_alpha": learner.log_alpha.detach().cpu(),
        "actor_observation_dim": int(actor_obs.size),
        "critic_observation_dim": int(critic_obs.size),
        "config": config.__dict__ if hasattr(config, "__dict__") else {
            field: getattr(config, field) for field in config.__dataclass_fields__
        },
    }
    torch.save(checkpoint, destination / "checkpoint.pt")
    _write_json(destination / "progress.json", {
        "status": "training_complete",
        "step": config.total_steps,
        "updates": updates,
        "completed_episodes": len(completed_episodes),
        "elapsed_s": training_elapsed,
        "checkpoint": str(destination / "checkpoint.pt"),
        "last_losses": losses,
    })

    evaluation_started = time.perf_counter()
    held_out_profiles = _held_out_profiles()
    raw_rows = evaluate_policy_pairs(
        ZeroPolicy(),
        validation_records,
        held_out_profiles,
        controller_name="Raw",
        seed=config.seed + 10_000,
    )
    controlled_rows = evaluate_policy_pairs(
        _TeacherPolicy(learner),
        validation_records,
        held_out_profiles,
        controller_name=f"{config.teacher_kind.upper()}-SAC",
        seed=config.seed + 20_000,
    )
    evaluation_summary = summarize_trial_evaluation(raw_rows, controlled_rows)
    report: dict[str, object] = {
        "status": "complete",
        "config": checkpoint["config"],
        "parameter_counts": learner.parameter_counts(),
        "training_aircraft_count": len(train_records),
        "training_command_count": len(profiles),
        "completed_episodes": completed_episodes,
        "updates": updates,
        "training_elapsed_s": training_elapsed,
        "evaluation_elapsed_s": time.perf_counter() - evaluation_started,
        "last_losses": losses,
        "evaluation_summary": evaluation_summary,
        "evaluation_rows": controlled_rows,
    }
    if device.type == "cuda":
        report["gpu"] = {
            "name": torch.cuda.get_device_name(device),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
    _write_json(destination / "report.json", report)
    return report
