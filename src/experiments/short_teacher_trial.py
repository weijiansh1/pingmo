"""Bounded multi-aircraft Teacher training with held-out matched evaluation."""

from __future__ import annotations

from collections import Counter
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
    total_steps: int = 160_000
    warmup_steps: int = 40_000
    batch_size: int = 256
    update_every_steps: int = 32
    replay_capacity: int = 160_000
    episode_duration_s: float = 1.25
    parallel_envs: int = 32
    evaluation_plants: int = 3
    evaluation_batch_size: int = 15
    seed: int = 20260828
    device: str = "cuda"
    progress_interval_steps: int = 10_000
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
            self.parallel_envs,
            self.evaluation_plants,
            self.evaluation_batch_size,
            self.progress_interval_steps,
            self.network_width,
            self.residual_blocks,
        ) <= 0:
            raise ValueError("trial counts and network dimensions must be positive")
        if self.replay_capacity < self.batch_size or self.episode_duration_s < 1.10:
            raise ValueError("replay must fit one batch and episodes must cover the 1 s sensitivity metric")
        horizon_steps = int(round(self.episode_duration_s / 0.001))
        episode_batch_steps = self.parallel_envs * horizon_steps
        if self.total_steps % episode_batch_steps != 0:
            raise ValueError("total_steps must contain complete parallel episode batches")
        if self.warmup_steps % self.parallel_envs != 0:
            raise ValueError("warmup_steps must align with parallel_envs")


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


def balanced_episode_assignments(
    records: list[PlantRecord],
    profiles: tuple[CommandProfile, ...],
    episode_count: int,
    seed: int,
) -> list[tuple[PlantRecord, CommandProfile]]:
    """Cover every quality-level/command pair before repeating either axis."""

    if episode_count <= 0 or not records or not profiles:
        raise ValueError("records, profiles, and episode_count must be non-empty")
    groups: dict[str, list[PlantRecord]] = {}
    for record in records:
        groups.setdefault(record.quality_region, []).append(record)
    levels = sorted(groups)
    rng = np.random.default_rng(seed)
    pools = {level: list(rng.permutation(group)) for level, group in groups.items()}
    positions = {level: 0 for level in levels}
    assignments: list[tuple[PlantRecord, CommandProfile]] = []
    for episode_index in range(episode_count):
        level = levels[episode_index % len(levels)]
        profile = profiles[(episode_index // len(levels)) % len(profiles)]
        if positions[level] >= len(pools[level]):
            pools[level] = list(rng.permutation(groups[level]))
            positions[level] = 0
        record = pools[level][positions[level]]
        positions[level] += 1
        assignments.append((record, profile))
    return assignments


class _TeacherPolicy:
    def __init__(self, learner: PrivilegedSAC | MoETeacher) -> None:
        self.learner = learner

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        single = tensor.ndim == 1
        if single:
            tensor = tensor.unsqueeze(0)
        actions = self.learner.act(tensor, deterministic=deterministic).numpy()
        return actions[0] if single else actions


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
    raw_good_cost_changes = []
    raw_good_action_rms = []
    for row in controlled_rows:
        raw = raw_by_pair[(row["plant_id"], row["command_id"])]
        cost_change = -float(row["episode_reward"]) + float(raw["episode_reward"])
        cost_changes.append(cost_change)
        if abs(float(raw["episode_reward"])) <= 1e-12:
            raw_good_cost_changes.append(cost_change)
            raw_good_action_rms.append(float(row["controlled_action_rms_n"]))
    action_rms = [float(row["controlled_action_rms_n"]) for row in controlled_rows]
    action_tv = [float(row["controlled_action_total_variation_n"]) for row in controlled_rows]
    saturation = [float(row["controlled_action_saturation_fraction"]) for row in controlled_rows]
    return {
        "evaluation_pairs": len(controlled_rows),
        "harm_rate": float(np.mean(np.asarray(cost_changes) > 0.0)),
        "median_episode_cost_change": float(np.median(cost_changes)),
        "raw_good_pairs": len(raw_good_cost_changes),
        "raw_good_harm_rate": (
            None if not raw_good_cost_changes else float(np.mean(np.asarray(raw_good_cost_changes) > 0.0))
        ),
        "raw_good_mean_action_rms_n": (
            None if not raw_good_action_rms else float(np.mean(raw_good_action_rms))
        ),
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
    episode_count = config.total_steps // horizon_steps
    assignments = balanced_episode_assignments(train_records, profiles, episode_count, config.seed)

    def start_rollout(assignment_index: int) -> dict[str, object]:
        record, profile = assignments[assignment_index]
        env = RollQualityEnv([record], horizon_steps=horizon_steps, command_profiles=(profile,))
        actor_observation, reset_info = env.reset(seed=config.seed + assignment_index)
        return {
            "env": env,
            "record": record,
            "profile": profile,
            "actor_obs": actor_observation,
            "critic_obs": np.asarray(reset_info["critic_state"], dtype=np.float32),
            "episode_reward": 0.0,
        }

    rollouts = [start_rollout(index) for index in range(config.parallel_envs)]
    next_assignment = config.parallel_envs
    first_actor_obs = np.asarray(rollouts[0]["actor_obs"], dtype=np.float32)
    first_critic_obs = np.asarray(rollouts[0]["critic_obs"], dtype=np.float32)
    learner = _build_learner(config, first_actor_obs.size, first_critic_obs.size)
    replay = TwoStreamReplayBuffer(
        config.replay_capacity,
        first_actor_obs.size,
        first_critic_obs.size,
        1,
        seed=config.seed,
    )
    completed_episodes: list[dict[str, object]] = []
    losses: dict[str, float] = {}
    updates = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    transition_steps = 0
    next_progress = config.progress_interval_steps
    while transition_steps < config.total_steps:
        actor_obs_batch = np.stack([np.asarray(rollout["actor_obs"]) for rollout in rollouts])
        critic_obs_batch = np.stack([np.asarray(rollout["critic_obs"]) for rollout in rollouts])
        if transition_steps < config.warmup_steps:
            actions = np_rng.uniform(-1.0, 1.0, size=(config.parallel_envs, 1)).astype(np.float32)
        else:
            actions = learner.act(torch.as_tensor(actor_obs_batch)).numpy().astype(np.float32)

        next_actor_obs_batch: list[np.ndarray] = []
        next_critic_obs_batch: list[np.ndarray] = []
        rewards = np.empty(config.parallel_envs, dtype=np.float32)
        dones = np.empty(config.parallel_envs, dtype=np.float32)
        finished_indices: list[int] = []
        for index, rollout in enumerate(rollouts):
            env = rollout["env"]
            next_actor_obs, reward, terminated, truncated, next_info = env.step(actions[index])
            done = terminated or truncated
            next_actor_obs_batch.append(next_actor_obs)
            next_critic_obs_batch.append(np.asarray(next_info["critic_state"], dtype=np.float32))
            rewards[index] = reward
            dones[index] = float(done)
            rollout["actor_obs"] = next_actor_obs
            rollout["critic_obs"] = next_critic_obs_batch[-1]
            rollout["episode_reward"] = float(rollout["episode_reward"]) + float(reward)
            if done:
                record = rollout["record"]
                profile = rollout["profile"]
                completed_episodes.append({
                    "plant_id": record.plant_id,
                    "quality_region": record.quality_region,
                    "command_id": profile.command_id,
                    "command_kind": profile.kind,
                    "episode_reward": rollout["episode_reward"],
                })
                finished_indices.append(index)

        replay.add_batch(
            actor_obs_batch,
            critic_obs_batch,
            actions,
            rewards,
            np.stack(next_actor_obs_batch),
            np.stack(next_critic_obs_batch),
            dones,
        )
        transition_steps += config.parallel_envs
        target_updates = max(0, transition_steps - config.warmup_steps) // config.update_every_steps
        while updates < target_updates and len(replay) >= config.batch_size:
            losses = learner.update(replay.sample(config.batch_size, learner.device))
            updates += 1

        for index in finished_indices:
            if next_assignment < len(assignments):
                rollouts[index] = start_rollout(next_assignment)
                next_assignment += 1

        if transition_steps >= next_progress:
            progress = {
                "step": transition_steps,
                "updates": updates,
                "completed_episodes": len(completed_episodes),
                "elapsed_s": time.perf_counter() - started,
                "last_losses": losses,
            }
            _write_json(destination / "progress.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            while next_progress <= transition_steps:
                next_progress += config.progress_interval_steps

    training_elapsed = time.perf_counter() - started
    checkpoint = {
        "teacher_kind": config.teacher_kind,
        "actor": learner.actor.state_dict(),
        "critic": learner.critic.state_dict(),
        "target_critic": learner.target_critic.state_dict(),
        "log_alpha": learner.log_alpha.detach().cpu(),
        "actor_observation_dim": int(first_actor_obs.size),
        "critic_observation_dim": int(first_critic_obs.size),
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
        inference_batch_size=config.evaluation_batch_size,
    )
    controlled_rows = evaluate_policy_pairs(
        _TeacherPolicy(learner),
        validation_records,
        held_out_profiles,
        controller_name=f"{config.teacher_kind.upper()}-SAC",
        seed=config.seed + 20_000,
        inference_batch_size=config.evaluation_batch_size,
    )
    evaluation_summary = summarize_trial_evaluation(raw_rows, controlled_rows)
    report: dict[str, object] = {
        "status": "complete",
        "config": checkpoint["config"],
        "parameter_counts": learner.parameter_counts(),
        "training_aircraft_count": len(train_records),
        "training_command_count": len(profiles),
        "completed_episodes": completed_episodes,
        "training_coverage": {
            "quality_regions": dict(Counter(row["quality_region"] for row in completed_episodes)),
            "command_kinds": dict(Counter(row["command_kind"] for row in completed_episodes)),
            "unique_plants": len({row["plant_id"] for row in completed_episodes}),
            "unique_commands": len({row["command_id"] for row in completed_episodes}),
        },
        "updates": updates,
        "training_elapsed_s": training_elapsed,
        "evaluation_elapsed_s": time.perf_counter() - evaluation_started,
        "last_losses": losses,
        "evaluation_summary": evaluation_summary,
        "raw_evaluation_rows": raw_rows,
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
