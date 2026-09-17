"""Comprehensive gymnasium-free distillation ablation (local, CPU).

Faithfully reuses the production ``DistillationDataset`` / ``losses`` / student
networks to run a controlled single-variable sweep over the methodology ladder on
synthetic teacher data. This isolates the four hypotheses behind the P0--P1 work
without needing the GPU environment or Teacher checkpoints:

  H1  V6 incremental-action (delta) representation reduces deployed total
      variation vs the v4 dense full-action representation (the 2.133 failure).
  H2  P1 advantage weighting improves holdout ("unseen aircraft") action MSE.
  H3  Odd-symmetry enforcement does not hurt on an odd ground-truth teacher.
  H4  Hard-case weighting sharpens accuracy on high-error rows.

The synthetic "teacher" is a smooth, odd MLP (mirroring a learned specialist),
so a well-distilled Student should be both accurate and smooth. Note: the
``advantage`` column is a *mechanism proxy* (|teacher action increment|), not a
real Q(s,u^T)-Q(s,u^S); the real P1 critic path is exercised separately by
``tests/test_v7_advantage_weighting_smoke.py``.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.dataset import (
    DistillationArrays,
    DistillationDataset,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
)
from src.distillation.losses import (
    incremental_student_losses,
    teacher_action_rate_mse,
    weighted_teacher_action_mse,
)
from src.student.dense.network import DenseConditionalStudent, IncrementalDenseStudent


# --------------------------------------------------------------------------- #
# Synthetic smooth odd teacher
# --------------------------------------------------------------------------- #
def make_teacher(observation_dim: int, theta_dim: int, hidden: int = 32, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)

    def layer(in_f, out_f):
        linear = nn.Linear(in_f, out_f)
        bound = 1.0 / math.sqrt(in_f)
        nn.init.uniform_(linear.weight, -bound, bound, generator=generator)
        nn.init.uniform_(linear.bias, -bound, bound, generator=generator)
        return linear

    trunk = nn.Sequential(
        layer(observation_dim + theta_dim, hidden),
        nn.Tanh(),
        layer(hidden, hidden),
        nn.Tanh(),
        layer(hidden, 1),
    )
    trunk.eval()

    def teacher(observations: np.ndarray, theta: np.ndarray) -> np.ndarray:
        obs = torch.as_tensor(observations, dtype=torch.float32)
        th = torch.as_tensor(theta, dtype=torch.float32)
        with torch.no_grad():
            positive = trunk(torch.cat((obs, th), dim=-1))
            negative = trunk(torch.cat((-obs, th), dim=-1))
        return torch.tanh(0.5 * (positive - negative)).numpy()

    return teacher


# --------------------------------------------------------------------------- #
# Synthetic trajectory generation
# --------------------------------------------------------------------------- #
def generate_arrays(
    teacher,
    *,
    observation_dim: int,
    theta_dim: int,
    n_train_aircraft: int,
    n_val_aircraft: int,
    episodes_per_aircraft: int,
    steps_per_episode: int,
    rho: float,
    seed: int,
    jitter_amp: float = 0.0,
    jitter_freq: float = 0.0,
) -> DistillationArrays:
    rng = np.random.default_rng(seed)
    n_aircraft = n_train_aircraft + n_val_aircraft
    observations: list[np.ndarray] = []
    thetas: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    advantages: list[np.ndarray] = []
    plant_indices: list[np.ndarray] = []
    command_indices: list[np.ndarray] = []
    split_codes: list[np.ndarray] = []
    episode_indices: list[np.ndarray] = []
    policy_steps: list[np.ndarray] = []
    episode_id = 0

    for aircraft in range(n_aircraft):
        theta = rng.normal(0.0, 0.3, size=theta_dim).astype(np.float32)
        split = TRAIN_SPLIT if aircraft < n_train_aircraft else VALIDATION_SPLIT
        for _ in range(episodes_per_aircraft):
            states = np.empty((steps_per_episode, observation_dim), dtype=np.float32)
            states[0] = rng.normal(0.0, 0.3, size=observation_dim).astype(np.float32)
            for step in range(1, steps_per_episode):
                states[step] = rho * states[step - 1] + math.sqrt(1.0 - rho * rho) * (
                    rng.normal(0.0, 0.3, size=observation_dim).astype(np.float32)
                )
            # Channel 3 is the "tracking error": amplify it so hard-case weighting
            # and advantage proxies carry a clean signal.
            states[:, 3] = 0.6 * np.sin(np.linspace(0.0, 2.0 * math.pi, steps_per_episode))

            theta_batch = np.broadcast_to(theta, (steps_per_episode, theta_dim)).astype(
                np.float32
            )
            action = teacher(states, theta_batch).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)
            if jitter_amp > 0.0:
                # High-frequency "SAC-style" action jitter on top of the smooth
                # teacher, so the dense baseline reproduces a TV blow-up.
                action[:, 0] += jitter_amp * np.sin(
                    jitter_freq * np.arange(steps_per_episode)
                )
                action = np.clip(action, -1.0, 1.0)

            # Mechanism proxy for Teacher advantage: |action increment|, the
            # transitions where the Teacher has to move fastest and the Student
            # is most likely to lag. Zero at episode start.
            increments = np.zeros((steps_per_episode, 1), dtype=np.float32)
            increments[1:, 0] = np.abs(np.diff(action[:, 0]))
            advantage = np.clip(increments / 0.05, 0.0, 1.0)

            observations.append(states)
            thetas.append(theta_batch)
            teacher_actions.append(action)
            advantages.append(advantage)
            plant_indices.append(np.full(steps_per_episode, aircraft, dtype=np.int32))
            command_indices.append(np.full(steps_per_episode, aircraft, dtype=np.int32))
            split_codes.append(np.full(steps_per_episode, int(split), dtype=np.uint8))
            episode_indices.append(np.full(steps_per_episode, episode_id, dtype=np.int64))
            policy_steps.append(np.arange(steps_per_episode, dtype=np.int32))
            episode_id += 1

    return DistillationArrays(
        observations=np.concatenate(observations),
        aircraft_parameters=np.concatenate(thetas),
        teacher_actions=np.concatenate(teacher_actions),
        plant_indices=np.concatenate(plant_indices),
        command_indices=np.concatenate(command_indices),
        split_codes=np.concatenate(split_codes),
        episode_indices=np.concatenate(episode_indices),
        policy_step_indices=np.concatenate(policy_steps),
        driver_actions=np.concatenate(teacher_actions),
        advantages=np.concatenate(advantages),
    )


# --------------------------------------------------------------------------- #
# Production-faithful training loop (mirrors train_dense_student loss assembly)
# --------------------------------------------------------------------------- #
@dataclass
class SweepConfig:
    name: str
    architecture: str
    odd_policy: bool = True
    hard_case_weight_boost: float = 0.0
    advantage_weight: float = 0.0
    advantage_clip: float = 1.0
    action_delta_weight: float = 1.0
    incremental_delta_weight: float = 1.0
    excess_weight: float = 1.0
    excess_margin: float = 0.05
    width: int = 128
    residual_blocks: int = 2


def build_model(config: SweepConfig, observation_dim: int, theta_dim: int):
    if config.architecture == "incremental":
        return IncrementalDenseStudent(
            observation_dim,
            theta_dim,
            width=config.width,
            residual_blocks=config.residual_blocks,
            enforce_odd_policy=config.odd_policy,
        )
    return DenseConditionalStudent(
        observation_dim,
        theta_dim,
        width=config.width,
        residual_blocks=config.residual_blocks,
        enforce_odd_policy=config.odd_policy,
    )


def train(
    model: nn.Module,
    config: SweepConfig,
    train_dataset: DistillationDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
) -> nn.Module:
    incremental = config.architecture == "incremental"
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            observation = batch["observation"].to(device)
            theta = batch["aircraft_parameters"].to(device)
            target = batch["teacher_action"].to(device)
            sample_weight = batch["sample_weight"].to(device)
            if incremental:
                previous_action = batch["previous_driver_action"].to(device)
                prediction, delta = model(observation, theta, previous_action)
                losses = incremental_student_losses(
                    prediction,
                    delta,
                    target,
                    batch["residual_delta_label"].to(device),
                    sample_weight,
                    config.excess_margin,
                )
                loss = (
                    losses["action_mse"]
                    + config.incremental_delta_weight * losses["delta_mse"]
                    + config.excess_weight * losses["excess"]
                )
            else:
                prediction = model(observation, theta)
                imitation = weighted_teacher_action_mse(prediction, target, sample_weight)
                previous_prediction = model(
                    batch["previous_observation"].to(device), theta
                )
                rate = teacher_action_rate_mse(
                    prediction,
                    previous_prediction,
                    target,
                    batch["previous_teacher_action"].to(device),
                    batch["policy_step_delta"].to(device),
                    batch["temporal_mask"].to(device),
                    sample_weight,
                )
                loss = imitation + config.action_delta_weight * rate
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return model


def measure_policy(
    model: nn.Module, arrays: DistillationArrays, device: torch.device
) -> dict[str, float]:
    """Autoregressive rollout on the (unseen) validation aircraft.

    Reports both the teacher-forced MSE (the training distribution) and the
    deployed MSE / total-variation ratio (generalization + smoothness).
    """

    incremental = isinstance(model, IncrementalDenseStudent)
    model.eval()
    observations = arrays.observations
    theta = arrays.aircraft_parameters
    teacher_actions = arrays.teacher_actions
    episodes = arrays.episode_indices

    forced_se = 0.0
    deployed_se = 0.0
    tv_teacher = 0.0
    tv_student = 0.0
    max_step_delta = 0.0
    prev_teacher = None
    prev_student = None
    current_episode = None
    count = 0

    with torch.no_grad():
        for i in range(len(observations)):
            episode = int(episodes[i])
            if episode != current_episode:
                current_episode = episode
                prev_teacher = None
                prev_student = None
            observation = torch.as_tensor(observations[i : i + 1], device=device)
            aircraft_parameters = torch.as_tensor(theta[i : i + 1], device=device)
            target = float(teacher_actions[i, 0])
            if incremental:
                forced_prev = 0.0 if prev_teacher is None else prev_teacher
                forced_action, _ = model(
                    observation,
                    aircraft_parameters,
                    torch.tensor([[forced_prev]], device=device),
                )
                deployed_prev = 0.0 if prev_student is None else prev_student
                deployed_action, _ = model(
                    observation,
                    aircraft_parameters,
                    torch.tensor([[deployed_prev]], device=device),
                )
            else:
                forced_action = model(observation, aircraft_parameters)
                deployed_action = forced_action
            forced = float(forced_action[0, 0].cpu())
            deployed = float(deployed_action[0, 0].cpu())
            forced_se += (forced - target) ** 2
            deployed_se += (deployed - target) ** 2
            if prev_teacher is not None:
                tv_teacher += abs(target - prev_teacher)
                tv_student += abs(deployed - prev_student)
                max_step_delta = max(max_step_delta, abs(deployed - prev_student))
            prev_teacher = target
            prev_student = deployed
            count += 1

    return {
        "forced_action_mse": forced_se / count,
        "deployed_action_mse": deployed_se / count,
        "tv_teacher": tv_teacher,
        "tv_student": tv_student,
        "tv_ratio": (tv_student / tv_teacher) if tv_teacher > 0 else float("inf"),
        "max_step_delta": max_step_delta,
    }


# --------------------------------------------------------------------------- #
# Sweep plumbing
# --------------------------------------------------------------------------- #
def _val_arrays(arrays: DistillationArrays) -> DistillationArrays:
    val_mask = arrays.split_codes == VALIDATION_SPLIT
    return DistillationArrays(
        observations=arrays.observations[val_mask],
        aircraft_parameters=arrays.aircraft_parameters[val_mask],
        teacher_actions=arrays.teacher_actions[val_mask],
        plant_indices=arrays.plant_indices[val_mask],
        command_indices=arrays.command_indices[val_mask],
        split_codes=arrays.split_codes[val_mask],
        episode_indices=arrays.episode_indices[val_mask],
        policy_step_indices=arrays.policy_step_indices[val_mask],
        driver_actions=arrays.driver_actions[val_mask],
        advantages=arrays.advantages[val_mask],
    )


def _run_configs(
    configs: list[SweepConfig],
    arrays: DistillationArrays,
    *,
    observation_dim: int,
    theta_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    seed: int,
    tag: str,
) -> list[dict]:
    val_arrays = _val_arrays(arrays)
    results = []
    for config in configs:
        torch.manual_seed(seed)
        np.random.seed(seed)
        train_dataset = DistillationDataset(
            arrays,
            "train",
            hard_case_weight_boost=config.hard_case_weight_boost,
            advantage_weight=config.advantage_weight,
            advantage_clip=config.advantage_clip,
        )
        model = build_model(config, observation_dim, theta_dim)
        started = time.perf_counter()
        train(
            model,
            config,
            train_dataset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
        )
        metrics = measure_policy(model, val_arrays, device)
        row = {
            "tag": tag,
            "config": config.name,
            "architecture": config.architecture,
            "odd_policy": config.odd_policy,
            "hard_case_weight_boost": config.hard_case_weight_boost,
            "advantage_weight": config.advantage_weight,
            "incremental_delta_weight": config.incremental_delta_weight,
            "excess_weight": config.excess_weight,
            "width": config.width,
            "residual_blocks": config.residual_blocks,
            "parameters": model.parameter_count,
            "wall_seconds": round(time.perf_counter() - started, 2),
            **{key: round(value, 6) for key, value in metrics.items()},
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    output = Path(ROOT / "results" / "ablation" / f"distillation_ablation_{tag}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output}\n")
    return results


# --------------------------------------------------------------------------- #
# Regime 1: smooth teacher (easy fit) -- representation + symmetry
# --------------------------------------------------------------------------- #
def run_sweep() -> None:
    device = torch.device("cpu")
    observation_dim = 7
    theta_dim = 8
    teacher = make_teacher(observation_dim, theta_dim, seed=0)
    arrays = generate_arrays(
        teacher,
        observation_dim=observation_dim,
        theta_dim=theta_dim,
        n_train_aircraft=8,
        n_val_aircraft=4,
        episodes_per_aircraft=5,
        steps_per_episode=40,
        rho=0.9,
        seed=2026,
    )
    configs = [
        SweepConfig("v4_dense_baseline", "dense"),
        SweepConfig("v6_incremental", "incremental"),
        SweepConfig("v6_incremental_odd_off", "incremental", odd_policy=False),
        SweepConfig("v6_hardcase", "incremental", hard_case_weight_boost=1.0),
        SweepConfig("v6_advantage_0.5", "incremental", advantage_weight=0.5),
        SweepConfig("v6_advantage_1.0", "incremental", advantage_weight=1.0),
    ]
    _run_configs(
        configs,
        arrays,
        observation_dim=observation_dim,
        theta_dim=theta_dim,
        epochs=60,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-6,
        device=device,
        seed=2026,
        tag="smooth",
    )


# --------------------------------------------------------------------------- #
# Regime 2: jittery teacher -- TV blow-up and the delta-weight trade-off
# --------------------------------------------------------------------------- #
def run_jitter_sweep() -> None:
    device = torch.device("cpu")
    observation_dim = 7
    theta_dim = 8
    teacher = make_teacher(observation_dim, theta_dim, seed=0)
    arrays = generate_arrays(
        teacher,
        observation_dim=observation_dim,
        theta_dim=theta_dim,
        n_train_aircraft=8,
        n_val_aircraft=4,
        episodes_per_aircraft=5,
        steps_per_episode=40,
        rho=0.9,
        seed=2026,
        jitter_amp=0.15,
        jitter_freq=6.0,
    )
    configs = [
        SweepConfig("dense", "dense"),
        SweepConfig("inc_dw_5.0", "incremental", incremental_delta_weight=5.0),
        SweepConfig("inc_dw_1.0", "incremental", incremental_delta_weight=1.0),
        SweepConfig("inc_dw_0.1", "incremental", incremental_delta_weight=0.1),
        SweepConfig("inc_dw_0.0", "incremental", incremental_delta_weight=0.0),
    ]
    _run_configs(
        configs,
        arrays,
        observation_dim=observation_dim,
        theta_dim=theta_dim,
        epochs=80,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-6,
        device=device,
        seed=2026,
        tag="jitter",
    )


# --------------------------------------------------------------------------- #
# Regime 3: under-parameterized student -- weighting headroom
# --------------------------------------------------------------------------- #
def run_capacity_sweep() -> None:
    device = torch.device("cpu")
    observation_dim = 7
    theta_dim = 8
    teacher = make_teacher(observation_dim, theta_dim, seed=0)
    arrays = generate_arrays(
        teacher,
        observation_dim=observation_dim,
        theta_dim=theta_dim,
        n_train_aircraft=8,
        n_val_aircraft=4,
        episodes_per_aircraft=5,
        steps_per_episode=40,
        rho=0.9,
        seed=2026,
        jitter_amp=0.06,
        jitter_freq=5.0,
    )
    configs = [
        SweepConfig("tiny_dense", "dense", width=16, residual_blocks=1),
        SweepConfig("tiny_inc", "incremental", width=16, residual_blocks=1),
        SweepConfig(
            "tiny_inc_adv_0.5", "incremental", width=16, residual_blocks=1, advantage_weight=0.5
        ),
        SweepConfig(
            "tiny_inc_adv_2.0", "incremental", width=16, residual_blocks=1, advantage_weight=2.0
        ),
        SweepConfig(
            "tiny_inc_hardcase", "incremental", width=16, residual_blocks=1, hard_case_weight_boost=2.0
        ),
    ]
    _run_configs(
        configs,
        arrays,
        observation_dim=observation_dim,
        theta_dim=theta_dim,
        epochs=80,
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-6,
        device=device,
        seed=2026,
        tag="capacity",
    )


if __name__ == "__main__":
    run_sweep()
    run_jitter_sweep()
    run_capacity_sweep()
