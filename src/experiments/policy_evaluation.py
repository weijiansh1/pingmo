"""Matched raw/controller evaluation on complete 1 ms response traces."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from src.aircraft.sampler import PlantRecord
from src.envs.commands import CommandProfile
from src.envs.p_channel_env import RollQualityEnv
from src.quality.modal_response import ModalResponseMetrics, evaluate_modal_response


class PredictivePolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> object: ...


class ZeroPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        shape = (observation.shape[0], 1) if observation.ndim == 2 else (1,)
        return np.zeros(shape, dtype=np.float32), None


def _policy_actions(policy: PredictivePolicy, observations: np.ndarray) -> np.ndarray:
    prediction = policy.predict(observations, deterministic=True)
    actions = np.asarray(prediction[0] if isinstance(prediction, tuple) else prediction, dtype=np.float32)
    batch_size = observations.shape[0]
    if actions.ndim == 1:
        if batch_size == 1:
            return actions.reshape(1, -1)
        if actions.size == batch_size:
            return actions.reshape(batch_size, 1)
    if actions.ndim != 2 or actions.shape[0] != batch_size:
        raise ValueError(f"policy returned action shape {actions.shape} for observation batch {observations.shape}")
    return actions


def _prefixed(metrics: ModalResponseMetrics, prefix: str) -> dict[str, float | str | None]:
    return {f"{prefix}_{name}": value for name, value in metrics.as_dict().items()}


def _matched_metrics(env: RollQualityEnv, profile: CommandProfile) -> tuple[ModalResponseMetrics, ModalResponseMetrics]:
    trace = env.trajectory()
    action_limit_n = env.correction_ratio * env.pilot_force_scale_n
    controlled = evaluate_modal_response(
        trace["time_s"],
        trace["f_pilot_n"],
        trace["p_rad_s"],
        trace["delta_f_n"],
        command_kind=profile.kind,
        frequency_hz=profile.frequency_hz if profile.kind == "sine" else None,
        action_limit_n=action_limit_n,
    )
    raw = evaluate_modal_response(
        trace["time_s"],
        trace["f_pilot_n"],
        trace["raw_p_rad_s"],
        np.zeros_like(trace["delta_f_n"]),
        command_kind=profile.kind,
        frequency_hz=profile.frequency_hz if profile.kind == "sine" else None,
        action_limit_n=action_limit_n,
    )
    return raw, controlled


def evaluate_policy_pairs(
    policy: PredictivePolicy,
    records: Iterable[PlantRecord],
    profiles: Iterable[CommandProfile],
    *,
    controller_name: str = "controller",
    plant_dt_s: float = 0.001,
    policy_dt_s: float = 0.001,
    seed: int = 0,
    inference_batch_size: int = 1,
) -> list[dict[str, float | int | str | None]]:
    """Evaluate every supplied pair, optionally batching Actor inference."""

    if inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")
    rows: list[dict[str, float | int | str | None]] = []
    record_list = list(records)
    profile_list = list(profiles)
    pairs = [
        (pair_index, record, profile)
        for pair_index, (record, profile) in enumerate(
            (record, profile) for record in record_list for profile in profile_list
        )
    ]
    for chunk_start in range(0, len(pairs), inference_batch_size):
        chunk = pairs[chunk_start:chunk_start + inference_batch_size]
        environments: list[RollQualityEnv] = []
        observations: list[np.ndarray] = []
        reward_traces: list[list[float]] = []
        final_infos: list[dict[str, object]] = []
        finished = np.zeros(len(chunk), dtype=bool)
        for pair_index, record, profile in chunk:
            duration_s = profile.duration_s or 10.0
            horizon_steps = int(round(duration_s / policy_dt_s))
            env = RollQualityEnv(
                [record],
                horizon_steps=horizon_steps,
                plant_dt_s=plant_dt_s,
                policy_dt_s=policy_dt_s,
                command_profiles=(profile,),
            )
            observation, _ = env.reset(seed=seed + pair_index)
            environments.append(env)
            observations.append(observation)
            reward_traces.append([])
            final_infos.append({})

        while not bool(np.all(finished)):
            active = np.flatnonzero(~finished)
            observation_batch = np.stack([observations[index] for index in active])
            action_batch = _policy_actions(policy, observation_batch)
            for batch_index, state_index in enumerate(active):
                observation, reward, terminated, truncated, info = environments[state_index].step(
                    action_batch[batch_index]
                )
                observations[state_index] = observation
                reward_traces[state_index].append(float(reward))
                final_infos[state_index] = info
                finished[state_index] = terminated or truncated

        for state_index, (_, record, profile) in enumerate(chunk):
            env = environments[state_index]
            rewards = reward_traces[state_index]
            final_info = final_infos[state_index]
            raw, controlled = _matched_metrics(env, profile)
            raw_onset = raw.response_onset_delay_s
            controlled_onset = controlled.response_onset_delay_s
            added_onset = None if raw_onset is None or controlled_onset is None else controlled_onset - raw_onset
            raw_oscillation = raw.oscillation_ratio_proxy
            controlled_oscillation = controlled.oscillation_ratio_proxy
            oscillation_change = None if raw_oscillation is None or controlled_oscillation is None else controlled_oscillation - raw_oscillation
            raw_sensitivity = raw.sensitivity_1s_deg_per_n
            controlled_sensitivity = controlled.sensitivity_1s_deg_per_n
            sensitivity_harm = bool(
                raw_sensitivity is not None
                and controlled_sensitivity is not None
                and raw_sensitivity <= 3.38
                and controlled_sensitivity > 3.38
            )
            row: dict[str, float | int | str | None] = {
                "plant_id": record.plant_id,
                "split": record.split,
                "quality_region": record.quality_region,
                "command_id": profile.command_id,
                "command_kind": profile.kind,
                "controller": controller_name,
                "policy_steps": len(rewards),
                "plant_samples": len(env.trajectory()["time_s"]) - 1,
                "episode_reward": float(np.sum(rewards)),
                "transport_delay_s": record.parameters.tau_p,
                "added_response_onset_delay_s": added_onset,
                "oscillation_ratio_change": oscillation_change,
                "sensitivity_level_1_harm": int(sensitivity_harm),
                "max_online_added_onset_delay_s": float(final_info["max_added_onset_delay_s"]),
                "cancel_index": float(final_info["cancel_index"]),
                "cancel_correlation": float(final_info["cancel_correlation"]),
            }
            row.update(_prefixed(raw, "raw"))
            row.update(_prefixed(controlled, "controlled"))
            rows.append(row)
    return rows


def evaluate_controller_set(
    controllers: dict[str, PredictivePolicy],
    records: Iterable[PlantRecord],
    profiles: Iterable[CommandProfile],
    *,
    seed: int = 0,
) -> list[dict[str, float | int | str | None]]:
    """Run raw, Linear, MLP-SAC, and MoE-SAC through one common interface."""

    record_list = list(records)
    profile_list = list(profiles)
    rows: list[dict[str, float | int | str | None]] = []
    for index, (name, controller) in enumerate(controllers.items()):
        rows.extend(evaluate_policy_pairs(controller, record_list, profile_list, controller_name=name, seed=seed + index * 1_000_000))
    return rows
