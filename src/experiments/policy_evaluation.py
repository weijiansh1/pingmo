"""Split-safe, pairwise evaluation for constrained roll-quality policies."""

from __future__ import annotations

from typing import Iterable, Protocol

import numpy as np

from src.aircraft.sampler import PlantRecord
from src.envs.commands import CommandProfile
from src.envs.p_channel_env import RollQualityEnv


class PredictivePolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> object: ...


def _policy_action(policy: PredictivePolicy, observation: np.ndarray) -> np.ndarray:
    prediction = policy.predict(observation, deterministic=True)
    return np.asarray(prediction[0] if isinstance(prediction, tuple) else prediction, dtype=np.float32)


def evaluate_policy_pairs(
    policy: PredictivePolicy,
    records: Iterable[PlantRecord],
    profiles: Iterable[CommandProfile],
    *,
    horizon_steps: int = 250,
    seed: int = 0,
) -> list[dict[str, float | int | str]]:
    """Evaluate every supplied aircraft-command pair without changing policy state."""
    rows: list[dict[str, float | int | str]] = []
    pair_index = 0
    for record in records:
        for profile in profiles:
            env = RollQualityEnv([record], horizon_steps=horizon_steps, command_profiles=(profile,))
            observation, _ = env.reset(seed=seed + pair_index)
            rates: list[float] = []
            references: list[float] = []
            command_increments: list[float] = []
            applied_increments: list[float] = []
            rewards: list[float] = []
            final_info: dict[str, object] = {}
            while True:
                observation, reward, terminated, truncated, info = env.step(_policy_action(policy, observation))
                rates.append(float(observation[1]))
                references.append(float(info["p_ref"]))
                scale = env.correction_ratio * env.pilot_force_scale_n
                command_increments.append(float(info["command_delta"]) * scale)
                applied_increments.append(float(info["applied_action_delta"]) * scale)
                rewards.append(float(reward))
                final_info = info
                if terminated or truncated:
                    break
            rows.append(
                {
                    "plant_id": record.plant_id,
                    "split": record.split,
                    "command_id": profile.command_id,
                    "steps": len(rewards),
                    "episode_reward": float(np.sum(rewards)),
                    "tracking_rmse": float(np.sqrt(np.mean(np.square(np.asarray(rates) - np.asarray(references))))),
                    "saturation_fraction": float(final_info["saturation_fraction"]),
                    "command_total_variation_n": float(np.sum(np.abs(command_increments))),
                    "applied_total_variation_n": float(np.sum(np.abs(applied_increments))),
                    "max_command_increment_n": float(np.max(np.abs(command_increments))),
                }
            )
            pair_index += 1
    return rows
