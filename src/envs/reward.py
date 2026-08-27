"""Dense tracking reward for non-formal control diagnostics."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    action_energy: float = 0.05
    command_delta: float = 0.75
    applied_delta: float = 0.15
    late_error: float = 0.50


def roll_quality_reward(
    error: float,
    action: float,
    action_delta: float,
    constraint: float = 0.0,
    applied_action_delta: float = 0.0,
    late_error: float = 0.0,
    *,
    weights: RewardWeights = RewardWeights(),
) -> float:
    return -(
        error * error
        + weights.action_energy * action * action
        + weights.command_delta * action_delta * action_delta
        + weights.applied_delta * applied_action_delta * applied_action_delta
        + weights.late_error * late_error * late_error
        + constraint
    )
