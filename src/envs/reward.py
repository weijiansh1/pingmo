"""Action-dependent response costs for reference-free roll-quality shaping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardWeights:
    wrong_way_response: float = 0.25
    added_onset_delay: float = 1.00
    sensitivity_violation: float = 1.00
    oscillation_response: float = 1.00
    spiral_recovery: float = 0.50
    action_energy: float = 0.02
    action_delta: float = 0.05

    def __post_init__(self) -> None:
        if any(value < 0 for value in (
            self.wrong_way_response,
            self.added_onset_delay,
            self.sensitivity_violation,
            self.oscillation_response,
            self.spiral_recovery,
            self.action_energy,
            self.action_delta,
        )):
            raise ValueError("reward weights must be non-negative")


def roll_quality_reward(
    *,
    wrong_way_cost: float,
    added_delay_cost: float,
    sensitivity_cost: float,
    oscillation_cost: float,
    spiral_recovery_cost: float,
    applied_action: float,
    normalized_command_delta: float,
    step_duration_s: float,
    weights: RewardWeights = RewardWeights(),
) -> tuple[float, dict[str, float]]:
    """Return a scalar reward and its dimensionless cost decomposition."""

    if step_duration_s <= 0:
        raise ValueError("step_duration_s must be positive")

    costs = {
        "wrong_way": weights.wrong_way_response * max(0.0, wrong_way_cost),
        "added_delay": weights.added_onset_delay * max(0.0, added_delay_cost),
        "sensitivity": weights.sensitivity_violation * max(0.0, sensitivity_cost),
        "oscillation": weights.oscillation_response * max(0.0, oscillation_cost),
        "spiral_recovery": weights.spiral_recovery * max(0.0, spiral_recovery_cost),
        "action_energy": weights.action_energy * applied_action * applied_action * step_duration_s,
        "action_delta": weights.action_delta * normalized_command_delta * normalized_command_delta * step_duration_s,
    }
    return -sum(costs.values()), costs
