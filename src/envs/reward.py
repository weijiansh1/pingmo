"""Dense tracking reward for non-formal control diagnostics."""


def roll_quality_reward(
    error: float,
    action: float,
    action_delta: float,
    constraint: float = 0.0,
    applied_action_delta: float = 0.0,
    late_error: float = 0.0,
) -> float:
    return -(
        error * error
        + 0.05 * action * action
        + 0.75 * action_delta * action_delta
        + 0.15 * applied_action_delta * applied_action_delta
        + 0.50 * late_error * late_error
        + constraint
    )
