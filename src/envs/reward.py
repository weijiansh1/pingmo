"""Dense reward without embedding certification metrics."""


def roll_quality_reward(error: float, action: float, action_delta: float, constraint: float = 0.0) -> float:
    return -(error * error + 0.02 * action * action + 0.05 * action_delta * action_delta + constraint)
