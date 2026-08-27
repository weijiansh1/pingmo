from src.envs.reward import RewardWeights, roll_quality_reward


def test_reward_penalizes_tracking_action_and_action_increment() -> None:
    near = roll_quality_reward(error=0.0, action=0.0, action_delta=0.0)
    far = roll_quality_reward(error=1.0, action=1.0, action_delta=1.0)
    assert near > far


def test_reward_penalizes_applied_action_motion_and_late_error() -> None:
    quiet = roll_quality_reward(
        error=0.0,
        action=0.0,
        action_delta=0.0,
        applied_action_delta=0.0,
        late_error=0.0,
    )
    chattering = roll_quality_reward(
        error=0.0,
        action=0.0,
        action_delta=0.2,
        applied_action_delta=0.1,
        late_error=0.0,
    )
    residual = roll_quality_reward(
        error=0.0,
        action=0.0,
        action_delta=0.0,
        applied_action_delta=0.0,
        late_error=0.3,
    )
    assert quiet > chattering
    assert quiet > residual


def test_reward_uses_supplied_action_energy_weight() -> None:
    low_cost = RewardWeights(action_energy=0.01)
    high_cost = RewardWeights(action_energy=0.20)
    assert roll_quality_reward(0.0, 1.0, 0.0, weights=high_cost) < roll_quality_reward(0.0, 1.0, 0.0, weights=low_cost)
