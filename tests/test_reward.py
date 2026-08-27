from src.envs.reward import roll_quality_reward


def test_reward_penalizes_tracking_action_and_action_increment() -> None:
    near = roll_quality_reward(error=0.0, action=0.0, action_delta=0.0)
    far = roll_quality_reward(error=1.0, action=1.0, action_delta=1.0)
    assert near > far
