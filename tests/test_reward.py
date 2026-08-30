import pytest

from src.envs.reward import RewardWeights, roll_quality_reward


def _reward(**overrides: float) -> tuple[float, dict[str, float]]:
    values = {
        "wrong_way_cost": 0.0,
        "added_delay_cost": 0.0,
        "sensitivity_cost": 0.0,
        "oscillation_cost": 0.0,
        "spiral_recovery_cost": 0.0,
        "applied_action": 0.0,
        "normalized_command_delta": 0.0,
        "step_duration_s": 0.001,
    }
    values.update(overrides)
    return roll_quality_reward(**values)


def test_reward_outputs_named_response_and_action_costs() -> None:
    quiet, quiet_costs = _reward()
    poor, costs = _reward(
        wrong_way_cost=0.2,
        added_delay_cost=0.1,
        sensitivity_cost=0.3,
        oscillation_cost=0.4,
        spiral_recovery_cost=0.5,
        applied_action=1.0,
        normalized_command_delta=1.0,
    )

    assert quiet == 0.0
    assert all(value == 0.0 for value in quiet_costs.values())
    assert poor < quiet
    assert set(costs) == {
        "wrong_way", "added_delay", "sensitivity", "oscillation",
        "spiral_recovery", "action_energy", "action_delta",
    }


def test_continuous_action_cost_is_invariant_to_time_discretization() -> None:
    one_second, _ = _reward(applied_action=1.0, step_duration_s=1.0)
    one_millisecond, _ = _reward(applied_action=1.0, step_duration_s=0.001)

    assert 1_000 * one_millisecond == pytest.approx(one_second)


def test_reward_uses_supplied_action_energy_weight() -> None:
    low, _ = roll_quality_reward(
        wrong_way_cost=0.0,
        added_delay_cost=0.0,
        sensitivity_cost=0.0,
        oscillation_cost=0.0,
        spiral_recovery_cost=0.0,
        applied_action=1.0,
        normalized_command_delta=0.0,
        step_duration_s=0.001,
        weights=RewardWeights(action_energy=0.01),
    )
    high, _ = roll_quality_reward(
        wrong_way_cost=0.0,
        added_delay_cost=0.0,
        sensitivity_cost=0.0,
        oscillation_cost=0.0,
        spiral_recovery_cost=0.0,
        applied_action=1.0,
        normalized_command_delta=0.0,
        step_duration_s=0.001,
        weights=RewardWeights(action_energy=0.20),
    )
    assert high < low
