from __future__ import annotations

import numpy as np
import pytest

from src.controllers.policy_wrappers import ForceSlewLimitedPolicy


class _SequencePolicy:
    def __init__(self, actions: list[list[float]]) -> None:
        self.actions = [np.asarray(action, dtype=np.float32) for action in actions]
        self.index = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.index = 0
        self.reset_count += 1

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        del observation, deterministic
        action = self.actions[self.index]
        self.index += 1
        return action


def test_force_slew_limiter_bounds_each_increment_from_zero() -> None:
    inner = _SequencePolicy([[1.0], [1.0], [-1.0]])
    policy = ForceSlewLimitedPolicy(
        inner,
        force_rate_limit_n_s=11.0,
        policy_dt_s=0.02,
        force_limit_n=22.0,
    )

    actions = [policy.predict(np.zeros(1)) for _ in range(3)]

    assert policy.maximum_normalized_increment == pytest.approx(0.01)
    assert np.asarray(actions)[:, 0] == pytest.approx([0.01, 0.02, 0.01])


def test_force_slew_limiter_reset_resets_inner_policy_and_action_state() -> None:
    inner = _SequencePolicy([[0.5], [0.5]])
    policy = ForceSlewLimitedPolicy(
        inner,
        force_rate_limit_n_s=22.0,
        policy_dt_s=0.02,
        force_limit_n=22.0,
    )

    first = policy.predict(np.zeros(1))
    policy.reset()
    restarted = policy.predict(np.zeros(1))

    assert inner.reset_count == 1
    assert first == pytest.approx([0.02])
    assert restarted == pytest.approx([0.02])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("force_rate_limit_n_s", 0.0),
        ("policy_dt_s", -0.02),
        ("force_limit_n", float("nan")),
    ],
)
def test_force_slew_limiter_rejects_invalid_contract(name: str, value: float) -> None:
    arguments = {
        "force_rate_limit_n_s": 11.0,
        "policy_dt_s": 0.02,
        "force_limit_n": 22.0,
    }
    arguments[name] = value

    with pytest.raises(ValueError):
        ForceSlewLimitedPolicy(_SequencePolicy([[0.0]]), **arguments)


def test_force_slew_limiter_rejects_action_shape_changes() -> None:
    policy = ForceSlewLimitedPolicy(
        _SequencePolicy([[0.0], [0.0, 0.0]]),
        force_rate_limit_n_s=11.0,
        policy_dt_s=0.02,
        force_limit_n=22.0,
    )
    policy.predict(np.zeros(1))

    with pytest.raises(ValueError, match="shape changed"):
        policy.predict(np.zeros(1))
