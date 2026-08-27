import pytest

from src.aircraft.constrained_oracle import ConstrainedDutchRollOracle
from src.aircraft.parameters import PChannelParameters


def test_constrained_oracle_respects_force_and_rate_limits() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    oracle = ConstrainedDutchRollOracle(parameters, augmentation_limit=0.3, normalized_rate_limit_s_inv=4.0, dt=0.02)
    corrections = [oracle.step(1.0) for _ in range(30)]

    assert max(abs(value) for value in corrections) <= 0.3
    assert max(abs(right - left) for left, right in zip(corrections, corrections[1:])) <= 0.3 * 4.0 * 0.02 + 1e-9
