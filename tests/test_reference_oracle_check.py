import numpy as np
import pytest

from src.aircraft.parameters import PChannelParameters
from src.experiments.reference_oracle_check import simulate_reference_oracle


def test_reference_oracle_trace_uses_same_force_and_constraint_contract() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)

    trace = simulate_reference_oracle(parameters, pilot_force_n=22.0, duration_s=0.4, correction_ratio=0.3, normalized_rate_limit_s_inv=4.0)

    assert trace.time_s.shape == (80,)
    assert trace.p_raw.shape == trace.p_ref.shape == trace.p_oracle.shape == trace.p_constrained.shape
    assert trace.f_pilot.max() == pytest.approx(22.0)
    assert np.max(np.abs(trace.delta_constrained)) <= 0.3 * 22.0 + 1e-9
    assert np.max(np.abs(np.diff(trace.delta_constrained))) <= 0.3 * 22.0 * 4.0 * 0.02 + 1e-9
    assert trace.metrics["constrained_tracking_rmse"] >= 0.0
    assert trace.metrics["oracle_gap_rmse"] >= 0.0
