import pytest

from src.aircraft.parameters import PChannelParameters
from src.aircraft.gain_calibration import sensitivity_1s_deg_per_n
from src.aircraft.reference import DutchRollOracle, ReferenceRollModel, diagnostic_reference_parameters


def test_reference_model_preserves_delay_and_produces_finite_roll_rate() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.2, 1.3, 0.0125)
    model = ReferenceRollModel(parameters, dt=0.005)
    outputs = [model.step(1.0)[0] for _ in range(8)]
    assert outputs[:2] == pytest.approx([0.0, 0.0])
    assert all(abs(value) < 1e6 for value in outputs)


def test_oracle_is_identity_when_zero_and_dutch_roll_factors_match() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.0, 1.0, 0.0)
    oracle = DutchRollOracle(parameters, dt=0.005)
    assert [oracle.step(1.0) for _ in range(4)] == pytest.approx([1.0] * 4)


def test_diagnostic_reference_preserves_raw_sensitivity_while_cancelling_mismatch() -> None:
    raw = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.45, 0.55, 0.0125)
    reference = diagnostic_reference_parameters(raw)

    assert reference.r_omega == 1.0
    assert reference.r_zeta == 1.0
    assert reference.tau_p == raw.tau_p
    assert sensitivity_1s_deg_per_n(reference) == pytest.approx(sensitivity_1s_deg_per_n(raw), rel=0.005)
