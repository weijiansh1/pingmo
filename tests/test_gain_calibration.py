from dataclasses import replace

import pytest

from src.aircraft.gain_calibration import calibrate_l_fa_for_sensitivity, sensitivity_1s_deg_per_n
from src.aircraft.parameters import PChannelParameters


def _parameters(l_fa: float = 1.0) -> PChannelParameters:
    return PChannelParameters(l_fa, -0.04, 0.5, 0.2, 2.0, 1.2, 1.3, 0.0125)


def test_roll_sensitivity_scales_linearly_with_l_fa() -> None:
    baseline = sensitivity_1s_deg_per_n(_parameters(1.0))
    doubled = sensitivity_1s_deg_per_n(_parameters(2.0))

    assert doubled == pytest.approx(2.0 * baseline, rel=1e-3)


def test_calibration_hits_requested_iv_a_sensitivity() -> None:
    calibrated = calibrate_l_fa_for_sensitivity(_parameters(), target_sensitivity_deg_per_n=3.38)

    assert sensitivity_1s_deg_per_n(calibrated) == pytest.approx(3.38, rel=0.005)
    assert calibrated.l_fa > 0
