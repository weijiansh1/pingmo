import numpy as np
import pytest

from src.aircraft.p_channel import PChannel
from src.aircraft.parameters import PChannelParameters


def test_parameters_derive_roll_mode_and_reject_invalid_damping() -> None:
    parameters = PChannelParameters(
        l_fa=1.2, lambda_s=-0.04, t_r=0.5, zeta_d=0.2,
        omega_d=2.0, r_omega=1.1, r_zeta=1.5, tau_p=0.0,
    )
    assert parameters.omega_phi == pytest.approx(2.2)
    assert parameters.zeta_phi == pytest.approx(0.3)
    with pytest.raises(ValueError, match="zeta_d"):
        PChannelParameters(1.0, -0.04, 0.5, 1.1, 2.0, 1.0, 1.0, 0.0)


def test_p_channel_is_linear_for_equal_initial_conditions() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.0, 1.0, 0.0)
    one = PChannel(parameters, dt=0.005)
    two = PChannel(parameters, dt=0.005)
    response_one = np.array([one.step(1.0)[0] for _ in range(300)])
    response_two = np.array([two.step(2.0)[0] for _ in range(300)])
    assert response_two == pytest.approx(2.0 * response_one, abs=1e-9)
