"""Exact continuous P-channel discretized with ZOH at the plant rate."""

import numpy as np
from scipy import signal

from src.aircraft.delay import FractionalDelay
from src.aircraft.parameters import PChannelParameters


def p_channel_polynomials(parameters: PChannelParameters) -> tuple[np.ndarray, np.ndarray]:
    """Return the corrected single-``s`` P-channel numerator and denominator."""
    p = parameters
    numerator = p.l_fa * np.array([
        1.0,
        2.0 * p.zeta_phi * p.omega_phi,
        p.omega_phi**2,
        0.0,
    ])
    denominator = np.polymul(
        np.polymul([1.0, -p.lambda_s], [1.0, 1.0 / p.t_r]),
        [1.0, 2.0 * p.zeta_d * p.omega_d, p.omega_d**2],
    )
    return numerator, denominator


class PChannel:
    def __init__(self, parameters: PChannelParameters, dt: float = 0.005) -> None:
        self.parameters, self.dt = parameters, dt
        numerator, denominator = p_channel_polynomials(parameters)
        a, b, c, d = signal.tf2ss(numerator, denominator)
        ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), dt, method="zoh")
        self._ad, self._bd, self._cd, self._dd = ad, bd, cd, dd
        self._state = np.zeros(ad.shape[0])
        self._delay = FractionalDelay(dt, parameters.tau_p)
        self._previous_output = 0.0

    def reset(self) -> None:
        self._state.fill(0.0)
        self._delay.reset()
        self._previous_output = 0.0

    def step(self, f_as: float) -> tuple[float, float]:
        delayed = self._delay.push(f_as)
        output = float((self._cd @ self._state + self._dd.squeeze() * delayed).item())
        self._state = self._ad @ self._state + self._bd[:, 0] * delayed
        p_dot = (output - self._previous_output) / self.dt
        self._previous_output = output
        return output, p_dot

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def privileged_state(self, delay_width: int) -> np.ndarray:
        return np.concatenate((self._state, self._delay.state(delay_width))).astype(np.float32)
