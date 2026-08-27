"""Reference roll model and analytical Dutch-roll input-shaping oracle."""

from dataclasses import replace

import numpy as np
from scipy import signal

from src.aircraft.delay import FractionalDelay
from src.aircraft.gain_calibration import calibrate_l_fa_for_sensitivity, sensitivity_1s_deg_per_n
from src.aircraft.parameters import PChannelParameters


class _DiscreteTransfer:
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray, dt: float) -> None:
        a, b, c, d = signal.tf2ss(numerator, denominator)
        self.ad, self.bd, self.cd, self.dd, _ = signal.cont2discrete((a, b, c, d), dt, method="zoh")
        self.state = np.zeros(self.ad.shape[0])

    def step(self, value: float) -> float:
        output = float((self.cd @ self.state + self.dd.squeeze() * value).item())
        self.state = self.ad @ self.state + self.bd[:, 0] * value
        return output

    def reset(self) -> None:
        self.state.fill(0.0)


def diagnostic_reference_parameters(parameters: PChannelParameters, dt: float = 0.005) -> PChannelParameters:
    """Cancel only pole-zero mismatch while retaining raw 1 s roll sensitivity and delay."""
    target_sensitivity = sensitivity_1s_deg_per_n(parameters, dt=dt)
    cancelled = replace(parameters, r_omega=1.0, r_zeta=1.0)
    return calibrate_l_fa_for_sensitivity(cancelled, target_sensitivity, dt=dt)


class ReferenceRollModel:
    """M*: preserves roll mode, sensitivity and delay, removes Dutch-roll factor."""
    def __init__(self, parameters: PChannelParameters, dt: float = 0.005) -> None:
        self.dt = dt
        self.parameters = diagnostic_reference_parameters(parameters, dt=dt)
        numerator = np.array([self.parameters.l_fa, 0.0, 0.0])
        denominator = np.polymul([1.0, -self.parameters.lambda_s], [1.0, 1.0 / self.parameters.t_r])
        self.transfer = _DiscreteTransfer(numerator, denominator, dt)
        self.delay = FractionalDelay(dt, self.parameters.tau_p)
        self.previous_output = 0.0

    def step(self, f_pilot: float) -> tuple[float, float]:
        output = self.transfer.step(self.delay.push(f_pilot))
        derivative = (output - self.previous_output) / self.dt
        self.previous_output = output
        return output, derivative

    def reset(self) -> None:
        self.transfer.reset(); self.delay.reset(); self.previous_output = 0.0

    def privileged_state(self, delay_width: int) -> np.ndarray:
        return np.concatenate((self.transfer.state, self.delay.state(delay_width))).astype(np.float32)


class DutchRollOracle:
    """C*=D_d/Z_phi; produces the unconstrained analytically shaped equivalent input."""
    def __init__(self, parameters: PChannelParameters, dt: float = 0.005) -> None:
        reference = diagnostic_reference_parameters(parameters, dt=dt)
        gain = reference.l_fa / parameters.l_fa
        numerator = gain * np.array([1.0, 2 * parameters.zeta_d * parameters.omega_d, parameters.omega_d**2])
        denominator = np.array([1.0, 2 * parameters.zeta_phi * parameters.omega_phi, parameters.omega_phi**2])
        self.transfer = _DiscreteTransfer(numerator, denominator, dt)

    def step(self, f_pilot: float) -> float:
        return self.transfer.step(f_pilot)

    def reset(self) -> None:
        self.transfer.reset()
