"""One aircraft: pitch short period, roll P-channel, pedal channels and kinematics.

State x = [V, alpha, q, theta, beta, p, r, phi, psi, X, Y, Z] in the mind-map
order.  Inputs are Fs (pull positive), Fas (right positive) and Frp (right
pedal positive), all in N.  ``step`` returns the state at the current sample
and then advances with the current input held for dt, the convention of
``PChannel.step``.

The LOES p and r are stability-axis rates (rolling about the velocity vector),
so a bank at constant alpha does not swing the velocity sideways.  They are
rotated through alpha into body rates for the attitude kinematics and the
inertial coupling term.  Attitude is propagated as a rotation matrix with an
exact per-step rotation, then read out as Euler angles; this satisfies
theta_dot = q cos(phi) - r sin(phi) of the mind map without Euler-angle
integration drift or the theta = 90 deg singularity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy import signal

from pinmo6.kinematics import body_to_ned, body_velocity, euler_from_dcm, rotation_step, stability_to_body
from pinmo6.lateral import DirectionalParameters, directional_polynomials
from pinmo6.longitudinal import LongitudinalParameters, short_period_matrices
from src.aircraft.delay import FractionalDelay
from src.aircraft.p_channel import PChannel
from src.aircraft.parameters import PChannelParameters

STATE_NAMES = ("V", "alpha", "q", "theta", "beta", "p", "r", "phi", "psi", "X", "Y", "Z")


@dataclass(frozen=True, slots=True)
class InertiaRatios:
    """Roll inertial coupling into pitch (mind map, 组合轴).

    Defaults sit inside the 0.8-1.0 and 0-0.05 spans of the Navion, F-16 and
    B747 inertias; they are not quality parameters.
    """
    k_zx: float = 0.9   # (Iz - Ix) / Iy
    k_xz: float = 0.02  # Ixz / Iy

    def pitch_acceleration(self, p: float, r: float) -> float:
        return self.k_zx * p * r + self.k_xz * (r * r - p * p)


@dataclass(frozen=True, slots=True)
class SixChannelParameters:
    longitudinal: LongitudinalParameters
    roll: PChannelParameters
    directional: DirectionalParameters
    inertia: InertiaRatios = field(default_factory=InertiaRatios)


class _Transfer:
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray, dt: float) -> None:
        a, b, c, d = signal.tf2ss(numerator, denominator)
        self._ad, self._bd, self._cd, self._dd, _ = signal.cont2discrete((a, b, c, d), dt, method="zoh")
        self._state = np.zeros(self._ad.shape[0])

    def reset(self) -> None:
        self._state.fill(0.0)

    def step(self, value: float) -> float:
        output = float((self._cd @ self._state + self._dd.squeeze() * value).item())
        self._state = self._ad @ self._state + self._bd[:, 0] * value
        return output


class SixChannelAircraft:
    def __init__(self, parameters: SixChannelParameters, dt: float = 0.001,
                 inertial_coupling: bool = True) -> None:
        self.parameters, self.dt = parameters, dt
        self.inertial_coupling = inertial_coupling
        lon, roll, directional = parameters.longitudinal, parameters.roll, parameters.directional
        a, b = short_period_matrices(lon)
        self._ad, self._bd, *_ = signal.cont2discrete((a, b, np.eye(2), np.zeros((2, 2))), dt, method="zoh")
        self._pitch_delay = FractionalDelay(dt, lon.tau_theta)
        self._roll = PChannel(roll, dt=dt)
        self._pedal_delay = FractionalDelay(dt, directional.tau_rp)
        polynomials = directional_polynomials(roll, directional)
        self._beta = _Transfer(*polynomials["beta"], dt)
        self._yaw = _Transfer(*polynomials["r"], dt)
        self.reset()

    def reset(self) -> None:
        self._short = np.zeros(2)
        self._attitude = np.array([0.0, self.parameters.longitudinal.alpha_trim, 0.0])
        self._dcm = body_to_ned(*self._attitude)
        self._position = np.zeros(3)
        self._pitch_delay.reset(); self._pedal_delay.reset(); self._roll.reset()
        self._beta.reset(); self._yaw.reset()
        self.nz_g = 0.0
        self.gamma = 0.0

    def step(self, fs: float, fas: float, frp: float) -> np.ndarray:
        lon = self.parameters.longitudinal
        alpha_dev, q = self._short
        p, _ = self._roll.step(fas)
        pedal = self._pedal_delay.push(frp)
        beta, r = self._beta.step(pedal), self._yaw.step(pedal)
        alpha = lon.alpha_trim + alpha_dev
        phi, theta, psi = self._attitude
        state = np.array([lon.v0, alpha, q, theta, beta, p, r, phi, psi, *self._position])

        ned = self._dcm @ body_velocity(lon.v0, alpha, beta)
        self.nz_g = lon.n_alpha * alpha_dev
        self.gamma = -math.asin(ned[2] / lon.v0)

        p_body, r_body = stability_to_body(p, r, alpha)
        c_q = self.parameters.inertia.pitch_acceleration(p_body, r_body) if self.inertial_coupling else 0.0
        self._short = self._ad @ self._short + self._bd @ np.array([self._pitch_delay.push(fs), c_q])
        self._dcm = self._dcm @ rotation_step(np.array([p_body, q, r_body]), self.dt)
        self._attitude = euler_from_dcm(self._dcm, self._attitude)
        self._position = self._position + self.dt * ned
        return state


def simulate(parameters: SixChannelParameters, inputs: np.ndarray, dt: float = 0.001,
             inertial_coupling: bool = True) -> dict[str, np.ndarray]:
    """Run ``inputs`` of shape (steps, 3) = [Fs, Fas, Frp]; returns states, nz and gamma."""
    aircraft = SixChannelAircraft(parameters, dt=dt, inertial_coupling=inertial_coupling)
    states = np.empty((len(inputs), len(STATE_NAMES)))
    nz, gamma = np.empty(len(inputs)), np.empty(len(inputs))
    for index, (fs, fas, frp) in enumerate(inputs):
        states[index] = aircraft.step(fs, fas, frp)
        nz[index], gamma[index] = aircraft.nz_g, aircraft.gamma
    return {"time": np.arange(len(inputs)) * dt, "states": states, "nz_g": nz, "gamma": gamma}
