"""Coupled 品模: LOES-exact linear aerodynamics with nonlinear gravity, kinematics and inertia.

Aerodynamic and control terms are the stability-axis derivatives of
``derivatives.py``; about wings-level trim they reproduce the GJB pitch short
period, the frozen roll channel and the cross channels and long period those
LOES leave out.  For large motions the force equations keep the wind-axis
kinematic and gravity terms (Stevens & Lewis, wind-axes force equations):

    V_dot     = aero_u + g_w[0]
    alpha_dot = q - tan(beta) p_w + aero_alpha + (g_w[2] - g) / (V cos beta)
    beta_dot  = -r_w + aero_beta + g_w[1] / V

where g_w is gravity in wind axes (zero perturbation at level trim).  The
states p_s, r_s are rates about the stability axes, fixed in the body at the
trim alpha (the axes the derivatives belong to); p_w, r_w are the same rates
about axes turned by the current alpha:

    p_w =  p_s cos(d_alpha) + r_s sin(d_alpha)
    r_w = -p_s sin(d_alpha) + r_s cos(d_alpha)

so a roll at an alpha off trim trades alpha for sideslip (kinematic coupling).
The moment equations add the full Euler inertial coupling of a body with I_xz.  Gusts enter only the aerodynamic terms,
through the air-relative u - u_g, alpha - alpha_g, beta - beta_g, p - p_g,
q - q_g, r - r_g.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal

from pinmo6 import G
from pinmo6.derivatives import LateralDerivatives, LongitudinalDerivatives
from pinmo6.longitudinal import LongitudinalParameters
from pinmo6.kinematics import body_to_ned, body_velocity, euler_from_dcm, rotation_step, stability_to_body
from src.aircraft.delay import FractionalDelay
from src.aircraft.parameters import PChannelParameters

STATE_NAMES = ("V", "alpha", "q", "theta", "beta", "p", "r", "phi", "psi", "X", "Y", "Z")


@dataclass(frozen=True, slots=True)
class InertiaMoments:
    """Moments of inertia normalized by I_x, body axes."""
    iy: float
    iz: float
    ixz: float = 0.0

    def gyroscopic(self, p: float, q: float, r: float) -> tuple[float, float, float]:
        """Body-axis angular accelerations from the inertial terms of Euler's equations."""
        ix, iy, iz, ixz = 1.0, self.iy, self.iz, self.ixz
        roll = (iy - iz) * q * r + ixz * p * q
        yaw = (ix - iy) * p * q - ixz * q * r
        det = ix * iz - ixz * ixz
        return ((iz * roll + ixz * yaw) / det,
                ((iz - ix) * p * r + ixz * (r * r - p * p)) / iy,
                (ixz * roll + ix * yaw) / det)


def _wind_from_body(alpha: float, beta: float) -> np.ndarray:
    ca, sa, cb, sb = math.cos(alpha), math.sin(alpha), math.cos(beta), math.sin(beta)
    return np.array([[ca * cb, sb, sa * cb],
                     [-ca * sb, cb, -sa * sb],
                     [-sa, 0.0, ca]])


class CoupledAircraft:
    """State [u, d_alpha, q, beta, p_s, r_s] plus attitude, position and engine thrust."""

    def __init__(self, loes: LongitudinalParameters, roll: PChannelParameters, tau_rp: float,
                 lateral: LateralDerivatives, longitudinal: LongitudinalDerivatives, inertia: InertiaMoments,
                 engine_lag_s: float = 1.0, dt: float = 0.001) -> None:
        """``loes`` supplies trim (V0, alpha), the pitch delay; ``roll`` the roll delay."""
        self.loes, self.roll, self.lateral, self.longitudinal = loes, roll, lateral, longitudinal
        self.inertia, self.engine_lag_s, self.dt = inertia, engine_lag_s, dt
        lon, lat = longitudinal, lateral
        aero = np.zeros((6, 6))
        aero[:3, :3] = [[lon.x_u, lon.x_a - G, 0.0], [lon.z_u, lon.z_a, 0.0], [lon.m_u, lon.m_a, lon.m_q]]
        aero[3:, 3:] = [[lat.y_b, 0.0, 0.0], [lat.l_b, lat.l_p, lat.l_r], [lat.n_b, lat.n_p, lat.n_r]]
        kinematic = np.zeros((6, 6))
        kinematic[1, 2], kinematic[3, 5] = 1.0, -1.0      # alpha_dot += q, beta_dot -= r_s
        controls = np.zeros((6, 3))                       # [F_s, F_as, F_rp]
        controls[2, 0] = lon.m_s
        controls[3:, 1] = [0.0, lat.l_a, lat.n_a]
        controls[3:, 2] = [lat.y_rp, lat.l_rp, lat.n_rp]
        self._aero = aero
        a = aero + kinematic
        b = np.hstack((controls, np.eye(6)))
        self._ad, self._bd, *_ = signal.cont2discrete((a, b, np.eye(6), np.zeros((6, 9))), dt, method="zoh")
        self._delays = [FractionalDelay(dt, loes.tau_theta), FractionalDelay(dt, roll.tau_p),
                        FractionalDelay(dt, tau_rp)]
        self.reset()

    def reset(self) -> None:
        trim = self.loes.alpha_trim
        self._x = np.zeros(6)
        self._dcm = body_to_ned(0.0, trim, 0.0)
        self._euler = np.array([0.0, trim, 0.0])
        self._position = np.zeros(3)
        self._thrust = 0.0
        for delay in self._delays:
            delay.reset()
        self.nz_g = 0.0
        self.gamma = 0.0

    def step(self, fs: float, fas: float, frp: float, throttle: float = 0.0,
             direct_lift: float = 0.0, gust: np.ndarray | None = None) -> np.ndarray:
        """Advance one dt; ``throttle`` and ``direct_lift`` are force increments per weight.

        ``gust`` = [u_g, alpha_g, q_g, beta_g, p_g, r_g] in state order (m/s, rad, rad/s), held over dt.
        """
        base = self.loes
        u, d_alpha, q, beta, p_s, r_s = self._x
        v = base.v0 + u
        alpha = base.alpha_trim + d_alpha
        phi, theta, psi = self._euler
        state = np.array([v, alpha, q, theta, beta, p_s, r_s, phi, psi, *self._position])

        ned = self._dcm @ body_velocity(v, alpha, beta)
        self.gamma = -math.asin(max(-1.0, min(1.0, ned[2] / v)))
        g_w = _wind_from_body(alpha, beta) @ (self._dcm.T @ np.array([0.0, 0.0, G]))
        gust = np.zeros(6) if gust is None else np.asarray(gust, dtype=float)
        aero_gust = -self._aero @ gust
        self.nz_g = (-(base.v0 / G) * (self.longitudinal.z_u * (u - gust[0])
                                        + self.longitudinal.z_a * (d_alpha - gust[1]))
                     + direct_lift)

        # stability axes are body-fixed at the trim alpha, so their rates need no transport terms
        p_b, r_b = stability_to_body(p_s, r_s, base.alpha_trim)
        dp_b, dq_b, dr_b = self.inertia.gyroscopic(p_b, q, r_b)
        ca, sa = math.cos(base.alpha_trim), math.sin(base.alpha_trim)
        cd, sd = math.cos(d_alpha), math.sin(d_alpha)
        p_w, r_w = p_s * cd + r_s * sd, -p_s * sd + r_s * cd
        accelerations = aero_gust + np.array([
            g_w[0] + G * self._thrust,
            -math.tan(beta) * p_w + (g_w[2] - G) / (v * math.cos(beta)) - G * direct_lift / v,
            dq_b,
            r_s - r_w + g_w[1] / v,                       # the linear part already holds -r_s
            dp_b * ca + dr_b * sa,
            -dp_b * sa + dr_b * ca,
        ])
        delayed = [delay.push(value) for delay, value in zip(self._delays, (fs, fas, frp))]
        self._x = self._ad @ self._x + self._bd @ np.concatenate((delayed, accelerations))
        self._thrust += self.dt * (throttle - self._thrust) / self.engine_lag_s
        self._dcm = self._dcm @ rotation_step(np.array([p_b, q, r_b]), self.dt)
        self._euler = euler_from_dcm(self._dcm, self._euler)
        self._position = self._position + self.dt * ned
        return state


def simulate_coupled(aircraft: CoupledAircraft, inputs: np.ndarray,
                     gusts: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """``inputs`` (steps, 3..5) = [F_s, F_as, F_rp, throttle, direct_lift]."""
    aircraft.reset()
    steps = len(inputs)
    padded = np.zeros((steps, 5))
    padded[:, :inputs.shape[1]] = inputs
    states, nz, gamma = np.empty((steps, len(STATE_NAMES))), np.empty(steps), np.empty(steps)
    for index in range(steps):
        states[index] = aircraft.step(*padded[index], gust=None if gusts is None else gusts[index])
        nz[index], gamma[index] = aircraft.nz_g, aircraft.gamma
    return {"time": np.arange(steps) * aircraft.dt, "states": states, "nz_g": nz, "gamma": gamma}
