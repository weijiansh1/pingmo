"""Pedal-force channels beta/Frp and r/Frp, sharing the roll plant's lateral poles.

Mind-map forms (Dutch-roll pair and spiral root taken from the roll plant):

    beta/Frp = K_beta e^{-tau s} / (s^2 + 2 zeta_d omega_d s + omega_d^2)
    r/Frp    = (r/delta_r)(delta_r/Frp)
             = A_r G_r (s^2 + 2 zeta_r omega_r s + omega_r^2) e^{-tau s}
               / ((s - lambda_s)(s^2 + 2 zeta_d omega_d s + omega_d^2))

Both are driven by the same yaw moment, so one gain k_rp = A_r G_r sets both:
in the Dutch-roll approximation beta_dot ~ -r, hence K_beta = -k_rp.  The two
channels share one pedal transport delay because they share the input path.
Signs: right pedal (Frp > 0) gives nose-right yaw (r > 0) and left sideslip.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.aircraft.parameters import PChannelParameters


@dataclass(frozen=True, slots=True)
class DirectionalParameters:
    sideslip_per_force_deg_per_n: float  # |beta_ss / Frp| of the second-order form
    r_omega_r: float                     # omega_r / omega_d
    zeta_r: float
    tau_rp: float                        # s

    def __post_init__(self) -> None:
        if self.sideslip_per_force_deg_per_n <= 0 or self.r_omega_r <= 0:
            raise ValueError("sideslip gradient and omega_r ratio must be positive")
        if not 0 < self.zeta_r < 1 or self.tau_rp < 0:
            raise ValueError("zeta_r must be in (0, 1) and tau_rp non-negative")


def pedal_gain(roll: PChannelParameters, d: DirectionalParameters) -> float:
    """k_rp = A_r G_r, initial yaw acceleration per pedal force (rad/s^2/N)."""
    return math.radians(d.sideslip_per_force_deg_per_n) * roll.omega_d**2


def directional_polynomials(roll: PChannelParameters, d: DirectionalParameters) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    k_rp = pedal_gain(roll, d)
    dutch = np.array([1.0, 2.0 * roll.zeta_d * roll.omega_d, roll.omega_d**2])
    omega_r = d.r_omega_r * roll.omega_d
    return {
        "beta": (np.array([-k_rp]), dutch),
        "r": (k_rp * np.array([1.0, 2.0 * d.zeta_r * omega_r, omega_r**2]),
              np.polymul([1.0, -roll.lambda_s], dutch)),
    }


DIRECTIONAL_UNIT_DIM = 4


def directional_from_unit(unit: np.ndarray) -> DirectionalParameters:
    """Research envelope, not a GJB grade.

    10 deg of sideslip needs 50-500 N of pedal; the r/delta_r zero pair sits at
    0.1-0.5 omega_d with damping 0.05-0.6, bracketing the Navion values (0.22, 0.14).
    """
    u_s, u_w, u_z, u_tau = map(float, unit)
    return DirectionalParameters(
        sideslip_per_force_deg_per_n=math.exp(math.log(0.02) + u_s * math.log(10.0)),
        r_omega_r=0.1 + 0.4 * u_w,
        zeta_r=0.05 + 0.55 * u_z,
        tau_rp=0.001 + 0.199 * u_tau,
    )
