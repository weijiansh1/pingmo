"""Short-period pitch LOES of GJB 2874-97 A5.1.1.1, realized on physical [alpha, q].

GJB (PDF 76-77) defines pitch rate and centre-of-rotation load factor by

    q/Fs   = K_theta (s + 1/T_theta2) e^{-tau s} / (s^2 + 2 zeta_sp omega_sp s + omega_sp^2)
    nz'/Fs = K_nz e^{-tau s} / (same denominator)

and the mind map adds theta = q/s (wings level), gamma/theta = 1/(T_theta2 s + 1)
and alpha = theta - gamma.  The two-state system

    alpha_dot = -alpha / T_theta2 + q
    q_dot     = M_alpha alpha + M_q q + K_theta Fs + c_q

reproduces q/Fs, alpha/Fs = K_theta e^{-tau s} / den and nz = (n/alpha) alpha
exactly.  c_q is an additive pitch acceleration (roll inertial coupling).

Speed is held at trim: the short-period form assumes no phugoid, so V = V0.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pinmo6 import G

IV_A_LOAD_LIMIT_G = 7.0  # n_L used by GJB Fig. A58 for the class IV centre stick

COMPONENTS = ("frequency", "damping", "delay", "stick_force")


@dataclass(frozen=True, slots=True)
class LongitudinalParameters:
    omega_sp: float      # rad/s
    zeta_sp: float
    n_alpha: float       # g/rad
    v0: float            # m/s, trim airspeed, held constant
    fs_per_g: float      # N/g, steady stick force per load factor
    tau_theta: float     # s, transport delay = equivalent delay of this LOES
    alpha_trim: float = 0.05  # rad; level trim, so theta_trim = alpha_trim

    def __post_init__(self) -> None:
        if min(self.omega_sp, self.n_alpha, self.v0, self.fs_per_g) <= 0:
            raise ValueError("omega_sp, n_alpha, v0 and fs_per_g must be positive")
        if not 0 < self.zeta_sp <= 5 or self.tau_theta < 0:
            raise ValueError("zeta_sp must be in (0, 5] and tau_theta non-negative")

    @property
    def t_theta2(self) -> float:
        return self.v0 / (G * self.n_alpha)

    @property
    def cap(self) -> float:
        return self.omega_sp**2 / self.n_alpha

    @property
    def k_theta(self) -> float:
        """Initial pitch acceleration per stick force, rad/s^2/N."""
        return self.cap / self.fs_per_g

    @property
    def k_nz(self) -> float:
        return self.k_theta * self.n_alpha

    @property
    def m_q(self) -> float:
        return 1.0 / self.t_theta2 - 2.0 * self.zeta_sp * self.omega_sp

    @property
    def m_alpha(self) -> float:
        return -self.omega_sp**2 - self.m_q / self.t_theta2


def short_period_matrices(p: LongitudinalParameters) -> tuple[np.ndarray, np.ndarray]:
    """Continuous A and B for x=[alpha, q], u=[Fs (delayed), c_q]."""
    a = np.array([[-1.0 / p.t_theta2, 1.0], [p.m_alpha, p.m_q]])
    b = np.array([[0.0, 0.0], [p.k_theta, 1.0]])
    return a, b


def longitudinal_polynomials(p: LongitudinalParameters) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Delay-free GJB forms; theta and gamma assume wings level."""
    den = np.array([1.0, 2.0 * p.zeta_sp * p.omega_sp, p.omega_sp**2])
    q = (p.k_theta * np.array([1.0, 1.0 / p.t_theta2]), den)
    theta = (q[0], np.polymul(den, [1.0, 0.0]))
    return {
        "q": q,
        "alpha": (np.array([p.k_theta]), den),
        "nz": (np.array([p.k_nz]), den),
        "theta": theta,
        "gamma": (np.array([p.k_theta / p.t_theta2]), np.polymul(den, [1.0, 0.0])),
    }


def stick_force_bounds(n_alpha: float, n_limit: float = IV_A_LOAD_LIMIT_G) -> dict[int, tuple[float, float]]:
    """GJB Table A10(a), centre stick, (Fs/n)min and (Fs/n)max in N/g."""
    if n_limit < 3:
        max1, max2 = 125.0, 190.0
    else:
        max1 = min(125.0, max(1090.0 / n_alpha, 250.0 / (n_limit - 1)))
        max2 = min(190.0, max(1630.0 / n_alpha, 385.0 / (n_limit - 1)))
    return {
        1: (max(100.0 / (n_limit - 1), 14.0), max1),
        2: (max(80.0 / (n_limit - 1), 14.0), max2),
        3: (max(50.0 / (n_limit - 1), 9.0), 250.0),
    }


def frequency_level(omega_sp: float, n_alpha: float) -> int:
    """GJB Fig. A3(a), category A; level 3 has no drawn boundary."""
    cap = omega_sp**2 / n_alpha
    if omega_sp >= 1.0 and 0.28 <= cap <= 3.6:
        return 1
    if omega_sp >= 0.6 and 0.16 <= cap <= 10.0:
        return 2
    return 3


def damping_level(zeta_sp: float) -> int | None:
    """GJB Table A6, categories A and C; only stable roots are sampled here."""
    if 0.35 <= zeta_sp <= 1.30:
        return 1
    if 0.25 <= zeta_sp <= 2.00:
        return 2
    return 3 if zeta_sp > 0 else None


def delay_level(tau: float) -> int | None:
    """GJB Table A7."""
    for level, limit in ((1, 0.10), (2, 0.20), (3, 0.30)):
        if tau <= limit:
            return level
    return None


def stick_force_level(fs_per_g: float, n_alpha: float) -> int | None:
    for level, (low, high) in stick_force_bounds(n_alpha).items():
        if low <= fs_per_g <= high:
            return level
    return None


def classify_longitudinal(p: LongitudinalParameters) -> dict[str, object]:
    levels = {
        "frequency": frequency_level(p.omega_sp, p.n_alpha),
        "damping": damping_level(p.zeta_sp),
        "delay": delay_level(p.tau_theta),
        "stick_force": stick_force_level(p.fs_per_g, p.n_alpha),
    }
    ungraded = [name for name, level in levels.items() if level is None]
    return {
        "component_levels": levels,
        "level": None if ungraded else max(levels.values()),
        "ungraded": ungraded,
    }


def _log_between(u: float, low: float, high: float) -> float:
    return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))


def _between(u: float, low: float, high: float) -> float:
    return low + u * (high - low)


_MARGIN = 0.01  # keep samples off the printed boundaries


def _cap(level: int, n_alpha: float, u: float, branch: float) -> float:
    low1 = max(0.28, 1.0 / n_alpha)
    low2 = max(0.16, 0.36 / n_alpha)
    if level == 1:
        return _log_between(u, low1 * (1 + _MARGIN), 3.6 * (1 - _MARGIN))
    if level == 2:
        if branch < 0.5:
            return _log_between(u, low2 * (1 + _MARGIN), low1 * (1 - _MARGIN))
        return _log_between(u, 3.6 * (1 + _MARGIN), 10.0 * (1 - _MARGIN))
    if branch < 0.5:
        return _log_between(u, 0.5 * low2, low2 * (1 - _MARGIN))
    return _log_between(u, 10.0 * (1 + _MARGIN), 20.0)


def _zeta(level: int, u: float, branch: float) -> float:
    if level == 1:
        return _between(u, 0.35 * (1 + _MARGIN), 1.30 * (1 - _MARGIN))
    if level == 2:
        return _between(u, 0.25, 0.35 * (1 - _MARGIN)) if branch < 0.5 else _between(u, 1.30 * (1 + _MARGIN), 2.0)
    return _between(u, 0.10, 0.25 * (1 - _MARGIN)) if branch < 0.5 else _between(u, 2.0 * (1 + _MARGIN), 2.6)


def _tau(level: int, u: float) -> float:
    low, high = {1: (0.005, 0.10), 2: (0.10 * (1 + _MARGIN), 0.20), 3: (0.20 * (1 + _MARGIN), 0.30)}[level]
    return _between(u, low, high)


def _fs_per_g(level: int, n_alpha: float, u: float, branch: float) -> float:
    bounds = stick_force_bounds(n_alpha)
    (min1, max1), (min2, max2), (min3, max3) = bounds[1], bounds[2], bounds[3]
    if level == 1:
        return _log_between(u, min1 * (1 + _MARGIN), max1 * (1 - _MARGIN))
    low_band = (min2, min1) if level == 2 else (min3, min2)
    high_band = (max1, max2) if level == 2 else (max2, max3)
    if branch < 0.5 and low_band[0] < low_band[1] * (1 - _MARGIN):
        return _log_between(u, low_band[0] * (1 + _MARGIN), low_band[1] * (1 - _MARGIN))
    return _log_between(u, high_band[0] * (1 + _MARGIN), high_band[1] * (1 - _MARGIN))


LONGITUDINAL_UNIT_DIM = 9


def longitudinal_from_unit(unit: np.ndarray, levels: dict[str, int]) -> LongitudinalParameters:
    """Map 9 unit latents to one aircraft whose components sit at ``levels``.

    V0 and 1/T_theta2 are the research envelope (60-250 m/s, 0.4-2.5 1/s), which
    keeps n/alpha inside the 1-100 g/rad range drawn in GJB Fig. A3.
    """
    u_v0, u_t2, u_cap, b_cap, u_zeta, b_zeta, u_tau, u_fs, b_fs = map(float, unit)
    v0 = _log_between(u_v0, 60.0, 250.0)
    inv_t2 = _log_between(u_t2, 0.4, 2.5)
    n_alpha = v0 * inv_t2 / G
    cap = _cap(levels["frequency"], n_alpha, u_cap, b_cap)
    return LongitudinalParameters(
        omega_sp=math.sqrt(cap * n_alpha),
        zeta_sp=_zeta(levels["damping"], u_zeta, b_zeta),
        n_alpha=n_alpha,
        v0=v0,
        fs_per_g=_fs_per_g(levels["stick_force"], n_alpha, u_fs, b_fs),
        tau_theta=_tau(levels["delay"], u_tau),
    )
