"""Stability-axis small-perturbation derivatives that reproduce the GJB LOES exactly.

Lateral-directional, state [beta, p, r, phi], inputs F_as and F_rp (Etkin & Reid,
Nelson ch. 5; Y_p = Y_r = 0, level trim so phi_dot = p):

    beta_dot = Y_b beta - r + (g/V) phi + Y_rp F_rp
    p_dot    = L_b beta + L_p p + L_r r + L_a F_as + L_rp F_rp
    r_dot    = N_b beta + N_p p + N_r r + N_a F_as + N_rp F_rp

Matching the frozen roll channel p/F_as (four poles, zeros zeta_phi/omega_phi,
gain L_Fa) gives six equations in eight unknowns.  With Y_b and L_r chosen,
the remaining six follow in closed form (the residual equation is linear in
eta = N_a / L_a), so every cross channel beta/F_as, r/F_as, p/F_rp ... follows
from the same poles.

Longitudinal, state [u, alpha, q, theta], inputs F_s and throttle dT (thrust
increment / weight):

    u_dot     = X_u u + X_a alpha - g theta + g dT
    alpha_dot = Z_u u + Z_a alpha + q
    q_dot     = M_u u + M_a alpha + M_q q + K_theta F_s

With X_u, X_a, Z_u, M_u given, keeping the short-period pair, the zero
1/T_theta2 and K_theta exactly is linear in the phugoid coefficients, so
omega_p, zeta_p and 1/T_theta1 are outputs of the physics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from pinmo6 import G
from pinmo6.longitudinal import LongitudinalParameters
from src.aircraft.parameters import PChannelParameters


@dataclass(frozen=True, slots=True)
class LateralDerivatives:
    g_over_v: float
    y_b: float
    l_b: float
    l_p: float
    l_r: float
    n_b: float
    n_p: float
    n_r: float
    l_a: float      # per N of F_as
    n_a: float
    y_rp: float     # per N of F_rp
    l_rp: float
    n_rp: float

    @property
    def eta(self) -> float:
        return self.n_a / self.l_a

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([[self.y_b, 0.0, -1.0, self.g_over_v],
                      [self.l_b, self.l_p, self.l_r, 0.0],
                      [self.n_b, self.n_p, self.n_r, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        b = np.array([[0.0, self.y_rp], [self.l_a, self.l_rp], [self.n_a, self.n_rp], [0.0, 0.0]])
        return a, b


def lateral_targets(roll: PChannelParameters) -> tuple[np.ndarray, float, float]:
    """Characteristic coefficients [a3, a2, a1, a0] and the p/F_as zero coefficients."""
    poles = np.polymul(np.polymul([1.0, -roll.lambda_s], [1.0, 1.0 / roll.t_r]),
                       [1.0, 2.0 * roll.zeta_d * roll.omega_d, roll.omega_d**2])
    return poles[1:], 2.0 * roll.zeta_phi * roll.omega_phi, roll.omega_phi**2


def solve_lateral(roll: PChannelParameters, v0: float, y_b: float, l_r: float,
                  n_rp: float = 0.0, l_rp_over_n_rp: float = 0.0,
                  y_rp_over_n_rp: float = 0.0) -> LateralDerivatives:
    (a3, a2, a1, a0), c1, c0 = lateral_targets(roll)
    g = G / v0
    if l_r == 0 or abs(y_b + c1) < 1e-12:
        raise ValueError("L_r must be non-zero and Y_b != -2 zeta_phi omega_phi")
    k1 = (l_r**2 * y_b**2 * c0 * g**2 + l_r**2 * y_b * c0 * c1 * g**2 + l_r**2 * c0**2 * g**2
          - l_r * y_b * a0 * c1 * g + 2 * l_r * a0 * c0 * g - l_r * a0 * c1**2 * g + a0**2)
    k0 = g * (y_b + c1) * (
        l_r**2 * y_b**2 * g**2 + l_r**2 * y_b * c1 * g**2 + l_r**2 * c0 * g**2 - l_r * y_b * a1 * g
        + l_r * y_b * a3 * c0 * g - l_r * y_b * c0 * c1 * g + l_r * a0 * g - l_r * a1 * c1 * g
        + l_r * a2 * c0 * g - l_r * c0**2 * g + a0 * a2 - a0 * a3 * c1 - a0 * c0 + a0 * c1**2)
    if abs(k1) < 1e-14:
        raise ValueError("degenerate lateral inversion")
    eta = -k0 / k1
    n_r = l_r * eta - y_b - c1
    l_p = -a3 - n_r - y_b
    n_b = (-l_r * y_b**2 * eta * g - l_r * y_b * c1 * eta * g - l_r * c0 * eta * g + y_b**3 * g
           + 2 * y_b**2 * c1 * g + y_b * c0 * g + y_b * c1**2 * g - a0 * eta + c0 * c1 * g) / (g * (y_b + c1))
    l_b = (-l_r * y_b**2 * g - l_r * y_b * c1 * g - l_r * c0 * g - a0) / (g * (y_b + c1))
    n_p = (l_p * n_r + l_p * y_b + n_b + n_r * y_b - a2) / l_r
    return LateralDerivatives(
        g_over_v=g, y_b=y_b, l_b=l_b, l_p=l_p, l_r=l_r, n_b=n_b, n_p=n_p, n_r=n_r,
        l_a=roll.l_fa, n_a=eta * roll.l_fa,
        y_rp=y_rp_over_n_rp * n_rp, l_rp=l_rp_over_n_rp * n_rp, n_rp=n_rp)


def dutch_roll_shape(d: LateralDerivatives) -> dict[str, float]:
    """|phi/beta|_d and the Dutch-roll eigenvector phase of p relative to beta."""
    a, _ = d.matrices()
    values, vectors = np.linalg.eig(a)
    k = int(np.argmax(values.imag))
    v = vectors[:, k]
    return {"phi_over_beta": float(abs(v[3] / v[0])),
            "p_phase_minus_beta_deg": float(np.degrees(np.angle(v[1] / v[0])))}


@dataclass(frozen=True, slots=True)
class LongitudinalDerivatives:
    g: float
    x_u: float
    x_a: float      # includes the trim-lift tilt term g; aerodynamic part is x_a - g
    z_u: float
    z_a: float
    m_u: float
    m_a: float
    m_q: float
    m_s: float      # K_theta, per N of F_s
    long_period: tuple[float, float]  # (p1, p0) of s^2 + p1 s + p0
    inv_t_theta1: float

    @property
    def long_period_roots(self) -> np.ndarray:
        return np.roots([1.0, *self.long_period])

    @property
    def phugoid(self) -> dict[str, float] | None:
        """omega_p and zeta_p when the long-period pair is oscillatory."""
        p1, p0 = self.long_period
        if p0 <= 0 or p1 * p1 >= 4 * p0:
            return None
        omega = math.sqrt(p0)
        return {"omega_p": omega, "zeta_p": p1 / (2.0 * omega)}

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array([[self.x_u, self.x_a, 0.0, -self.g],
                      [self.z_u, self.z_a, 1.0, 0.0],
                      [self.m_u, self.m_a, self.m_q, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        b = np.array([[0.0, self.g], [0.0, 0.0], [self.m_s, 0.0], [0.0, 0.0]])
        return a, b


def solve_longitudinal(lon: LongitudinalParameters, x_u: float, x_a_aero: float,
                       m_u: float = 0.0) -> LongitudinalDerivatives:
    """Keep the short-period pair, 1/T_theta2 and K_theta of ``lon`` exactly."""
    g = G
    v = lon.v0
    z_u = -2.0 * g / v**2               # lift proportional to V^2, alpha_dot equation
    x_a = g + x_a_aero
    q1, q0 = 2.0 * lon.zeta_sp * lon.omega_sp, lon.omega_sp**2
    z2 = 1.0 / lon.t_theta2
    if abs(z2 + x_u) < 1e-12:
        raise ValueError("X_u cannot equal -1/T_theta2")
    z_a = (x_a * z_u - z2**2 - x_u * z2) / (z2 + x_u)
    # M_a = alpha_a * p0 + alpha_b ; M_q = -p1 + mb
    alpha_a, alpha_b = q0 / (g * z_u), m_u * z_a / z_u
    mb = -q1 - x_u - z_a
    s2, c2 = x_u + z_a, x_a * z_u - x_u * z_a
    matrix = np.array([[-alpha_a - 1.0, -s2 - q1],
                       [alpha_a * x_u - q1, -c2 - q0]])
    rhs = np.array([q0 + alpha_b - mb * s2 + x_a * z_u - x_u * z_a,
                    -alpha_b * x_u - mb * c2 + m_u * x_a - m_u * g])
    p0, p1 = np.linalg.solve(matrix, rhs)
    m_a, m_q = alpha_a * p0 + alpha_b, -p1 + mb
    zeros = np.roots([1.0, -(x_u + z_a), x_u * z_a - x_a * z_u])
    other = zeros[np.argmax(np.abs(zeros + z2))]
    return LongitudinalDerivatives(
        g=g, x_u=x_u, x_a=x_a, z_u=z_u, z_a=z_a, m_u=m_u, m_a=m_a, m_q=m_q, m_s=lon.k_theta,
        long_period=(float(p1), float(p0)), inv_t_theta1=float(-other.real))


def lateral_from_modes(roll_modes: tuple[float, float, float, float], v0: float, y_b: float,
                       n_p_over_l_p: float, l_r_over_n_r: float) -> list[LateralDerivatives]:
    """All real derivative sets with the given four lateral poles (control terms left zero).

    ``roll_modes`` = (lambda_s, T_R, zeta_d, omega_d).  With Y_b and the shape
    ratios c1 = N'_p/L'_p, c2 = L'_r/N'_r fixed, the characteristic polynomial
    gives N'_r = S - L'_p, N'_b quadratic and L'_b rational in L'_p, and the
    constant coefficient becomes a quartic in L'_p.
    """
    lambda_s, t_r, zeta_d, omega_d = roll_modes
    a3, a2, a1, a0 = map(float, np.polymul(np.polymul([1.0, -lambda_s], [1.0, 1.0 / t_r]),
                                           [1.0, 2.0 * zeta_d * omega_d, omega_d**2])[1:])
    g = G / v0
    # plain floats: a numpy scalar on the left of a poly1d broadcasts instead of multiplying
    y_b, c1, c2 = float(y_b), float(n_p_over_l_p), float(l_r_over_n_r)
    total = -a3 - y_b                                   # L'_p + N'_r
    lp = np.poly1d([1.0, 0.0])
    n_r = total - lp
    n_p = c1 * lp
    l_r = c2 * n_r
    n_b = a2 - lp * n_r - y_b * total + l_r * n_p
    lb_num = a1 + lp * n_b + lp * n_r * y_b - l_r * n_p * y_b
    quartic = g * (lb_num * n_r - l_r * n_b * (n_p - g)) - a0 * (n_p - g)
    out = []
    for root in quartic.roots:
        if abs(root.imag) > 1e-9 * max(1.0, abs(root.real)):
            continue
        x = float(root.real)
        denominator = c1 * x - g
        if abs(denominator) < 1e-12:
            continue
        out.append(LateralDerivatives(
            g_over_v=g, y_b=y_b, l_b=float(lb_num(x) / denominator), l_p=x, l_r=float(l_r(x)),
            n_b=float(n_b(x)), n_p=float(n_p(x)), n_r=float(n_r(x)),
            l_a=0.0, n_a=0.0, y_rp=0.0, l_rp=0.0, n_rp=0.0))
    return out


def roll_zeros(d: LateralDerivatives, eta: float) -> tuple[float, float]:
    """(2 zeta_phi omega_phi, omega_phi^2) of p/F_as for aileron yaw ratio eta."""
    return (-d.n_r - d.y_b + d.l_r * eta,
            d.n_b + d.n_r * d.y_b - eta * (d.l_b + d.l_r * d.y_b))
