"""Kinematic transform from the dynamic outputs (V, alpha, beta, p, q, r) to attitude and position.

Body axes x forward, y right, z down; earth axes north-east-down.
"""
from __future__ import annotations

import math

import numpy as np


def euler_rates(phi: float, theta: float, p: float, q: float, r: float) -> tuple[float, float, float]:
    sin_phi, cos_phi = math.sin(phi), math.cos(phi)
    lateral = q * sin_phi + r * cos_phi
    return (p + lateral * math.tan(theta),
            q * cos_phi - r * sin_phi,
            lateral / math.cos(theta))


def stability_to_body(p: float, r: float, alpha: float) -> tuple[float, float]:
    """Rotate stability-axis roll and yaw rates through alpha into body axes."""
    cos_a, sin_a = math.cos(alpha), math.sin(alpha)
    return p * cos_a - r * sin_a, p * sin_a + r * cos_a


def body_velocity(v: float, alpha: float, beta: float) -> np.ndarray:
    """u = V cos(alpha) cos(beta), the mind map's V = u / (cos(alpha) cos(beta))."""
    return np.array([v * math.cos(alpha) * math.cos(beta),
                     v * math.sin(beta),
                     v * math.sin(alpha) * math.cos(beta)])


def body_to_ned(phi: float, theta: float, psi: float) -> np.ndarray:
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array([
        [ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
        [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
        [-st, sf * ct, cf * ct],
    ])


def rotation_step(omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Exact rotation over dt for body rates held constant (Rodrigues)."""
    rate = float(np.linalg.norm(omega_body))
    if rate * dt < 1e-15:
        return np.eye(3)
    x, y, z = omega_body / rate
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    angle = rate * dt
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def euler_from_dcm(rotation: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Body-to-NED matrix to (phi, theta, psi); phi and psi unwrapped about ``previous``."""
    phi = math.atan2(rotation[2, 1], rotation[2, 2])
    theta = -math.asin(max(-1.0, min(1.0, rotation[2, 0])))
    psi = math.atan2(rotation[1, 0], rotation[0, 0])
    unwrap = lambda angle, last: last + (angle - last + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
    return np.array([unwrap(phi, previous[0]), theta, unwrap(psi, previous[2])])

