"""Dryden continuous turbulence as shaping filters driven by white noise.

Translational spectra are the Dryden forms of MIL-STD-1797A 4.9.2 (PDF p.804),
written with full scale lengths (MIL-STD-1797A uses L_u = 2L_v = 2L_w, i.e.
L_v, L_w are half of the lengths used here); rotary spectra follow the
MIL-F-8785C filters.  Spectra are one-sided in temporal frequency, so a
filter H(s) driven by white noise of two-sided density pi gives
sigma^2 = integral_0^inf Phi(omega) d omega; a discrete sequence of variance
pi/dt approximates that density.

Gusts enter the aircraft through the aerodynamic terms only (MIL-STD-1797A
4.9.3), evaluated on the air-relative state x - x_g with
x_g = [u_g, alpha_g, q_g, beta_g, p_g, r_g], alpha_g = w_g/V, beta_g = v_g/V.
With that subtraction, the tail's velocity relative to the local air gives
q_g = -dw_g/dx and r_g = +dv_g/dx (frozen field, d/dx = (1/V) d/dt); the
MIL text states the opposite signs because it defines gust rates as
equivalent aircraft rates, which add instead of subtract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy import signal

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x and 1.x names


@dataclass(frozen=True, slots=True)
class DrydenScales:
    l_u: float      # m
    l_v: float
    l_w: float
    sigma_u: float  # m/s
    sigma_v: float
    sigma_w: float
    span: float     # m, wing span b for the rotary gusts


FT = 0.3048
KT = 0.514444


def high_altitude_scales(intensity: str, span: float, altitude_ft: float = 10000.0) -> DrydenScales:
    """MIL-STD-1797A medium/high altitude: isotropic, L_u = 1750 ft, sigma from Fig. 262.

    Fig. 262 is read only at its plateaus (PDF p.799, digitized): light 5 ft/s
    up to about 9,000 ft, moderate 10 ft/s up to about 11,000 ft, severe about
    21 ft/s from 4,000 to 20,000 ft; above those ranges use a measured value.
    """
    plateau_ft = {"light": (5.0, 9000.0), "moderate": (10.0, 11000.0), "severe": (21.2, 20000.0)}
    sigma_ft, top = plateau_ft[intensity]
    if not 2000.0 <= altitude_ft <= top:
        raise ValueError("altitude outside the plateau of Fig. 262 for this intensity")
    length, sigma = 1750.0 * FT, sigma_ft * FT
    return DrydenScales(length, length, length, sigma, sigma, sigma, span)


def low_altitude_scales(altitude_ft: float, wind_20ft_kt: float, span: float) -> DrydenScales:
    """MIL-STD-1797A below 1000 ft (Figs. 266-267): L_w = h, sigma_w = 0.1 u_20."""
    if not 10.0 < altitude_ft < 1000.0:
        raise ValueError("low-altitude model applies for 10 < h < 1000 ft")
    base = 0.177 + 0.000823 * altitude_ft
    length_uv = altitude_ft / base**1.2 * FT
    sigma_w = 0.1 * wind_20ft_kt * KT
    sigma_uv = sigma_w / base**0.4
    return DrydenScales(length_uv, length_uv, altitude_ft * FT, sigma_uv, sigma_uv, sigma_w, span)


def dryden_filters(scales: DrydenScales, v: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Continuous (numerator, denominator) of each shaping filter at airspeed v."""
    s = scales
    tu, tv, tw = s.l_u / v, s.l_v / v, s.l_w / v
    h_u = (s.sigma_u * math.sqrt(2.0 * s.l_u / (math.pi * v)) * np.array([1.0]), np.array([tu, 1.0]))
    h_v = (s.sigma_v * math.sqrt(s.l_v / (math.pi * v)) * np.array([math.sqrt(3.0) * tv, 1.0]),
           np.polymul([tv, 1.0], [tv, 1.0]))
    h_w = (s.sigma_w * math.sqrt(s.l_w / (math.pi * v)) * np.array([math.sqrt(3.0) * tw, 1.0]),
           np.polymul([tw, 1.0], [tw, 1.0]))
    tp = 4.0 * s.span / (math.pi * v)
    h_p = (s.sigma_w * math.sqrt(0.8 / v) * (math.pi / (4.0 * s.span)) ** (1.0 / 6.0)
           / s.l_w ** (1.0 / 3.0) * np.array([1.0]), np.array([tp, 1.0]))
    # rotary gusts from spatial gradients: q_g = -dw_g/dx, r_g = +dv_g/dx (see module note)
    h_q = (np.polymul([-1.0 / v, 0.0], h_w[0]), np.polymul([tp, 1.0], h_w[1]))
    tr = 3.0 * s.span / (math.pi * v)
    h_r = (np.polymul([1.0 / v, 0.0], h_v[0]), np.polymul([tr, 1.0], h_v[1]))
    return {"u": h_u, "v": h_v, "w": h_w, "p": h_p, "q": h_q, "r": h_r}


def filter_variance(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """integral_0^inf |H(j omega)|^2 d omega, i.e. the variance under this normalization."""
    omega = np.concatenate(([0.0], np.logspace(-6, 4, 200000)))
    _, response = signal.freqs(numerator, denominator, worN=omega)
    return float(_trapezoid(np.abs(response) ** 2, omega))


class DrydenGust:
    """Discrete generator; one ``step`` per plant sample."""

    def __init__(self, scales: DrydenScales, v: float, dt: float, seed: int) -> None:
        self.v, self.dt = v, dt
        self._rng = np.random.default_rng(seed)
        filters = dryden_filters(scales, v)
        self._names = ("u", "v", "w", "p", "q", "r")
        self._systems = {}
        for name in self._names:
            a, b, c, d = signal.tf2ss(*filters[name])
            ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), dt, method="zoh")
            self._systems[name] = [ad, bd[:, 0], cd[0], float(dd.squeeze()), np.zeros(ad.shape[0])]
        self._noise_scale = math.sqrt(math.pi / dt)

    def step(self) -> np.ndarray:
        # w drives w and q; v drives v and r: they share one noise sample each
        noise = self._rng.standard_normal(4) * self._noise_scale
        drive = {"u": noise[0], "v": noise[1], "w": noise[2], "p": noise[3], "q": noise[2], "r": noise[1]}
        out = {}
        for name in self._names:
            ad, bd, cd, dd, x = self._systems[name]
            out[name] = float(cd @ x + dd * drive[name])
            self._systems[name][4] = ad @ x + bd * drive[name]
        return np.array([out["u"], out["w"] / self.v, out["q"], out["v"] / self.v, out["p"], out["r"]])
