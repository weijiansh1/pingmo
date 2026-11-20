"""Choose the two free lateral derivatives so the derived set looks like a real aircraft.

For each roll object the closed-form inversion leaves Y_beta and L'_r free.
A grid over their literature ranges is scored against envelopes of
dimensionless derivative ratios taken from real aircraft (literature/
envelopes.json); the library builder draws among the points inside every
envelope, or keeps the least-violating point and reports what it violates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from pinmo6.derivatives import LateralDerivatives, solve_lateral
from src.aircraft.parameters import PChannelParameters


@dataclass(frozen=True, slots=True)
class LateralEnvelope:
    """Closed intervals on dimensionless or per-second quantities."""
    y_b: tuple[float, float]
    l_r: tuple[float, float]
    ratios: dict[str, tuple[float, float]] = field(default_factory=dict)


def lateral_ratios(d: LateralDerivatives, roll: PChannelParameters) -> dict[str, float]:
    """Quantities compared with real aircraft; dimensional ones normalized by the object's modes."""
    safe = lambda a, b: a / b if b != 0 else math.copysign(math.inf, a)  # noqa: E731
    return {
        "dihedral_ratio": safe(-d.l_b, d.n_b),
        "n_b_over_omega_d2": d.n_b / roll.omega_d**2,
        "l_p_times_t_r": d.l_p * roll.t_r,
        "n_r_over_2_zeta_omega": d.n_r / (2.0 * roll.zeta_d * roll.omega_d),
        "n_p_over_l_p": safe(d.n_p, d.l_p),
        "l_r_over_n_r": safe(d.l_r, d.n_r),
        "eta": d.eta,
    }


def violation(values: dict[str, float], ratios: dict[str, tuple[float, float]]) -> tuple[float, list[str]]:
    total, names = 0.0, []
    for name, (low, high) in ratios.items():
        value = values[name]
        if not math.isfinite(value):
            total += 1e6
            names.append(name)
            continue
        width = max(high - low, 1e-9)
        excess = max(low - value, value - high, 0.0) / width
        if excess > 0:
            total += excess * excess
            names.append(name)
    return total, names


def lateral_candidates(roll: PChannelParameters, v0: float, envelope: LateralEnvelope,
                       pedal: dict, grid: tuple[int, int] = (9, 81)):
    """All grid points inside the envelope, and the least-violating point."""
    feasible, best = [], None
    for y_b in np.linspace(*envelope.y_b, grid[0]):
        for l_r in np.linspace(*envelope.l_r, grid[1]):
            try:
                d = solve_lateral(roll, v0, float(y_b), float(l_r), **pedal)
            except ValueError:
                continue
            score, names = violation(lateral_ratios(d, roll), envelope.ratios)
            if score == 0.0:
                feasible.append(d)
            if best is None or score < best[0]:
                best = (score, names, d)
    if best is None:
        raise ValueError("no non-degenerate lateral inversion on the grid")
    return feasible, best
