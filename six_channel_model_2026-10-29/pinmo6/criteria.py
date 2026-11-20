"""Cross-axis and long-period handling-qualities checks that need the coupled channels.

Recommended values for class IV, category A.  Each item cites MIL-STD-1797A
(Notice 3, literature/MIL-STD-1797A_notice3_2004.pdf) and its GJB 2874-97
counterpart, both as PDF pages; the GJB values were read independently and
agree (literature/notes_gjb2874_lateral.md):

    Dutch roll and |phi/beta| correction   MIL Table XL p.652     GJB Table A35 p.303
    roll oscillation p_osc/p_av vs psi_b   MIL Fig. 156 p.551     GJB Fig. A116 p.240
    sideslip excursion delta-beta/k        MIL Fig. 235 p.686     GJB Figs. A186-A189 p.334-336
                                           (GJB Fig. A184 on p.331 is misprinted as a bandwidth chart)
    psi_beta, k, delta-beta definitions    MIL p.135-142          GJB p.12-13, 241
    dihedral sense in steady sideslip      MIL 4.5.5 p.161        GJB 5.4.6 p.33
    long period                            MIL 4.2.1.1 p.297      GJB Table A5 p.77

Figure boundaries are digitized as straight segments between breakpoints.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from pinmo6.derivatives import LateralDerivatives, LongitudinalDerivatives
from src.aircraft.parameters import PChannelParameters
from src.benchmark.time_domain import extract_gjb_roll_peaks, no_spiral_step_response

# Fig. 156 (PDF p.551), p_osc/p_av, categories A and C; upper psi_beta scale
ROLL_OSCILLATION = {
    1: ((0, 0.05), (-130, 0.05), (-200, 0.25), (-270, 0.25), (-340, 0.05), (-360, 0.05)),
    2: ((0, 0.10), (-120, 0.10), (-200, 0.60), (-270, 0.60), (-350, 0.10), (-360, 0.10)),
}
# Fig. 235 (PDF p.686), delta-beta / k in degrees, category A level 1 and all-category level 2
SIDESLIP_EXCURSION = {
    1: ((0, 2.0), (-130, 2.0), (-205, 6.0), (-270, 6.0), (-340, 2.0), (-360, 2.0)),
    2: ((0, 4.0), (-110, 4.0), (-205, 15.0), (-272, 15.0), (-360, 4.0)),
}
# Table XL (PDF p.652), class IV category A other than CO/GA/RR/TF/RC/FF/AS
DUTCH_ROLL = {1: (0.19, 0.35, 1.0), 2: (0.02, 0.05, 0.4), 3: (0.0, 0.0, 0.4)}
DUTCH_ROLL_CORRECTION = {1: 0.014, 2: 0.009, 3: 0.005}
# Table XXX (PDF p.583), class IV category A level 1 roll performance by speed range (bank deg, s)
ROLL_PERFORMANCE = {"VL": (30.0, 1.1), "L": (30.0, 1.1), "M": (90.0, 1.3), "H": (50.0, 1.1)}


def _boundary(points, psi_deg: float) -> float:
    phases = np.array([p for p, _ in points], dtype=float)[::-1]
    limits = np.array([v for _, v in points], dtype=float)[::-1]
    return float(np.interp(psi_deg, phases, limits))


def _wrap(angle_deg: float) -> float:
    """Map onto (-360, 0]."""
    wrapped = -((-angle_deg) % 360.0)
    return 0.0 if wrapped == -0.0 else wrapped


def dutch_roll_pole(roll: PChannelParameters) -> complex:
    return complex(-roll.zeta_d * roll.omega_d, roll.omega_d * math.sqrt(1.0 - roll.zeta_d**2))


def lateral_transfer(d: LateralDerivatives, output: int, column: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = d.matrices()
    c = np.zeros((1, 4))
    c[0, output] = 1.0
    num, den = signal.ss2tf(a, b[:, [column]], c, np.zeros((1, 1)))
    return np.trim_zeros(num[0], "f"), den


def dutch_roll_eigenvector(d: LateralDerivatives, roll: PChannelParameters) -> dict[str, float]:
    a, _ = d.matrices()
    values, vectors = np.linalg.eig(a)
    k = int(np.argmin(np.abs(values - dutch_roll_pole(roll))))
    v = vectors[:, k]
    return {"phi_over_beta": float(abs(v[3] / v[0])),
            "phi_over_beta_phase_deg": float(np.degrees(np.angle(v[3] / v[0]))),
            "p_lead_beta_deg": float(np.degrees(np.angle(v[1] / v[0])) % 360.0)}


def psi_beta_deg(d: LateralDerivatives, roll: PChannelParameters) -> float:
    """Phase of the Dutch-roll cosine in sideslip after a right step of F_as (MIL 3.4.6, PDF p.140).

    Residue of beta/F_as * 1/s at the Dutch-roll pole; the transport delay is a
    time shift, so it lags the phase by 360 tau / T_d (t_nbeta is measured from
    the control input).
    """
    num, den = lateral_transfer(d, 0, 0)
    pole = dutch_roll_pole(roll)
    step_den = np.polymul(den, [1.0, 0.0])
    residue = np.polyval(num, pole) / np.polyval(np.polyder(step_den), pole)
    damped = pole.imag
    return _wrap(math.degrees(np.angle(residue)) - math.degrees(damped * roll.tau_p))


def dutch_roll_level(roll: PChannelParameters, phi_over_beta: float) -> dict[str, object]:
    product = roll.zeta_d * roll.omega_d
    excess = max(roll.omega_d**2 * phi_over_beta - 20.0, 0.0)
    level = None
    for candidate in (1, 2, 3):
        zeta_min, product_min, omega_min = DUTCH_ROLL[candidate]
        product_min += DUTCH_ROLL_CORRECTION[candidate] * excess
        if roll.zeta_d >= zeta_min and product >= product_min and roll.omega_d >= omega_min:
            level = candidate
            break
    return {"level": level, "omega_d2_phi_over_beta": roll.omega_d**2 * phi_over_beta,
            "corrected_product_min_level_1": DUTCH_ROLL[1][1] + DUTCH_ROLL_CORRECTION[1] * excess}


def roll_oscillation(roll: PChannelParameters, psi_beta: float, p_lead_beta: float,
                     phi_over_beta: float) -> dict[str, object]:
    """4.5.1.4 with p from the spiral-free step response (project convention, GJB Fig. A120)."""
    t_d = 2.0 * math.pi / dutch_roll_pole(roll).imag
    time = np.linspace(0.0, max(4.0 * t_d, 6.0), 6001)
    peaks = extract_gjb_roll_peaks(time, no_spiral_step_response(roll, time))
    position = psi_beta if 45.0 <= p_lead_beta <= 225.0 else _wrap(psi_beta + 180.0)
    limits = {level: _boundary(points, position) for level, points in ROLL_OSCILLATION.items()}
    out = {"psi_beta_on_figure_deg": position, "limit_level_1": limits[1], "limit_level_2": limits[2],
           "applicable": phi_over_beta >= 1.5}
    if peaks.status != "assessable":
        # no peak-valley-peak sequence: the roll rate does not oscillate
        return {**out, "status": f"no_oscillation_{peaks.reason}", "p_osc_over_p_av": 0.0, "level": 1,
                "first_minimum_ratio": 1.0, "first_minimum_level": 1}
    p1, p2, p3 = peaks.p1, peaks.p2, peaks.p3
    ratio = (p1 - p2) / (p1 + p2) if roll.zeta_d > 0.2 else (p1 + p3 - 2 * p2) / (p1 + p3 + 2 * p2)
    level = 1 if ratio <= limits[1] else 2 if ratio <= limits[2] else 3
    minimum = p2 / p1
    minimum_level = 1 if minimum >= 0.60 else 2 if minimum >= 0.05 else 3
    return {**out, "status": "assessable", "p_osc_over_p_av": ratio, "level": level,
            "first_minimum_ratio": minimum, "first_minimum_level": minimum_level}


def sideslip_excursion(d: LateralDerivatives, roll: PChannelParameters, psi_beta: float,
                       dt: float = 0.002, speed_range: str = "M") -> dict[str, object]:
    """4.6.2 alternative 1(a): delta-beta_max / k for a small right step of F_as (pedals free).

    k uses the level-1 roll performance of ``speed_range``; in the linear range
    delta-beta / k does not depend on the step size.
    """
    bank_required, bank_time = ROLL_PERFORMANCE[speed_range]
    t_d = 2.0 * math.pi / dutch_roll_pole(roll).imag
    window = max(2.0, 0.5 * t_d)
    a, b = d.matrices()
    steps = int(math.ceil(max(window, bank_time) / dt)) + 1
    time = np.arange(steps) * dt
    delay = int(round(roll.tau_p / dt))
    forcing = (time >= delay * dt).astype(float)
    _, out, _ = signal.lsim((a, b[:, [0]], np.eye(4), np.zeros((4, 1))), forcing, time)
    beta_deg, phi_deg = np.degrees(out[:, 0]), np.degrees(out[:, 3])
    inside = beta_deg[time <= window]
    extrema = np.concatenate(([0.0], inside[1:-1][(np.diff(np.sign(np.diff(inside))) != 0)], [inside[-1]]))
    excursion = float(np.sum(np.abs(np.diff(extrema))))
    bank = float(np.interp(bank_time, time, phi_deg))
    if bank <= 0:
        return {"status": "nonpositive_bank_response", "level": None}
    ratio = excursion / (bank / bank_required)
    limits = {level: _boundary(points, psi_beta) for level, points in SIDESLIP_EXCURSION.items()}
    level = 1 if ratio <= limits[1] else 2 if ratio <= limits[2] else 3
    return {"status": "assessable", "delta_beta_over_k_deg": ratio, "level": level, "speed_range": speed_range,
            "limit_level_1": limits[1], "limit_level_2": limits[2], "window_s": window}


def steady_sideslip(d: LateralDerivatives) -> dict[str, float | bool]:
    """Straight steady sideslip (p = r = 0): controls and bank per degree of right sideslip.

    4.5.5 (PDF p.161): right bank or none, and right or zero roll control, with
    right sideslip; 4.6.1.2 (PDF p.164): right pedal gives left sideslip.
    """
    controls = np.array([[d.l_a, d.l_rp], [d.n_a, d.n_rp]])
    f_as, f_rp = np.linalg.solve(controls, -np.array([d.l_b, d.n_b]) * math.radians(1.0))
    bank_deg = -math.degrees((d.y_b * math.radians(1.0) + d.y_rp * f_rp) / d.g_over_v)
    return {"roll_force_n_per_deg": float(f_as), "pedal_force_n_per_deg": float(f_rp),
            "bank_deg_per_deg": bank_deg,
            "dihedral_sense_ok": bool(f_as >= 0.0 and bank_deg >= 0.0),
            "pedal_sense_ok": bool(f_rp < 0.0)}


def long_period_level(d: LongitudinalDerivatives) -> dict[str, object]:
    """4.2.1.1 (PDF p.297): zeta_p > 0.04 / > 0, level 3 time to double > 55 s."""
    phugoid = d.phugoid
    if phugoid is not None:
        zeta = phugoid["zeta_p"]
        if zeta > 0.04:
            return {"level": 1, "zeta_p": zeta, "form": "oscillatory"}
        if zeta > 0.0:
            return {"level": 2, "zeta_p": zeta, "form": "oscillatory"}
    growth = max(float(np.max(d.long_period_roots.real)), 0.0)
    double = math.inf if growth <= 0 else math.log(2.0) / growth
    if phugoid is None and growth <= 0:
        return {"level": 1, "zeta_p": None, "form": "two_stable_real_roots"}
    return {"level": 3 if double > 55.0 else None, "zeta_p": None if phugoid is None else phugoid["zeta_p"],
            "time_to_double_s": double, "form": "divergent"}
