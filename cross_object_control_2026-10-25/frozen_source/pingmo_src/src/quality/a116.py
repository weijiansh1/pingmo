"""Digitized Figure A116 boundaries and a roll-rate phase proxy.

The current SISO P-channel does not contain a sideslip state.  Figure A119
permits the roll-rate phase ``psi_p`` to stand in for ``psi_beta``; here it is
obtained from the Dutch-roll complex-pole residue of the unit-step response.
"""

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from scipy import signal

from src.aircraft.parameters import PChannelParameters
from src.aircraft.p_channel import p_channel_polynomials
from src.benchmark.time_domain import (
    extract_gjb_roll_peaks,
    gjb_roll_oscillation_ratio,
    held_step_roll_rate_response,
    no_spiral_step_response,
)


@dataclass(frozen=True, slots=True)
class A116Assessment:
    phase_group: str
    psi_p_deg: float
    p_osc_over_p_av: float
    level: int | str
    level_1_limit: float | None
    level_2_limit: float | None
    source_pdf_page: int
    source_print_page: int


@dataclass(frozen=True, slots=True)
class A116StepCalibration:
    """Per-aircraft held-step amplitude for the A116 small-input condition."""

    status: str
    reason: str | None
    dutch_roll_time_constant_s: float
    target_time_s: float
    step_force_n: float | None
    bank_angle_at_target_deg: float | None
    target_error_deg: float | None


def _bank_angle_deg(time: np.ndarray, roll_rate: np.ndarray) -> np.ndarray:
    increments = 0.5 * (roll_rate[:-1] + roll_rate[1:]) * np.diff(time)
    return np.rad2deg(np.concatenate(([0.0], np.cumsum(increments))))


def calibrate_a116_step(
    parameters: PChannelParameters,
    max_step_force_n: float | None,
    dt: float,
) -> A116StepCalibration:
    """Set a held roll step to reach 60 degrees at ``1.7*T_d``.

    ``T_d`` is the Dutch-roll decay time constant of the current P-channel.
    The full delay-aware plant determines reachability; the model is linear, so
    one unit-force simulation determines the exact required amplitude.
    """
    if (max_step_force_n is not None and max_step_force_n <= 0.0) or dt <= 0.0:
        raise ValueError("an optional max_step_force_n and dt must be positive")
    dutch_roll_time_constant_s = 1.0 / (parameters.zeta_d * parameters.omega_d)
    target_time_s = 1.7 * dutch_roll_time_constant_s
    steps = int(np.ceil(target_time_s / dt)) + 1
    time = np.arange(steps, dtype=float) * dt
    unit_response = held_step_roll_rate_response(parameters, time, force_n=1.0)
    unit_bank_deg = float(np.interp(target_time_s, time, _bank_angle_deg(time, unit_response)))
    if unit_bank_deg <= 0.0:
        return A116StepCalibration(
            "not_assessable", "nonpositive_unit_bank_response", dutch_roll_time_constant_s,
            target_time_s, None, None, None,
        )
    step_force_n = 60.0 / unit_bank_deg
    if max_step_force_n is not None and step_force_n > max_step_force_n:
        return A116StepCalibration(
            "not_assessable", "input_not_achievable", dutch_roll_time_constant_s,
            target_time_s, None, None, None,
        )
    bank_angle_at_target_deg = step_force_n * unit_bank_deg
    return A116StepCalibration(
        "assessable", None, dutch_roll_time_constant_s, target_time_s, step_force_n,
        bank_angle_at_target_deg, bank_angle_at_target_deg - 60.0,
    )


class A116BoundarySet:
    def __init__(self, curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], source_pages: set[int], source_print_pages: set[int]) -> None:
        self._curves = curves
        self.source_pdf_pages = source_pages
        self.source_print_pages = source_print_pages

    @classmethod
    def from_csv(cls, path: str | Path) -> "A116BoundarySet":
        curves: dict[tuple[str, int], list[tuple[float, float]]] = {}
        source_pages: set[int] = set()
        source_print_pages: set[int] = set()
        with Path(path).open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (row["phase_group"], int(row["level"]))
                curves.setdefault(key, []).append((float(row["psi_p_deg"]), float(row["p_osc_over_p_av"])))
                source_pages.add(int(row["source_pdf_page"]))
                source_print_pages.add(int(row["source_print_page"]))
        normalized: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for key, points in curves.items():
            ordered = sorted(points)
            phases = np.array([point[0] for point in ordered], dtype=float)
            limits = np.array([point[1] for point in ordered], dtype=float)
            if len(phases) < 2 or np.any(np.diff(phases) <= 0):
                raise ValueError(f"A116 curve {key} needs strictly increasing unique phase points")
            normalized[key] = phases, limits
        return cls(normalized, source_pages, source_print_pages)

    def interpolate(self, phase_group: str, level: int, psi_p_deg: float) -> float | None:
        curve = self._curves.get((phase_group, level))
        if curve is None or not np.isfinite(psi_p_deg):
            return None
        phases, limits = curve
        if psi_p_deg < phases[0] or psi_p_deg > phases[-1]:
            return None
        return float(np.interp(psi_p_deg, phases, limits))


def roll_rate_phase_proxy_deg(parameters: PChannelParameters) -> float:
    """Return the Figure A119-permitted ``psi_p`` proxy in ``[-360, 0)``.

    A unit step adds the pole at zero.  The positive-imaginary Dutch-roll pole
    residue is the complex amplitude in the real roll-rate response, so its
    angle is the phase of that component.  Pure delay shifts its onset but is
    deliberately not inserted into this residue phase.
    """
    numerator, denominator = p_channel_polynomials(parameters)
    residues, poles, _ = signal.residue(numerator, np.polymul(denominator, [1.0, 0.0]))
    candidates = [(residue, pole) for residue, pole in zip(residues, poles) if np.imag(pole) > 0]
    if not candidates:
        raise ValueError("P-channel has no positive-imaginary Dutch-roll pole")
    residue, _ = min(candidates, key=lambda pair: abs(np.imag(pair[1]) - parameters.omega_d * np.sqrt(1.0 - parameters.zeta_d**2)))
    phase = float(np.angle(residue, deg=True))
    return phase if phase < 0.0 else phase - 360.0


def assess_a116(boundaries: A116BoundarySet, phase_group: str, psi_p_deg: float, p_osc_over_p_av: float) -> A116Assessment:
    """Classify only when both A116 boundaries cover the requested phase."""
    level_1 = boundaries.interpolate(phase_group, 1, psi_p_deg)
    level_2 = boundaries.interpolate(phase_group, 2, psi_p_deg)
    source_pdf_page = min(boundaries.source_pdf_pages) if boundaries.source_pdf_pages else -1
    source_print_page = min(boundaries.source_print_pages) if boundaries.source_print_pages else -1
    if level_1 is None or level_2 is None or not np.isfinite(p_osc_over_p_av):
        level: int | str = "not_available"
    elif p_osc_over_p_av <= level_1:
        level = 1
    elif p_osc_over_p_av <= level_2:
        level = 2
    else:
        level = "above_level_2"
    return A116Assessment(phase_group, float(psi_p_deg), float(p_osc_over_p_av), level, level_1, level_2, source_pdf_page, source_print_page)


def audit_a116_parameters(
    boundaries: A116BoundarySet,
    parameters: PChannelParameters,
    phase_group: str,
    time: np.ndarray,
) -> dict[str, float | int | str | None]:
    """Run the status-first Figure A116 audit for one raw aircraft."""
    dt = float(time[1] - time[0]) if len(time) >= 2 else 0.0
    calibration = calibrate_a116_step(parameters, max_step_force_n=None, dt=dt)
    psi_p_deg = roll_rate_phase_proxy_deg(parameters)
    level_1_limit = boundaries.interpolate(phase_group, 1, psi_p_deg)
    level_2_limit = boundaries.interpolate(phase_group, 2, psi_p_deg)
    source_pdf_page = min(boundaries.source_pdf_pages) if boundaries.source_pdf_pages else -1
    source_print_page = min(boundaries.source_print_pages) if boundaries.source_print_pages else -1
    base = {
        "phase_group": phase_group,
        "psi_p_deg": psi_p_deg,
        "a116_level_1_limit": level_1_limit,
        "a116_level_2_limit": level_2_limit,
        "source_pdf_page": source_pdf_page,
        "source_print_page": source_print_page,
        "spiral_mode_removed": True,
        "dutch_roll_time_constant_s": calibration.dutch_roll_time_constant_s,
        "target_time_s": calibration.target_time_s,
        "step_force_n": calibration.step_force_n,
        "bank_angle_at_target_deg": calibration.bank_angle_at_target_deg,
        "target_error_deg": calibration.target_error_deg,
    }
    if calibration.status != "assessable" or calibration.step_force_n is None:
        return {
            **base,
            "p1": None, "p2": None, "p3": None,
            "t1_s": None, "t2_s": None, "t3_s": None,
            "p_osc_over_p_av": None,
            "a116_level": "not_available",
            "a116_status": "not_assessable",
            "a116_reason": calibration.reason,
        }

    response = calibration.step_force_n * no_spiral_step_response(parameters, time)
    peaks = extract_gjb_roll_peaks(time, response)
    if peaks.status != "assessable" or None in (peaks.p1, peaks.p2, peaks.p3):
        return {
            **base,
            "p1": peaks.p1, "p2": peaks.p2, "p3": peaks.p3,
            "t1_s": peaks.t1_s, "t2_s": peaks.t2_s, "t3_s": peaks.t3_s,
            "p_osc_over_p_av": None,
            "a116_level": "not_available",
            "a116_status": "not_assessable",
            "a116_reason": peaks.reason,
        }
    try:
        ratio = gjb_roll_oscillation_ratio(peaks.p1, peaks.p2, peaks.p3)
    except ValueError:
        return {
            **base,
            "p1": peaks.p1, "p2": peaks.p2, "p3": peaks.p3,
            "t1_s": peaks.t1_s, "t2_s": peaks.t2_s, "t3_s": peaks.t3_s,
            "p_osc_over_p_av": None,
            "a116_level": "not_available",
            "a116_status": "not_assessable",
            "a116_reason": "nonpositive_a120_denominator",
        }
    assessment = assess_a116(boundaries, phase_group, psi_p_deg, ratio)
    in_boundary_domain = assessment.level_1_limit is not None and assessment.level_2_limit is not None
    return {
        **base,
        "p1": peaks.p1, "p2": peaks.p2, "p3": peaks.p3,
        "t1_s": peaks.t1_s, "t2_s": peaks.t2_s, "t3_s": peaks.t3_s,
        "p_osc_over_p_av": ratio,
        "a116_level": assessment.level,
        "a116_status": "assessable" if in_boundary_domain else "not_assessable",
        "a116_reason": None if in_boundary_domain else "outside_digitized_boundary_domain",
    }
