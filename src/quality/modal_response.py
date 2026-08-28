"""Time-domain modal-response proxies for the SISO roll channel."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class ModalResponseMetrics:
    response_onset_delay_s: float | None
    sensitivity_1s_deg_per_n: float | None
    roll_peak_abs_rad_s: float
    roll_rate_rms_rad_s: float
    oscillation_ratio_proxy: float | None
    oscillation_status: str
    post_release_roll_rms_rad_s: float | None
    post_release_bank_drift_deg: float | None
    sine_gain_rad_s_per_n: float | None
    sine_phase_lag_deg: float | None
    action_rms_n: float
    action_total_variation_n: float
    action_saturation_fraction: float

    def as_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


def _validate_trace(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    values = tuple(np.asarray(array, dtype=float) for array in arrays)
    if not values or any(value.ndim != 1 for value in values):
        raise ValueError("response traces must be one-dimensional")
    if len({len(value) for value in values}) != 1 or len(values[0]) < 3:
        raise ValueError("response traces must have equal length and at least three samples")
    return values


def _first_edge(force: np.ndarray) -> int | None:
    changes = np.flatnonzero(np.abs(np.diff(force)) > 1e-12)
    if len(changes):
        return int(changes[0] + 1)
    return 0 if abs(force[0]) > 1e-12 else None


def _onset_delay(time_s: np.ndarray, force_n: np.ndarray, roll_rate: np.ndarray) -> float | None:
    edge = _first_edge(force_n)
    if edge is None:
        return None
    baseline = roll_rate[max(0, edge - 1)]
    response = np.abs(roll_rate[edge:] - baseline)
    peak = float(np.max(response))
    if peak <= 1e-12:
        return None
    crossings = np.flatnonzero(response >= max(0.01 * peak, 1e-10))
    return None if not len(crossings) else float(time_s[edge + int(crossings[0])] - time_s[edge])


def _sensitivity_1s(time_s: np.ndarray, force_n: np.ndarray, roll_rate: np.ndarray) -> float | None:
    edge = _first_edge(force_n)
    if edge is None:
        return None
    before = force_n[max(0, edge - 1)]
    delta_force = force_n[edge] - before
    if abs(delta_force) <= 1e-12:
        return None
    end_time = time_s[edge] + 1.0
    end = int(np.searchsorted(time_s, end_time, side="left"))
    if end >= len(time_s) or not np.allclose(force_n[edge:end + 1], force_n[edge], atol=1e-10):
        return None
    bank_change = float(np.trapezoid(roll_rate[edge:end + 1], time_s[edge:end + 1]))
    return abs(math.degrees(bank_change) / delta_force)


def _oscillation_proxy(force_n: np.ndarray, roll_rate: np.ndarray) -> tuple[float | None, str]:
    edge = _first_edge(force_n)
    if edge is None:
        return None, "not_assessable_no_edge"
    direction = np.sign(force_n[edge] - force_n[max(0, edge - 1)])
    response = direction * (roll_rate[edge:] - roll_rate[max(0, edge - 1)])
    maxima, _ = find_peaks(response)
    if len(maxima) < 2:
        return None, "not_assessable_missing_two_peaks"
    first, third = int(maxima[0]), int(maxima[1])
    minima, _ = find_peaks(-response[first:third + 1])
    if not len(minima):
        return None, "not_assessable_missing_valley"
    second = first + int(minima[0])
    p1, p2, p3 = float(response[first]), float(response[second]), float(response[third])
    denominator = p1 + p3 + 2.0 * p2
    if denominator <= 1e-12:
        return None, "not_assessable_nonpositive_denominator"
    return (p1 + p3 - 2.0 * p2) / denominator, "assessable_siso_proxy"


def _post_release(time_s: np.ndarray, force_n: np.ndarray, roll_rate: np.ndarray) -> tuple[float | None, float | None]:
    active = np.flatnonzero(np.abs(force_n) > 1e-12)
    if not len(active) or active[-1] >= len(force_n) - 2:
        return None, None
    release = int(active[-1] + 1)
    bank = np.concatenate(([0.0], np.cumsum(0.5 * (roll_rate[1:] + roll_rate[:-1]) * np.diff(time_s))))
    return (
        float(np.sqrt(np.mean(np.square(roll_rate[release:])))),
        float(abs(math.degrees(bank[-1] - bank[release]))),
    )


def _sine_response(time_s: np.ndarray, force_n: np.ndarray, roll_rate: np.ndarray, frequency_hz: float | None) -> tuple[float | None, float | None]:
    if frequency_hz is None or frequency_hz <= 0:
        return None, None
    omega = 2.0 * math.pi * frequency_hz
    start = int(len(time_s) // 3)
    basis = np.column_stack((np.sin(omega * time_s[start:]), np.cos(omega * time_s[start:]), np.ones(len(time_s) - start)))
    force_coefficients, *_ = np.linalg.lstsq(basis, force_n[start:], rcond=None)
    rate_coefficients, *_ = np.linalg.lstsq(basis, roll_rate[start:], rcond=None)
    force_amplitude = float(np.hypot(force_coefficients[0], force_coefficients[1]))
    if force_amplitude <= 1e-12:
        return None, None
    rate_amplitude = float(np.hypot(rate_coefficients[0], rate_coefficients[1]))
    force_phase = math.atan2(force_coefficients[1], force_coefficients[0])
    rate_phase = math.atan2(rate_coefficients[1], rate_coefficients[0])
    lag = math.degrees(force_phase - rate_phase)
    lag = (lag + 180.0) % 360.0 - 180.0
    return rate_amplitude / force_amplitude, lag


def evaluate_modal_response(
    time_s: np.ndarray,
    force_n: np.ndarray,
    roll_rate_rad_s: np.ndarray,
    applied_delta_f_n: np.ndarray,
    *,
    command_kind: str | None = None,
    frequency_hz: float | None = None,
    action_limit_n: float | None = None,
) -> ModalResponseMetrics:
    """Extract matched response metrics without assigning a formal GJB grade.

    Dutch-roll quantities from this SISO trace remain proxies because ``beta``
    and ``r`` are absent.  Formal equivalent parameters require a separate LOES
    identification with a fit-quality gate.
    """

    time_s, force_n, roll_rate_rad_s, applied_delta_f_n = _validate_trace(
        time_s, force_n, roll_rate_rad_s, applied_delta_f_n
    )
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be strictly increasing")
    if command_kind in {None, "step"}:
        onset_delay = _onset_delay(time_s, force_n, roll_rate_rad_s)
        sensitivity = _sensitivity_1s(time_s, force_n, roll_rate_rad_s)
        oscillation, oscillation_status = _oscillation_proxy(force_n, roll_rate_rad_s)
    else:
        onset_delay = None
        sensitivity = None
        oscillation, oscillation_status = None, "not_assessable_requires_held_step"
    if command_kind in {"pulse", "doublet", "staircase", "piecewise"}:
        post_roll, post_bank = _post_release(time_s, force_n, roll_rate_rad_s)
    else:
        post_roll, post_bank = None, None
    sine_gain, sine_phase = _sine_response(
        time_s,
        force_n,
        roll_rate_rad_s,
        frequency_hz if command_kind in {None, "sine"} else None,
    )
    if action_limit_n is None or action_limit_n <= 0:
        saturation = 0.0
    else:
        saturation = float(np.mean(np.abs(applied_delta_f_n) >= action_limit_n - 1e-9))
    return ModalResponseMetrics(
        response_onset_delay_s=onset_delay,
        sensitivity_1s_deg_per_n=sensitivity,
        roll_peak_abs_rad_s=float(np.max(np.abs(roll_rate_rad_s))),
        roll_rate_rms_rad_s=float(np.sqrt(np.mean(np.square(roll_rate_rad_s)))),
        oscillation_ratio_proxy=oscillation,
        oscillation_status=oscillation_status,
        post_release_roll_rms_rad_s=post_roll,
        post_release_bank_drift_deg=post_bank,
        sine_gain_rad_s_per_n=sine_gain,
        sine_phase_lag_deg=sine_phase,
        action_rms_n=float(np.sqrt(np.mean(np.square(applied_delta_f_n)))),
        action_total_variation_n=float(np.sum(np.abs(np.diff(applied_delta_f_n)))),
        action_saturation_fraction=saturation,
    )
