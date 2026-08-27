"""Shared roll-response quantities for plant and controller benchmarks."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from scipy import signal

from src.aircraft.p_channel import PChannel, p_channel_polynomials
from src.aircraft.parameters import PChannelParameters


@dataclass(frozen=True, slots=True)
class GJBRollPeaks:
    """The Figure A120 first-peak, first-valley, second-peak sequence."""

    status: str
    reason: str | None
    p1: float | None
    p2: float | None
    p3: float | None
    t1_s: float | None
    t2_s: float | None
    t3_s: float | None


def extract_gjb_roll_peaks(time: np.ndarray, roll_rate: np.ndarray) -> GJBRollPeaks:
    """Extract Figure A120's signed first peak, valley, and second peak.

    The response must contain the ordered ``P1 -> P2 -> P3`` structure.
    Missing features are reported rather than synthesized from absolute values.
    """
    time = np.asarray(time, dtype=float)
    roll_rate = np.asarray(roll_rate, dtype=float)
    if time.ndim != 1 or roll_rate.ndim != 1 or len(time) != len(roll_rate) or len(time) < 3:
        raise ValueError("time and roll_rate must be equal-length one-dimensional arrays")

    maxima, _ = find_peaks(roll_rate)
    if len(maxima) < 2:
        return GJBRollPeaks("not_assessable", "missing_two_roll_rate_peaks", None, None, None, None, None, None)
    first_peak, second_peak = (int(maxima[0]), int(maxima[1]))
    minima, _ = find_peaks(-roll_rate[first_peak:second_peak + 1])
    if not len(minima):
        return GJBRollPeaks("not_assessable", "missing_valley_between_peaks", None, None, None, None, None, None)
    valley = first_peak + int(minima[0])
    return GJBRollPeaks(
        "assessable",
        None,
        float(roll_rate[first_peak]),
        float(roll_rate[valley]),
        float(roll_rate[second_peak]),
        float(time[first_peak]),
        float(time[valley]),
        float(time[second_peak]),
    )


def gjb_roll_oscillation_ratio(p1: float, p2: float, p3: float) -> float:
    """Figure A120's response-level roll-rate oscillation ratio."""
    denominator = p1 + p3 + 2.0 * p2
    if denominator <= 0.0:
        raise ValueError("Figure A120 ratio denominator must be positive")
    return (p1 + p3 - 2.0 * p2) / denominator


def evaluate_roll_response(time: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Extract legacy absolute-peak diagnostics and generic time-domain metrics.

    This helper supports plant-library construction and is not the Figure A120
    evaluator.  Formal A116 assessment uses :func:`extract_gjb_roll_peaks` and
    :func:`gjb_roll_oscillation_ratio` only after validating their prerequisites.
    """
    time = np.asarray(time, dtype=float)
    p = np.asarray(p, dtype=float)
    if time.ndim != 1 or p.ndim != 1 or len(time) != len(p) or len(time) < 3:
        raise ValueError("time and p must be equal-length one-dimensional arrays")
    indices, _ = find_peaks(np.abs(p))
    peaks = np.abs(p[indices])
    if len(peaks) < 3:
        peaks = np.pad(peaks, (0, 3 - len(peaks)))
    p1, p2, p3 = (float(value) for value in peaks[:3])
    legacy_denominator = p1 + p3 + 2.0 * p2
    legacy_ratio = (p1 + p3 - 2.0 * p2) / legacy_denominator if legacy_denominator > 0.0 else 0.0
    threshold = 0.02 * max(float(np.max(np.abs(p))), np.finfo(float).eps)
    beyond = np.flatnonzero(np.abs(p) > threshold)
    settling = float(time[beyond[-1]]) if len(beyond) else 0.0
    return {
        "p1": p1, "p2": p2, "p3": p3,
        "p_osc_over_p_av": legacy_ratio,
        "peak": float(np.max(np.abs(p))),
        "settling_time_s": settling,
        "iae": float(np.trapezoid(np.abs(p), time)),
        "itae": float(np.trapezoid(time * np.abs(p), time)),
    }


def held_step_roll_rate_response(parameters: PChannelParameters, time: np.ndarray, force_n: float) -> np.ndarray:
    """Simulate the delay-aware full P-channel under a held pilot-force step."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or len(time) < 2 or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be a strictly increasing one-dimensional array")
    dt = float(time[1] - time[0])
    if not np.allclose(np.diff(time), dt):
        raise ValueError("held-step simulation requires a uniform time grid")
    channel = PChannel(parameters, dt=dt)
    response = np.empty(len(time), dtype=float)
    for index in range(len(time)):
        response[index], _ = channel.step(force_n)
    return response


def no_spiral_step_response(parameters: PChannelParameters, time: np.ndarray) -> np.ndarray:
    """Unit-step roll-rate response with the spiral mode removed for A120.

    A120 explicitly defines ``P`` as the response with the spiral mode removed.
    Replacing the spiral factor by ``s`` gives the required no-spiral transfer;
    pure delay is omitted because it shifts time but does not change peak levels.
    """
    p = parameters
    numerator, _ = p_channel_polynomials(p)
    denominator = np.polymul(
        np.polymul([1.0, 0.0], [1.0, 1.0 / p.t_r]),
        [1.0, 2 * p.zeta_d * p.omega_d, p.omega_d**2],
    )
    _, response = signal.step((numerator, denominator), T=np.asarray(time, dtype=float))
    return np.asarray(response, dtype=float)
