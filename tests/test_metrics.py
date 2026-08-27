import numpy as np
import pytest

from src.benchmark.time_domain import extract_gjb_roll_peaks, evaluate_roll_response, gjb_roll_oscillation_ratio


def test_extract_gjb_roll_peaks_uses_first_peak_first_valley_second_peak() -> None:
    time = np.arange(7.0)
    roll_rate = np.array([0.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0])

    peaks = extract_gjb_roll_peaks(time, roll_rate)

    assert peaks.status == "assessable"
    assert (peaks.p1, peaks.p2, peaks.p3) == pytest.approx((2.0, 1.0, 3.0))
    assert (peaks.t1_s, peaks.t2_s, peaks.t3_s) == pytest.approx((1.0, 2.0, 3.0))


def test_extract_gjb_roll_peaks_reports_no_second_peak_without_inventing_values() -> None:
    peaks = extract_gjb_roll_peaks(np.arange(4.0), np.array([0.0, 1.0, 0.7, 0.5]))

    assert peaks.status == "not_assessable"
    assert peaks.reason == "missing_two_roll_rate_peaks"
    assert peaks.p1 is None


def test_generic_roll_response_metrics_keep_zero_signal_as_a_non_gjb_diagnostic() -> None:
    metrics = evaluate_roll_response(np.arange(4.0), np.zeros(4))

    assert metrics["p_osc_over_p_av"] == 0.0
    assert metrics["peak"] == 0.0


def test_roll_response_metrics_extract_finite_oscillation_values() -> None:
    time = np.linspace(0.0, 12.0, 2401)
    response = np.exp(-0.25 * time) * np.sin(3.0 * time)
    metrics = evaluate_roll_response(time, response)
    assert metrics["p1"] > 0.0
    assert metrics["p2"] > 0.0
    assert metrics["p3"] > 0.0
    assert np.isfinite(metrics["p_osc_over_p_av"])
    assert metrics["settling_time_s"] >= 0.0


def test_gjb_roll_oscillation_ratio_uses_the_a120_formula_without_damping_branch() -> None:
    assert gjb_roll_oscillation_ratio(1.0, 0.5, 0.8) == pytest.approx((1.0 + 0.8 - 2 * 0.5) / (1.0 + 0.8 + 2 * 0.5))
