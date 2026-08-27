from pathlib import Path

import numpy as np
import pytest

from src.aircraft.parameters import PChannelParameters
from src.aircraft.p_channel import PChannel
from src.quality.a116 import A116BoundarySet, assess_a116, audit_a116_parameters, calibrate_a116_step, roll_rate_phase_proxy_deg


def test_calibrate_a116_step_hits_60_degrees_at_1_point_7_dutch_roll_time_constants() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)

    calibration = calibrate_a116_step(parameters, max_step_force_n=22.0, dt=0.005)

    assert calibration.status == "assessable"
    assert calibration.target_time_s == pytest.approx(1.7 / (parameters.zeta_d * parameters.omega_d))
    assert calibration.bank_angle_at_target_deg == pytest.approx(60.0, abs=0.05)
    assert calibration.step_force_n is not None
    assert 0.0 < calibration.step_force_n <= 22.0


def test_calibrate_a116_step_does_not_assign_force_when_60_degree_target_exceeds_limit() -> None:
    parameters = PChannelParameters(0.001, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)

    calibration = calibrate_a116_step(parameters, max_step_force_n=22.0, dt=0.005)

    assert calibration.status == "not_assessable"
    assert calibration.reason == "input_not_achievable"
    assert calibration.step_force_n is None


def test_calibrate_a116_step_allows_an_uncapped_standard_audit_input() -> None:
    parameters = PChannelParameters(0.001, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)

    calibration = calibrate_a116_step(parameters, max_step_force_n=None, dt=0.005)

    assert calibration.status == "assessable"
    assert calibration.step_force_n is not None
    assert calibration.step_force_n > 22.0
    assert calibration.bank_angle_at_target_deg == pytest.approx(60.0, abs=0.05)


def test_a116_ac_boundaries_are_source_labelled_and_interpolated() -> None:
    root = Path(__file__).parents[1]
    boundaries = A116BoundarySet.from_csv(root / "data/gjb_a116_boundary.csv")

    assert boundaries.source_pdf_pages == {240}
    assert boundaries.interpolate("A_C", 1, -200.0) == pytest.approx(0.25)
    assert boundaries.interpolate("A_C", 2, -200.0) == pytest.approx(0.60)


def test_a116_assessment_classifies_only_inside_the_traced_domain() -> None:
    root = Path(__file__).parents[1]
    boundaries = A116BoundarySet.from_csv(root / "data/gjb_a116_boundary.csv")

    assert assess_a116(boundaries, "A_C", -200.0, 0.24).level == 1
    assert assess_a116(boundaries, "A_C", -200.0, 0.40).level == 2
    assert assess_a116(boundaries, "A_C", -200.0, 0.61).level == "above_level_2"
    assert assess_a116(boundaries, "A_C", -380.0, 0.10).level == "not_available"


def test_roll_rate_phase_proxy_is_wrapped_for_a116_and_reported_with_metric() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    phase = roll_rate_phase_proxy_deg(parameters)
    report = assess_a116(A116BoundarySet.from_csv(Path(__file__).parents[1] / "data/gjb_a116_boundary.csv"), "A_C", phase, 0.2)

    assert -360.0 <= phase < 0.0
    assert report.psi_p_deg == pytest.approx(phase)
    assert report.source_pdf_page == 240


def test_a116_audit_removes_the_spiral_mode_before_using_the_phase_proxy() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    audit = audit_a116_parameters(
        A116BoundarySet.from_csv(Path(__file__).parents[1] / "data/gjb_a116_boundary.csv"),
        parameters,
        "A_C",
        np.arange(1000) * 0.005,
    )

    assert -360.0 <= audit["psi_p_deg"] < 0.0
    assert audit["source_pdf_page"] == 240
    assert audit["spiral_mode_removed"] is True
    assert audit["a116_status"] in {"assessable", "not_assessable"}
    assert "p_osc_over_p_av_unvalidated_peak_proxy" not in audit


def test_a116_audit_reports_strict_peaks_and_never_uses_the_legacy_proxy() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    boundaries = A116BoundarySet.from_csv(Path(__file__).parents[1] / "data/gjb_a116_boundary.csv")

    audit = audit_a116_parameters(boundaries, parameters, "A_C", np.arange(4000) * 0.005)

    assert audit["a116_status"] in {"assessable", "not_assessable"}
    assert "p_osc_over_p_av_unvalidated_peak_proxy" not in audit
    if audit["a116_status"] == "assessable":
        assert audit["p1"] is not None
        assert audit["p2"] is not None
        assert audit["p3"] is not None
        assert audit["a116_level"] in {1, 2, "above_level_2", "not_available"}
