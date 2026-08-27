from src.aircraft.parameters import PChannelParameters
from src.envs.commands import CommandProfile
from src.experiments.feasibility_audit import audit_pair, summarize_audit_rows


def test_audit_pair_retains_split_command_and_constrained_metrics() -> None:
    parameters = PChannelParameters(1.0, -0.04, 0.5, 0.2, 2.0, 1.5, 0.7, 0.0125)
    profile = CommandProfile("step-1.00", "step", amplitude=1.0)

    row = audit_pair("id_test-0000", "id_test", parameters, profile, duration_s=0.4)

    assert row["plant_id"] == "id_test-0000"
    assert row["split"] == "id_test"
    assert row["command_id"] == "step-1.00"
    assert row["raw_tracking_rmse"] >= 0.0
    assert row["constrained_tracking_rmse"] >= 0.0
    assert isinstance(row["constrained_improves_raw"], bool)


def test_audit_summary_keeps_splits_separate() -> None:
    rows = [
        {"split": "validation", "command_id": "step-1.00", "constrained_improves_raw": True, "raw_tracking_rmse": 2.0, "constrained_tracking_rmse": 1.0, "constrained_saturation_fraction": 0.0},
        {"split": "id_test", "command_id": "step-1.00", "constrained_improves_raw": False, "raw_tracking_rmse": 1.0, "constrained_tracking_rmse": 2.0, "constrained_saturation_fraction": 1.0},
    ]

    summary = summarize_audit_rows(rows)

    assert set(summary["by_split"]) == {"validation", "id_test"}
    assert set(summary["by_command"]) == {"step-1.00"}
    assert summary["by_split"]["validation"]["improvement_rate"] == 1.0
    assert summary["by_split"]["id_test"]["improvement_rate"] == 0.0
