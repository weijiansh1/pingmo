from src.aircraft.parameters import PChannelParameters
from src.envs.commands import CommandProfile
from src.experiments.feasibility_audit import (
    FeasibilityPolicy,
    audit_pair,
    classify_feasibility,
    iter_audit_library,
    summarize_audit_rows,
)


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
    assert row["feasibility_label"] in {
        "control_not_needed",
        "constrained_feasible",
        "authority_limited",
        "oracle_unreachable",
    }
    assert isinstance(row["oracle_feasible"], bool)
    assert isinstance(row["controller_improvable"], bool)
    assert row["constrained_max_increment_n"] <= row["constrained_increment_limit_n"] + 1e-9


def test_feasibility_labels_separate_authority_from_oracle_limits() -> None:
    policy = FeasibilityPolicy(relative_tracking_rmse_limit=0.10)

    assert classify_feasibility(0.05, 0.01, 0.03, policy) == "control_not_needed"
    assert classify_feasibility(0.30, 0.01, 0.08, policy) == "constrained_feasible"
    assert classify_feasibility(0.30, 0.01, 0.20, policy) == "authority_limited"
    assert classify_feasibility(0.30, 0.20, 0.25, policy) == "oracle_unreachable"


def test_audit_summary_keeps_splits_separate() -> None:
    rows = [
        {"split": "validation", "command_id": "step-1.00", "constrained_improves_raw": True, "oracle_feasible": True, "controller_improvable": True, "feasibility_label": "constrained_feasible", "raw_tracking_rmse": 2.0, "constrained_tracking_rmse": 1.0, "raw_relative_tracking_rmse": 0.2, "constrained_relative_tracking_rmse": 0.1, "constrained_saturation_fraction": 0.0},
        {"split": "id_test", "command_id": "step-1.00", "constrained_improves_raw": False, "oracle_feasible": False, "controller_improvable": False, "feasibility_label": "authority_limited", "raw_tracking_rmse": 1.0, "constrained_tracking_rmse": 2.0, "raw_relative_tracking_rmse": 0.1, "constrained_relative_tracking_rmse": 0.2, "constrained_saturation_fraction": 1.0},
    ]

    summary = summarize_audit_rows(rows)

    assert set(summary["by_split"]) == {"validation", "id_test"}
    assert set(summary["by_command"]) == {"step-1.00"}
    assert summary["by_split"]["validation"]["improvement_rate"] == 1.0
    assert summary["by_split"]["id_test"]["improvement_rate"] == 0.0
    assert summary["by_split"]["validation"]["oracle_feasible_rate"] == 1.0


def test_audit_iterator_skips_checkpointed_pairs(tmp_path) -> None:
    library = tmp_path / "plants.jsonl"
    library.write_text(
        '{"plant_id":"validation-0000","split":"validation","parameters":'
        '{"l_fa":1.0,"lambda_s":-0.04,"t_r":0.5,"zeta_d":0.2,'
        '"omega_d":2.0,"r_omega":1.5,"r_zeta":0.7,"tau_p":0.0125}}\n',
        encoding="utf-8",
    )
    profiles = (
        CommandProfile("step-1.00", "step", amplitude=1.0),
        CommandProfile("step--1.00", "step", amplitude=-1.0),
    )

    rows = list(
        iter_audit_library(
            library,
            profiles,
            duration_s=0.2,
            completed_keys={("validation-0000", "step-1.00")},
        )
    )

    assert [(row["plant_id"], row["command_id"]) for row in rows] == [
        ("validation-0000", "step--1.00")
    ]
