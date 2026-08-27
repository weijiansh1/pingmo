import json

import pytest

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import (
    IV_A_L_FA_EVIDENCE_MAX,
    build_iv_a_library,
    build_stratified_library,
    classify_iv_a_quality,
    generate_plant_library,
    persist_plant_library,
)


def test_sobol_library_is_seeded_labeled_and_derives_mode_parameters() -> None:
    records = generate_plant_library(seed=7, split_counts={"train_core": 3, "train_boundary": 2})
    again = generate_plant_library(seed=7, split_counts={"train_core": 3, "train_boundary": 2})
    assert [record.plant_id for record in records] == [record.plant_id for record in again]
    assert [record.split for record in records].count("train_core") == 3
    assert all(record.parameters.omega_phi == record.parameters.r_omega * record.parameters.omega_d for record in records)
    assert {record.quality_region for record in records} == {"core", "boundary"}


def test_persisted_library_contains_all_splits_and_manifest(tmp_path) -> None:
    counts = {"train_core": 3, "train_boundary": 2, "validation": 1, "id_test": 1, "ood_test": 1, "extreme_test": 1}
    destination = persist_plant_library(tmp_path, seed=9, split_counts=counts)
    lines = (destination / "plants.jsonl").read_text(encoding="utf-8").splitlines()
    manifest = (destination / "manifest.json").read_text(encoding="utf-8")
    assert len(lines) == sum(counts.values())
    assert '"total_plants": 9' in manifest
    assert '"validation": 1' in manifest


def test_stratified_builder_keeps_candidates_metrics_and_selection_reason(tmp_path) -> None:
    targets = {"train_core": 4, "train_boundary": 3, "validation": 1, "id_test": 1, "ood_test": 2, "extreme_test": 1}
    destination = build_stratified_library(tmp_path, seed=12, candidate_count=64, target_counts=targets)
    selected = (destination / "plants.jsonl").read_text(encoding="utf-8").splitlines()
    candidates = (destination / "candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(selected) == sum(targets.values())
    assert len(candidates) == 64
    assert '"selection_reason"' in selected[0]
    assert '"raw_metrics"' in candidates[0]


def test_iv_a_library_calibrates_and_persists_response_sensitivity(tmp_path) -> None:
    targets = {"train_core": 4, "train_boundary": 3, "validation": 1, "id_test": 1, "ood_test": 2, "extreme_test": 1}
    destination = build_iv_a_library(tmp_path, seed=15, candidate_count=64, target_counts=targets)
    rows = [json.loads(line) for line in (destination / "plants.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == sum(targets.values())
    assert manifest["model_version"] == "GJB_s1_corrected"
    assert {row["aircraft_class"] for row in rows} == {"IV"}
    assert {row["flight_phase"] for row in rows} == {"A"}
    assert {row["profile"] for row in rows} == {"IV-A"}
    assert {row["task_subtype"] for row in rows} == {"ordinary_not_CO_GA"}
    assert {row["inceptor_assumption"] for row in rows} == {"stick_controlled_for_A31_gate"}
    assert {row["delay_definition"] for row in rows} == {"pure_transport_delay_before_p_channel"}
    assert all(row["sensitivity_measured_deg_per_n"] == pytest.approx(row["sensitivity_target_deg_per_n"], rel=0.005) for row in rows)
    assert manifest["level_counts"] == {"L1": 3, "L2": 3, "L3": 3, "OOD": 3}
    assert {row["sampling_bucket"] for row in rows} == {"L1", "L2", "L3", "OOD"}
    assert all(row["parameters"]["l_fa"] <= IV_A_L_FA_EVIDENCE_MAX for row in rows if row["sampling_bucket"] != "OOD")
    assert all(row["gjb_level"] in {1, 2, 3} for row in rows if row["sampling_bucket"] != "OOD")
    assert all(row["ood_reasons"] for row in rows if row["sampling_bucket"] == "OOD")


def test_iv_a_quality_uses_worst_direct_gjb_component_and_keeps_ood_separate() -> None:
    level_1 = PChannelParameters(.1, -.05, .8, .3, 2.0, 1.0, 1.0, .05)
    level_2 = PChannelParameters(.1, -.05, 1.2, .3, 2.0, 1.0, 1.0, .05)
    level_3 = PChannelParameters(.1, .10, .8, .01, 1.0, 1.0, 20.0, .05)
    outside = PChannelParameters(.1, .20, .8, .3, 2.0, 1.0, 1.0, .05)

    assert classify_iv_a_quality(level_1, 3.0)["sampling_bucket"] == "L1"
    assert classify_iv_a_quality(level_2, 3.0)["sampling_bucket"] == "L2"
    level_3_grade = classify_iv_a_quality(level_3, 4.0)
    assert level_3_grade["sampling_bucket"] == "L3"
    assert level_3_grade["gjb_component_levels"]["spiral"] == 3
    assert level_3_grade["gjb_component_levels"]["dutch_roll"] == 3
    outside_grade = classify_iv_a_quality(outside, 3.0)
    assert outside_grade["sampling_bucket"] == "OOD"
    assert outside_grade["gjb_level"] is None
    assert "spiral_beyond_level_3" in outside_grade["ood_reasons"]
    with pytest.raises(ValueError, match="positive and finite"):
        classify_iv_a_quality(level_1, 0.0)
