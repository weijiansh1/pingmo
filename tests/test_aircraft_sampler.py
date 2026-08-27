import json

import pytest

from src.aircraft.sampler import build_iv_a_library, build_stratified_library, generate_plant_library, persist_plant_library


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

    assert len(rows) == sum(targets.values())
    assert {row["aircraft_class"] for row in rows} == {"IV"}
    assert {row["flight_phase"] for row in rows} == {"A"}
    assert {row["profile"] for row in rows} == {"IV-A"}
    assert all(row["sensitivity_measured_deg_per_n"] == pytest.approx(row["sensitivity_target_deg_per_n"], rel=0.005) for row in rows)
