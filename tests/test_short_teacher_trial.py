from pathlib import Path

import pytest
import torch

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import PlantRecord
from src.experiments.short_teacher_trial import (
    ShortTrialConfig,
    _write_actor_snapshot,
    balanced_episode_assignments,
    library_fingerprint,
    require_balanced_quality_levels,
    short_training_command_suite,
    summarize_trial_evaluation,
)
from src.teacher.sac.teacher import PrivilegedSAC


def _record(plant_id: str, level: str) -> PlantRecord:
    return PlantRecord(
        plant_id,
        "train_core",
        level,
        "IV",
        "A",
        PChannelParameters(0.2, -0.05, 0.8, 0.3, 2.0, 1.0, 1.0, 0.02),
    )


def test_short_training_commands_cover_all_families_and_one_second_step() -> None:
    profiles = short_training_command_suite(1.25)
    assert {profile.kind for profile in profiles} == {
        "step", "pulse", "doublet", "square", "sine", "chirp", "staircase", "piecewise",
    }
    assert all(profile.duration_s == pytest.approx(1.25) for profile in profiles)
    assert all(len(profile.samples(0.001, 1.25, 22.0)) == 1_250 for profile in profiles)


def test_short_trial_config_rejects_episode_too_short_for_sensitivity() -> None:
    with pytest.raises(ValueError, match="1 s sensitivity"):
        ShortTrialConfig(episode_duration_s=1.0)


def test_actor_snapshot_interval_must_end_complete_parallel_episodes() -> None:
    with pytest.raises(ValueError, match="complete parallel episode batches"):
        ShortTrialConfig(actor_snapshot_interval_steps=1_000)

    assert ShortTrialConfig(actor_snapshot_interval_steps=40_000).actor_snapshot_interval_steps == 40_000


def test_actor_snapshot_contains_one_shared_actor_and_provenance(tmp_path: Path) -> None:
    learner = PrivilegedSAC(
        13,
        25,
        1,
        actor_width=16,
        actor_residual_blocks=1,
        critic_width=16,
        critic_residual_blocks=1,
    )
    config = ShortTrialConfig(actor_snapshot_interval_steps=40_000, device="cpu")

    path = _write_actor_snapshot(
        tmp_path,
        learner,
        config,
        {"plants_sha256": "plants"},
        {"commit": "abc", "tracked_dirty": False},
        transition_steps=40_000,
        completed_episodes=32,
        actor_observation_dim=13,
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert path.name == "actor_step_000040000.pt"
    assert payload["transition_steps"] == 40_000
    assert payload["completed_episodes"] == 32
    assert payload["aircraft_library"]["plants_sha256"] == "plants"
    assert payload["source"]["commit"] == "abc"
    assert "critic" not in payload


def test_balanced_assignments_cover_every_level_command_pair_before_repeat() -> None:
    records = [
        _record(f"p-{level}-{index}", level)
        for level in ("level_1", "level_2", "level_3")
        for index in range(2)
    ]
    profiles = short_training_command_suite()
    pair_count = 3 * len(profiles)

    assignments = balanced_episode_assignments(records, profiles, pair_count, seed=17)

    assert len({(record.quality_region, profile.command_id) for record, profile in assignments}) == pair_count
    assert len({record.plant_id for record, _ in assignments}) == len(records)


def test_full_fleet_assignment_visits_all_1800_training_plants_once() -> None:
    records = [
        _record(f"p-{level}-{index:03d}", level)
        for level in ("level_1", "level_2", "level_3")
        for index in range(600)
    ]

    assignments = balanced_episode_assignments(
        records,
        short_training_command_suite(),
        episode_count=1_800,
        seed=20260828,
    )

    assert len(assignments) == 1_800
    assert len({record.plant_id for record, _ in assignments}) == 1_800
    assert {
        level: sum(record.quality_region == level for record, _ in assignments)
        for level in ("level_1", "level_2", "level_3")
    } == {"level_1": 600, "level_2": 600, "level_3": 600}


def test_level_balance_guard_rejects_stale_core_boundary_library() -> None:
    records = [_record("core", "core"), _record("boundary", "boundary")]

    with pytest.raises(ValueError, match="level_1/level_2/level_3"):
        require_balanced_quality_levels(records, "training source")


def test_library_fingerprint_requires_level_balanced_manifest(tmp_path) -> None:
    plants = tmp_path / "plants.jsonl"
    manifest = tmp_path / "manifest.json"
    plants.write_text("{}\n", encoding="utf-8")
    manifest.write_text('{"sampling_policy": "old_core_boundary"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="level-balanced"):
        library_fingerprint(plants)


def test_trial_summary_compares_matched_episode_costs() -> None:
    raw = [{"plant_id": "p1", "command_id": "step", "episode_reward": -2.0}]
    controlled = [{
        "plant_id": "p1",
        "command_id": "step",
        "episode_reward": -1.0,
        "controlled_action_rms_n": 0.5,
        "controlled_action_total_variation_n": 2.0,
        "controlled_action_saturation_fraction": 0.0,
        "raw_response_onset_delay_s": 0.1,
        "controlled_response_onset_delay_s": 0.08,
        "raw_sensitivity_1s_deg_per_n": 3.0,
        "controlled_sensitivity_1s_deg_per_n": 2.5,
        "raw_oscillation_ratio_proxy": None,
        "controlled_oscillation_ratio_proxy": None,
        "raw_post_release_roll_rms_rad_s": None,
        "controlled_post_release_roll_rms_rad_s": None,
    }]
    summary = summarize_trial_evaluation(raw, controlled)
    assert summary["harm_rate"] == 0.0
    assert summary["median_episode_cost_change"] == pytest.approx(-1.0)
    assert summary["median_onset_delay_change_s"] == pytest.approx(-0.02)
    assert summary["raw_good_pairs"] == 0
    assert summary["raw_good_harm_rate"] is None
