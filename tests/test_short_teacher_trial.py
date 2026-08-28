import pytest

from src.experiments.short_teacher_trial import (
    ShortTrialConfig,
    short_training_command_suite,
    summarize_trial_evaluation,
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
