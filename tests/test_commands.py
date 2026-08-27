import numpy as np
import pytest

from src.envs.commands import DEFAULT_COMMAND_DURATION_S, CommandProfile, default_command_suite


def test_doublet_profile_reverses_sign_at_half_duration() -> None:
    profile = CommandProfile("doublet-1.00", "doublet", amplitude=1.0)

    samples = profile.samples(action_dt=0.02, duration_s=4.0, nominal_force_n=22.0)

    assert samples.shape == (200,)
    assert samples[0] == 22.0
    assert samples[50] == -22.0
    assert samples[100] == 0.0


def test_command_profile_samples_are_deterministic_and_force_scaled() -> None:
    profile = CommandProfile("sine-0.50hz", "sine", amplitude=0.5, frequency_hz=0.5)

    first = profile.samples(action_dt=0.02, duration_s=2.0, nominal_force_n=22.0)
    second = profile.samples(action_dt=0.02, duration_s=2.0, nominal_force_n=22.0)

    assert np.array_equal(first, second)
    assert np.max(np.abs(first)) <= 11.0 + 1e-12


def test_default_command_suite_contains_all_required_families() -> None:
    profiles = default_command_suite()
    kinds = {profile.kind for profile in profiles}

    assert kinds == {"step", "doublet", "sine", "chirp"}
    assert {profile.duration_s for profile in profiles} == {DEFAULT_COMMAND_DURATION_S}
    assert [profile.command_id for profile in profiles[:6]] == [
        "step-pos-0.25",
        "step-pos-0.50",
        "step-pos-1.00",
        "step-neg-0.25",
        "step-neg-0.50",
        "step-neg-1.00",
    ]


def test_default_command_cannot_silently_change_its_force_history() -> None:
    profile = default_command_suite()[0]

    with pytest.raises(ValueError, match="is defined for"):
        profile.samples(action_dt=0.02, duration_s=5.0, nominal_force_n=22.0)
