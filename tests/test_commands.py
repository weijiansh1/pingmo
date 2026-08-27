import numpy as np

from src.envs.commands import CommandProfile, default_command_suite


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
    kinds = {profile.kind for profile in default_command_suite()}

    assert kinds == {"step", "doublet", "sine", "chirp"}
