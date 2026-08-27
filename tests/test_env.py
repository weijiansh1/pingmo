import numpy as np
import pytest

from src.aircraft.sampler import generate_plant_library
from src.envs.commands import CommandProfile
from src.envs.p_channel_env import RollQualityEnv


def test_environment_holds_each_action_for_four_plant_steps() -> None:
    env = RollQualityEnv(generate_plant_library(3, {"train_core": 1}), horizon_steps=3, pilot_signal="step", correction_ratio=0.3, pilot_force_scale_n=22.0, normalized_rate_limit_s_inv=4.0)
    observation, info = env.reset(seed=3)
    assert observation.shape == (142,)
    assert observation[0] == pytest.approx(22.0)
    next_observation, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert next_observation.shape == (142,)
    assert np.isfinite(reward)
    assert info["plant_substeps"] == 4
    assert info["f_pilot"] == pytest.approx(22.0)
    actuator_alpha = 1.0 - np.exp(-0.02 / 0.08)
    assert info["delta_f"] == pytest.approx(0.3 * 22.0 * actuator_alpha * 4.0 * 0.02)
    assert info["f_eq"] == pytest.approx(info["f_pilot"] + info["delta_f"])
    assert np.isfinite(info["p_ref"])
    assert info["critic_state"].shape == (env.critic_state_dim,)


def test_environment_exposes_fixed_width_privileged_state_at_reset_and_step() -> None:
    env = RollQualityEnv(generate_plant_library(7, {"train_core": 1}), horizon_steps=3, pilot_signal="step")
    observation, reset_info = env.reset(seed=7)
    next_observation, _, _, _, step_info = env.step(np.array([0.0], dtype=np.float32))

    assert observation.shape == (142,)
    assert next_observation.shape == (142,)
    assert reset_info["critic_state"].shape == (env.critic_state_dim,)
    assert step_info["critic_state"].shape == (env.critic_state_dim,)
    assert env.critic_state_dim > 19
    assert np.isfinite(reset_info["critic_state"]).all()
    assert np.isfinite(step_info["critic_state"]).all()


def test_environment_reports_nonreward_cancellation_and_constraint_diagnostics() -> None:
    env = RollQualityEnv(generate_plant_library(11, {"train_core": 1}), horizon_steps=4, pilot_signal="step", correction_ratio=0.3)
    env.reset(seed=11)
    _, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))

    assert info["saturation_fraction"] == pytest.approx(0.0)
    assert info["cancel_index"] > 0.0
    actuator_alpha = 1.0 - np.exp(-0.02 / 0.08)
    assert info["action_rate_n_per_s"] == pytest.approx(-0.3 * 22.0 * 4.0 * actuator_alpha)
    assert "cancel_correlation" in info


def test_environment_exposes_rate_limited_command_and_lagged_applied_action() -> None:
    env = RollQualityEnv(
        generate_plant_library(19, {"train_core": 1}),
        horizon_steps=3,
        pilot_signal="step",
        correction_ratio=0.3,
        pilot_force_scale_n=22.0,
        normalized_rate_limit_s_inv=4.0,
        actuator_time_constant_s=0.08,
    )
    observation, _ = env.reset(seed=19)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))

    rate_limited_command = 4.0 * 0.02
    expected_applied = (1.0 - np.exp(-0.02 / 0.08)) * rate_limited_command
    assert observation.shape == (142,)
    assert info["commanded_action"] == pytest.approx(rate_limited_command)
    assert info["applied_action"] == pytest.approx(expected_applied)
    assert info["delta_f"] == pytest.approx(expected_applied * 0.3 * 22.0)
    assert info["command_delta"] == pytest.approx(rate_limited_command)
    assert info["applied_action_delta"] == pytest.approx(expected_applied)


def test_environment_runs_a_named_multi_command_profile() -> None:
    profile = CommandProfile("doublet-1.00", "doublet", amplitude=1.0)
    env = RollQualityEnv(
        generate_plant_library(23, {"train_core": 1}),
        horizon_steps=4,
        command_profiles=(profile,),
    )

    observation, reset_info = env.reset(seed=23)
    next_observation, _, _, _, step_info = env.step(np.array([0.0], dtype=np.float32))

    assert reset_info["command_id"] == "doublet-1.00"
    assert observation[0] == pytest.approx(22.0)
    assert step_info["command_id"] == "doublet-1.00"
    assert next_observation[0] == pytest.approx(-22.0)
