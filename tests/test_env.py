import numpy as np
import pytest

from src.aircraft.sampler import generate_plant_library
from src.envs.p_channel_env import RollQualityEnv


def test_environment_holds_each_action_for_four_plant_steps() -> None:
    env = RollQualityEnv(generate_plant_library(3, {"train_core": 1}), horizon_steps=3, pilot_signal="step", correction_ratio=0.3, pilot_force_scale_n=22.0, normalized_rate_limit_s_inv=4.0)
    observation, info = env.reset(seed=3)
    assert observation.shape == (141,)
    assert observation[0] == pytest.approx(22.0)
    next_observation, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))
    assert next_observation.shape == (141,)
    assert np.isfinite(reward)
    assert info["plant_substeps"] == 4
    assert info["f_pilot"] == pytest.approx(22.0)
    assert info["delta_f"] == pytest.approx(0.3 * 22.0 * 4.0 * 0.02)
    assert info["f_eq"] == pytest.approx(info["f_pilot"] + info["delta_f"])
    assert np.isfinite(info["p_ref"])
    assert info["critic_state"].shape == (env.critic_state_dim,)


def test_environment_exposes_fixed_width_privileged_state_at_reset_and_step() -> None:
    env = RollQualityEnv(generate_plant_library(7, {"train_core": 1}), horizon_steps=3, pilot_signal="step")
    observation, reset_info = env.reset(seed=7)
    next_observation, _, _, _, step_info = env.step(np.array([0.0], dtype=np.float32))

    assert observation.shape == (141,)
    assert next_observation.shape == (141,)
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
    assert info["action_rate_n_per_s"] == pytest.approx(-0.3 * 22.0 * 4.0)
    assert "cancel_correlation" in info
