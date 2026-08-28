import numpy as np
import pytest

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import PlantRecord, generate_plant_library
from src.envs.commands import CommandProfile
from src.envs.p_channel_env import RollQualityEnv


def _record(delay_s: float) -> PlantRecord:
    return PlantRecord(
        plant_id=f"delay-{delay_s}",
        split="train_core",
        quality_region="level_1",
        aircraft_class="IV",
        flight_phase="A",
        parameters=PChannelParameters(0.2, -0.05, 0.8, 0.3, 2.0, 1.0, 1.0, delay_s),
    )


def test_environment_makes_one_decision_per_one_ms_plant_sample() -> None:
    env = RollQualityEnv(
        generate_plant_library(3, {"train_core": 1}),
        horizon_steps=3,
        pilot_signal="step",
        correction_ratio=0.3,
        pilot_force_scale_n=22.0,
        normalized_rate_limit_s_inv=4.0,
    )
    observation, info = env.reset(seed=3)

    assert observation.shape == (268,)
    assert observation[0] == pytest.approx(1.0)
    next_observation, reward, terminated, truncated, info = env.step(np.array([1.0], dtype=np.float32))

    assert next_observation.shape == (268,)
    assert np.isfinite(reward)
    assert not terminated and not truncated
    assert info["plant_substeps"] == 1
    assert info["plant_dt_s"] == pytest.approx(0.001)
    assert info["policy_dt_s"] == pytest.approx(0.001)
    assert info["commanded_action"] == pytest.approx(0.004)
    assert info["applied_action"] == pytest.approx(0.004)
    assert info["delta_f"] == pytest.approx(0.004 * 0.3 * 22.0)
    assert info["f_eq"] == pytest.approx(info["f_pilot"] + info["delta_f"])
    assert info["critic_state"].shape == (env.critic_state_dim,)


def test_actor_receives_current_modal_and_delay_response_costs() -> None:
    env = RollQualityEnv([_record(0.020)], horizon_steps=30, pilot_signal="step")
    observation, _ = env.reset(seed=7)

    assert np.allclose(observation[6:10], 0.0)
    observation, _, _, _, first_info = env.step(np.zeros(1, dtype=np.float32))
    observation, _, _, _, second_info = env.step(np.zeros(1, dtype=np.float32))

    assert second_info["current_response_wait_s"] == pytest.approx(0.001)
    assert observation[9] > 0.0
    assert second_info["actor_response_feedback"]["delay_response_cost"] == pytest.approx(observation[9])
    assert set(second_info["actor_response_feedback"]) == {
        "roll_cost", "oscillation_cost", "spiral_recovery_cost", "delay_response_cost",
    }
    assert first_info["current_response_wait_s"] == 0.0


def test_critic_state_does_not_expose_a_transport_delay_fifo() -> None:
    short = RollQualityEnv([_record(0.001)], horizon_steps=3, pilot_signal="step")
    long = RollQualityEnv([_record(0.200)], horizon_steps=3, pilot_signal="step")
    _, short_info = short.reset(seed=1)
    _, long_info = long.reset(seed=1)

    assert short.critic_state_dim == long.critic_state_dim == 273
    assert short_info["critic_state"].shape == long_info["critic_state"].shape


def test_environment_reports_cancellation_and_constraint_diagnostics() -> None:
    env = RollQualityEnv(
        generate_plant_library(11, {"train_core": 1}),
        horizon_steps=4,
        pilot_signal="step",
        correction_ratio=0.3,
    )
    env.reset(seed=11)
    _, _, _, _, info = env.step(np.array([-1.0], dtype=np.float32))

    assert info["saturation_fraction"] == pytest.approx(0.0)
    assert info["cancel_index"] > 0.0
    assert info["action_rate_n_per_s"] == pytest.approx(-0.3 * 22.0 * 4.0)
    assert "cancel_correlation" in info
    assert set(info["reward_costs"]) == {
        "wrong_way", "added_delay", "sensitivity", "oscillation",
        "spiral_recovery", "action_energy", "action_delta",
    }


def test_optional_augmentation_lag_is_integrated_at_one_ms() -> None:
    env = RollQualityEnv(
        generate_plant_library(19, {"train_core": 1}),
        horizon_steps=3,
        pilot_signal="step",
        actuator_time_constant_s=0.08,
    )
    env.reset(seed=19)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))

    command = 4.0 * 0.001
    expected_applied = (1.0 - np.exp(-0.001 / 0.08)) * command
    assert info["commanded_action"] == pytest.approx(command)
    assert info["applied_action"] == pytest.approx(expected_applied)


def test_environment_runs_a_named_doublet_on_the_plant_grid() -> None:
    profile = CommandProfile(
        "doublet-1.00",
        "doublet",
        amplitude=1.0,
        onset_s=0.0,
        segment_duration_s=0.001,
    )
    env = RollQualityEnv(
        generate_plant_library(23, {"train_core": 1}),
        horizon_steps=4,
        command_profiles=(profile,),
    )

    observation, reset_info = env.reset(seed=23)
    next_observation, _, _, _, step_info = env.step(np.array([0.0], dtype=np.float32))

    assert reset_info["command_id"] == "doublet-1.00"
    assert observation[0] == pytest.approx(1.0)
    assert step_info["command_id"] == "doublet-1.00"
    assert next_observation[0] == pytest.approx(-1.0)


def test_stage_one_rejects_slower_policy_periods() -> None:
    with pytest.raises(ValueError, match="one SAC decision per plant sample"):
        RollQualityEnv(
            generate_plant_library(29, {"train_core": 1}),
            horizon_steps=3,
            pilot_signal="step",
            plant_dt_s=0.001,
            policy_dt_s=0.020,
        )
