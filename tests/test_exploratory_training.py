from pathlib import Path

import numpy as np

from src.experiments.exploratory_sac import DEFAULT_TRAIN_PLANT_ID, build_fixed_env, collect_response_trace, response_metrics, reward_axis_limits
from src.experiments.privileged_sac import train_fixed_privileged_sac


def test_fixed_training_env_loads_a_persisted_plant() -> None:
    root = Path(__file__).parents[1]
    env = build_fixed_env(root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl", "id_test-2152", horizon_steps=8)
    observation, info = env.reset(seed=2)
    assert observation.shape == (142,)
    assert info["plant_id"] == "id_test-2152"


def test_default_exploratory_plant_is_from_the_training_split() -> None:
    root = Path(__file__).parents[1]
    env = build_fixed_env(root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl", DEFAULT_TRAIN_PLANT_ID, horizon_steps=8)
    _, info = env.reset(seed=2)
    assert info["plant_id"].startswith("train_core-")


def test_fixed_training_env_accepts_reference_tracking_experiment_settings() -> None:
    root = Path(__file__).parents[1]
    env = build_fixed_env(root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl", "train_core-0000", horizon_steps=8, correction_ratio=0.5, pilot_signal="step")
    env.reset(seed=2)
    _, _, _, _, info = env.step(env.action_space.sample())
    assert info["f_pilot"] == 22.0
    assert "p_ref" in info


def test_collect_response_trace_records_raw_reference_and_control_effort() -> None:
    class ZeroPolicy:
        def predict(self, observation, deterministic: bool):
            return np.zeros(1, dtype=np.float32), None

    root = Path(__file__).parents[1]
    env = build_fixed_env(
        root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl",
        "train_core-0000",
        horizon_steps=8,
        correction_ratio=0.3,
        pilot_signal="step",
    )
    trace = collect_response_trace(ZeroPolicy(), env, seed=3)

    assert trace["time_s"].shape == (8,)
    assert trace["p"].shape == (8,)
    assert trace["p_ref"].shape == (8,)
    assert np.allclose(trace["delta_f"], 0.0)
    assert trace["commanded_delta_f"].shape == (8,)
    assert np.allclose(trace["commanded_delta_f"], 0.0)


def test_response_metrics_include_tracking_and_effort() -> None:
    trace = {
        "p": np.array([0.0, 1.0]),
        "p_ref": np.array([0.0, 0.0]),
        "delta_f": np.array([0.0, 3.0]),
        "commanded_delta_f": np.array([0.0, 4.0]),
    }
    metrics = response_metrics(trace)
    assert metrics == {
        "tracking_rmse": 2 ** -0.5,
        "applied_delta_f_rms_n": 3 / 2 ** 0.5,
        "commanded_delta_f_total_variation_n": 4.0,
    }


def test_reward_axis_limits_make_small_negative_rewards_visible() -> None:
    assert reward_axis_limits(np.array([-0.004, -0.013, -0.008])) == (-0.03, 0.0)


def test_fixed_privileged_sac_cpu_smoke_persists_two_stream_checkpoint(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    report = train_fixed_privileged_sac(
        root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl",
        "train_core-0000",
        tmp_path,
        timesteps=96,
        warmup_steps=16,
        batch_size=16,
        seed=13,
    )

    assert report["actor_observation_dim"] == 142
    assert report["critic_observation_dim"] > 19
    assert report["updates"] > 0
    assert (tmp_path / "privileged_sac.pt").exists()
    assert (tmp_path / "report.json").exists()
