from pathlib import Path

from src.experiments.exploratory_sac import DEFAULT_TRAIN_PLANT_ID, build_fixed_env
from src.experiments.privileged_sac import train_fixed_privileged_sac


def test_fixed_training_env_loads_a_persisted_plant() -> None:
    root = Path(__file__).parents[1]
    env = build_fixed_env(root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl", "id_test-2152", horizon_steps=8)
    observation, info = env.reset(seed=2)
    assert observation.shape == (141,)
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

    assert report["actor_observation_dim"] == 141
    assert report["critic_observation_dim"] > 19
    assert report["updates"] > 0
    assert (tmp_path / "privileged_sac.pt").exists()
    assert (tmp_path / "report.json").exists()
