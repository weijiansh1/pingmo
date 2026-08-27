"""Short, explicitly non-formal GPU SAC experiment on one persisted plant."""

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import PlantRecord
from src.envs.p_channel_env import RollQualityEnv

DEFAULT_TRAIN_PLANT_ID = "train_core-0000"


def build_fixed_env(library_path: str | Path, plant_id: str, horizon_steps: int = 250, correction_ratio: float = 0.5, pilot_signal: str = "step") -> RollQualityEnv:
    for line in Path(library_path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["plant_id"] == plant_id:
            values = {name: row["parameters"][name] for name in ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")}
            record = PlantRecord(plant_id=row["plant_id"], split=row["split"], quality_region=row["quality_region"], aircraft_class=row["aircraft_class"], flight_phase=row["flight_phase"], parameters=PChannelParameters(**values))
            return RollQualityEnv([record], horizon_steps=horizon_steps, correction_ratio=correction_ratio, pilot_signal=pilot_signal)
    raise ValueError(f"plant_id {plant_id!r} not found in {library_path}")


def evaluate(model: SAC, env: RollQualityEnv, seed: int = 0) -> dict[str, float]:
    observation, _ = env.reset(seed=seed)
    rewards, actions = [], []
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward)); actions.append(float(action[0]))
        if terminated or truncated:
            break
    return {"episode_reward": float(np.sum(rewards)), "action_rms": float(np.sqrt(np.mean(np.square(actions)))), "action_total_variation": float(np.sum(np.abs(np.diff(actions))))}


def train_short_experiment(library_path: str | Path, plant_id: str, output_dir: str | Path, timesteps: int = 20_000, seed: int = 20260827, device: str = "auto", correction_ratio: float = 0.5) -> dict[str, float]:
    env = build_fixed_env(library_path, plant_id, correction_ratio=correction_ratio, pilot_signal="step")
    model = SAC("MlpPolicy", env, learning_starts=1_000, buffer_size=50_000, batch_size=128, train_freq=1, gradient_steps=1, learning_rate=3e-4, policy_kwargs={"net_arch": [128, 128]}, seed=seed, device=device, verbose=1)
    before = evaluate(model, env, seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    after = evaluate(model, env, seed)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / "fixed_plant_sac")
    report = {"plant_id": plant_id, "timesteps": timesteps, "seed": seed, "correction_ratio": correction_ratio, "before_reward": before["episode_reward"], "after_reward": after["episode_reward"], "before_action_rms": before["action_rms"], "after_action_rms": after["action_rms"], "after_action_total_variation": after["action_total_variation"]}
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
