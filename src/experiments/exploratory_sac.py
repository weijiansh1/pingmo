"""Short, explicitly non-formal GPU SAC experiment on one persisted plant."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import PlantRecord
from src.envs.p_channel_env import RollQualityEnv
from src.envs.reward import RewardWeights

DEFAULT_TRAIN_PLANT_ID = "train_core-0000"


def load_persisted_records(library_path: str | Path, plant_ids: list[str]) -> list[PlantRecord]:
    """Load requested P-channel records in the caller-specified order."""
    rows = {row["plant_id"]: row for row in (json.loads(line) for line in Path(library_path).read_text(encoding="utf-8").splitlines())}
    missing = [plant_id for plant_id in plant_ids if plant_id not in rows]
    if missing:
        raise ValueError(f"plant IDs not found in {library_path}: {missing}")
    values = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")
    return [
        PlantRecord(
            plant_id=rows[plant_id]["plant_id"],
            split=rows[plant_id]["split"],
            quality_region=rows[plant_id]["quality_region"],
            aircraft_class=rows[plant_id]["aircraft_class"],
            flight_phase=rows[plant_id]["flight_phase"],
            parameters=PChannelParameters(**{name: rows[plant_id]["parameters"][name] for name in values}),
        )
        for plant_id in plant_ids
    ]


def build_multi_env(library_path: str | Path, plant_ids: list[str], horizon_steps: int = 250, correction_ratio: float = 0.5, pilot_signal: str = "step", reward_weights: RewardWeights = RewardWeights()) -> RollQualityEnv:
    return RollQualityEnv(
        load_persisted_records(library_path, plant_ids),
        horizon_steps=horizon_steps,
        correction_ratio=correction_ratio,
        pilot_signal=pilot_signal,
        reward_weights=reward_weights,
    )


def build_fixed_env(library_path: str | Path, plant_id: str, horizon_steps: int = 250, correction_ratio: float = 0.5, pilot_signal: str = "step", reward_weights: RewardWeights = RewardWeights()) -> RollQualityEnv:
    return build_multi_env(library_path, [plant_id], horizon_steps, correction_ratio, pilot_signal, reward_weights)


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


def collect_response_trace(model, env: RollQualityEnv, seed: int = 0) -> dict[str, np.ndarray]:
    """Roll out one deterministic episode and retain response-level diagnostics."""
    observation, _ = env.reset(seed=seed)
    traces: dict[str, list[float]] = {name: [] for name in ("time_s", "p", "p_ref", "delta_f", "commanded_delta_f", "f_eq", "reward")}
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        traces["time_s"].append(env._episode_step * env._action_dt)
        traces["p"].append(env._p)
        traces["p_ref"].append(float(info["p_ref"]))
        traces["delta_f"].append(float(info["delta_f"]))
        traces["commanded_delta_f"].append(float(info["commanded_action"]) * env.correction_ratio * env.pilot_force_scale_n)
        traces["f_eq"].append(float(info["f_eq"]))
        traces["reward"].append(float(reward))
        if terminated or truncated:
            break
    return {name: np.asarray(values, dtype=float) for name, values in traces.items()}


def response_metrics(trace: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute tracking and effort measures from one matched response trace."""
    return {
        "tracking_rmse": float(np.sqrt(np.mean(np.square(trace["p"] - trace["p_ref"]))),),
        "applied_delta_f_rms_n": float(np.sqrt(np.mean(np.square(trace["delta_f"]))),),
        "commanded_delta_f_total_variation_n": float(np.sum(np.abs(np.diff(trace["commanded_delta_f"]))),),
    }


def reward_axis_limits(rewards: np.ndarray) -> tuple[float, float]:
    """Choose a compact negative reward axis without clipping the trace."""
    return min(-0.03, 1.1 * float(np.min(rewards))), 0.0


def save_response_comparison(raw: dict[str, np.ndarray], sac: dict[str, np.ndarray], output: str | Path) -> Path:
    """Save the raw/reference/SAC response and SAC effort for one matched rollout."""
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, (response_axis, effort_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    response_axis.plot(raw["time_s"], raw["p"], label="raw：原始飞机", color="#1f77b4", linewidth=2)
    response_axis.plot(raw["time_s"], raw["p_ref"], label="ref：参考响应", color="#2ca02c", linestyle="--", linewidth=2)
    response_axis.plot(sac["time_s"], sac["p"], label="SAC：训练后控制", color="#d62728", linewidth=2)
    response_axis.set_title("GPU 单机 SAC 探索：raw / ref / SAC 阶跃响应")
    response_axis.set_ylabel("滚转角速度 p")
    response_axis.grid(alpha=.25)
    response_axis.legend()
    effort_axis.plot(sac["time_s"], sac["commanded_delta_f"], label="ΔF_命令", color="#7f7f7f", linestyle="--")
    effort_axis.plot(sac["time_s"], sac["delta_f"], label="ΔF_RL（实际）", color="#9467bd")
    effort_axis.set_xlabel("时间（s）")
    effort_axis.set_ylabel("控制增量（N）")
    effort_axis.grid(alpha=.25)
    reward_axis = effort_axis.twinx()
    reward_axis.plot(sac["time_s"], sac["reward"], label="每步奖励", color="#ff7f0e", alpha=.8)
    reward_axis.set_ylabel("每步奖励")
    reward_axis.set_ylim(*reward_axis_limits(sac["reward"]))
    effort_handles, effort_labels = effort_axis.get_legend_handles_labels()
    reward_handles, reward_labels = reward_axis.get_legend_handles_labels()
    effort_axis.legend(effort_handles + reward_handles, effort_labels + reward_labels)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


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
