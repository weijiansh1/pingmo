"""Bounded reference-free SAC experiments and response diagnostics."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np

from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import PlantRecord
from src.envs.commands import CommandProfile
from src.envs.p_channel_env import RollQualityEnv
from src.envs.reward import RewardWeights


DEFAULT_TRAIN_PLANT_ID = "train_core-0000"


class PredictivePolicy(Protocol):
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> object: ...


def load_persisted_records(library_path: str | Path, plant_ids: list[str]) -> list[PlantRecord]:
    """Load requested P-channel records in the caller-specified order."""

    rows = {
        row["plant_id"]: row
        for row in (json.loads(line) for line in Path(library_path).read_text(encoding="utf-8").splitlines())
    }
    missing = [plant_id for plant_id in plant_ids if plant_id not in rows]
    if missing:
        raise ValueError(f"plant IDs not found in {library_path}: {missing}")
    names = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")
    return [
        PlantRecord(
            plant_id=rows[plant_id]["plant_id"],
            split=rows[plant_id]["split"],
            quality_region=rows[plant_id]["quality_region"],
            aircraft_class=rows[plant_id]["aircraft_class"],
            flight_phase=rows[plant_id]["flight_phase"],
            parameters=PChannelParameters(**{name: rows[plant_id]["parameters"][name] for name in names}),
        )
        for plant_id in plant_ids
    ]


def build_multi_env(
    library_path: str | Path,
    plant_ids: list[str],
    horizon_steps: int = 10_000,
    correction_ratio: float = 0.3,
    pilot_signal: str | None = "step",
    reward_weights: RewardWeights = RewardWeights(),
    command_profiles: tuple[CommandProfile, ...] | None = None,
) -> RollQualityEnv:
    return RollQualityEnv(
        load_persisted_records(library_path, plant_ids),
        horizon_steps=horizon_steps,
        correction_ratio=correction_ratio,
        pilot_signal=pilot_signal,
        reward_weights=reward_weights,
        command_profiles=command_profiles,
    )


def build_fixed_env(
    library_path: str | Path,
    plant_id: str,
    horizon_steps: int = 10_000,
    correction_ratio: float = 0.3,
    pilot_signal: str | None = "step",
    reward_weights: RewardWeights = RewardWeights(),
    command_profiles: tuple[CommandProfile, ...] | None = None,
) -> RollQualityEnv:
    return build_multi_env(
        library_path,
        [plant_id],
        horizon_steps,
        correction_ratio,
        pilot_signal,
        reward_weights,
        command_profiles,
    )


def _predict(model: PredictivePolicy, observation: np.ndarray) -> np.ndarray:
    prediction = model.predict(observation, deterministic=True)
    return np.asarray(prediction[0] if isinstance(prediction, tuple) else prediction, dtype=np.float32)


def evaluate(model: PredictivePolicy, env: RollQualityEnv, seed: int = 0) -> dict[str, float]:
    observation, _ = env.reset(seed=seed)
    rewards: list[float] = []
    actions: list[float] = []
    while True:
        action = _predict(model, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        actions.append(float(action[0]))
        if terminated or truncated:
            break
    values = np.asarray(actions)
    return {
        "episode_reward": float(np.sum(rewards)),
        "action_rms": float(np.sqrt(np.mean(np.square(values)))),
        "action_total_variation": float(np.sum(np.abs(np.diff(values)))),
    }


def collect_response_trace(model: PredictivePolicy, env: RollQualityEnv, seed: int = 0) -> dict[str, np.ndarray]:
    """Roll out one deterministic episode and return the complete 1 ms trace."""

    observation, _ = env.reset(seed=seed)
    while True:
        observation, _, terminated, truncated, _ = env.step(_predict(model, observation))
        if terminated or truncated:
            break
    trace = env.trajectory()
    return {
        "time_s": trace["time_s"],
        "f_pilot": trace["f_pilot_n"],
        "p": trace["p_rad_s"],
        "raw_p": trace["raw_p_rad_s"],
        "delta_f": trace["delta_f_n"],
        "commanded_delta_f": trace["commanded_delta_f_n"],
        "reward": trace["reward"],
        **{name: values for name, values in trace.items() if name.endswith("_cost")},
    }


def response_metrics(trace: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute response-energy and effort diagnostics without a reference model."""

    return {
        "episode_cost": float(-np.sum(trace["reward"])),
        "roll_rate_rms_rad_s": float(np.sqrt(np.mean(np.square(trace["p"]))),),
        "applied_delta_f_rms_n": float(np.sqrt(np.mean(np.square(trace["delta_f"]))),),
        "commanded_delta_f_total_variation_n": float(np.sum(np.abs(np.diff(trace["commanded_delta_f"]))),),
    }


def summarize_held_out_metrics(records: list[dict[str, dict[str, float]]]) -> dict[str, float]:
    """Summarize increases in the common response/action cost on held-out pairs."""

    changes = np.asarray(
        [record["sac"]["episode_cost"] - record["raw"]["episode_cost"] for record in records],
        dtype=float,
    )
    return {
        "harm_rate": float(np.mean(changes > 0.0)),
        "median_episode_cost_change": float(np.median(changes)),
    }


def load_completed_screening_report(run_dir: str | Path) -> dict[str, object] | None:
    report_path = Path(run_dir) / "screening_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def reward_axis_limits(rewards: np.ndarray) -> tuple[float, float]:
    return min(-0.03, 1.1 * float(np.min(rewards))), 0.0


def save_response_comparison(raw: dict[str, np.ndarray], controlled: dict[str, np.ndarray], output: str | Path) -> Path:
    """Save matched raw/controlled response and augmentation-effort traces."""

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, (response_axis, effort_axis) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, layout="constrained")
    response_axis.plot(raw["time_s"], raw["p"], label="raw：原始飞机", color="#1f77b4", linewidth=2)
    response_axis.plot(controlled["time_s"], controlled["p"], label="SAC：修正后响应", color="#d62728", linewidth=2)
    response_axis.set_title("单通道滚转品质：raw / SAC 响应")
    response_axis.set_ylabel("滚转角速度 p（rad/s）")
    response_axis.grid(alpha=0.25)
    response_axis.legend()
    effort_axis.plot(controlled["time_s"], controlled["commanded_delta_f"], label="ΔF 命令", color="#7f7f7f", linestyle="--")
    effort_axis.plot(controlled["time_s"], controlled["delta_f"], label="ΔF 实际", color="#9467bd")
    effort_axis.set_xlabel("时间（s）")
    effort_axis.set_ylabel("修正力（N）")
    effort_axis.grid(alpha=0.25)
    reward_axis = effort_axis.twinx()
    reward_axis.plot(controlled["time_s"], controlled["reward"], label="每步奖励", color="#ff7f0e", alpha=0.8)
    reward_axis.set_ylabel("每步奖励")
    reward_axis.set_ylim(*reward_axis_limits(controlled["reward"]))
    effort_handles, effort_labels = effort_axis.get_legend_handles_labels()
    reward_handles, reward_labels = reward_axis.get_legend_handles_labels()
    effort_axis.legend(effort_handles + reward_handles, effort_labels + reward_labels)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def train_short_experiment(
    library_path: str | Path,
    plant_id: str,
    output_dir: str | Path,
    timesteps: int = 20_000,
    seed: int = 20260827,
    device: str = "auto",
    correction_ratio: float = 0.3,
    plant_ids: list[str] | None = None,
    reward_weights: RewardWeights = RewardWeights(),
) -> dict[str, object]:
    """Run a bounded SB3 compatibility experiment; this is not a Teacher run."""

    from stable_baselines3 import SAC

    training_plant_ids = plant_ids or [plant_id]
    env = build_multi_env(
        library_path,
        training_plant_ids,
        correction_ratio=correction_ratio,
        pilot_signal="step",
        reward_weights=reward_weights,
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=1_000,
        buffer_size=200_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.9999,
        tau=0.005,
        policy_kwargs={"net_arch": [256, 256]},
        seed=seed,
        device=device,
        verbose=1,
    )
    before = evaluate(model, env, seed)
    model.learn(total_timesteps=timesteps, progress_bar=False)
    after = evaluate(model, env, seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save(destination / "fixed_plant_sac")
    report: dict[str, object] = {
        "plant_id": plant_id,
        "training_plant_ids": training_plant_ids,
        "timesteps": timesteps,
        "seed": seed,
        "correction_ratio": correction_ratio,
        "reward_weights": asdict(reward_weights),
        **{f"before_{name}": value for name, value in before.items()},
        **{f"after_{name}": value for name, value in after.items()},
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
