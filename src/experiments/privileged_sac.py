"""Bounded fixed-plant training loop for the two-stream privileged SAC Teacher."""

import json
from pathlib import Path

import numpy as np
import torch

from src.experiments.exploratory_sac import build_fixed_env
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.sac.teacher import PrivilegedSAC


def evaluate_fixed_privileged_sac(learner: PrivilegedSAC, env, seed: int = 0) -> dict[str, float]:
    actor_obs, info = env.reset(seed=seed)
    critic_obs = info["critic_state"]
    rewards, actions, errors = [], [], []
    while True:
        action = learner.act(torch.as_tensor(actor_obs, dtype=torch.float32).unsqueeze(0), deterministic=True).numpy()[0]
        actor_obs, reward, terminated, truncated, info = env.step(action)
        critic_obs = info["critic_state"]
        scale = max(abs(env._record.parameters.l_fa), 1.0)
        errors.append((env._p - env._p_ref) / scale)
        rewards.append(float(reward)); actions.append(float(action[0]))
        if terminated or truncated:
            break
    action_values = np.asarray(actions)
    return {
        "episode_reward": float(np.sum(rewards)),
        "action_rms": float(np.sqrt(np.mean(np.square(action_values)))),
        "action_total_variation": float(np.sum(np.abs(np.diff(action_values)))),
        "tracking_nrmse": float(np.sqrt(np.mean(np.square(errors)))),
        "critic_observation_dim": float(np.asarray(critic_obs).size),
    }


def train_fixed_privileged_sac(library_path: str | Path, plant_id: str, output_dir: str | Path, timesteps: int = 20_000, warmup_steps: int = 1_000, batch_size: int = 128, seed: int = 20260827, device: str = "cpu", correction_ratio: float = 0.5) -> dict[str, float | int | str]:
    """Train only on one persisted train plant; this is not a formal global Teacher run."""
    if timesteps <= warmup_steps or batch_size <= 0:
        raise ValueError("timesteps must exceed warmup_steps and batch_size must be positive")
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = build_fixed_env(library_path, plant_id, correction_ratio=correction_ratio, pilot_signal="step")
    actor_obs, info = env.reset(seed=seed)
    learner = PrivilegedSAC(actor_obs.size, info["critic_state"].size, env.action_space.shape[0], device=device)
    replay = TwoStreamReplayBuffer(50_000, actor_obs.size, info["critic_state"].size, env.action_space.shape[0], seed=seed)
    before = evaluate_fixed_privileged_sac(learner, env, seed=seed)
    critic_obs = info["critic_state"]
    losses: dict[str, float] = {}
    updates = 0
    for step in range(timesteps):
        if step < warmup_steps:
            action = np_rng.uniform(-1.0, 1.0, size=env.action_space.shape).astype(np.float32)
        else:
            action = learner.act(torch.as_tensor(actor_obs, dtype=torch.float32).unsqueeze(0)).numpy()[0].astype(np.float32)
        next_actor_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        next_critic_obs = next_info["critic_state"]
        replay.add(actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done)
        actor_obs, critic_obs = next_actor_obs, next_critic_obs
        if done:
            actor_obs, reset_info = env.reset()
            critic_obs = reset_info["critic_state"]
        if step >= warmup_steps and len(replay) >= batch_size:
            losses = learner.update(replay.sample(batch_size, learner.device))
            updates += 1
    after = evaluate_fixed_privileged_sac(learner, env, seed=seed)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "actor": learner.actor.state_dict(),
        "critic": learner.critic.state_dict(),
        "actor_observation_dim": actor_obs.size,
        "critic_observation_dim": critic_obs.size,
        "plant_id": plant_id,
        "correction_ratio": correction_ratio,
    }
    torch.save(checkpoint, destination / "privileged_sac.pt")
    report: dict[str, float | int | str] = {
        "plant_id": plant_id,
        "timesteps": timesteps,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "seed": seed,
        "correction_ratio": correction_ratio,
        "actor_observation_dim": int(actor_obs.size),
        "critic_observation_dim": int(critic_obs.size),
        "updates": updates,
        **{f"before_{key}": value for key, value in before.items()},
        **{f"after_{key}": value for key, value in after.items()},
        **losses,
    }
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
