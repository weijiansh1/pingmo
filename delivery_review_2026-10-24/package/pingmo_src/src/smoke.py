"""CPU-only integration smoke run; deliberately not a convergence experiment."""

from pathlib import Path
import math

import numpy as np
import torch

from src.aircraft.sampler import generate_plant_library
from src.envs.p_channel_env import RollQualityEnv
from src.teacher.moe.teacher import MoETeacher
from src.teacher.sac.teacher import PrivilegedSAC


def _batch(seed: int, size: int = 32) -> dict[str, torch.Tensor]:
    env = RollQualityEnv(
        generate_plant_library(seed, {"train_core": 2, "train_boundary": 2}),
        horizon_steps=20,
        pilot_signal="step",
    )
    observation, info = env.reset(seed=seed)
    critic_state = info["critic_state"]
    rows: list[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]] = []
    rng = np.random.default_rng(seed)
    for _ in range(size):
        action = rng.uniform(-0.5, 0.5, size=1).astype(np.float32)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        next_critic_state = next_info["critic_state"]
        rows.append((observation, critic_state, action, reward, next_observation, next_critic_state, float(terminated or truncated)))
        if terminated or truncated:
            observation, reset_info = env.reset()
            critic_state = reset_info["critic_state"]
        else:
            observation, critic_state = next_observation, next_critic_state
    actor_obs = torch.tensor(np.stack([row[0] for row in rows]))
    next_actor_obs = torch.tensor(np.stack([row[4] for row in rows]))
    return {
        "actor_obs": actor_obs, "critic_obs": torch.tensor(np.stack([row[1] for row in rows])),
        "action": torch.tensor(np.stack([row[2] for row in rows])), "reward": torch.tensor([[row[3]] for row in rows]),
        "next_actor_obs": next_actor_obs, "next_critic_obs": torch.tensor(np.stack([row[5] for row in rows])),
        "done": torch.tensor([[row[6]] for row in rows]), "obs": actor_obs, "next_obs": next_actor_obs,
    }


def run_local_smoke(output_dir: str | Path, updates: int = 4, seed: int = 0) -> dict[str, dict[str, float | bool]]:
    """Run fixed-budget CPU updates and persist only local smoke checkpoints."""
    torch.manual_seed(seed)
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    batch = _batch(seed)
    actor_dim = batch["actor_obs"].shape[1]
    critic_dim = batch["critic_obs"].shape[1]
    sac = PrivilegedSAC(
        actor_dim,
        critic_dim,
        1,
        actor_width=32,
        actor_residual_blocks=2,
        critic_width=32,
        critic_residual_blocks=2,
    )
    moe = MoETeacher(
        actor_dim,
        critic_dim,
        1,
        experts=4,
        actor_width=32,
        shared_residual_blocks=2,
        expert_residual_blocks=1,
        expert_bottleneck_width=16,
        critic_width=32,
        critic_residual_blocks=2,
    )
    sac_losses, moe_losses = {}, {}
    for _ in range(updates):
        sac_losses, moe_losses = sac.update(batch), moe.update(batch)
    torch.save(sac.actor.state_dict(), destination / "sac_smoke.pt")
    torch.save(moe.actor.state_dict(), destination / "moe_smoke.pt")
    finite_sac = all(math.isfinite(value) for value in sac_losses.values())
    finite_moe = all(math.isfinite(value) for value in moe_losses.values())
    return {"sac": {**sac_losses, "finite": finite_sac}, "moe": {**moe_losses, "finite": finite_moe}}
