"""PPO Teacher training on the batched P-channel environment.

    python scripts/50_train_ppo_teacher.py --plant-id train_core-0312 ...

One Teacher per aircraft.  The rollout is one full 8 s episode per environment;
advantages use GAE(lambda) with bootstrapping across episode truncation (the
episode resets to a zero state, so the bootstrap value is evaluated on the reset
observation rather than assumed to be zero).
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from src.aircraft.parameters import PChannelParameters
from src.envs.batched_p_channel import (
    COMMAND_SCALE_RAD_S,
    DT_POLICY,
    EPISODE_STEPS,
    OBS_DIM,
    BatchedRollRateEnv,
    build_command_set,
)
from src.teacher.ppo.networks import (
    PPOTeacherActor,
    PPOTeacherCritic,
    parameter_count,
)


def _autocast(device: torch.device, enabled: bool):
    """fp16 tensor cores on Turing; no-op on CPU."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@dataclass
class PPOTrainingConfig:
    total_rollouts: int = 40
    n_envs: int = 512
    epochs: int = 5
    minibatch: int = 16384
    amp: bool = True
    learning_rate: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    max_grad_norm: float = 0.5
    width: int = 512
    blocks: int = 8
    critic_width: int = 512
    critic_blocks: int = 4
    conv_width: int = 24
    initial_log_std: float = -2.0
    device: str = "cuda"
    seed: int = 20261022
    eval_commands: int = 6
    eval_envs_per_command: int = 8
    log_every: int = 5
    checkpoint_every: int = 10


def _gae(rewards: torch.Tensor, values: torch.Tensor, bootstrap: torch.Tensor,
         gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """rewards/values: (T, N).  bootstrap: (N,) value of the state after the last step."""
    steps, envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last = torch.zeros(envs, device=rewards.device)
    for t in reversed(range(steps)):
        next_value = bootstrap if t == steps - 1 else values[t + 1]
        next_adv = 0.0 if t == steps - 1 else advantages[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * next_adv
    returns = advantages + values
    return advantages, returns


def evaluate(actor: PPOTeacherActor, env: BatchedRollRateEnv, config: PPOTrainingConfig):
    """Deterministic closed-loop metrics per evaluation command.

    Two subtleties that used to corrupt the numbers:

    * the episode truncates and resets on the final step, so ``step`` must stop
      one short of ``EPISODE_STEPS``; otherwise the last error mixes the old
      output with the reset reference and shows up as a peak equal to the
      command amplitude;
    * the reference has to be read before ``env.step`` advances the clock,
      otherwise every error is paired with the reference one step ahead.
    """
    actor.eval()
    n_envs = env.n_envs
    with torch.no_grad():
        env.reset()
        command_index = torch.arange(n_envs, device=env.device) % env.n_cmd
        env.command_index = command_index
        env.t = torch.zeros(n_envs, dtype=torch.long, device=env.device)
        sums = torch.zeros(n_envs, device=env.device)
        peaks = torch.zeros(n_envs, device=env.device)
        requested_prev = torch.zeros(n_envs, device=env.device)
        tv = torch.zeros(n_envs, device=env.device)
        count = 0
        first_active = int(round(2.0 / DT_POLICY))
        for step in range(EPISODE_STEPS - 1):
            reference = env.reference.clone()
            observation = env.observation()
            with _autocast(env.device, config.amp):
                action, _, _ = actor.act(observation, env.actor_privileged(),
                                         deterministic=True)
            observation, reward, output = env.step(action)
            error = reference - output
            if step >= first_active:
                sums += error.pow(2)
                peaks = torch.maximum(peaks, error.abs())
                tv += (env.requested - requested_prev).abs()
                count += 1
            requested_prev = env.requested.clone()
        rmse = (sums / max(count, 1)).sqrt() / COMMAND_SCALE_RAD_S
    actor.train()
    return {
        "rmse_deg_s": float(np.degrees((sums / max(count, 1)).sqrt().mean().cpu())),
        "rmse_normalized": float(rmse.mean()),
        "peak_deg_s": float(np.degrees(peaks.max().cpu())),
        "tv_rate_n_s": float((tv / max(count, 1)).max() / DT_POLICY),
        "per_env_rmse": (sums / max(count, 1)).sqrt().cpu().tolist(),
    }


def train(parameters: PChannelParameters, config: PPOTrainingConfig, output: Path,
          verbose: bool = True) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    env = BatchedRollRateEnv(parameters, config.n_envs, device=config.device,
                             seed=config.seed)
    actor_privileged_dim = env.actor_privileged().shape[-1]
    actor = PPOTeacherActor(width=config.width, blocks=config.blocks,
                            conv_width=config.conv_width,
                            privileged_dim=actor_privileged_dim,
                            initial_log_std=config.initial_log_std).to(device)
    privileged_dim = env.privileged().shape[-1]
    critic = PPOTeacherCritic(privileged_dim, width=config.critic_width,
                              blocks=config.critic_blocks).to(device)
    optimiser = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=config.learning_rate)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train_log.jsonl"
    history = []
    started = time.time()

    for rollout in range(1, config.total_rollouts + 1):
        observations, actor_privileges, privileges = [], [], []
        actions, log_probs = [], []
        rewards, values = [], []
        env.reset()
        for _ in range(EPISODE_STEPS):
            observation = env.observation()
            actor_privileged = env.actor_privileged()
            privileged = env.privileged()
            with torch.no_grad(), _autocast(device, config.amp):
                action, log_prob, _ = actor.act(observation, actor_privileged)
                value = critic(observation, privileged)
            obs_next, reward, _ = env.step(action)
            observations.append(observation)
            actor_privileges.append(actor_privileged)
            privileges.append(privileged)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze(-1))
        obs_t = torch.stack(observations)          # (T, N, obs)
        actor_priv_t = torch.stack(actor_privileges)
        priv_t = torch.stack(privileges)
        act_t = torch.stack(actions)
        logp_t = torch.stack(log_probs).squeeze(-1)
        rew_t = torch.stack(rewards)
        val_t = torch.stack(values)
        with torch.no_grad(), _autocast(device, config.amp):
            bootstrap = critic(env.observation(), env.privileged()).squeeze(-1)
        advantages, returns = _gae(rew_t, val_t, bootstrap, config.gamma, config.gae_lambda)
        adv = advantages.reshape(-1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat_obs = obs_t.reshape(-1, obs_t.shape[-1])
        flat_actor_priv = actor_priv_t.reshape(-1, actor_priv_t.shape[-1])
        flat_priv = priv_t.reshape(-1, priv_t.shape[-1])
        flat_act = act_t.reshape(-1, act_t.shape[-1])
        flat_logp = logp_t.reshape(-1)
        flat_ret = returns.reshape(-1)
        total = flat_obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "count": 0}
        for _ in range(config.epochs):
            permutation = torch.randperm(total, device=device)
            for start in range(0, total, config.minibatch):
                index = permutation[start:start + config.minibatch]
                with _autocast(device, config.amp):
                    new_logp, entropy, _ = actor.evaluate(
                        flat_obs[index], flat_actor_priv[index], flat_act[index])
                    value = critic(flat_obs[index], flat_priv[index]).squeeze(-1)
                ratio = (new_logp.squeeze(-1) - flat_logp[index]).exp()
                unclipped = ratio * adv[index]
                clipped = ratio.clamp(1 - config.clip_ratio, 1 + config.clip_ratio) * adv[index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - flat_ret[index]).pow(2).mean()
                entropy_mean = entropy.mean()
                loss = (policy_loss + config.value_coef * value_loss
                        - config.entropy_coef * entropy_mean)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm)
                optimiser.step()
                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(value_loss.detach())
                stats["entropy"] += float(entropy_mean.detach())
                stats["count"] += 1
        metrics = evaluate(actor, env, config)
        record = {
            "rollout": rollout,
            "steps": rollout * EPISODE_STEPS * config.n_envs,
            "policy_loss": stats["policy_loss"] / max(stats["count"], 1),
            "value_loss": stats["value_loss"] / max(stats["count"], 1),
            "entropy": stats["entropy"] / max(stats["count"], 1),
            "mean_reward": float(rew_t.mean()),
            "elapsed_s": time.time() - started,
            **{k: v for k, v in metrics.items() if k != "per_env_rmse"},
        }
        history.append(record)
        if verbose and (rollout % config.log_every == 0 or rollout == 1):
            print(json.dumps(record, ensure_ascii=False))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if rollout % config.checkpoint_every == 0 or rollout == config.total_rollouts:
            torch.save({
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": asdict(config),
                "parameters": asdict(parameters),
                "privileged_dim": privileged_dim,
                "rollout": rollout,
            }, output / "checkpoint.pt")
    final = history[-1]
    (output / "summary.json").write_text(json.dumps({
        "plant_id": str(output.name),
        "actor_parameters": parameter_count(actor),
        "critic_parameters": parameter_count(critic),
        "obs_dim": OBS_DIM,
        "privileged_dim": privileged_dim,
        "final": final,
        "best_rmse_deg_s": min(r["rmse_deg_s"] for r in history),
        "rollouts": len(history),
        "wall_clock_s": time.time() - started,
        "config": asdict(config),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return final

