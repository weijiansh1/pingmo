"""PPO Teacher training on the batched P-channel environment.

    python scripts/50_train_ppo_teacher.py --plant-id train_core-0312 ...

One Teacher per aircraft. The default task is a finite 8 s episode, with zero
terminal bootstrap. The explicit legacy_reset mode reproduces the archived
reset-state continuation objective; it is not time-limit truncation bootstrap.
Training-command monitoring is not an independent generalization test.
"""
from __future__ import annotations

import json
import hashlib
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
from src.envs.roll_rate_commands import RollRateCommand
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
    reference_mode: str = "fixed"
    episode_end_mode: str = "finite_horizon"
    evaluation_alignment: str = "same_time"
    sample_alignment: str = "same_time"
    episode_steps: int = EPISODE_STEPS
    target_kl: float | None = None
    initial_output_std_deg_s: float = 0.0
    disturbance_std_n: float = 0.0
    disturbance_duration_s: float = 0.2

    def __post_init__(self):
        if min(self.total_rollouts, self.n_envs, self.epochs, self.minibatch,
               self.log_every, self.checkpoint_every) <= 0:
            raise ValueError("training budgets and logging intervals must be positive")
        if self.reference_mode not in ("fixed", "legacy_plant"):
            raise ValueError("invalid reference_mode")
        if self.episode_end_mode not in ("finite_horizon", "legacy_reset"):
            raise ValueError("episode_end_mode must be finite_horizon or legacy_reset")
        if self.evaluation_alignment not in ("same_time", "legacy_interval"):
            raise ValueError("invalid evaluation_alignment")
        if self.sample_alignment not in ("same_time", "legacy_interval"):
            raise ValueError("invalid sample_alignment")
        if self.episode_steps < 2 or int(self.episode_steps) != self.episode_steps:
            raise ValueError("episode_steps must be an integer >= 2")
        if self.target_kl is not None and self.target_kl <= 0:
            raise ValueError("target_kl must be positive or None")
        if min(self.initial_output_std_deg_s, self.disturbance_std_n) < 0:
            raise ValueError("randomization scales must be nonnegative")
        if self.disturbance_duration_s <= 0 or self.eval_envs_per_command <= 0:
            raise ValueError("disturbance duration and evaluation replication must be positive")


def _gae(rewards: torch.Tensor, values: torch.Tensor, bootstrap: torch.Tensor,
         gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """rewards/values: (T, N).  bootstrap: (N,) value of the state after the last step."""
    steps, envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    for t in reversed(range(steps)):
        next_value = bootstrap if t == steps - 1 else values[t + 1]
        next_adv = 0.0 if t == steps - 1 else advantages[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = delta + gamma * lam * next_adv
    returns = advantages + values
    return advantages, returns


def _rollout_bootstrap(critic, env, mode: str) -> torch.Tensor:
    if mode == "finite_horizon":
        return torch.zeros(env.n_envs, device=env.device)
    if mode == "legacy_reset":
        return critic(env.observation(), env.privileged()).squeeze(-1)
    raise ValueError("Unsupported episode end; truncation requires pre-reset final state")


def evaluate(actor: PPOTeacherActor, env: BatchedRollRateEnv, config: PPOTrainingConfig,
             scope: str = "deterministic_training_command_bank_monitor"):
    """Deterministic training-bank monitor; not an independent held-out test.

    Same-time mode compares p(t_{k+1}) with reference(t_{k+1}), over 2.00–7.98 s.
    The final auto-reset transition is excluded. legacy_interval preserves the
    archived comparison against reference(t_k), over outputs at 2.02–7.98 s.
    Legacy eval_commands/eval_envs_per_command fields do not select this bank.
    """
    was_training = actor.training
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
        for step in range(getattr(env, "episode_steps", EPISODE_STEPS) - 1):
            reference = env.reference.clone()
            observation = env.observation()
            with _autocast(env.device, config.amp):
                action, _, _ = actor.act(observation, env.actor_privileged(),
                                         deterministic=True)
            observation, reward, output = env.step(action)
            if config.evaluation_alignment == "same_time":
                reference = env.reference.clone()
            error = reference - output
            active = (step + 1 >= first_active if config.evaluation_alignment == "same_time"
                      else step >= first_active)
            if active:
                sums += error.pow(2)
                peaks = torch.maximum(peaks, error.abs())
                tv += (env.requested - requested_prev).abs()
                count += 1
            requested_prev = env.requested.clone()
        rmse = (sums / max(count, 1)).sqrt() / COMMAND_SCALE_RAD_S
    actor.train(was_training)
    per_command = []
    for index in torch.unique(command_index).tolist():
        mask = command_index == index
        per_command.append({"bank_index": index, "n_envs": int(mask.sum()),
            "rmse_deg_s": float(np.degrees((sums[mask]/max(count,1)).sqrt().mean().cpu())),
            "peak_deg_s": float(np.degrees(peaks[mask].max().cpu())),
            "tv_rate_n_s": float((tv[mask]/max(count,1)).max()/DT_POLICY)})
    return {
        "evaluation_scope": scope,
        "evaluation_alignment": config.evaluation_alignment,
        "per_command": per_command,
        "worst_command_rmse_deg_s": max(r["rmse_deg_s"] for r in per_command),
        "rmse_deg_s": float(np.degrees((sums / max(count, 1)).sqrt().mean().cpu())),
        "rmse_normalized": float(rmse.mean()),
        "peak_deg_s": float(np.degrees(peaks.max().cpu())),
        "tv_rate_n_s": float((tv / max(count, 1)).max() / DT_POLICY),
        "per_env_rmse": (sums / max(count, 1)).sqrt().cpu().tolist(),
    }


def train(parameters: PChannelParameters, config: PPOTrainingConfig, output: Path,
          verbose: bool = True, *,
          training_commands: list[RollRateCommand] | None = None,
          validation_commands: list[RollRateCommand] | None = None) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    train_commands = training_commands or build_command_set()
    if validation_commands is not None:
        if not validation_commands:
            raise ValueError("validation_commands must not be empty")
        # Command IDs alone are insufficient: reject duplicate waveforms even
        # if their names differ. This is command holdout, not plant holdout.
        train_waves = {hashlib.sha256(c.samples(DT_POLICY).tobytes()).hexdigest()
                       for c in train_commands}
        validation_waves = {hashlib.sha256(c.samples(DT_POLICY).tobytes()).hexdigest()
                            for c in validation_commands}
        if train_waves & validation_waves:
            raise ValueError("training and validation command waveforms overlap")
    env = BatchedRollRateEnv(parameters, config.n_envs, commands=train_commands,
        device=config.device, seed=config.seed, reference_mode=config.reference_mode,
        episode_steps=config.episode_steps, sample_alignment=config.sample_alignment)
    monitor = env
    monitor_scope = "deterministic_training_command_bank_monitor"
    if validation_commands is not None:
        monitor = BatchedRollRateEnv(parameters,
            len(validation_commands)*config.eval_envs_per_command,
            commands=validation_commands, device=config.device, seed=config.seed+10000,
            reference_mode=config.reference_mode, episode_steps=config.episode_steps,
            sample_alignment=config.sample_alignment)
        monitor_scope = "independent_validation_commands_on_training_plant"
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
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")
    if any((output/name).exists() for name in ["train_log.jsonl", "checkpoint.pt", "protocol.json"]):
        raise FileExistsError("Use a fresh output directory to preserve the previous run")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "config": asdict(config), "parameters": asdict(parameters),
        "action_probability_contract": "Gaussian sample retained; environment clips physical action",
        "episode_end_mode": config.episode_end_mode,
        "monitor_scope": monitor_scope,
        "legacy_unused_config_fields": (["eval_commands"] if validation_commands is not None
                                         else ["eval_commands", "eval_envs_per_command"]),
        "commands": [asdict(c) for c in train_commands],
        "validation_commands": [asdict(c) for c in validation_commands or []],
        "source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in {"trainer": __file__,
                "networks": __import__(PPOTeacherActor.__module__, fromlist=['__file__']).__file__,
                "environment": __import__(BatchedRollRateEnv.__module__, fromlist=['__file__']).__file__}.items()},
    }
    (output / "protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    log_path = output / "train_log.jsonl"
    history = []
    best_monitor_rmse = float("inf")
    started = time.time()
    randomization_rng = torch.Generator(device="cpu").manual_seed(config.seed + 20000)

    for rollout in range(1, config.total_rollouts + 1):
        observations, actor_privileges, privileges = [], [], []
        actions, log_probs = [], []
        rewards, values = [], []
        env.reset()
        if config.initial_output_std_deg_s > 0:
            initial = torch.randn(config.n_envs, generator=randomization_rng)
            env.set_initial_output(initial.to(device) * np.radians(config.initial_output_std_deg_s))
        disturbance_amplitude = None
        if config.disturbance_std_n > 0:
            disturbance_amplitude = (torch.randn(config.n_envs, generator=randomization_rng)
                * config.disturbance_std_n).to(device)
            pulse_start = torch.randint(int(.25*config.episode_steps),
                max(int(.25*config.episode_steps)+1, int(.6*config.episode_steps)),
                (config.n_envs,), generator=randomization_rng).to(device)
            pulse_length = max(1, int(round(config.disturbance_duration_s/DT_POLICY)))
        for step_index in range(config.episode_steps):
            observation = env.observation()
            actor_privileged = env.actor_privileged()
            privileged = env.privileged()
            with torch.no_grad(), _autocast(device, config.amp):
                action, log_prob, _ = actor.act(observation, actor_privileged)
                value = critic(observation, privileged)
            if disturbance_amplitude is None:
                obs_next, reward, _ = env.step(action)
            else:
                disturbance = disturbance_amplitude * ((step_index >= pulse_start)
                    & (step_index < pulse_start+pulse_length))
                obs_next, reward, _ = env.step(action, external_force_n=disturbance)
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
            bootstrap = _rollout_bootstrap(critic, env, config.episode_end_mode)
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
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0, "count": 0}
        early_stop = False
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
                approx_kl = ((ratio-1)-(new_logp.squeeze(-1)-flat_logp[index])).mean()
                if config.target_kl is not None and float(approx_kl.detach()) > config.target_kl:
                    early_stop = True
                    break
                loss = (policy_loss + config.value_coef * value_loss
                        - config.entropy_coef * entropy_mean)
                optimiser.zero_grad(set_to_none=True)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite PPO loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), config.max_grad_norm,
                    error_if_nonfinite=True)
                scaler.step(optimiser)
                scaler.update()
                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(value_loss.detach())
                stats["entropy"] += float(entropy_mean.detach())
                log_ratio = new_logp.squeeze(-1) - flat_logp[index]
                stats["approx_kl"] += float(((ratio-1)-log_ratio).mean().detach())
                stats["clip_fraction"] += float(((ratio-1).abs()>config.clip_ratio).float().mean())
                stats["grad_norm"] += float(grad_norm)
                stats["count"] += 1
            if early_stop:
                break
        metrics = evaluate(actor, monitor, config, scope=monitor_scope)
        record = {
            "rollout": rollout,
            "steps": rollout * config.episode_steps * config.n_envs,
            "optimizer_updates": stats["count"],
            "kl_early_stop": early_stop,
            "policy_loss": stats["policy_loss"] / max(stats["count"], 1),
            "value_loss": stats["value_loss"] / max(stats["count"], 1),
            "entropy": stats["entropy"] / max(stats["count"], 1),
            "approx_kl": stats["approx_kl"] / max(stats["count"], 1),
            "clip_fraction": stats["clip_fraction"] / max(stats["count"], 1),
            "grad_norm": stats["grad_norm"] / max(stats["count"], 1),
            "action_clip_fraction": float((act_t.abs()>1).float().mean()),
            "terminal_bootstrap_mean": float(bootstrap.mean()),
            "mean_reward": float(rew_t.mean()),
            "elapsed_s": time.time() - started,
            **{k: v for k, v in metrics.items() if k != "per_env_rmse"},
        }
        history.append(record)
        if verbose and (rollout % config.log_every == 0 or rollout == 1):
            print(json.dumps(record, ensure_ascii=False))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        checkpoint = {
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "config": asdict(config),
                "parameters": asdict(parameters),
                "privileged_dim": privileged_dim,
                "actor_privileged_dim": actor_privileged_dim,
                "rollout": rollout,
                "monitor_metrics": metrics,
                "action_probability_contract": protocol["action_probability_contract"],
            }
        if metrics["rmse_deg_s"] < best_monitor_rmse:
            best_monitor_rmse = metrics["rmse_deg_s"]
            torch.save(checkpoint, output / "best_monitor_checkpoint.pt")
            if validation_commands is not None:
                torch.save(checkpoint, output / "best_validation_checkpoint.pt")
        if rollout % config.checkpoint_every == 0 or rollout == config.total_rollouts:
            torch.save(checkpoint, output / f"checkpoint_rollout_{rollout:04d}.pt")
            torch.save(checkpoint, output / "checkpoint.pt")
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

