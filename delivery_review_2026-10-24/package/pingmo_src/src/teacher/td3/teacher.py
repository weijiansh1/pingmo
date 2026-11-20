"""Privileged-critic TD3 with a deterministic squashed Actor."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from src.teacher.sac.actor import SquashedGaussianActor
from src.teacher.sac.critic import TwinQCritic


class PrivilegedTD3:
    """TD3 learner whose deployable Actor never receives privileged plant state."""

    def __init__(
        self,
        actor_observation_dim: int,
        critic_observation_dim: int,
        action_dim: int,
        *,
        gamma: float = 0.995,
        tau: float = 0.005,
        target_policy_noise: float = 0.10,
        target_noise_clip: float = 0.25,
        policy_delay: int = 2,
        actor_learning_rate: float = 3e-5,
        critic_learning_rate: float = 3e-4,
        gradient_norm_limit: float = 10.0,
        reward_scale: float = 1.0,
        behavior_regularization_weight: float = 100.0,
        q_normalization_scale: float = 2.5,
        maximum_q_coefficient: float = 1.0,
        actor_width: int = 128,
        actor_residual_blocks: int = 2,
        actor_enforce_odd_symmetry: bool = False,
        actor: nn.Module | None = None,
        critic_width: int = 128,
        critic_residual_blocks: int = 2,
        device: str | torch.device = "cpu",
    ) -> None:
        positive = (
            tau,
            target_policy_noise,
            target_noise_clip,
            policy_delay,
            actor_learning_rate,
            critic_learning_rate,
            gradient_norm_limit,
            reward_scale,
            q_normalization_scale,
            maximum_q_coefficient,
        )
        if (
            not 0 < gamma <= 1
            or min(positive) <= 0
            or tau > 1
            or behavior_regularization_weight < 0
        ):
            raise ValueError("invalid TD3 hyperparameters")
        self.device = torch.device(device)
        self.actor = (
            actor
            if actor is not None
            else SquashedGaussianActor(
                actor_observation_dim,
                action_dim,
                width=actor_width,
                residual_blocks=actor_residual_blocks,
                enforce_odd_symmetry=actor_enforce_odd_symmetry,
            )
        ).to(self.device)
        if hasattr(self.actor, "log_std"):
            self.actor.log_std.requires_grad_(False)
        self.critic = TwinQCritic(
            critic_observation_dim,
            action_dim,
            width=critic_width,
            residual_blocks=critic_residual_blocks,
        ).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device).eval()
        self.target_critic = deepcopy(self.critic).to(self.device).eval()
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            (
                parameter
                for parameter in self.actor.parameters()
                if parameter.requires_grad
            ),
            lr=actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.gamma = gamma
        self.tau = tau
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.gradient_norm_limit = gradient_norm_limit
        self.reward_scale = reward_scale
        self.behavior_regularization_weight = behavior_regularization_weight
        self.q_normalization_scale = q_normalization_scale
        self.maximum_q_coefficient = maximum_q_coefficient
        self._update_count = 0
        self._actor_anchor: tuple[torch.Tensor, ...] | None = None
        self._actor_trust_region_radius: float | None = None

    def set_actor_trust_region(self, radius: float) -> None:
        """Anchor subsequent TD3 updates to the verified cloned Actor."""

        if radius <= 0:
            raise ValueError("Actor trust-region radius must be positive")
        self._actor_anchor = tuple(
            parameter.detach().clone() for parameter in self.actor.parameters()
        )
        self._actor_trust_region_radius = float(radius)

    def _project_actor_to_trust_region(self) -> tuple[float, bool]:
        if self._actor_anchor is None or self._actor_trust_region_radius is None:
            return 0.0, False
        with torch.no_grad():
            squared_norm = torch.zeros((), device=self.device)
            for parameter, anchor in zip(
                self.actor.parameters(), self._actor_anchor, strict=True
            ):
                squared_norm += (parameter - anchor).square().sum()
            delta_norm = float(squared_norm.sqrt())
            projected = delta_norm > self._actor_trust_region_radius
            if projected:
                scale = self._actor_trust_region_radius / max(delta_norm, 1e-12)
                for parameter, anchor in zip(
                    self.actor.parameters(), self._actor_anchor, strict=True
                ):
                    parameter.copy_(anchor + scale * (parameter - anchor))
                delta_norm = self._actor_trust_region_radius
        return delta_norm, projected

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.actor.sample(
                observation.to(self.device), deterministic=True
            )
        return action.cpu()

    def behavior_clone(
        self,
        observations: torch.Tensor,
        target_actions: torch.Tensor,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        seed: int,
    ) -> dict[str, float]:
        if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
            raise ValueError(
                "behavior cloning epochs, batch size, and learning rate must be positive"
            )
        observations = observations.to(self.device)
        target_actions = target_actions.to(self.device)
        if (
            observations.ndim != 2
            or target_actions.ndim != 2
            or len(observations) != len(target_actions)
            or not len(observations)
        ):
            raise ValueError(
                "behavior cloning arrays must be aligned non-empty matrices"
            )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        optimizer = torch.optim.Adam(
            (
                parameter
                for parameter in self.actor.parameters()
                if parameter.requires_grad
            ),
            lr=learning_rate,
        )
        final_loss = 0.0
        optimizer_steps = 0
        for _ in range(epochs):
            order = torch.randperm(len(observations), generator=generator)
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size].to(self.device)
                prediction, _ = self.actor.sample(
                    observations[indices], deterministic=True
                )
                loss = nn.functional.mse_loss(prediction, target_actions[indices])
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.gradient_norm_limit
                )
                optimizer.step()
                final_loss = float(loss.detach())
                optimizer_steps += 1
        self.target_actor.load_state_dict(self.actor.state_dict())
        return {
            "final_mse": final_loss,
            "samples": float(len(observations)),
            "epochs": float(epochs),
            "optimizer_steps": float(optimizer_steps),
        }

    def update(
        self,
        batch: dict[str, torch.Tensor],
        *,
        allow_actor_update: bool = True,
    ) -> dict[str, float]:
        actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done = (
            batch[key].to(self.device)
            for key in (
                "actor_obs",
                "critic_obs",
                "action",
                "reward",
                "next_actor_obs",
                "next_critic_obs",
                "done",
            )
        )
        with torch.no_grad():
            next_action, _ = self.target_actor.sample(
                next_actor_obs, deterministic=True
            )
            noise = torch.randn_like(next_action) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_critic(next_critic_obs, next_action)
            target = self.reward_scale * reward + self.gamma * (1.0 - done) * (
                torch.minimum(target_q1, target_q2)
            )

        q1, q2 = self.critic(critic_obs, action)
        critic_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(
            q2, target
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_norm_limit)
        self.critic_optimizer.step()

        self._update_count += 1
        actor_loss_value = 0.0
        behavior_loss_value = 0.0
        q_coefficient_value = 0.0
        actor_delta_l2 = 0.0
        trust_region_projected = False
        delayed_update = self._update_count % self.policy_delay == 0
        actor_updated = allow_actor_update and delayed_update
        if actor_updated:
            policy_action, _ = self.actor.sample(actor_obs, deterministic=True)
            features = torch.cat((critic_obs, policy_action), dim=1)
            self.critic.requires_grad_(False)
            try:
                policy_q = self.critic.q1(features)
                q_coefficient = self.q_normalization_scale / (
                    policy_q.abs().mean().detach().clamp_min(1e-6)
                )
                q_coefficient = q_coefficient.clamp(max=self.maximum_q_coefficient)
                behavior_loss = torch.zeros((), device=self.device)
                if "behavior_actor_obs" in batch and "behavior_action" in batch:
                    behavior_observation = batch["behavior_actor_obs"].to(self.device)
                    behavior_target = batch["behavior_action"].to(self.device)
                    behavior_action, _ = self.actor.sample(
                        behavior_observation, deterministic=True
                    )
                    behavior_loss = nn.functional.mse_loss(
                        behavior_action, behavior_target
                    )
                actor_loss = (
                    -q_coefficient * policy_q.mean()
                    + self.behavior_regularization_weight * behavior_loss
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.gradient_norm_limit
                )
                self.actor_optimizer.step()
                actor_delta_l2, trust_region_projected = (
                    self._project_actor_to_trust_region()
                )
            finally:
                self.critic.requires_grad_(True)
            actor_loss_value = float(actor_loss.detach())
            behavior_loss_value = float(behavior_loss.detach())
            q_coefficient_value = float(q_coefficient.detach())
            self._soft_update(self.target_actor, self.actor)
        if delayed_update:
            self._soft_update(self.target_critic, self.critic)

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": float(critic_loss.detach()),
            "behavior_loss": behavior_loss_value if actor_updated else 0.0,
            "q_coefficient": q_coefficient_value,
            "actor_delta_l2": actor_delta_l2,
            "trust_region_projected": float(trust_region_projected),
            "actor_updated": float(actor_updated),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(parameter, self.tau)

    def parameter_counts(self) -> dict[str, int]:
        actor = self.actor.parameter_count
        actor_trainable = sum(
            parameter.numel()
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        )
        critics = self.critic.parameter_count
        return {
            "actor": actor,
            "actor_trainable": actor_trainable,
            "twin_critics": critics,
            "trainable_total": actor_trainable + critics,
            "frozen_targets": actor + critics,
        }
