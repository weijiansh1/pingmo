"""Privileged-critic SAC with a large residual MLP actor."""

from __future__ import annotations

from copy import deepcopy
import math

import torch

from src.teacher.sac.actor import SquashedGaussianActor
from src.teacher.sac.critic import TwinQCritic


class PrivilegedSAC:
    def __init__(
        self,
        actor_observation_dim: int,
        critic_observation_dim: int,
        action_dim: int,
        gamma: float = 0.9999,
        tau: float = 0.005,
        initial_alpha: float = 0.1,
        target_entropy: float | None = None,
        learning_rate: float = 3e-4,
        gradient_norm_limit: float = 10.0,
        actor_width: int = 896,
        actor_residual_blocks: int = 14,
        critic_width: int = 896,
        critic_residual_blocks: int = 14,
        residual_scale: float = 0.1,
        device: str | torch.device = "cpu",
    ) -> None:
        if (
            not 0 < gamma <= 1
            or not 0 < tau <= 1
            or initial_alpha <= 0
            or learning_rate <= 0
            or gradient_norm_limit <= 0
        ):
            raise ValueError("invalid SAC hyperparameters")
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy
        self.gradient_norm_limit = gradient_norm_limit
        self.actor = SquashedGaussianActor(
            actor_observation_dim,
            action_dim,
            width=actor_width,
            residual_blocks=actor_residual_blocks,
            residual_scale=residual_scale,
        ).to(self.device)
        self.critic = TwinQCritic(
            critic_observation_dim,
            action_dim,
            width=critic_width,
            residual_blocks=critic_residual_blocks,
            residual_scale=residual_scale,
        ).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device).eval()
        self.target_critic.requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(initial_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=learning_rate)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, actor_observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.actor.sample(actor_observation.to(self.device), deterministic=deterministic)
        return action.cpu()

    def q_values(self, critic_observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(critic_observation.to(self.device), action.to(self.device))

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done = (
            batch[key].to(self.device)
            for key in ("actor_obs", "critic_obs", "action", "reward", "next_actor_obs", "next_critic_obs", "done")
        )
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_actor_obs)
            next_q1, next_q2 = self.target_critic(next_critic_obs, next_action)
            target = reward + self.gamma * (1.0 - done) * (
                torch.minimum(next_q1, next_q2) - self.alpha.detach() * next_log_prob
            )
        q1, q2 = self.critic(critic_obs, action)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_norm_limit)
        self.critic_optim.step()
        self.critic_optim.zero_grad(set_to_none=True)

        policy_action, log_prob = self.actor.sample(actor_obs)
        self.critic.requires_grad_(False)
        try:
            q1_pi, q2_pi = self.critic(critic_obs, policy_action)
            actor_loss = (self.alpha.detach() * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_norm_limit)
            self.actor_optim.step()
            self.actor_optim.zero_grad(set_to_none=True)
        finally:
            self.critic.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha_optim.zero_grad(set_to_none=True)

        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return {
            "actor_loss": float(actor_loss.detach()),
            "critic_loss": float(critic_loss.detach()),
            "alpha_loss": float(alpha_loss.detach()),
            "alpha": float(self.alpha.detach()),
        }

    def parameter_counts(self) -> dict[str, int]:
        actor = self.actor.parameter_count
        critics = self.critic.parameter_count
        return {
            "actor": actor,
            "twin_critics": critics,
            "entropy_temperature": 1,
            "trainable_total": actor + critics + 1,
            "frozen_target_critics": critics,
            "stored_network_total_excluding_optimizer": actor + 2 * critics,
        }
