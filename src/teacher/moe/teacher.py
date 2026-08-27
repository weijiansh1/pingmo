"""Global MoE actor with twin critics over a separate privileged state."""

from copy import deepcopy

import torch

from src.teacher.moe.actor import MoEActor
from src.teacher.sac.critic import TwinQCritic


class MoETeacher:
    def __init__(self, actor_observation_dim: int, critic_observation_dim: int, action_dim: int, experts: int = 4, balance_coefficient: float = 1e-3, gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2) -> None:
        self.actor = MoEActor(actor_observation_dim, action_dim, experts)
        self.critic = TwinQCritic(critic_observation_dim, action_dim)
        self.target_critic = deepcopy(self.critic).eval()
        self.balance_coefficient, self.gamma, self.tau, self.alpha = balance_coefficient, gamma, tau, alpha
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def q_values(self, critic_observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(critic_observation, action)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done = (batch[key] for key in ("actor_obs", "critic_obs", "action", "reward", "next_actor_obs", "next_critic_obs", "done"))
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(next_actor_obs)
            target_q1, target_q2 = self.target_critic(next_critic_obs, next_action)
            target = reward + self.gamma * (1 - done) * (torch.minimum(target_q1, target_q2) - self.alpha * next_log_prob)
        q1, q2 = self.critic(critic_obs, action)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optim.zero_grad(); critic_loss.backward(); self.critic_optim.step()
        policy_action, log_prob, weights = self.actor.sample(actor_obs)
        policy_q1, policy_q2 = self.critic(critic_obs, policy_action)
        policy_q = torch.minimum(policy_q1, policy_q2)
        balance = self.actor.balance_loss(weights)
        actor_loss = (self.alpha * log_prob - policy_q).mean() + self.balance_coefficient * balance
        self.actor_optim.zero_grad(); actor_loss.backward(); self.actor_optim.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()), "balance_loss": float(balance.detach()), "router_max_mean": float(weights.mean(0).max().detach())}
