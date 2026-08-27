"""SAC with deployment observations for the actor and privileged states for Q values."""

from copy import deepcopy

import torch

from src.teacher.sac.actor import SquashedGaussianActor
from src.teacher.sac.critic import TwinQCritic


class PrivilegedSAC:
    def __init__(self, actor_observation_dim: int, critic_observation_dim: int, action_dim: int, gamma: float = 0.99, tau: float = 0.005, alpha: float = 0.2, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.gamma, self.tau, self.alpha = gamma, tau, alpha
        self.actor = SquashedGaussianActor(actor_observation_dim, action_dim).to(self.device)
        self.critic = TwinQCritic(critic_observation_dim, action_dim).to(self.device)
        self.target_critic = deepcopy(self.critic).to(self.device).eval()
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

    def act(self, actor_observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        with torch.no_grad():
            action, _ = self.actor.sample(actor_observation.to(self.device), deterministic=deterministic)
        return action.cpu()

    def q_values(self, critic_observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic(critic_observation.to(self.device), action.to(self.device))

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done = (batch[key].to(self.device) for key in ("actor_obs", "critic_obs", "action", "reward", "next_actor_obs", "next_critic_obs", "done"))
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_actor_obs)
            next_q1, next_q2 = self.target_critic(next_critic_obs, next_action)
            target = reward + self.gamma * (1 - done) * (torch.minimum(next_q1, next_q2) - self.alpha * next_log_prob)
        q1, q2 = self.critic(critic_obs, action)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optim.zero_grad(); critic_loss.backward(); self.critic_optim.step()
        policy_action, log_prob = self.actor.sample(actor_obs)
        q1_pi, q2_pi = self.critic(critic_obs, policy_action)
        actor_loss = (self.alpha * log_prob - torch.minimum(q1_pi, q2_pi)).mean()
        self.actor_optim.zero_grad(); actor_loss.backward(); self.actor_optim.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return {"actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach())}
