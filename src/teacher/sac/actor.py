"""Squashed Gaussian actor over the Teacher deployment observation."""

import torch
from torch import nn


class SquashedGaussianActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.mean = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observation)
        mean, log_std = self.mean(hidden), self.log_std(hidden).clamp(-5, 2)
        if deterministic:
            action = mean.tanh()
            return action, torch.zeros((observation.shape[0], 1), device=observation.device)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = raw.tanh()
        log_prob = normal.log_prob(raw).sum(-1, keepdim=True) - torch.log(1 - action.square() + 1e-6).sum(-1, keepdim=True)
        return action, log_prob
