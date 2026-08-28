"""Large residual squashed-Gaussian actor for the MLP Teacher."""

from __future__ import annotations

import torch
from torch import nn

from src.teacher.residual import ResidualMLPTrunk


class SquashedGaussianActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        width: int = 896,
        residual_blocks: int = 14,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.body = ResidualMLPTrunk(
            observation_dim,
            width,
            residual_blocks,
            residual_scale=residual_scale,
        )
        self.mean = nn.Linear(width, action_dim)
        self.log_std = nn.Linear(width, action_dim)
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.bias, -3e-3, 3e-3)

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observation)
        mean, log_std = self.mean(hidden), self.log_std(hidden).clamp(-5, 2)
        if deterministic:
            action = mean.tanh()
            return action, torch.zeros((observation.shape[0], 1), device=observation.device)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = raw.tanh()
        correction = torch.log(1.0 - action.square() + 1e-6).sum(-1, keepdim=True)
        return action, normal.log_prob(raw).sum(-1, keepdim=True) - correction

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
