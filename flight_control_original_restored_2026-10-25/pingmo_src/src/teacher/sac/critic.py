"""Twin Q networks over a privileged critic state and action."""

import torch
from torch import nn

from src.teacher.residual import ResidualMLPTrunk


class _ResidualQNetwork(nn.Module):
    def __init__(self, input_dim: int, width: int, residual_blocks: int, residual_scale: float) -> None:
        super().__init__()
        self.body = ResidualMLPTrunk(
            input_dim,
            width,
            residual_blocks,
            residual_scale=residual_scale,
        )
        self.value = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(self.body(features))


class TwinQCritic(nn.Module):
    def __init__(
        self,
        critic_observation_dim: int,
        action_dim: int,
        width: int = 896,
        residual_blocks: int = 14,
        residual_scale: float = 0.1,
    ) -> None:
        super().__init__()
        input_dim = critic_observation_dim + action_dim
        self.q1 = _ResidualQNetwork(input_dim, width, residual_blocks, residual_scale)
        self.q2 = _ResidualQNetwork(input_dim, width, residual_blocks, residual_scale)

    def forward(self, critic_observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((critic_observation, action), dim=1)
        return self.q1(features), self.q2(features)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
