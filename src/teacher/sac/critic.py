"""Twin Q networks over a privileged critic state and action."""

import torch
from torch import nn


class TwinQCritic(nn.Module):
    def __init__(self, critic_observation_dim: int, action_dim: int) -> None:
        super().__init__()
        input_dim = critic_observation_dim + action_dim
        self.q1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, critic_observation: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((critic_observation, action), dim=1)
        return self.q1(features), self.q2(features)
