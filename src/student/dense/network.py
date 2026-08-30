"""Parameter-conditioned dense Student for universal P-channel control."""

from __future__ import annotations

import torch
from torch import nn

from src.teacher.residual import ResidualMLPTrunk


class DenseConditionalStudent(nn.Module):
    """Map deployment observation and normalized aircraft theta to full F_as."""

    def __init__(
        self,
        observation_dim: int,
        aircraft_parameter_dim: int = 8,
        action_dim: int = 1,
        *,
        width: int = 512,
        residual_blocks: int = 8,
        residual_scale: float = 0.1,
        enforce_odd_policy: bool = True,
    ) -> None:
        super().__init__()
        if min(observation_dim, aircraft_parameter_dim, action_dim, width, residual_blocks) <= 0:
            raise ValueError("student network dimensions must be positive")
        self.observation_dim = observation_dim
        self.aircraft_parameter_dim = aircraft_parameter_dim
        self.action_dim = action_dim
        self.enforce_odd_policy = enforce_odd_policy
        self.body = ResidualMLPTrunk(
            observation_dim + aircraft_parameter_dim,
            width,
            residual_blocks,
            residual_scale=residual_scale,
        )
        self.action_head = nn.Linear(width, action_dim)
        nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.action_head.bias, -3e-3, 3e-3)

    def forward(self, observation: torch.Tensor, aircraft_parameters: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("student observation dimension mismatch")
        if aircraft_parameters.shape[-1] != self.aircraft_parameter_dim:
            raise ValueError("student aircraft-parameter dimension mismatch")
        hidden = self.body(torch.cat((observation, aircraft_parameters), dim=-1))
        action_logits = self.action_head(hidden)
        if self.enforce_odd_policy:
            mirrored_hidden = self.body(
                torch.cat((-observation, aircraft_parameters), dim=-1)
            )
            action_logits = 0.5 * (
                action_logits - self.action_head(mirrored_hidden)
            )
        return action_logits.tanh()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
