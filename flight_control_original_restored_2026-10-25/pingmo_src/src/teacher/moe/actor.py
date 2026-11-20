"""Four-expert, theta-routed Gaussian actor with continuous soft routing."""

from __future__ import annotations

import torch
from torch import nn

from src.teacher.residual import ResidualMLPBlock, ResidualMLPTrunk


class MoEActor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        experts: int = 4,
        model_width: int = 896,
        shared_residual_blocks: int = 10,
        expert_residual_blocks: int = 2,
        expert_bottleneck_width: int = 448,
        residual_scale: float = 0.1,
        theta_widths: tuple[int, int] = (512, 256),
        router_hidden_width: int = 512,
    ) -> None:
        super().__init__()
        if observation_dim <= 8:
            raise ValueError("observation must end with the eight normalized plant parameters")
        if experts <= 1:
            raise ValueError("MoE actor requires at least two experts")
        if min(model_width, shared_residual_blocks, expert_residual_blocks, expert_bottleneck_width, router_hidden_width, *theta_widths) <= 0:
            raise ValueError("MoE dimensions must be positive")
        self.expert_count = experts
        theta_first, theta_second = theta_widths
        self.theta_encoder = nn.Sequential(
            nn.Linear(8, theta_first),
            nn.SiLU(),
            nn.Linear(theta_first, theta_second),
            nn.SiLU(),
        )
        self.router = nn.Sequential(
            nn.Linear(theta_second, router_hidden_width),
            nn.SiLU(),
            nn.Linear(router_hidden_width, experts),
        )
        self.state_encoder = ResidualMLPTrunk(
            observation_dim - 8,
            model_width,
            shared_residual_blocks,
            residual_scale=residual_scale,
            output_norm=False,
        )
        self.experts = nn.ModuleList(
            nn.Sequential(*(
                ResidualMLPBlock(model_width, expert_bottleneck_width, residual_scale)
                for _ in range(expert_residual_blocks)
            ))
            for _ in range(experts)
        )
        self.output_norm = nn.LayerNorm(model_width)
        self.mean = nn.Linear(model_width, action_dim)
        self.log_std = nn.Linear(model_width, action_dim)
        nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std.bias, -3e-3, 3e-3)

    def sample(self, observation: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, theta = observation[:, :-8], observation[:, -8:]
        weights = torch.softmax(self.router(self.theta_encoder(theta)), dim=-1)
        encoded_state = self.state_encoder(state)
        features = torch.stack([expert(encoded_state) for expert in self.experts], dim=1)
        mixed = self.output_norm((weights.unsqueeze(-1) * features).sum(dim=1))
        mean, log_std = self.mean(mixed), self.log_std(mixed).clamp(-5, 2)
        if deterministic:
            action = mean.tanh()
            return action, torch.zeros((observation.shape[0], 1), device=observation.device), weights
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample()
        action = raw.tanh()
        correction = torch.log(1.0 - action.square() + 1e-6).sum(-1, keepdim=True)
        return action, normal.log_prob(raw).sum(-1, keepdim=True) - correction, weights

    @staticmethod
    def balance_loss(weights: torch.Tensor) -> torch.Tensor:
        mean_weight = weights.mean(0)
        uniform = torch.full_like(mean_weight, 1.0 / mean_weight.numel())
        return torch.sum(mean_weight * (mean_weight.clamp_min(1e-8).log() - uniform.log()))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
