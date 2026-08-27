"""Feature-level, theta-routed Gaussian MoE actor."""

import torch
from torch import nn


class MoEActor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, experts: int = 4) -> None:
        super().__init__()
        if observation_dim < 8:
            raise ValueError("observation must end with the eight privileged parameters")
        self.theta_encoder = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.router = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, experts))
        self.state_encoder = nn.Sequential(nn.Linear(observation_dim - 8, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU()) for _ in range(experts))
        self.mean, self.log_std = nn.Linear(64, action_dim), nn.Linear(64, action_dim)

    def sample(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, theta = observation[:, :-8], observation[:, -8:]
        weights = torch.softmax(self.router(self.theta_encoder(theta)), dim=-1)
        features = torch.stack([expert(self.state_encoder(state)) for expert in self.experts], dim=1)
        mixed = (weights.unsqueeze(-1) * features).sum(dim=1)
        mean, log_std = self.mean(mixed), self.log_std(mixed).clamp(-5, 2)
        normal = torch.distributions.Normal(mean, log_std.exp())
        raw = normal.rsample(); action = raw.tanh()
        log_prob = normal.log_prob(raw).sum(-1, keepdim=True) - torch.log(1 - action.square() + 1e-6).sum(-1, keepdim=True)
        return action, log_prob, weights

    @staticmethod
    def balance_loss(weights: torch.Tensor) -> torch.Tensor:
        mean_weight = weights.mean(0)
        uniform = torch.full_like(mean_weight, 1.0 / mean_weight.numel())
        return torch.sum(mean_weight * (mean_weight.log() - uniform.log()))
