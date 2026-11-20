"""No-history, theta-routed linear mixture-of-experts Student."""

from __future__ import annotations

import math

import torch
from torch import nn


class ThetaRoutedLinearMoEStudent(nn.Module):
    """Blend smooth linear control laws using aircraft parameters only.

    The router never sees the response observation, so its weights are constant
    throughout a rollout for a fixed aircraft. Each bias-free expert is an odd
    current-state control law; the final hard clamp preserves odd symmetry.
    """

    architecture_name = "theta_routed_sparse_linear_soft_moe_v2"

    def __init__(
        self,
        observation_dim: int,
        aircraft_parameter_dim: int,
        action_dim: int,
        anchor_prototypes: torch.Tensor,
        *,
        router_temperature: float = 0.2,
        prototype_movement_limit: float = 0.05,
        control_feature_indices: tuple[int, ...] = (3, 4, 5),
    ) -> None:
        super().__init__()
        prototypes = torch.as_tensor(anchor_prototypes, dtype=torch.float32)
        if prototypes.ndim != 2:
            raise ValueError("MoE anchor prototypes must have shape [experts, theta_dim]")
        expert_count, prototype_dim = prototypes.shape
        if min(
            observation_dim,
            aircraft_parameter_dim,
            action_dim,
            expert_count,
        ) <= 0:
            raise ValueError("MoE Student dimensions must be positive")
        if prototype_dim != aircraft_parameter_dim:
            raise ValueError("MoE prototype dimension does not match theta")
        if not torch.isfinite(prototypes).all():
            raise ValueError("MoE anchor prototypes must be finite")
        if router_temperature <= 0:
            raise ValueError("MoE router temperature must be positive")
        if prototype_movement_limit < 0:
            raise ValueError("MoE prototype movement limit cannot be negative")
        if (
            not control_feature_indices
            or len(set(control_feature_indices)) != len(control_feature_indices)
            or min(control_feature_indices) < 0
            or max(control_feature_indices) >= observation_dim
        ):
            raise ValueError("MoE control feature indices are invalid")

        self.observation_dim = observation_dim
        self.aircraft_parameter_dim = aircraft_parameter_dim
        self.action_dim = action_dim
        self.expert_count = expert_count
        self.control_feature_indices = tuple(control_feature_indices)
        self.control_feature_dim = len(self.control_feature_indices)
        self.router_temperature = float(router_temperature)
        self.prototype_movement_limit = float(prototype_movement_limit)
        self.enforce_odd_policy = True

        self.register_buffer("anchor_prototypes", prototypes.clone())
        self.prototype_offsets = nn.Parameter(torch.zeros_like(prototypes))
        self.expert_weights = nn.Parameter(
            torch.zeros(expert_count, action_dim, self.control_feature_dim)
        )

    @property
    def prototypes(self) -> torch.Tensor:
        return self.anchor_prototypes + self.prototype_movement_limit * torch.tanh(
            self.prototype_offsets
        )

    def routing_logits(self, aircraft_parameters: torch.Tensor) -> torch.Tensor:
        if aircraft_parameters.shape[-1] != self.aircraft_parameter_dim:
            raise ValueError("student aircraft-parameter dimension mismatch")
        difference = aircraft_parameters.unsqueeze(-2) - self.prototypes
        mean_squared_distance = difference.square().mean(dim=-1)
        return -mean_squared_distance / self.router_temperature

    def routing_weights(self, aircraft_parameters: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.routing_logits(aircraft_parameters), dim=-1)

    def forward_with_routing(
        self,
        observation: torch.Tensor,
        aircraft_parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("student observation dimension mismatch")
        if observation.shape[:-1] != aircraft_parameters.shape[:-1]:
            raise ValueError("Student observation and theta batches must align")
        logits = self.routing_logits(aircraft_parameters)
        route_weights = torch.softmax(logits, dim=-1)
        control_observation = observation[..., self.control_feature_indices]
        expert_actions = torch.einsum(
            "...o,eao->...ea", control_observation, self.expert_weights
        )
        action = torch.einsum("...e,...ea->...a", route_weights, expert_actions)
        return action.clamp(-1.0, 1.0), route_weights, logits

    def forward(
        self,
        observation: torch.Tensor,
        aircraft_parameters: torch.Tensor,
    ) -> torch.Tensor:
        action, _, _ = self.forward_with_routing(observation, aircraft_parameters)
        return action

    def router_regularization(
        self,
        aircraft_parameters: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return differentiable collapse controls and route diagnostics."""

        logits = self.routing_logits(aircraft_parameters)
        weights = torch.softmax(logits, dim=-1)
        mean_usage = weights.mean(dim=0)
        balance_loss = self.expert_count * mean_usage.square().sum() - 1.0
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        prototype_anchor_loss = (
            self.prototypes - self.anchor_prototypes
        ).square().mean()
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        normalized_entropy = entropy / max(math.log(self.expert_count), 1.0)
        return {
            "router_balance_loss": balance_loss,
            "router_z_loss": z_loss,
            "prototype_anchor_loss": prototype_anchor_loss,
            "router_entropy": entropy,
            "router_normalized_entropy": normalized_entropy,
            "router_max_mean_usage": mean_usage.max(),
            "router_min_mean_usage": mean_usage.min(),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
