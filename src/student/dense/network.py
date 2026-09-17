"""Parameter-conditioned dense Student for universal P-channel control."""

from __future__ import annotations

import torch
from torch import nn

from src.teacher.residual import ResidualMLPTrunk


def _small_head(linear: nn.Linear, scale: float = 3e-3) -> None:
    nn.init.uniform_(linear.weight, -scale, scale)
    nn.init.zeros_(linear.bias)


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


class IncrementalDenseStudent(nn.Module):
    """Incremental-action Student: maps (o, theta, u_{t-1}) to an action delta.

    Instead of re-guessing the full force every policy step (v4's
    ``DenseConditionalStudent``), this Student emits ``delta`` and reconstructs
    ``u_t = clip(u_{t-1} + delta, -1, 1)``. The previous action ``u_{t-1}`` is
    the controller's own previous *requested* action (its last output), which is
    both what the environment exposes as ``previous_force_normalized`` and what
    a deployed policy keeps in memory. The 88 N/s actuator rate limit stays in
    the environment layer and is applied uniformly to every controller.
    """

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
        delta_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if min(
            observation_dim,
            aircraft_parameter_dim,
            action_dim,
            width,
            residual_blocks,
        ) <= 0:
            raise ValueError("incremental Student network dimensions must be positive")
        if delta_scale <= 0:
            raise ValueError("incremental Student delta_scale must be positive")
        self.observation_dim = observation_dim
        self.aircraft_parameter_dim = aircraft_parameter_dim
        self.action_dim = action_dim
        self.enforce_odd_policy = enforce_odd_policy
        self.delta_scale = delta_scale
        self.body = ResidualMLPTrunk(
            observation_dim + aircraft_parameter_dim + action_dim,
            width,
            residual_blocks,
            residual_scale=residual_scale,
        )
        self.delta_head = nn.Linear(width, action_dim)
        _small_head(self.delta_head)

    def _delta_logits(
        self,
        observation: torch.Tensor,
        aircraft_parameters: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("incremental Student observation dimension mismatch")
        if aircraft_parameters.shape[-1] != self.aircraft_parameter_dim:
            raise ValueError("incremental Student aircraft-parameter dimension mismatch")
        if previous_action.shape[-1] != self.action_dim:
            raise ValueError("incremental Student previous-action dimension mismatch")
        hidden = self.body(
            torch.cat((observation, aircraft_parameters, previous_action), dim=-1)
        )
        return self.delta_head(hidden)

    def forward(
        self,
        observation: torch.Tensor,
        aircraft_parameters: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_t, delta_t)`` where ``action_t = clip(u_{t-1} + delta_t)``.

        Odd-policy symmetry is enforced on the delta in the joint input
        ``(observation, previous_action)``: mirroring both the observation and the
        previous action negates the delta, so the reconstructed action is odd too.
        """
        delta_logits = self._delta_logits(
            observation, aircraft_parameters, previous_action
        )
        if self.enforce_odd_policy:
            mirrored_logits = self._delta_logits(
                -observation, aircraft_parameters, -previous_action
            )
            delta_logits = 0.5 * (delta_logits - mirrored_logits)
        delta = self.delta_scale * delta_logits.tanh()
        action = (previous_action + delta).clamp(-1.0, 1.0)
        return action, delta

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
