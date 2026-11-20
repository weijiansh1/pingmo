"""Deterministic TD3 Actors with optional control priors."""

from __future__ import annotations

import torch
from torch import nn

from src.teacher.residual import ResidualMLPTrunk


class DeterministicActor(nn.Module):
    """Direct bounded control policy without entropy or a control prior."""

    architecture_name = "odd_deterministic_residual_mlp_v1"

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        width: int = 128,
        residual_blocks: int = 2,
        residual_scale: float = 0.1,
        enforce_odd_symmetry: bool = True,
    ) -> None:
        super().__init__()
        if observation_dim <= 0 or action_dim <= 0:
            raise ValueError("deterministic Actor dimensions must be positive")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.enforce_odd_symmetry = enforce_odd_symmetry
        self.body = ResidualMLPTrunk(
            observation_dim,
            width,
            residual_blocks,
            residual_scale=residual_scale,
        )
        self.action_head = nn.Linear(width, action_dim)
        nn.init.uniform_(self.action_head.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.action_head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("deterministic Actor observation dimension mismatch")
        logits = self.action_head(self.body(observation))
        if self.enforce_odd_symmetry:
            mirrored_logits = self.action_head(self.body(-observation))
            logits = 0.5 * (logits - mirrored_logits)
        return logits.tanh()

    def sample(
        self, observation: torch.Tensor, deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del deterministic
        action = self(observation)
        zeros = torch.zeros(
            (observation.shape[0], 1),
            dtype=observation.dtype,
            device=observation.device,
        )
        return action, zeros

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class PIDResidualActor(nn.Module):
    """Apply a bounded learned residual around an embedded PID force law.

    The prior coefficients are stored in the checkpoint as a non-trainable
    tensor.  Deployment therefore needs only this module and the seven Actor
    observations; it does not instantiate or call the PID controller.
    """

    architecture_name = "pid_initialized_bounded_residual_mlp_v1"

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        control_prior_coefficients: torch.Tensor,
        *,
        width: int = 704,
        residual_blocks: int = 10,
        residual_scale: float = 0.1,
        residual_action_limit: float = 0.05,
        enforce_odd_symmetry: bool = True,
    ) -> None:
        super().__init__()
        coefficients = torch.as_tensor(control_prior_coefficients, dtype=torch.float32)
        if coefficients.shape != (action_dim, observation_dim):
            raise ValueError(
                "control-prior coefficients must have shape "
                "(action_dim, observation_dim)"
            )
        if not torch.isfinite(coefficients).all():
            raise ValueError("control-prior coefficients must be finite")
        if not 0 < residual_action_limit <= 1:
            raise ValueError("residual_action_limit must be in (0, 1]")

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.residual_action_limit = float(residual_action_limit)
        self.enforce_odd_symmetry = enforce_odd_symmetry
        self.body = ResidualMLPTrunk(
            observation_dim,
            width,
            residual_blocks,
            residual_scale=residual_scale,
        )
        self.residual_head = nn.Linear(width, action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        self.register_buffer(
            "control_prior_coefficients", coefficients.detach().clone()
        )

    def _residual(self, observation: torch.Tensor) -> torch.Tensor:
        residual_logits = self.residual_head(self.body(observation))
        if self.enforce_odd_symmetry:
            mirrored_logits = self.residual_head(self.body(-observation))
            residual_logits = 0.5 * (residual_logits - mirrored_logits)
        return self.residual_action_limit * residual_logits.tanh()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError("PID-residual Actor observation dimension mismatch")
        prior = nn.functional.linear(
            observation, self.control_prior_coefficients
        ).clamp(-1.0, 1.0)
        return (prior + self._residual(observation)).clamp(-1.0, 1.0)

    def sample(
        self, observation: torch.Tensor, deterministic: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del deterministic
        action = self(observation)
        zeros = torch.zeros(
            (observation.shape[0], 1),
            dtype=observation.dtype,
            device=observation.device,
        )
        return action, zeros

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
