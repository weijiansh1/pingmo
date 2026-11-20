"""Asymmetric PPO networks for the roll-rate Teacher.

Actor   : proprioception only (current signals + 16-step history + recent actions)
Critic  : proprioception + privileged plant state, delay queue and progress

The temporal front-end is shared in spirit with the Student: a 1-D convolution
over the history window, followed by "last step + time mean".  This keeps the
deployment input identical in form (history in, features out), so the Student can
be a narrower copy instead of a different architecture.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from src.envs.batched_p_channel import (
    THETA_DIM,
    CURRENT_DIM,
    HISTORY_CHANNELS,
    HISTORY_STEPS,
    OBS_DIM,
    OLDER_ACTIONS,
)
from src.teacher.residual import ResidualMLPTrunk


class HistoryConv(nn.Module):
    """(N, steps, channels) -> (N, 2 * width) by convolutions over time.

    Dilations (1, 4, 8) give a receptive field of 27 policy steps (540 ms), which
    spans the transport-delay range (0 to 0.5 s) the Actor has to compensate for:
    the encoder can compare when the command moved with when the aircraft answer
    arrived.  A flat MLP cannot express that comparison, and a 2-layer k=3 stack
    only reaches 100 ms.
    """

    def __init__(self, channels: int = HISTORY_CHANNELS, width: int = 24,
                 dilations: tuple[int, ...] = (1, 4, 8)) -> None:
        super().__init__()
        layers = []
        in_channels = channels
        for dilation in dilations:
            layers.append(nn.Conv1d(in_channels, width, kernel_size=3,
                                    padding=dilation, dilation=dilation))
            layers.append(nn.SiLU())
            in_channels = width
        self.layers = nn.Sequential(*layers)
        self.width = width
        self.receptive_field = 1 + 2 * sum(dilations)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.layers(history.transpose(1, 2))          # (N, C, T)
        return torch.cat([x[:, :, -1], x.mean(dim=2)], dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class PPOTeacherActor(nn.Module):
    """Gaussian policy over normalised force-rate, mean squashed to (-1, 1).

    The Teacher is never deployed, so its Actor may read theta and the plant
    internal state; the Student has to infer the same information from history.

    The mean is projected onto odd symmetry in (observation, plant state) and the
    output head has no bias.  Both are required by the force-rate action: any
    offset survives the rate integrator and becomes a standing force bias, which
    then favours one command sign over the other.
    """

    def __init__(self, width: int = 512, blocks: int = 8, conv_width: int = 24,
                 privileged_dim: int = THETA_DIM + 4, theta_dim: int = THETA_DIM,
                 enforce_odd_symmetry: bool = True,
                 initial_log_std: float = -2.0) -> None:
        super().__init__()
        self.encoder = HistoryConv(width=conv_width)
        self.privileged_dim = privileged_dim
        self.theta_dim = theta_dim
        self.enforce_odd_symmetry = enforce_odd_symmetry
        feature_dim = (2 * conv_width + CURRENT_DIM + OLDER_ACTIONS + privileged_dim)
        self.feature_dim = feature_dim
        self.body = ResidualMLPTrunk(feature_dim, width, blocks)
        self.mean_head = nn.Linear(width, 1, bias=False)
        _orthogonal(self.mean_head, 0.01)
        self.log_std = nn.Parameter(torch.full((1,), initial_log_std))

    def split_observation(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = observation[..., :CURRENT_DIM]
        history_start = CURRENT_DIM
        history_end = history_start + HISTORY_STEPS * HISTORY_CHANNELS
        history = observation[..., history_start:history_end].reshape(
            -1, HISTORY_STEPS, HISTORY_CHANNELS)
        older = observation[..., history_end:history_end + OLDER_ACTIONS]
        return current, history, older

    def features(self, observation: torch.Tensor,
                 privileged: torch.Tensor) -> torch.Tensor:
        current, history, older = self.split_observation(observation)
        return torch.cat([self.encoder(history), current, older, privileged], dim=-1)

    def _mirror_privileged(self, privileged: torch.Tensor) -> torch.Tensor:
        """Negate the plant state; theta keeps its sign (it is not a signed signal)."""
        theta = privileged[..., :self.theta_dim]
        signed = privileged[..., self.theta_dim:]
        return torch.cat([theta, -signed], dim=-1)

    def _logits(self, observation: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        return self.mean_head(self.body(self.features(observation, privileged)))

    def forward(self, observation: torch.Tensor,
                privileged: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._logits(observation, privileged)
        if self.enforce_odd_symmetry:
            mirrored = self._logits(-observation, self._mirror_privileged(privileged))
            logits = 0.5 * (logits - mirrored)
        mean = torch.tanh(logits.float())
        log_std = self.log_std.expand(observation.shape[0], 1)
        return mean, log_std

    def act(self, observation: torch.Tensor, privileged: torch.Tensor,
            deterministic: bool = False):
        """Return the bounded environment action, sample log-probability and mean.

        PPO collectors use ``sample_for_ppo`` to retain the raw Gaussian draw
        alongside its log-probability. This three-value interface is retained
        for existing evaluation and teacher-label callers.
        """
        action, log_prob, mean, _ = self.sample_for_ppo(
            observation, privileged, deterministic=deterministic)
        return action, log_prob, mean

    def sample_for_ppo(self, observation: torch.Tensor, privileged: torch.Tensor,
                       deterministic: bool = False):
        """Return environment action, log-probability, mean and raw sample.

        The environment receives the clipped draw. PPO evaluates the same raw
        draw under both the rollout policy and the updated Gaussian policy.
        """
        mean, log_std = self(observation, privileged)
        if deterministic:
            return mean, torch.zeros_like(mean), mean, mean
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        sample = normal.rsample()
        action = sample.clamp(-1.0, 1.0)
        log_prob = normal.log_prob(sample).sum(-1, keepdim=True)
        return action, log_prob, mean, sample

    def evaluate(self, observation: torch.Tensor, privileged: torch.Tensor,
                 raw_action: torch.Tensor):
        """Evaluate stored raw Gaussian draws, before environment clipping."""
        mean, log_std = self(observation, privileged)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        log_prob = normal.log_prob(raw_action).sum(-1, keepdim=True)
        entropy = normal.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, mean


class PPOTeacherCritic(nn.Module):
    """Privileged value function."""

    def __init__(self, privileged_dim: int, width: int = 512, blocks: int = 4) -> None:
        super().__init__()
        self.body = ResidualMLPTrunk(OBS_DIM + privileged_dim, width, blocks)
        self.value_head = nn.Linear(width, 1)
        _orthogonal(self.value_head, 1.0)

    def forward(self, observation: torch.Tensor, privileged: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.body(torch.cat([observation, privileged], dim=-1))).float()


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

