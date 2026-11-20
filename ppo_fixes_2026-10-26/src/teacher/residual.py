"""Pre-normalized residual MLP building blocks for large SAC Teachers."""

from __future__ import annotations

import math

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    """A stable pre-LayerNorm residual block with an optional bottleneck."""

    def __init__(self, width: int, bottleneck_width: int | None = None, residual_scale: float = 0.1) -> None:
        super().__init__()
        hidden = width if bottleneck_width is None else bottleneck_width
        if width <= 0 or hidden <= 0 or residual_scale <= 0:
            raise ValueError("residual widths and scale must be positive")
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, hidden)
        self.activation = nn.SiLU()
        self.linear_out = nn.Linear(hidden, width)
        self.residual_scale = residual_scale
        nn.init.xavier_uniform_(self.linear_in.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.linear_in.bias)
        nn.init.xavier_uniform_(self.linear_out.weight, gain=0.01)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.linear_out(self.activation(self.linear_in(self.norm(values))))
        return values + self.residual_scale * residual


class ResidualMLPTrunk(nn.Module):
    """Project an input vector and refine it through repeated residual blocks."""

    def __init__(
        self,
        input_dim: int,
        width: int = 896,
        blocks: int = 14,
        *,
        bottleneck_width: int | None = None,
        residual_scale: float = 0.1,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or width <= 0 or blocks <= 0:
            raise ValueError("residual trunk dimensions must be positive")
        self.projection = nn.Linear(input_dim, width)
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(width, bottleneck_width, residual_scale)
            for _ in range(blocks)
        )
        self.output_norm = nn.LayerNorm(width) if output_norm else nn.Identity()
        nn.init.xavier_uniform_(self.projection.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(self.projection.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.activation(self.projection(values))
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_norm(hidden)
