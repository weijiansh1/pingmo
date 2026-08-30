"""No-reference MoE-TD3 residual damper for the basic P-channel study.

The controller does not receive transfer-function parameters or a reference-model
trajectory.  A fixed pilot-force command is preserved and the learned policy adds
only a bounded residual force.  Causal filters turn measured roll rate into an
online oscillation signal for both the observation and reward.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from scipy import signal
import torch
from torch import nn
from torch.nn import functional as F

from basic.plant import Plant
from src.aircraft.delay import FractionalDelay


@dataclass(frozen=True, slots=True)
class DampingConfig:
    plant_dt_s: float = 0.001
    control_dt_s: float = 0.020
    duration_s: float = 10.0
    base_force_n: float = 3.0
    force_limit_n: float = 22.0
    residual_limit_n: float = 6.6
    residual_rate_limit_n_s: float = 88.0
    p_scale_rad_s: float = float(np.deg2rad(15.0))
    filter_centers_rad_s: tuple[float, ...] = (0.4, 0.7, 1.2, 2.0, 3.5, 6.0)
    safety_filter_centers_rad_s: tuple[float, ...] = (10.0, 18.0, 30.0)
    filter_damping: float = 0.45
    energy_time_constant_s: float = 0.25
    slow_time_constant_s: float = 1.0
    oscillation_warmup_s: float = 0.25
    oscillation_weight: float = 4.0
    high_frequency_weight: float = 8.0
    growth_weight: float = 2.0
    intent_weight: float = 1.0
    effort_weight: float = 0.02
    variation_weight: float = 0.20
    failure_limit_rad_s: float = float(np.deg2rad(120.0))
    failure_penalty: float = 25.0

    def __post_init__(self) -> None:
        positive = (
            self.plant_dt_s,
            self.control_dt_s,
            self.duration_s,
            self.force_limit_n,
            self.residual_limit_n,
            self.residual_rate_limit_n_s,
            self.p_scale_rad_s,
            self.filter_damping,
            self.energy_time_constant_s,
            self.slow_time_constant_s,
            self.failure_limit_rad_s,
        )
        if min(positive) <= 0:
            raise ValueError("time, scale, filter, and force settings must be positive")
        all_filter_centers = (
            self.filter_centers_rad_s + self.safety_filter_centers_rad_s
        )
        if not all_filter_centers or min(all_filter_centers) <= 0:
            raise ValueError("filter centers must contain positive frequencies")
        if self.residual_limit_n > self.force_limit_n:
            raise ValueError("residual force limit cannot exceed total force limit")
        ratio = self.control_dt_s / self.plant_dt_s
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("control_dt_s must be an integer multiple of plant_dt_s")
        if self.oscillation_warmup_s < 0:
            raise ValueError("oscillation warmup cannot be negative")
        weights = (
            self.oscillation_weight,
            self.high_frequency_weight,
            self.growth_weight,
            self.intent_weight,
            self.effort_weight,
            self.variation_weight,
            self.failure_penalty,
        )
        if min(weights) < 0:
            raise ValueError("reward weights cannot be negative")


class CausalOscillationFilterBank:
    """Second-order band-pass filters plus causal energy estimates."""

    def __init__(
        self,
        dt_s: float,
        centers_rad_s: Sequence[float],
        damping: float,
        energy_time_constant_s: float,
    ) -> None:
        self.dt_s = float(dt_s)
        self.centers_rad_s = np.asarray(centers_rad_s, dtype=float)
        if self.dt_s <= 0 or damping <= 0 or energy_time_constant_s <= 0:
            raise ValueError("filter time and damping values must be positive")
        if self.centers_rad_s.ndim != 1 or np.any(self.centers_rad_s <= 0):
            raise ValueError(
                "centers_rad_s must be a positive one-dimensional sequence"
            )

        matrices: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for omega in self.centers_rad_s:
            numerator = [2.0 * damping * omega, 0.0]
            denominator = [1.0, 2.0 * damping * omega, omega**2]
            a, b, c, d = signal.tf2ss(numerator, denominator)
            ad, bd, cd, dd, _ = signal.cont2discrete(
                (a, b, c, d), self.dt_s, method="zoh"
            )
            matrices.append((ad, bd[:, 0], cd[0], dd.reshape(())))

        self._ad = np.stack([item[0] for item in matrices])
        self._bd = np.stack([item[1] for item in matrices])
        self._cd = np.stack([item[2] for item in matrices])
        self._dd = np.asarray([item[3] for item in matrices], dtype=float)
        self._energy_alpha = 1.0 - np.exp(-self.dt_s / energy_time_constant_s)
        self.reset()

    @property
    def size(self) -> int:
        return int(self.centers_rad_s.size)

    def reset(self) -> None:
        self.states = np.zeros((self.size, 2), dtype=float)
        self.outputs = np.zeros(self.size, dtype=float)
        self.energies = np.zeros(self.size, dtype=float)

    def update(self, value: float, scale: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(value) or scale <= 0:
            raise ValueError("filter input must be finite and scale must be positive")
        self.states = np.einsum("nij,nj->ni", self._ad, self.states) + self._bd * value
        self.outputs = np.einsum("ni,ni->n", self._cd, self.states) + self._dd * value
        normalized_square = np.square(self.outputs / scale)
        self.energies += self._energy_alpha * (normalized_square - self.energies)
        return self.outputs.copy(), self.energies.copy()


class TemporalObservation:
    """Build a fixed causal history without exposing plant parameters."""

    base_observation_dim = 20
    history_length = 256
    history_indices = (0, 1, 6, 7)
    history_channels = len(history_indices)
    observation_dim = base_observation_dim + history_length * history_channels

    def __init__(self) -> None:
        self._history = np.zeros(
            (self.history_length, self.history_channels), dtype=np.float32
        )

    def reset(self, observation: np.ndarray) -> np.ndarray:
        self._history.fill(0.0)
        return self.append(observation)

    def append(self, observation: np.ndarray) -> np.ndarray:
        current = np.asarray(observation, dtype=np.float32)
        if current.shape != (self.base_observation_dim,):
            raise ValueError(
                f"base observation must have shape ({self.base_observation_dim},)"
            )
        self._history[:-1] = self._history[1:]
        self._history[-1] = current[list(self.history_indices)]
        return np.concatenate((current, self._history.reshape(-1))).astype(
            np.float32, copy=False
        )


class _CausalTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = self.convolution(F.pad(values, (self.left_padding, 0)))
        hidden = self.pointwise(self.activation(hidden))
        return values + 0.1 * hidden


class CausalTCNEncoder(nn.Module):
    """Encode 5.12 s of measured action-response history into a latent context."""

    def __init__(
        self,
        input_channels: int = TemporalObservation.history_channels,
        channels: int = 32,
        context_dim: int = 32,
        layers: int = 8,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_channels, channels, 1)
        self.blocks = nn.ModuleList(
            _CausalTCNBlock(channels, 2**index) for index in range(layers)
        )
        self.output = nn.Linear(channels, context_dim)

    def encode_sequence(self, history: torch.Tensor) -> torch.Tensor:
        hidden = self.input_projection(history)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return self.output(self.encode_sequence(history)[:, :, -1])


def split_temporal_observation(
    observation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if observation.shape[-1] != TemporalObservation.observation_dim:
        raise ValueError(
            f"temporal observation must end in {TemporalObservation.observation_dim} values"
        )
    current = observation[:, : TemporalObservation.base_observation_dim]
    history = observation[:, TemporalObservation.base_observation_dim :].reshape(
        observation.shape[0],
        TemporalObservation.history_length,
        TemporalObservation.history_channels,
    )
    return current, history.transpose(1, 2)


class ResidualDampingEnv:
    """Short-horizon plant simulation driven by a residual damping action.

    The normalized action is the target residual force.  The pilot's base force is
    never rate-limited or replaced, so an all-zero policy exactly means "raw plant".
    """

    observation_dim = 20
    action_dim = 1

    def __init__(self, plant: Plant, config: DampingConfig | None = None) -> None:
        self.plant = plant
        self.config = DampingConfig() if config is None else config
        self._substeps = int(round(self.config.control_dt_s / self.config.plant_dt_s))
        a, b, c, d = signal.tf2ss(plant.numerator, plant.denominator)
        self._ad, self._bd, self._cd, self._dd, _ = signal.cont2discrete(
            (a, b, c, d), self.config.plant_dt_s, method="zoh"
        )
        self._filter_bank = CausalOscillationFilterBank(
            self.config.plant_dt_s,
            self.config.filter_centers_rad_s,
            self.config.filter_damping,
            self.config.energy_time_constant_s,
        )
        self._safety_filter_bank = CausalOscillationFilterBank(
            self.config.plant_dt_s,
            self.config.safety_filter_centers_rad_s,
            self.config.filter_damping,
            self.config.energy_time_constant_s,
        )
        expected_dim = 8 + 2 * self._filter_bank.size
        if expected_dim != self.observation_dim:
            raise ValueError(
                f"the observation contract expects six filters, got {self._filter_bank.size}"
            )
        self._slow_alpha = 1.0 - np.exp(
            -self.config.plant_dt_s / self.config.slow_time_constant_s
        )
        self.reset()

    def reset(self, *, base_force_n: float | None = None) -> np.ndarray:
        base_force = (
            self.config.base_force_n if base_force_n is None else float(base_force_n)
        )
        if not np.isfinite(base_force) or abs(base_force) > self.config.force_limit_n:
            raise ValueError(
                "base force must be finite and inside the total force limit"
            )
        self._base_force_n = base_force
        self._state = np.zeros(self._ad.shape[0], dtype=float)
        self._delay = FractionalDelay(self.config.plant_dt_s, self.plant.tau_p)
        self._filter_bank.reset()
        self._safety_filter_bank.reset()
        self._time_s = 0.0
        self._p_rad_s = 0.0
        self._p_dot_rad_s2 = 0.0
        self._p_slow_rad_s = 0.0
        self._residual_force_n = 0.0
        self._residual_slow_n = 0.0
        self._previous_action = 0.0
        self._previous_energy = 0.0
        return self._observation()

    @property
    def time_s(self) -> float:
        return self._time_s

    def _observation(self) -> np.ndarray:
        config = self.config
        normalized_filters = self._filter_bank.outputs / config.p_scale_rad_s
        features = np.concatenate(
            (
                np.array(
                    [
                        self._p_rad_s / config.p_scale_rad_s,
                        self._p_dot_rad_s2
                        / (config.p_scale_rad_s * max(config.filter_centers_rad_s)),
                        self._p_slow_rad_s / config.p_scale_rad_s,
                        self._residual_slow_n / max(abs(self._base_force_n), 1.0),
                        self._base_force_n / config.force_limit_n,
                        self._residual_force_n / config.residual_limit_n,
                        (self._base_force_n + self._residual_force_n)
                        / config.force_limit_n,
                        self._previous_action,
                    ],
                    dtype=float,
                ),
                normalized_filters,
                self._filter_bank.energies,
            )
        )
        return np.clip(features, -20.0, 20.0).astype(np.float32)

    def step(
        self, action: float | np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        scalar_action = float(np.asarray(action, dtype=float).reshape(-1)[0])
        if not np.isfinite(scalar_action):
            raise ValueError("action must be finite")
        scalar_action = float(np.clip(scalar_action, -1.0, 1.0))
        target_residual = scalar_action * self.config.residual_limit_n
        target_residual = float(
            np.clip(
                target_residual,
                -self.config.force_limit_n - self._base_force_n,
                self.config.force_limit_n - self._base_force_n,
            )
        )
        max_substep_change = (
            self.config.residual_rate_limit_n_s * self.config.plant_dt_s
        )
        previous_control_p = self._p_rad_s

        for _ in range(self._substeps):
            difference = target_residual - self._residual_force_n
            self._residual_force_n += float(
                np.clip(difference, -max_substep_change, max_substep_change)
            )
            total_force = self._base_force_n + self._residual_force_n
            delayed_force = self._delay.push(total_force)
            self._state = self._ad @ self._state + self._bd[:, 0] * delayed_force
            self._p_rad_s = float(
                (self._cd @ self._state + self._dd.squeeze() * delayed_force).item()
            )
            self._p_slow_rad_s += self._slow_alpha * (
                self._p_rad_s - self._p_slow_rad_s
            )
            self._residual_slow_n += self._slow_alpha * (
                self._residual_force_n - self._residual_slow_n
            )
            self._filter_bank.update(self._p_rad_s, self.config.p_scale_rad_s)
            self._safety_filter_bank.update(self._p_rad_s, self.config.p_scale_rad_s)

        self._time_s += self.config.control_dt_s
        self._p_dot_rad_s2 = (
            self._p_rad_s - previous_control_p
        ) / self.config.control_dt_s
        primary_energy = float(np.mean(self._filter_bank.energies))
        high_frequency_energy = float(np.mean(self._safety_filter_bank.energies))
        broadband_energy = primary_energy + high_frequency_energy
        energy_growth = max(0.0, broadband_energy - self._previous_energy)
        gate = float(
            np.clip(
                (self._time_s - self.config.oscillation_warmup_s)
                / max(self.config.oscillation_warmup_s, self.config.control_dt_s),
                0.0,
                1.0,
            )
        )
        intent_cost = (self._residual_slow_n / max(abs(self._base_force_n), 1.0)) ** 2
        effort_cost = (self._residual_force_n / self.config.residual_limit_n) ** 2
        variation_cost = (scalar_action - self._previous_action) ** 2
        oscillation_cost = gate * primary_energy
        high_frequency_cost = gate * high_frequency_energy
        growth_cost = gate * energy_growth
        total_cost = (
            self.config.oscillation_weight * oscillation_cost
            + self.config.high_frequency_weight * high_frequency_cost
            + self.config.growth_weight * growth_cost
            + self.config.intent_weight * intent_cost
            + self.config.effort_weight * effort_cost
            + self.config.variation_weight * variation_cost
        )
        failed = abs(self._p_rad_s) > self.config.failure_limit_rad_s
        reward = -total_cost - (self.config.failure_penalty if failed else 0.0)
        done = failed or self._time_s >= self.config.duration_s - 1e-12
        self._previous_action = scalar_action
        self._previous_energy = broadband_energy
        observation = self._observation()
        info = {
            "p_rad_s": self._p_rad_s,
            "base_force_n": self._base_force_n,
            "residual_force_n": self._residual_force_n,
            "total_force_n": self._base_force_n + self._residual_force_n,
            "oscillation_energy": broadband_energy,
            "primary_energy": primary_energy,
            "high_frequency_energy": high_frequency_energy,
            "oscillation_cost": oscillation_cost,
            "high_frequency_cost": high_frequency_cost,
            "growth_cost": growth_cost,
            "intent_cost": intent_cost,
            "effort_cost": effort_cost,
            "variation_cost": variation_cost,
            "failed": float(failed),
        }
        return observation, float(reward), done, info

    def rollout(
        self,
        policy: Callable[[np.ndarray], float | np.ndarray] | None = None,
        *,
        base_force_n: float | None = None,
    ) -> dict[str, np.ndarray]:
        observation = self.reset(base_force_n=base_force_n)
        records: dict[str, list[float]] = {
            "time_s": [0.0],
            "p_rad_s": [0.0],
            "base_force_n": [self._base_force_n],
            "residual_force_n": [0.0],
            "total_force_n": [self._base_force_n],
            "oscillation_energy": [0.0],
            "primary_energy": [0.0],
            "high_frequency_energy": [0.0],
            "reward": [0.0],
        }
        done = False
        while not done:
            action = 0.0 if policy is None else policy(observation.copy())
            observation, reward, done, info = self.step(action)
            records["time_s"].append(self._time_s)
            records["p_rad_s"].append(info["p_rad_s"])
            records["base_force_n"].append(info["base_force_n"])
            records["residual_force_n"].append(info["residual_force_n"])
            records["total_force_n"].append(info["total_force_n"])
            records["oscillation_energy"].append(info["oscillation_energy"])
            records["primary_energy"].append(info["primary_energy"])
            records["high_frequency_energy"].append(info["high_frequency_energy"])
            records["reward"].append(reward)
        return {key: np.asarray(value, dtype=float) for key, value in records.items()}


@dataclass(frozen=True, slots=True)
class PlantCase:
    plant_id: str
    gjb_level: int
    dutch_roll_level: int
    plant: Plant


def load_balanced_plant_cases(
    path: str | Path,
    *,
    per_level: int | None = None,
    seed: int = 20260828,
    split_prefix: str = "train",
    balance_component: str | None = "dutch_roll",
) -> list[PlantCase]:
    """Load L1/L2/L3 plants balanced by a selected GJB component."""

    groups: dict[int, list[PlantCase]] = {1: [], 2: [], 3: []}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("sampling_bucket") == "OOD":
                continue
            component_levels = row.get("gjb_component_levels", {})
            level = (
                row.get("gjb_level")
                if balance_component is None
                else component_levels.get(balance_component)
            )
            if level not in groups or not str(row.get("split", "")).startswith(
                split_prefix
            ):
                continue
            parameters = row["parameters"]
            groups[level].append(
                PlantCase(
                    plant_id=str(row["plant_id"]),
                    gjb_level=int(row["gjb_level"]),
                    dutch_roll_level=int(component_levels["dutch_roll"]),
                    plant=Plant(
                        parameters["l_fa"],
                        parameters["lambda_s"],
                        parameters["t_r"],
                        parameters["zeta_d"],
                        parameters["omega_d"],
                        parameters["r_omega"],
                        parameters["r_zeta"],
                        parameters["tau_p"],
                    ),
                )
            )
    if any(not cases for cases in groups.values()):
        raise ValueError("plant bank must contain cases for all three selected levels")

    available = min(len(cases) for cases in groups.values())
    count = available if per_level is None else min(per_level, available)
    if count <= 0:
        raise ValueError("per_level must be positive")
    rng = np.random.default_rng(seed)
    selected: list[PlantCase] = []
    for level in (1, 2, 3):
        indices = rng.choice(len(groups[level]), size=count, replace=False)
        selected.extend(groups[level][int(index)] for index in indices)
    return selected


class SparseMoEActor(nn.Module):
    """Four-expert actor with a causal TCN and loss-free top-2 routing."""

    def __init__(
        self,
        observation_dim: int = TemporalObservation.observation_dim,
        action_dim: int = ResidualDampingEnv.action_dim,
        *,
        expert_count: int = 4,
        shared_width: int = 512,
        expert_width: int = 512,
        expert_bottleneck: int = 256,
        router_width: int = 128,
        top_k: int = 2,
        context_dim: int = 32,
        router_noise_std: float = 0.0,
    ) -> None:
        super().__init__()
        if not 1 <= top_k <= expert_count:
            raise ValueError("top_k must be between one and expert_count")
        if observation_dim != TemporalObservation.observation_dim:
            raise ValueError("SparseMoEActor requires the causal temporal observation")
        self.expert_count = expert_count
        self.top_k = top_k
        self.router_noise_std = router_noise_std
        if self.router_noise_std < 0:
            raise ValueError("router_noise_std cannot be negative")
        self.temporal_encoder = CausalTCNEncoder(context_dim=context_dim)
        controller_input_dim = TemporalObservation.base_observation_dim + context_dim
        self.shared = nn.Sequential(
            nn.Linear(controller_input_dim, shared_width),
            nn.SiLU(),
            nn.Linear(shared_width, expert_width),
            nn.SiLU(),
        )
        self.router = nn.Sequential(
            nn.Linear(controller_input_dim, router_width),
            nn.SiLU(),
            nn.Linear(router_width, expert_count),
        )
        nn.init.orthogonal_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)
        self.register_buffer("routing_bias", torch.zeros(expert_count))
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(expert_width, expert_width),
                nn.SiLU(),
                nn.Linear(expert_width, expert_bottleneck),
                nn.SiLU(),
                nn.Linear(expert_bottleneck, action_dim),
            )
            for _ in range(expert_count)
        )
        for expert in self.experts:
            output = expert[-1]
            nn.init.uniform_(output.weight, -3e-3, 3e-3)
            nn.init.uniform_(output.bias, -3e-3, 3e-3)

    def forward(
        self,
        observation: torch.Tensor,
        *,
        noisy_routing: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        current, history = split_temporal_observation(observation)
        context = self.temporal_encoder(history)
        controller_input = torch.cat((current, context), dim=-1)
        features = self.shared(controller_input)
        router_logits = self.router(controller_input).float()
        dense_weights = torch.softmax(router_logits, dim=-1)
        selection_scores = router_logits
        if noisy_routing and self.router_noise_std > 0:
            selection_scores = selection_scores + torch.randn_like(selection_scores) * (
                self.router_noise_std
            )
        selection_scores = selection_scores + self.routing_bias
        _, top_indices = torch.topk(selection_scores, self.top_k, dim=-1)
        top_values = torch.gather(dense_weights, -1, top_indices)
        sparse_weights = torch.zeros_like(dense_weights).scatter(
            -1, top_indices, top_values
        )
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True)
        expert_actions = torch.stack(
            [expert(features) for expert in self.experts], dim=1
        )
        mixed_action = (sparse_weights.unsqueeze(-1) * expert_actions).sum(dim=1)
        return torch.tanh(mixed_action), sparse_weights, dense_weights, router_logits

    @torch.no_grad()
    def update_routing_bias(
        self, sparse_weights: torch.Tensor, update_rate: float
    ) -> None:
        """DeepSeek loss-free balancing update from the previous mini-batch."""

        if update_rate <= 0:
            return
        counts = sparse_weights.gt(0).sum(dim=0).to(self.routing_bias.dtype)
        error = counts.mean() - counts
        self.routing_bias.add_(update_rate * torch.sign(error))
        self.routing_bias.sub_(self.routing_bias.mean())

    @staticmethod
    def load_balance_loss(
        dense_weights: torch.Tensor, sparse_weights: torch.Tensor
    ) -> torch.Tensor:
        """Switch-style loss coupling soft importance to actual Top-k dispatch."""

        selected = sparse_weights.detach().gt(0).to(dense_weights.dtype)
        load_fraction = selected.sum(dim=0)
        load_fraction = load_fraction / load_fraction.sum().clamp_min(1.0)
        importance = dense_weights.mean(dim=0)
        return dense_weights.shape[-1] * torch.sum(load_fraction * importance)

    @staticmethod
    def router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
        """ST-MoE z-loss prevents the router logits from growing without bound."""

        return torch.logsumexp(router_logits, dim=-1).square().mean()

    def simbal_loss(self) -> torch.Tensor:
        """Keep the final router projection close to row-orthonormal."""

        weight = self.router[-1].weight.float()
        identity = torch.eye(weight.shape[0], device=weight.device)
        return (weight @ weight.T - identity).abs().sum()

    @staticmethod
    def dispatch_cv_squared(sparse_weights: torch.Tensor) -> torch.Tensor:
        """Non-differentiable diagnostic of actual expert-selection imbalance."""

        load = sparse_weights.detach().gt(0).float().mean(dim=0)
        return load.var(unbiased=False) / load.mean().square().clamp_min(1e-8)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class _QNetwork(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        width: int,
        context_dim: int = 32,
    ) -> None:
        super().__init__()
        if observation_dim != TemporalObservation.observation_dim:
            raise ValueError("Q network requires the causal temporal observation")
        self.temporal_encoder = CausalTCNEncoder(context_dim=context_dim)
        critic_input_dim = (
            TemporalObservation.base_observation_dim + context_dim + action_dim
        )
        self.network = nn.Sequential(
            nn.Linear(critic_input_dim, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        current, history = split_temporal_observation(observation)
        context = self.temporal_encoder(history)
        return self.network(torch.cat((current, context, action), dim=-1))


class TwinQCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int = TemporalObservation.observation_dim,
        action_dim: int = ResidualDampingEnv.action_dim,
        *,
        width: int = 1147,
    ) -> None:
        super().__init__()
        self.q1 = _QNetwork(observation_dim, action_dim, width)
        self.q2 = _QNetwork(observation_dim, action_dim, width)

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action), self.q2(observation, action)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class MoETD3:
    """TD3 learner using a sparse deterministic MoE actor."""

    def __init__(
        self,
        *,
        observation_dim: int = TemporalObservation.observation_dim,
        action_dim: int = ResidualDampingEnv.action_dim,
        gamma: float = 0.995,
        tau: float = 0.005,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        balance_coefficient: float = 0.0,
        simbal_coefficient: float = 0.1,
        router_z_coefficient: float = 0.001,
        router_noise_std: float = 0.0,
        routing_bias_update_rate: float = 0.001,
        actor_learning_rate: float = 1e-4,
        critic_learning_rate: float = 3e-4,
        gradient_norm_limit: float = 10.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if not 0 < gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("gamma and tau must be in (0, 1]")
        if policy_delay <= 0 or min(actor_learning_rate, critic_learning_rate) <= 0:
            raise ValueError("optimizer settings must be positive")
        if (
            balance_coefficient < 0
            or simbal_coefficient < 0
            or router_z_coefficient < 0
            or routing_bias_update_rate < 0
        ):
            raise ValueError("router loss coefficients cannot be negative")
        self.device = torch.device(device)
        self.actor = SparseMoEActor(
            observation_dim,
            action_dim,
            router_noise_std=router_noise_std,
        ).to(self.device)
        self.critic = TwinQCritic(observation_dim, action_dim).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device).eval()
        self.target_critic = deepcopy(self.critic).to(self.device).eval()
        self.target_actor.requires_grad_(False)
        self.target_critic.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_learning_rate
        )
        self.gamma = gamma
        self.tau = tau
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.balance_coefficient = balance_coefficient
        self.simbal_coefficient = simbal_coefficient
        self.router_z_coefficient = router_z_coefficient
        self.routing_bias_update_rate = routing_bias_update_rate
        self.gradient_norm_limit = gradient_norm_limit
        self._update_count = 0

    def act(
        self,
        observation: np.ndarray,
        *,
        exploration_std: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32)[None, :], device=self.device
        )
        with torch.no_grad():
            action = (
                self.actor(tensor, noisy_routing=exploration_std > 0)[0][0]
                .cpu()
                .numpy()
            )
        if exploration_std > 0:
            generator = np.random.default_rng() if rng is None else rng
            action = action + generator.normal(0.0, exploration_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        observation, action, reward, next_observation, done = (
            batch[key].to(self.device)
            for key in ("observation", "action", "reward", "next_observation", "done")
        )
        with torch.no_grad():
            next_action = self.target_actor(next_observation, noisy_routing=False)[0]
            noise = torch.randn_like(next_action) * self.target_policy_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.target_critic(next_observation, next_action)
            target = reward + self.gamma * (1.0 - done) * torch.minimum(
                target_q1, target_q2
            )

        q1, q2 = self.critic(observation, action)
        critic_loss = (q1 - target).square().mean() + (q2 - target).square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.gradient_norm_limit
        )
        self.critic_optimizer.step()

        self._update_count += 1
        actor_loss_value = 0.0
        balance_value = 0.0
        simbal_value = 0.0
        router_z_value = 0.0
        dispatch_cv_squared_value = 0.0
        router_entropy_value = 0.0
        if self._update_count % self.policy_delay == 0:
            policy_action, sparse_weights, dense_weights, router_logits = self.actor(
                observation, noisy_routing=True
            )
            self.critic.requires_grad_(False)
            try:
                actor_q = self.critic.q1(observation, policy_action)
                balance_loss = self.actor.load_balance_loss(
                    dense_weights, sparse_weights
                )
                simbal_loss = self.actor.simbal_loss()
                router_z_loss = self.actor.router_z_loss(router_logits)
                actor_loss = (
                    -actor_q.mean()
                    + self.balance_coefficient * balance_loss
                    + self.simbal_coefficient * simbal_loss
                    + self.router_z_coefficient * router_z_loss
                )
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.gradient_norm_limit
                )
                self.actor_optimizer.step()
            finally:
                self.critic.requires_grad_(True)
            self.actor.update_routing_bias(
                sparse_weights, self.routing_bias_update_rate
            )
            actor_loss_value = float(actor_loss.detach())
            balance_value = float(balance_loss.detach())
            simbal_value = float(simbal_loss.detach())
            router_z_value = float(router_z_loss.detach())
            dispatch_cv_squared_value = float(
                self.actor.dispatch_cv_squared(sparse_weights).detach()
            )
            router_entropy_value = float(
                (
                    -(dense_weights * dense_weights.clamp_min(1e-8).log())
                    .sum(-1)
                    .mean()
                ).detach()
            )
            self._soft_update(self.target_actor, self.actor)
            self.target_actor.routing_bias.copy_(self.actor.routing_bias)
            self._soft_update(self.target_critic, self.critic)

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": actor_loss_value,
            "balance_loss": balance_value,
            "simbal_loss": simbal_value,
            "router_z_loss": router_z_value,
            "dispatch_cv_squared": dispatch_cv_squared_value,
            "router_entropy": router_entropy_value,
            "routing_bias_span": float(
                (self.actor.routing_bias.max() - self.actor.routing_bias.min())
                .detach()
                .cpu()
            ),
            "actor_updated": float(self._update_count % self.policy_delay == 0),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target.parameters(), source.parameters()
            ):
                target_parameter.lerp_(parameter, self.tau)

    def parameter_counts(self) -> dict[str, int]:
        actor = self.actor.parameter_count
        critics = self.critic.parameter_count
        return {
            "actor": actor,
            "twin_critics": critics,
            "trainable_online_total": actor + critics,
            "frozen_targets": actor + critics,
        }

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "target_actor": self.target_actor.state_dict(),
                "target_critic": self.target_critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_count": self._update_count,
            },
            Path(path),
        )

    def load(self, path: str | Path, *, load_optimizers: bool = False) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        actor_status = self.actor.load_state_dict(checkpoint["actor"], strict=False)
        self.critic.load_state_dict(checkpoint["critic"])
        target_actor_status = self.target_actor.load_state_dict(
            checkpoint["target_actor"], strict=False
        )
        allowed_missing = {"routing_bias"}
        for status in (actor_status, target_actor_status):
            if set(status.missing_keys) - allowed_missing or status.unexpected_keys:
                raise RuntimeError(f"incompatible actor checkpoint: {status}")
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self._update_count = int(checkpoint["update_count"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_dim: int = TemporalObservation.observation_dim,
        action_dim: int = ResidualDampingEnv.action_dim,
        *,
        seed: int = 20260828,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self.observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self._position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index, 0] = reward
        self.next_observations[index] = next_observation
        self.dones[index, 0] = done
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, device: str | torch.device = "cpu"
    ) -> dict[str, torch.Tensor]:
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than stored")
        indices = self._rng.integers(self._size, size=batch_size)
        return {
            "observation": torch.as_tensor(self.observations[indices], device=device),
            "action": torch.as_tensor(self.actions[indices], device=device),
            "reward": torch.as_tensor(self.rewards[indices], device=device),
            "next_observation": torch.as_tensor(
                self.next_observations[indices], device=device
            ),
            "done": torch.as_tensor(self.dones[indices], device=device),
        }


def train_moe_td3(
    controller: MoETD3,
    cases: Sequence[PlantCase],
    *,
    total_steps: int,
    config: DampingConfig | None = None,
    replay_capacity: int = 50_000,
    random_steps: int = 5_000,
    batch_size: int = 256,
    exploration_std: float = 0.15,
    seed: int = 20260828,
    progress_every: int = 5_000,
) -> list[dict[str, float]]:
    """Train across a balanced plant list; plant parameters never enter the policy."""

    if not cases or total_steps <= 0:
        raise ValueError("training requires plant cases and a positive step count")
    rng = np.random.default_rng(seed)
    replay = ReplayBuffer(replay_capacity, seed=seed)
    episode_returns: list[dict[str, float]] = []
    case = cases[int(rng.integers(len(cases)))]
    environment = ResidualDampingEnv(case.plant, config)
    temporal_observation = TemporalObservation()
    observation = temporal_observation.reset(
        environment.reset(
            base_force_n=environment.config.base_force_n * rng.choice((-1.0, 1.0))
        )
    )
    episode_return = 0.0
    episode_steps = 0
    latest_update: dict[str, float] = {}

    for step in range(1, total_steps + 1):
        if step <= random_steps:
            action = rng.uniform(-1.0, 1.0, size=1).astype(np.float32)
        else:
            action = controller.act(
                observation, exploration_std=exploration_std, rng=rng
            )
        next_base_observation, reward, done, info = environment.step(action)
        next_observation = temporal_observation.append(next_base_observation)
        replay.add(observation, action, reward, next_observation, done)
        observation = next_observation
        episode_return += reward
        episode_steps += 1

        if len(replay) >= batch_size and step > random_steps:
            latest_update = controller.update(
                replay.sample(batch_size, controller.device)
            )

        if done:
            episode_returns.append(
                {
                    "step": float(step),
                    "return": float(episode_return),
                    "episode_steps": float(episode_steps),
                    "gjb_level": float(case.gjb_level),
                    "dutch_roll_level": float(case.dutch_roll_level),
                    "failed": info["failed"],
                }
            )
            case = cases[int(rng.integers(len(cases)))]
            environment = ResidualDampingEnv(case.plant, config)
            observation = temporal_observation.reset(
                environment.reset(
                    base_force_n=environment.config.base_force_n
                    * rng.choice((-1.0, 1.0))
                )
            )
            episode_return = 0.0
            episode_steps = 0

        if progress_every > 0 and step % progress_every == 0:
            recent = episode_returns[-10:]
            mean_return = (
                float(np.mean([row["return"] for row in recent]))
                if recent
                else float("nan")
            )
            router_status = ""
            if latest_update:
                router_status = (
                    f" load_loss={latest_update['balance_loss']:.4f}"
                    f" simbal={latest_update['simbal_loss']:.4f}"
                    f" load_cv2={latest_update['dispatch_cv_squared']:.4f}"
                    f" router_h={latest_update['router_entropy']:.4f}"
                    f" bias_span={latest_update['routing_bias_span']:.4f}"
                )
            print(
                f"step={step}/{total_steps} replay={len(replay)} "
                f"episodes={len(episode_returns)} recent_return={mean_return:.4f}"
                f"{router_status}",
                flush=True,
            )
    return episode_returns
