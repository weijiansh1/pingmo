"""Fixed-aircraft model-reference environment for specialist SAC Teachers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

from src.aircraft.p_channel import PChannel
from src.aircraft.sampler import PlantRecord
from src.envs.reference_model import SecondOrderReferenceConfig, SecondOrderRollRateReference
from src.envs.roll_rate_commands import RollRateCommandProfile, specialist_step_commands


@dataclass(frozen=True, slots=True)
class TrackingRewardWeights:
    tracking_error: float = 1.0
    force_energy: float = 0.02
    force_delta: float = 0.05

    def __post_init__(self) -> None:
        if min(self.tracking_error, self.force_energy, self.force_delta) < 0:
            raise ValueError("tracking reward weights must be non-negative")


class SpecialistRollRateEnv(gym.Env[np.ndarray, np.ndarray]):
    """One fixed P-channel, one direct F_as action, and no theta in Actor input."""

    metadata = {"render_modes": []}
    current_observation_names = (
        "p_command_normalized",
        "p_reference_normalized",
        "p_normalized",
        "tracking_error_normalized",
        "p_dot_normalized",
        "previous_force_normalized",
    )
    history_observation_names = (
        "p_command_normalized",
        "p_reference_normalized",
        "p_normalized",
        "tracking_error_normalized",
        "force_normalized",
    )

    def __init__(
        self,
        plant: PlantRecord,
        *,
        command_profiles: tuple[RollRateCommandProfile, ...] | None = None,
        dt_s: float = 0.001,
        history_steps: int = 250,
        command_scale_deg_s: float = 30.0,
        force_limit_n: float = 22.0,
        force_rate_limit_n_s: float = 88.0,
        actuator_time_constant_s: float = 0.0,
        roll_acceleration_scale_rad_s2: float = 5.0,
        reference_config: SecondOrderReferenceConfig = SecondOrderReferenceConfig(),
        reward_weights: TrackingRewardWeights = TrackingRewardWeights(),
    ) -> None:
        profiles = command_profiles or specialist_step_commands()
        if not profiles:
            raise ValueError("specialist environment requires at least one command")
        durations = {profile.duration_s for profile in profiles}
        if len(durations) != 1:
            raise ValueError("all specialist commands must use one episode duration")
        if min(dt_s, history_steps, command_scale_deg_s, force_limit_n, force_rate_limit_n_s, roll_acceleration_scale_rad_s2) <= 0:
            raise ValueError("specialist environment scales must be positive")
        if actuator_time_constant_s < 0:
            raise ValueError("actuator time constant cannot be negative")

        self.record = plant
        self.command_profiles = tuple(profiles)
        self.dt_s = dt_s
        self.history_steps = history_steps
        self.command_scale_rad_s = math.radians(command_scale_deg_s)
        self.force_limit_n = force_limit_n
        self.force_rate_limit_n_s = force_rate_limit_n_s
        self.max_force_increment_n = force_rate_limit_n_s * dt_s
        self.actuator_time_constant_s = actuator_time_constant_s
        self.actuator_alpha = (
            1.0
            if actuator_time_constant_s == 0
            else 1.0 - math.exp(-dt_s / actuator_time_constant_s)
        )
        self.roll_acceleration_scale_rad_s2 = roll_acceleration_scale_rad_s2
        self.reference_config = reference_config
        self.reward_weights = reward_weights
        self.episode_duration_s = next(iter(durations))
        self.horizon_steps = int(round(self.episode_duration_s / dt_s))
        if not np.isclose(self.horizon_steps * dt_s, self.episode_duration_s):
            raise ValueError("episode duration must be an integer multiple of dt_s")

        self._current_width = len(self.current_observation_names)
        self._history_width = len(self.history_observation_names)
        observation_dim = self._current_width + history_steps * self._history_width
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_dim,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        plant_state_width = PChannel(plant.parameters, dt=dt_s).state.size
        self.critic_state_dim = observation_dim + plant_state_width + 1

        self._rng = np.random.default_rng()
        self._history = np.zeros((history_steps, self._history_width), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._profile = self.command_profiles[int(self._rng.integers(len(self.command_profiles)))]
        self._command = self._profile.samples(self.dt_s)
        reference = SecondOrderRollRateReference(self.reference_config, dt_s=self.dt_s)
        self._reference = reference.rollout(self._command)
        self._plant = PChannel(self.record.parameters, dt=self.dt_s)
        self._episode_step = 0
        self._p = 0.0
        self._p_dot = 0.0
        self._commanded_force_n = 0.0
        self._applied_force_n = 0.0
        self._history[:] = self._history_row()
        self._saturation_count = 0
        self._trace: dict[str, list[float]] = {
            "time_s": [0.0],
            "p_command_rad_s": [float(self._command[0])],
            "p_reference_rad_s": [0.0],
            "p_rad_s": [0.0],
            "f_as_n": [0.0],
            "commanded_f_as_n": [0.0],
            "reward": [0.0],
            "tracking_error_cost": [0.0],
            "force_energy_cost": [0.0],
            "force_delta_cost": [0.0],
        }
        return self._observation(), self._info()

    def _current_command(self) -> float:
        return float(self._command[min(self._episode_step, self.horizon_steps - 1)])

    def _current_reference(self) -> float:
        return float(self._reference[self._episode_step])

    def _normalized_current(self) -> np.ndarray:
        reference = self._current_reference()
        error = reference - self._p
        return np.asarray(
            [
                self._current_command() / self.command_scale_rad_s,
                reference / self.command_scale_rad_s,
                self._p / self.command_scale_rad_s,
                error / self.command_scale_rad_s,
                self._p_dot / self.roll_acceleration_scale_rad_s2,
                self._applied_force_n / self.force_limit_n,
            ],
            dtype=np.float32,
        )

    def _history_row(self) -> np.ndarray:
        reference = self._current_reference()
        error = reference - self._p
        return np.asarray(
            [
                self._current_command() / self.command_scale_rad_s,
                reference / self.command_scale_rad_s,
                self._p / self.command_scale_rad_s,
                error / self.command_scale_rad_s,
                self._applied_force_n / self.force_limit_n,
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        return np.concatenate((self._normalized_current(), self._history.ravel())).astype(np.float32)

    def _critic_state(self) -> np.ndarray:
        return np.concatenate(
            (
                self._observation(),
                self._plant.state.astype(np.float32),
                np.asarray([self._episode_step / self.horizon_steps], dtype=np.float32),
            )
        ).astype(np.float32)

    def _info(self) -> dict[str, object]:
        return {
            "plant_id": self.record.plant_id,
            "command_id": self._profile.command_id,
            "critic_state": self._critic_state(),
            "f_as_n": self._applied_force_n,
            "commanded_f_as_n": self._commanded_force_n,
            "action_saturation_fraction": self._saturation_count / max(self._episode_step, 1),
            "actor_receives_theta": False,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_values = np.asarray(action, dtype=float).reshape(-1)
        if action_values.size != 1 or not np.isfinite(action_values[0]):
            raise ValueError("specialist action must contain one finite normalized force")
        requested_normalized = float(np.clip(action_values[0], -1.0, 1.0))
        requested_force_n = requested_normalized * self.force_limit_n
        previous_commanded = self._commanded_force_n
        previous_applied = self._applied_force_n
        self._commanded_force_n = float(
            np.clip(
                requested_force_n,
                previous_commanded - self.max_force_increment_n,
                previous_commanded + self.max_force_increment_n,
            )
        )
        self._applied_force_n += self.actuator_alpha * (self._commanded_force_n - self._applied_force_n)
        self._p, self._p_dot = self._plant.step(self._applied_force_n)
        self._episode_step += 1

        reference = self._current_reference()
        normalized_error = (reference - self._p) / self.command_scale_rad_s
        normalized_force = self._applied_force_n / self.force_limit_n
        normalized_delta = (self._applied_force_n - previous_applied) / self.max_force_increment_n
        costs = {
            "tracking_error": self.reward_weights.tracking_error * normalized_error**2 * self.dt_s,
            "force_energy": self.reward_weights.force_energy * normalized_force**2 * self.dt_s,
            "force_delta": self.reward_weights.force_delta * normalized_delta**2 * self.dt_s,
        }
        reward = -sum(costs.values())
        self._saturation_count += int(abs(self._commanded_force_n) >= self.force_limit_n - 1e-9)

        self._history[:-1] = self._history[1:]
        self._history[-1] = self._history_row()
        time_s = self._episode_step * self.dt_s
        self._trace["time_s"].append(time_s)
        self._trace["p_command_rad_s"].append(self._current_command())
        self._trace["p_reference_rad_s"].append(reference)
        self._trace["p_rad_s"].append(self._p)
        self._trace["f_as_n"].append(self._applied_force_n)
        self._trace["commanded_f_as_n"].append(self._commanded_force_n)
        self._trace["reward"].append(reward)
        for name, value in costs.items():
            self._trace[f"{name}_cost"].append(value)

        truncated = self._episode_step >= self.horizon_steps
        return self._observation(), float(reward), False, truncated, self._info()

    def trajectory(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(values, dtype=float).copy() for name, values in self._trace.items()}


class CommandForceBaseline:
    """Map normalized p_c directly to normalized full force for an open-loop baseline."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        values = np.asarray(observation, dtype=float)
        if values.ndim == 1:
            return np.asarray([np.clip(values[0], -1.0, 1.0)], dtype=np.float32)
        return np.clip(values[:, :1], -1.0, 1.0).astype(np.float32)
