"""Stateful PID baseline for the specialist roll-rate observation contract."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize


@dataclass(frozen=True, slots=True)
class PIDGains:
    """Physical gains mapping roll-rate error in radians to force in newtons."""

    proportional: float
    integral: float
    derivative: float

    def __post_init__(self) -> None:
        values = (self.proportional, self.integral, self.derivative)
        if min(values) < 0 or not np.all(np.isfinite(values)):
            raise ValueError("PID gains must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PIDTuningConfig:
    log10_bounds: tuple[tuple[float, float], ...] = (
        (-1.0, 3.0),
        (-2.0, 3.0),
        (-3.0, 2.0),
    )
    max_iterations: int = 12
    population_size: int = 6
    seed: int = 20260828

    def __post_init__(self) -> None:
        if len(self.log10_bounds) != 3:
            raise ValueError("PID tuning requires three gain bounds")
        if self.max_iterations <= 0 or self.population_size <= 0:
            raise ValueError("PID tuning iteration and population counts must be positive")
        if any(low >= high for low, high in self.log10_bounds):
            raise ValueError("PID tuning bounds must increase")


class RollRatePIDPolicy:
    """PID controller reading the specialist Actor's explicit error states.

    The derivative term acts on measured roll acceleration, avoiding derivative
    kick when ``p_ref`` changes. The environment remains responsible for the
    physical force magnitude and slew-rate limits.
    """

    def __init__(
        self,
        gains: PIDGains,
        *,
        policy_dt_s: float,
        command_scale_rad_s: float,
        integral_error_scale_rad: float,
        roll_acceleration_scale_rad_s2: float,
        force_limit_n: float,
    ) -> None:
        scales = (
            policy_dt_s,
            command_scale_rad_s,
            integral_error_scale_rad,
            roll_acceleration_scale_rad_s2,
            force_limit_n,
        )
        if min(scales) <= 0 or not np.all(np.isfinite(scales)):
            raise ValueError("PID timing, observation scales, and force limit must be positive")
        self.gains = gains
        self.policy_dt_s = float(policy_dt_s)
        self.command_scale_rad_s = float(command_scale_rad_s)
        self.integral_error_scale_rad = float(integral_error_scale_rad)
        self.roll_acceleration_scale_rad_s2 = float(
            roll_acceleration_scale_rad_s2
        )
        self.force_limit_n = float(force_limit_n)
        self.reset()

    def reset(self) -> None:
        """Keep the policy interface state-reset compatible."""

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        del deterministic
        values = np.asarray(observation, dtype=float)
        if values.ndim != 1 or values.size < 7 or not np.all(np.isfinite(values[:7])):
            raise ValueError("PID policy requires one finite specialist observation")

        error_rad_s = float(values[3]) * self.command_scale_rad_s
        integral_error_rad = float(values[4]) * self.integral_error_scale_rad
        roll_acceleration_rad_s2 = (
            float(values[5]) * self.roll_acceleration_scale_rad_s2
        )
        requested_force_n = (
            self.gains.proportional * error_rad_s
            + self.gains.integral * integral_error_rad
            - self.gains.derivative * roll_acceleration_rad_s2
        )
        limited_force_n = float(
            np.clip(requested_force_n, -self.force_limit_n, self.force_limit_n)
        )
        return np.asarray([limited_force_n / self.force_limit_n], dtype=np.float32)


def tune_pid_gains(
    objective: Callable[[PIDGains], float],
    config: PIDTuningConfig = PIDTuningConfig(),
) -> tuple[PIDGains, float]:
    """Tune positive physical PID gains against a caller-owned control objective."""

    def objective_from_log_gains(log_gains: np.ndarray) -> float:
        gains = np.power(10.0, log_gains)
        value = float(objective(PIDGains(*map(float, gains))))
        return value if np.isfinite(value) else 1e12

    result = optimize.differential_evolution(
        objective_from_log_gains,
        bounds=config.log10_bounds,
        seed=config.seed,
        maxiter=config.max_iterations,
        popsize=config.population_size,
        polish=True,
        workers=1,
    )
    tuned = np.power(10.0, result.x)
    return PIDGains(*map(float, tuned)), float(result.fun)
