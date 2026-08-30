"""Traditional PID baseline for the P-channel aircraft model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal

from basic.plant import Plant
from src.aircraft.delay import FractionalDelay


@dataclass(frozen=True, slots=True)
class PIDGains:
    kp: float
    ki: float
    kd: float

    def __post_init__(self) -> None:
        if min(self.kp, self.ki, self.kd) < 0:
            raise ValueError("PID gains must be non-negative")


class PIDPlant:
    """Close a constrained PID loop around ``F_as -> p``."""

    def __init__(
        self,
        plant: Plant,
        gains: PIDGains,
        *,
        dt_s: float = 0.001,
        force_limit_n: float = 22.0,
        force_rate_limit_n_s: float = 88.0,
        derivative_filter_time_constant_s: float = 0.02,
    ) -> None:
        if min(dt_s, force_limit_n, force_rate_limit_n_s) <= 0:
            raise ValueError("PID sample time and force limits must be positive")
        if derivative_filter_time_constant_s < 0:
            raise ValueError("derivative filter time constant cannot be negative")

        self.plant = plant
        self.gains = gains
        self.dt_s = float(dt_s)
        self.force_limit_n = float(force_limit_n)
        self.force_rate_limit_n_s = float(force_rate_limit_n_s)
        self.derivative_filter_time_constant_s = float(derivative_filter_time_constant_s)

        a, b, c, d = signal.tf2ss(plant.numerator, plant.denominator)
        self._ad, self._bd, self._cd, self._dd, _ = signal.cont2discrete(
            (a, b, c, d),
            self.dt_s,
            method="zoh",
        )

    def simulate_step(
        self,
        command_deg_s: float,
        duration_s: float,
    ) -> dict[str, np.ndarray]:
        """Simulate roll-rate command tracking from zero initial conditions."""

        if not np.isfinite(command_deg_s) or command_deg_s == 0:
            raise ValueError("command_deg_s must be finite and non-zero")
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        steps = int(round(duration_s / self.dt_s))
        if not np.isclose(steps * self.dt_s, duration_s):
            raise ValueError("duration_s must be an integer multiple of dt_s")

        time = np.arange(steps + 1, dtype=float) * self.dt_s
        command = np.full(steps + 1, np.deg2rad(command_deg_s), dtype=float)
        response = np.zeros(steps + 1, dtype=float)
        force = np.zeros(steps + 1, dtype=float)
        error = np.zeros(steps + 1, dtype=float)
        state = np.zeros(self._ad.shape[0], dtype=float)
        delay = FractionalDelay(self.dt_s, self.plant.tau_p)

        integral = 0.0
        filtered_p_dot = 0.0
        derivative_alpha = (
            1.0
            if self.derivative_filter_time_constant_s == 0
            else self.dt_s / (self.derivative_filter_time_constant_s + self.dt_s)
        )
        max_force_change = self.force_rate_limit_n_s * self.dt_s

        for index in range(steps):
            current_error = command[index] - response[index]
            error[index] = current_error
            raw_p_dot = 0.0 if index == 0 else (response[index] - response[index - 1]) / self.dt_s
            filtered_p_dot += derivative_alpha * (raw_p_dot - filtered_p_dot)

            candidate_integral = integral + current_error * self.dt_s
            candidate_force = (
                self.gains.kp * current_error
                + self.gains.ki * candidate_integral
                - self.gains.kd * filtered_p_dot
            )
            limited_target = float(np.clip(candidate_force, -self.force_limit_n, self.force_limit_n))
            limited_force = float(
                np.clip(
                    limited_target,
                    force[index] - max_force_change,
                    force[index] + max_force_change,
                )
            )

            drives_further_into_limit = (
                candidate_force > limited_force and current_error > 0
            ) or (
                candidate_force < limited_force and current_error < 0
            )
            if not drives_further_into_limit:
                integral = candidate_integral
            force[index + 1] = limited_force

            delayed_force = delay.push(limited_force)
            next_response = float(
                (self._cd @ state + self._dd.squeeze() * delayed_force).item()
            )
            state = self._ad @ state + self._bd[:, 0] * delayed_force
            response[index + 1] = next_response

        error[-1] = command[-1] - response[-1]
        return {
            "time_s": time,
            "p_command_rad_s": command,
            "p_rad_s": response,
            "error_rad_s": error,
            "f_as_n": force,
        }


def pid_tracking_cost(trace: dict[str, np.ndarray], force_limit_n: float) -> float:
    """Time-domain tuning cost with tracking as the primary objective."""

    time = trace["time_s"]
    command = trace["p_command_rad_s"]
    response = trace["p_rad_s"]
    force = trace["f_as_n"]
    target = abs(float(command[0]))
    direction = np.sign(command[0])
    normalized_error = (command - response) / target
    signed_response = direction * response / target
    tail = normalized_error[int(0.8 * len(normalized_error)) :]
    overshoot = max(0.0, float(np.max(signed_response)) - 1.0)
    wrong_way = max(0.0, -float(np.min(signed_response)))

    mean_squared_error = float(np.trapezoid(normalized_error**2, time) / time[-1])
    mean_absolute_error = float(np.trapezoid(np.abs(normalized_error), time) / time[-1])
    tail_error = float(np.mean(tail**2))
    force_energy = float(np.mean((force / force_limit_n) ** 2))
    force_variation = float(np.sum(np.abs(np.diff(force))) / (2.0 * force_limit_n))
    saturation = float(np.mean(np.abs(force) >= 0.999 * force_limit_n))
    return (
        mean_squared_error
        + 0.25 * mean_absolute_error
        + 4.0 * tail_error
        + 3.0 * overshoot**2
        + 2.0 * wrong_way**2
        + 0.02 * force_energy
        + 0.002 * force_variation
        + saturation
    )


def tune_pid(
    plant: Plant,
    *,
    command_deg_s: float = 15.0,
    duration_s: float = 10.0,
    dt_s: float = 0.005,
    force_limit_n: float = 22.0,
    force_rate_limit_n_s: float = 88.0,
    seed: int = 20260828,
    max_iterations: int = 24,
) -> tuple[PIDGains, float]:
    """Tune positive PID gains using a bounded time-domain search."""

    def objective(log_gains: np.ndarray) -> float:
        values = np.power(10.0, log_gains)
        gains = PIDGains(float(values[0]), float(values[1]), float(values[2]))
        loop = PIDPlant(
            plant,
            gains,
            dt_s=dt_s,
            force_limit_n=force_limit_n,
            force_rate_limit_n_s=force_rate_limit_n_s,
        )
        return pid_tracking_cost(loop.simulate_step(command_deg_s, duration_s), force_limit_n)

    result = optimize.differential_evolution(
        objective,
        bounds=((-2.0, 2.0), (-3.0, 2.5), (-4.0, 1.5)),
        seed=seed,
        maxiter=max_iterations,
        popsize=8,
        polish=True,
        workers=1,
    )
    tuned = np.power(10.0, result.x)
    return PIDGains(float(tuned[0]), float(tuned[1]), float(tuned[2])), float(result.fun)
