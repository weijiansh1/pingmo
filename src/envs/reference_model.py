"""Command-to-roll-rate reference dynamics for specialist Teachers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True, slots=True)
class SecondOrderReferenceConfig:
    natural_frequency_rad_s: float = 2.0
    damping_ratio: float = 0.7

    def __post_init__(self) -> None:
        if self.natural_frequency_rad_s <= 0 or self.damping_ratio <= 0:
            raise ValueError("reference natural frequency and damping must be positive")


class SecondOrderRollRateReference:
    """Unity-DC-gain second-order reference model discretized with ZOH."""

    def __init__(
        self,
        config: SecondOrderReferenceConfig = SecondOrderReferenceConfig(),
        *,
        dt_s: float = 0.001,
    ) -> None:
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        omega = config.natural_frequency_rad_s
        numerator = np.array([omega**2], dtype=float)
        denominator = np.array([1.0, 2.0 * config.damping_ratio * omega, omega**2], dtype=float)
        a, b, c, d = signal.tf2ss(numerator, denominator)
        self._ad, self._bd, self._cd, self._dd, _ = signal.cont2discrete(
            (a, b, c, d),
            dt_s,
            method="zoh",
        )
        self.config = config
        self.dt_s = dt_s
        self._state = np.zeros(self._ad.shape[0], dtype=float)

    def reset(self) -> None:
        self._state.fill(0.0)

    def step(self, command_rad_s: float) -> float:
        output = float((self._cd @ self._state + self._dd.squeeze() * command_rad_s).item())
        self._state = self._ad @ self._state + self._bd[:, 0] * command_rad_s
        return output

    def rollout(self, command_rad_s: np.ndarray) -> np.ndarray:
        """Return the initial output followed by one reference sample per command."""

        command = np.asarray(command_rad_s, dtype=float)
        if command.ndim != 1 or not len(command):
            raise ValueError("reference command must be a non-empty one-dimensional array")
        self.reset()
        response = np.empty(len(command) + 1, dtype=float)
        response[0] = 0.0
        for index, value in enumerate(command):
            response[index + 1] = self.step(float(value))
        return response

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()
