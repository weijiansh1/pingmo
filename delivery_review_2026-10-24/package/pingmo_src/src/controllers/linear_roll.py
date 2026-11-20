"""Fixed linear augmentation baseline with the same normalized action interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LinearRollPolicy:
    """A train-split-tunable force/feedforward and rate/acceleration feedback law."""

    force_gain: float = 0.15
    roll_rate_gain: float = 0.20
    roll_acceleration_gain: float = 0.10

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        values = np.asarray(observation, dtype=float)
        if values.ndim != 1 or values.size < 3:
            raise ValueError("linear roll policy expects a one-dimensional environment observation")
        force_normalized, roll_rate, roll_acceleration_normalized = values[:3]
        action = (
            self.force_gain * force_normalized
            - self.roll_rate_gain * roll_rate
            - self.roll_acceleration_gain * roll_acceleration_normalized
        )
        return np.array([np.clip(action, -1.0, 1.0)], dtype=np.float32), None
