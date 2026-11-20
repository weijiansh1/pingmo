"""Causal oracle projection onto the same augmentation force/rate limits as SAC."""

import numpy as np

from src.aircraft.parameters import PChannelParameters
from src.aircraft.reference import DutchRollOracle


class ConstrainedDutchRollOracle:
    def __init__(self, parameters: PChannelParameters, augmentation_limit: float, normalized_rate_limit_s_inv: float, dt: float = 0.02) -> None:
        if augmentation_limit <= 0 or normalized_rate_limit_s_inv <= 0 or dt <= 0:
            raise ValueError("augmentation_limit, normalized_rate_limit_s_inv, and dt must be positive")
        self.augmentation_limit = augmentation_limit
        self.max_increment = augmentation_limit * normalized_rate_limit_s_inv * dt
        self.oracle = DutchRollOracle(parameters, dt=dt)
        self.previous_correction = 0.0

    def reset(self) -> None:
        self.oracle.reset()
        self.previous_correction = 0.0

    def step(self, pilot_force: float) -> float:
        desired = self.oracle.step(pilot_force) - pilot_force
        rate_limited = float(np.clip(desired, self.previous_correction - self.max_increment, self.previous_correction + self.max_increment))
        self.previous_correction = float(np.clip(rate_limited, -self.augmentation_limit, self.augmentation_limit))
        return self.previous_correction
