"""Surface position and slew limits, expressed in SI units.

This is an ideal position servo after limits, not an identified hardware servo.
For DHC-6, the ARX input is da_act; adding another fitted lag would double-count
dynamics already present in the data. Limits must come from the experiment's
configuration and must not be mistaken for certified aircraft specifications.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class SurfaceActuatorConfig:
    dt_s: float = 0.02
    deflection_limit_rad: float = math.radians(10)
    rate_limit_rad_s: float = math.radians(40)

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (
                self.dt_s, self.deflection_limit_rad, self.rate_limit_rad_s)):
            raise ValueError("surface limits and period must be finite and positive")


@dataclass(frozen=True, slots=True)
class SurfaceActuatorState:
    requested_rad: float = 0.0
    commanded_rad: float = 0.0
    applied_rad: float = 0.0
    amplitude_limited: bool = False
    rate_limited: bool = False

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (
                self.requested_rad, self.commanded_rad, self.applied_rad)):
            raise ValueError("surface feedback must contain finite angles in radians")


class SurfaceActuator:
    def __init__(self, config: SurfaceActuatorConfig = SurfaceActuatorConfig()) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.state = SurfaceActuatorState()

    def step(self, requested_rad: float) -> SurfaceActuatorState:
        if not math.isfinite(requested_rad):
            raise ValueError("surface request must be finite, in radians")
        limit = self.config.deflection_limit_rad
        bounded = min(limit, max(-limit, requested_rad))
        increment = self.config.rate_limit_rad_s * self.config.dt_s
        change = bounded - self.state.commanded_rad
        commanded = self.state.commanded_rad + min(increment, max(-increment, change))
        self.state = SurfaceActuatorState(
            requested_rad=float(requested_rad), commanded_rad=commanded,
            applied_rad=commanded,
            amplitude_limited=abs(requested_rad) > limit,
            rate_limited=abs(change) > increment + 1e-14,
        )
        return self.state
