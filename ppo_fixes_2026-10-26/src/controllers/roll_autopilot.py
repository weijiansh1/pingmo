"""Conventional roll-loop periphery around the frozen Pingmo rate controller.

The external bus carries bank/body rates and aileron angles in SI units. The
35-feature / 8-context network, desired-response reference and virtual force
coordinates remain inside the central controller compatibility adapter. No
policy weights or learned normalization are changed by this module.

RollStateEstimate is a boundary for an estimator, not an implemented IMU/EKF.
The separate DHC-6 reduced-model experiment supplies simulated state feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from src.aircraft.actuator import SurfaceActuator, SurfaceActuatorState
from src.envs.reference_model import SecondOrderReferenceConfig, SecondOrderRollRateReference
from src.transfer import fine_tune as legacy


@dataclass(frozen=True, slots=True)
class RollStateEstimate:
    """Euler angles and FRD body rates; positive roll lowers the right wing."""

    time_s: float
    roll_rad: float
    p_rad_s: float
    pitch_rad: float = 0.0
    q_rad_s: float = 0.0
    r_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (
                self.time_s, self.roll_rad, self.p_rad_s,
                self.pitch_rad, self.q_rad_s, self.r_rad_s)):
            raise ValueError("estimated roll state must be finite and use SI units")


@dataclass(frozen=True, slots=True)
class RollOuterLoopConfig:
    time_constant_s: float = 2.0
    bank_limit_rad: float = math.radians(30)
    body_rate_limit_rad_s: float = math.radians(30)

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (
                self.time_constant_s, self.bank_limit_rad, self.body_rate_limit_rad_s)):
            raise ValueError("outer-loop time constant and limits must be finite and positive")
        if self.bank_limit_rad >= math.pi / 2:
            raise ValueError("this conventional bank loop requires bank limit below 90 degrees")


class RollOuterLoop:
    def __init__(self, config: RollOuterLoopConfig = RollOuterLoopConfig()) -> None:
        self.config = config

    def limit_bank(self, setpoint_rad: float) -> float:
        if not math.isfinite(setpoint_rad):
            raise ValueError("bank setpoint must be finite, in radians")
        limit = self.config.bank_limit_rad
        return min(limit, max(-limit, setpoint_rad))

    def limit_rate(self, setpoint_rad_s: float) -> float:
        if not math.isfinite(setpoint_rad_s):
            raise ValueError("body-rate setpoint must be finite, in rad/s")
        limit = self.config.body_rate_limit_rad_s
        return min(limit, max(-limit, setpoint_rad_s))

    def rate_setpoint(self, bank_setpoint_rad: float, state: RollStateEstimate) -> float:
        target = self.limit_bank(bank_setpoint_rad)
        if abs(math.cos(state.pitch_rad)) < 1e-3:
            raise ValueError("Euler bank control is singular near 90 degree pitch")
        error = target - state.roll_rad
        error = math.atan2(math.sin(error), math.cos(error))
        desired_roll_dot = error / self.config.time_constant_s
        # phi_dot = p + tan(pitch) * (q sin(phi) + r cos(phi)).
        body_p = desired_roll_dot - math.tan(state.pitch_rad) * (
            state.q_rad_s * math.sin(state.roll_rad)
            + state.r_rad_s * math.cos(state.roll_rad))
        return self.limit_rate(body_p)


@dataclass(frozen=True, slots=True)
class RateControllerOutput:
    requested_aileron_rad: float
    normalized_action: float
    reference_rad_s: float
    next_reference_rad_s: float


class PingmoRateController:
    """Keep the trained rate control law intact behind a physical I/O boundary.

    `policy` is the frozen student's callable with theta cached internally (e.g.
    DenseStudentPolicy.predict). For the present DHC-6 mapping, action +1 means
    a -10 degree aileron request. The original virtual +/-22 N is a feature
    coordinate of this controller, never a command sent to a physical servo.
    """

    def __init__(self, policy: Callable[[np.ndarray], np.ndarray], *,
                 dt_s: float = 0.02, aileron_full_scale_rad: float = math.radians(10),
                 aileron_polarity: float = -1.0) -> None:
        if not math.isclose(dt_s, legacy.DT, rel_tol=0, abs_tol=1e-12):
            raise ValueError("the frozen controller requires a fixed 20 ms clock")
        if not math.isfinite(aileron_full_scale_rad) or aileron_full_scale_rad <= 0:
            raise ValueError("aileron calibration must be finite and positive")
        if aileron_polarity not in (-1.0, 1.0):
            raise ValueError("aileron polarity must be +1 or -1")
        self.policy = policy
        self.dt_s = dt_s
        self.aileron_full_scale_rad = aileron_full_scale_rad
        self.aileron_polarity = aileron_polarity
        self._reference = SecondOrderRollRateReference(
            SecondOrderReferenceConfig(), dt_s=dt_s, delay_s=legacy.TAU_P_REFERENCE_S)
        self.reset()

    def reset(self) -> None:
        self._reference.reset()
        self._ref = self._ref_dot = self._integral = 0.0
        self._previous_p = self._previous_ref = 0.0
        self._last_time: float | None = None
        self._history = np.zeros(legacy.HISTORY_STEPS, dtype=float)

    def command(self, p_setpoint_rad_s: float, state: RollStateEstimate,
                feedback: SurfaceActuatorState) -> RateControllerOutput:
        if not math.isfinite(p_setpoint_rad_s):
            raise ValueError("rate request must be finite")
        integral = self._integral
        p_dot = 0.0
        if self._last_time is not None:
            if not math.isclose(state.time_s - self._last_time, self.dt_s,
                                rel_tol=0, abs_tol=1e-7):
                raise ValueError("state time must advance on the fixed 20 ms clock")
            p_dot = (state.p_rad_s - self._previous_p) / self.dt_s
            integral += 0.5 * (
                self._previous_ref - self._previous_p + self._ref - state.p_rad_s) * self.dt_s
        force_per_rad = legacy.FORCE_LIMIT_N / (
            self.aileron_polarity * self.aileron_full_scale_rad)
        observation = legacy.build_observation(
            p_setpoint_rad_s, self._ref, self._ref_dot, state.p_rad_s, p_dot, integral,
            feedback.requested_rad * force_per_rad,
            feedback.commanded_rad * force_per_rad,
            feedback.applied_rad * force_per_rad, self._history)
        values = np.asarray(self.policy(observation), dtype=float).reshape(-1)
        if values.size != 1 or not np.isfinite(values).all():
            raise ValueError("central policy must return one finite normalized action")
        action = float(np.clip(values[0], -1.0, 1.0))
        next_ref = self._reference.step(p_setpoint_rad_s)
        result = RateControllerOutput(
            self.aileron_polarity * self.aileron_full_scale_rad * action,
            action, self._ref, next_ref)
        self._previous_p, self._previous_ref = state.p_rad_s, self._ref
        self._ref_dot = (next_ref - self._ref) / self.dt_s
        self._ref, self._integral, self._last_time = next_ref, integral, state.time_s
        self._history[1:] = self._history[:-1]
        self._history[0] = action
        return result


@dataclass(frozen=True, slots=True)
class RollControlOutput:
    p_setpoint_rad_s: float
    bank_setpoint_rad: float | None
    surface: SurfaceActuatorState
    inner: RateControllerOutput


class RollAutopilot:
    """Bank or rate command -> shared periphery -> unchanged central policy."""

    def __init__(self, rate_controller: PingmoRateController, *,
                 outer_config: RollOuterLoopConfig = RollOuterLoopConfig(),
                 actuator: SurfaceActuator | None = None) -> None:
        self.rate_controller = rate_controller
        self.outer_loop = RollOuterLoop(outer_config)
        self.actuator = actuator if actuator is not None else SurfaceActuator()
        if not math.isclose(self.actuator.config.dt_s, rate_controller.dt_s, abs_tol=1e-12):
            raise ValueError("controller and surface actuator periods must agree")
        if not math.isclose(self.actuator.config.deflection_limit_rad,
                            rate_controller.aileron_full_scale_rad, abs_tol=1e-12):
            raise ValueError("surface limit must agree with the controller calibration")

    def reset(self) -> None:
        self.rate_controller.reset()
        self.actuator.reset()

    def step(self, state: RollStateEstimate, *, roll_setpoint_rad: float | None = None,
             p_setpoint_rad_s: float | None = None) -> RollControlOutput:
        if (roll_setpoint_rad is None) == (p_setpoint_rad_s is None):
            raise ValueError("select exactly one of bank mode or body-rate mode")
        bank = None
        if roll_setpoint_rad is not None:
            bank = self.outer_loop.limit_bank(roll_setpoint_rad)
            rate = self.outer_loop.rate_setpoint(bank, state)
        else:
            rate = self.outer_loop.limit_rate(p_setpoint_rad_s)
        inner = self.rate_controller.command(rate, state, self.actuator.state)
        surface = self.actuator.step(inner.requested_aileron_rad)
        return RollControlOutput(rate, bank, surface, inner)
