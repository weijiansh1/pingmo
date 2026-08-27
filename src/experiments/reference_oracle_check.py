"""Same-input G2 diagnostic for raw, reference, and oracle responses."""

from dataclasses import dataclass

import numpy as np

from src.aircraft.constrained_oracle import ConstrainedDutchRollOracle
from src.aircraft.p_channel import PChannel
from src.aircraft.parameters import PChannelParameters
from src.aircraft.reference import DutchRollOracle, ReferenceRollModel


@dataclass(frozen=True, slots=True)
class ReferenceOracleTrace:
    """A 200 Hz response record driven by a 50 Hz held augmentation command."""

    time_s: np.ndarray
    f_pilot: np.ndarray
    delta_oracle: np.ndarray
    delta_constrained: np.ndarray
    f_eq_oracle: np.ndarray
    f_eq_constrained: np.ndarray
    p_raw: np.ndarray
    p_ref: np.ndarray
    p_oracle: np.ndarray
    p_constrained: np.ndarray
    metrics: dict[str, float]


def _rmse(response: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((response - reference) ** 2)))


def simulate_reference_oracle(
    parameters: PChannelParameters,
    *,
    pilot_force_n: float = 22.0,
    duration_s: float = 10.0,
    correction_ratio: float = 0.3,
    normalized_rate_limit_s_inv: float = 4.0,
    plant_dt: float = 0.005,
    action_dt: float = 0.02,
) -> ReferenceOracleTrace:
    """Run the G2 comparison under one pilot step and identical plant physics.

    The reference sees the pilot command directly.  Raw sees the same command;
    oracle traces use the same plant but replace it with an unconstrained or
    force/rate-constrained equivalent input at the 50 Hz action rate.
    """
    if pilot_force_n <= 0 or duration_s <= 0 or not 0 < correction_ratio <= 1:
        raise ValueError("pilot force, duration, and correction ratio must be positive")
    action_substeps = int(round(action_dt / plant_dt))
    if action_substeps <= 0 or not np.isclose(action_substeps * plant_dt, action_dt):
        raise ValueError("action_dt must be an integer multiple of plant_dt")
    n_steps = int(round(duration_s / plant_dt))
    if n_steps <= 0 or not np.isclose(n_steps * plant_dt, duration_s):
        raise ValueError("duration_s must be an integer multiple of plant_dt")

    raw_plant = PChannel(parameters, dt=plant_dt)
    oracle_plant = PChannel(parameters, dt=plant_dt)
    constrained_plant = PChannel(parameters, dt=plant_dt)
    reference = ReferenceRollModel(parameters, dt=plant_dt)
    oracle = DutchRollOracle(parameters, dt=action_dt)
    augmentation_limit = correction_ratio * pilot_force_n
    constrained_oracle = ConstrainedDutchRollOracle(
        parameters,
        augmentation_limit=augmentation_limit,
        normalized_rate_limit_s_inv=normalized_rate_limit_s_inv,
        dt=action_dt,
    )

    p_raw = np.empty(n_steps)
    p_ref = np.empty(n_steps)
    p_oracle = np.empty(n_steps)
    p_constrained = np.empty(n_steps)
    f_pilot = np.full(n_steps, pilot_force_n)
    delta_oracle = np.empty(n_steps)
    delta_constrained = np.empty(n_steps)
    f_eq_oracle = np.empty(n_steps)
    f_eq_constrained = np.empty(n_steps)
    unconstrained_delta = 0.0
    constrained_delta = 0.0

    for index in range(n_steps):
        if index % action_substeps == 0:
            unconstrained_delta = oracle.step(pilot_force_n) - pilot_force_n
            constrained_delta = constrained_oracle.step(pilot_force_n)
        p_raw[index], _ = raw_plant.step(pilot_force_n)
        p_ref[index], _ = reference.step(pilot_force_n)
        p_oracle[index], _ = oracle_plant.step(pilot_force_n + unconstrained_delta)
        p_constrained[index], _ = constrained_plant.step(pilot_force_n + constrained_delta)
        delta_oracle[index] = unconstrained_delta
        delta_constrained[index] = constrained_delta
        f_eq_oracle[index] = pilot_force_n + unconstrained_delta
        f_eq_constrained[index] = pilot_force_n + constrained_delta

    constrained_deltas_50hz = delta_constrained[::action_substeps]
    limit = augmentation_limit
    max_rate_increment = limit * normalized_rate_limit_s_inv * action_dt
    return ReferenceOracleTrace(
        time_s=np.arange(n_steps, dtype=float) * plant_dt,
        f_pilot=f_pilot,
        delta_oracle=delta_oracle,
        delta_constrained=delta_constrained,
        f_eq_oracle=f_eq_oracle,
        f_eq_constrained=f_eq_constrained,
        p_raw=p_raw,
        p_ref=p_ref,
        p_oracle=p_oracle,
        p_constrained=p_constrained,
        metrics={
            "raw_tracking_rmse": _rmse(p_raw, p_ref),
            "oracle_tracking_rmse": _rmse(p_oracle, p_ref),
            "constrained_tracking_rmse": _rmse(p_constrained, p_ref),
            "oracle_gap_rmse": _rmse(p_constrained, p_oracle),
            "constrained_saturation_fraction": float(np.mean(np.isclose(np.abs(constrained_deltas_50hz), limit, atol=1e-9))),
            "constrained_max_increment_n": float(np.max(np.abs(np.diff(constrained_deltas_50hz))) if len(constrained_deltas_50hz) > 1 else 0.0),
            "constrained_increment_limit_n": max_rate_increment,
        },
    )
