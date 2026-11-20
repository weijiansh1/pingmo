"""Same-input G2 diagnostic for raw, reference, and oracle responses."""

from dataclasses import dataclass

import numpy as np

from src.aircraft.constrained_oracle import ConstrainedDutchRollOracle
from src.aircraft.p_channel import PChannel
from src.aircraft.parameters import PChannelParameters
from src.aircraft.reference import DutchRollOracle, ReferenceRollModel
from src.envs.commands import CommandProfile


@dataclass(frozen=True, slots=True)
class ReferenceOracleTrace:
    """A 200 Hz response record driven by a 50 Hz held augmentation command."""

    time_s: np.ndarray
    f_pilot: np.ndarray
    delta_oracle: np.ndarray
    delta_constrained_command: np.ndarray
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


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def simulate_reference_oracle(
    parameters: PChannelParameters,
    *,
    pilot_force_n: float = 22.0,
    duration_s: float = 10.0,
    correction_ratio: float = 0.3,
    normalized_rate_limit_s_inv: float = 4.0,
    actuator_time_constant_s: float = 0.08,
    plant_dt: float = 0.005,
    action_dt: float = 0.02,
    command_profile: CommandProfile | None = None,
) -> ReferenceOracleTrace:
    """Run the G2 comparison under one pilot step and identical plant physics.

    The reference sees the pilot command directly.  Raw sees the same command;
    oracle traces use the same plant but replace it with an unconstrained or
    force/rate-constrained equivalent input at the 50 Hz action rate.
    """
    if pilot_force_n <= 0 or duration_s <= 0 or not 0 < correction_ratio <= 1 or actuator_time_constant_s <= 0:
        raise ValueError("pilot force, duration, correction ratio, and actuator time constant must be positive")
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
    action_count = n_steps // action_substeps
    action_commands = (
        np.full(action_count, pilot_force_n, dtype=float)
        if command_profile is None
        else command_profile.samples(action_dt, duration_s, pilot_force_n)
    )
    f_pilot = np.repeat(action_commands, action_substeps)
    delta_oracle = np.empty(n_steps)
    delta_constrained_command = np.empty(n_steps)
    delta_constrained = np.empty(n_steps)
    f_eq_oracle = np.empty(n_steps)
    f_eq_constrained = np.empty(n_steps)
    unconstrained_delta = 0.0
    constrained_command = 0.0
    constrained_delta = 0.0
    actuator_alpha = 1.0 - np.exp(-action_dt / actuator_time_constant_s)

    for index in range(n_steps):
        if index % action_substeps == 0:
            pilot_command = action_commands[index // action_substeps]
            unconstrained_delta = oracle.step(pilot_command) - pilot_command
            constrained_command = constrained_oracle.step(pilot_command)
            constrained_delta += actuator_alpha * (constrained_command - constrained_delta)
        p_raw[index], _ = raw_plant.step(f_pilot[index])
        p_ref[index], _ = reference.step(f_pilot[index])
        p_oracle[index], _ = oracle_plant.step(f_pilot[index] + unconstrained_delta)
        p_constrained[index], _ = constrained_plant.step(f_pilot[index] + constrained_delta)
        delta_oracle[index] = unconstrained_delta
        delta_constrained_command[index] = constrained_command
        delta_constrained[index] = constrained_delta
        f_eq_oracle[index] = f_pilot[index] + unconstrained_delta
        f_eq_constrained[index] = f_pilot[index] + constrained_delta

    oracle_deltas_50hz = delta_oracle[::action_substeps]
    constrained_commands_50hz = delta_constrained_command[::action_substeps]
    constrained_deltas_50hz = delta_constrained[::action_substeps]
    limit = augmentation_limit
    max_rate_increment = limit * normalized_rate_limit_s_inv * action_dt
    oracle_increments = np.diff(oracle_deltas_50hz, prepend=0.0)
    constrained_command_increments = np.diff(constrained_commands_50hz, prepend=0.0)
    constrained_applied_increments = np.diff(constrained_deltas_50hz, prepend=0.0)
    reference_rms = _rms(p_ref)
    normalization = max(reference_rms, np.finfo(float).eps)
    raw_rmse = _rmse(p_raw, p_ref)
    oracle_rmse = _rmse(p_oracle, p_ref)
    constrained_rmse = _rmse(p_constrained, p_ref)
    return ReferenceOracleTrace(
        time_s=np.arange(n_steps, dtype=float) * plant_dt,
        f_pilot=f_pilot,
        delta_oracle=delta_oracle,
        delta_constrained_command=delta_constrained_command,
        delta_constrained=delta_constrained,
        f_eq_oracle=f_eq_oracle,
        f_eq_constrained=f_eq_constrained,
        p_raw=p_raw,
        p_ref=p_ref,
        p_oracle=p_oracle,
        p_constrained=p_constrained,
        metrics={
            "reference_response_rms": reference_rms,
            "reference_tracking_rmse": 0.0,
            "raw_tracking_rmse": raw_rmse,
            "oracle_tracking_rmse": oracle_rmse,
            "constrained_tracking_rmse": constrained_rmse,
            "raw_relative_tracking_rmse": raw_rmse / normalization,
            "oracle_relative_tracking_rmse": oracle_rmse / normalization,
            "constrained_relative_tracking_rmse": constrained_rmse / normalization,
            "oracle_gap_rmse": _rmse(p_constrained, p_oracle),
            "oracle_augmentation_rms_n": _rms(oracle_deltas_50hz),
            "oracle_peak_augmentation_n": float(np.max(np.abs(oracle_deltas_50hz))),
            "oracle_total_variation_n": float(np.sum(np.abs(oracle_increments))),
            "oracle_authority_exceedance_fraction": float(np.mean(np.abs(oracle_deltas_50hz) > limit + 1e-9)),
            "oracle_slew_exceedance_fraction": float(np.mean(np.abs(oracle_increments) > max_rate_increment + 1e-9)),
            "constrained_saturation_fraction": float(np.mean(np.isclose(np.abs(constrained_commands_50hz), limit, atol=1e-9, rtol=0.0))),
            "constrained_slew_bound_fraction": float(np.mean(np.isclose(np.abs(constrained_command_increments), max_rate_increment, atol=1e-9, rtol=0.0))),
            "constrained_command_rms_n": _rms(constrained_commands_50hz),
            "constrained_applied_rms_n": _rms(constrained_deltas_50hz),
            "constrained_max_increment_n": float(np.max(np.abs(constrained_command_increments))),
            "constrained_increment_limit_n": max_rate_increment,
            "constrained_command_total_variation_n": float(np.sum(np.abs(constrained_command_increments))),
            "constrained_applied_total_variation_n": float(np.sum(np.abs(constrained_applied_increments))),
        },
    )
