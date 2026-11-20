"""Shared aircraft-parameter contract for conditional student policies."""

from __future__ import annotations

import math

import numpy as np

from src.aircraft.parameters import PChannelParameters


AIRCRAFT_PARAMETER_NAMES = (
    "l_fa",
    "lambda_s",
    "t_r",
    "zeta_d",
    "omega_d",
    "r_omega",
    "r_zeta",
    "tau_p",
)
AIRCRAFT_PARAMETER_NORMALIZATION = "log_linear_bounds_v1_unclipped"

_PARAMETER_LOW = np.array([0.04, -0.15, 0.18, 0.005, 0.40, 0.65, 0.149, 0.001], dtype=float)
_PARAMETER_HIGH = np.array(
    [0.47083875, math.log(2.0) / 4.0, 10.0, 0.70, 6.0, 1.35, 128.0, 0.25],
    dtype=float,
)
_LOG_COLUMNS = np.array([True, False, True, True, True, True, True, False])


def aircraft_parameter_vector(parameters: PChannelParameters) -> np.ndarray:
    """Return theta in the single canonical order used by student models."""

    return np.asarray(
        [
            parameters.l_fa,
            parameters.lambda_s,
            parameters.t_r,
            parameters.zeta_d,
            parameters.omega_d,
            parameters.r_omega,
            parameters.r_zeta,
            parameters.tau_p,
        ],
        dtype=np.float32,
    )


def normalize_aircraft_parameters(
    parameters: PChannelParameters,
    *,
    clip: bool = False,
) -> np.ndarray:
    """Normalize theta to the training envelope while preserving OOD excursions."""

    values = aircraft_parameter_vector(parameters).astype(float)
    low = _PARAMETER_LOW.copy()
    high = _PARAMETER_HIGH.copy()
    values[_LOG_COLUMNS] = np.log(values[_LOG_COLUMNS])
    low[_LOG_COLUMNS] = np.log(low[_LOG_COLUMNS])
    high[_LOG_COLUMNS] = np.log(high[_LOG_COLUMNS])
    normalized = 2.0 * (values - low) / (high - low) - 1.0
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.astype(np.float32)
