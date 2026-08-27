"""Response-level IV-A gain calibration for the GJB-original roll plant."""

from dataclasses import replace

import numpy as np

from src.aircraft.p_channel import PChannel
from src.aircraft.parameters import PChannelParameters


def sensitivity_1s_deg_per_n(parameters: PChannelParameters, force_n: float = 1.0, dt: float = 0.005) -> float:
    """Measure ``abs(phi(1 s))/abs(F_step)`` by integrating roll rate in radians."""
    if force_n == 0 or dt <= 0:
        raise ValueError("force_n and dt must be non-zero and positive respectively")
    steps = round(1.0 / dt)
    if not np.isclose(steps * dt, 1.0):
        raise ValueError("dt must divide one second exactly")
    plant = PChannel(parameters, dt=dt)
    roll_rate = np.empty(steps + 1)
    roll_rate[0] = 0.0
    for index in range(1, steps + 1):
        roll_rate[index] = plant.step(force_n)[0]
    bank_angle_rad = np.trapezoid(roll_rate, dx=dt)
    return float(abs(np.degrees(bank_angle_rad) / force_n))


def calibrate_l_fa_for_sensitivity(parameters: PChannelParameters, target_sensitivity_deg_per_n: float, dt: float = 0.005) -> PChannelParameters:
    """Use plant linearity to calibrate ``L_Fa`` to a specified 1 s sensitivity."""
    if target_sensitivity_deg_per_n <= 0:
        raise ValueError("target_sensitivity_deg_per_n must be positive")
    baseline = sensitivity_1s_deg_per_n(parameters, dt=dt)
    if baseline <= np.finfo(float).eps:
        raise ValueError("cannot calibrate a plant with zero 1 s sensitivity")
    return replace(parameters, l_fa=parameters.l_fa * target_sensitivity_deg_per_n / baseline)
