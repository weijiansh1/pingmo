"""P-channel aircraft transfer function."""

from __future__ import annotations

import numpy as np
from scipy import signal


class Plant:
    """Aircraft model from pilot lateral force ``F_as`` to roll rate ``p``.

    The model is

        p(s) / F_as(s) = G0(s) * exp(-tau_p * s)

    where ``G0`` is the rational transfer function represented by
    ``transfer_function`` and ``tau_p`` is the pure transport delay.
    """

    def __init__(
        self,
        l_fa: float,
        lambda_s: float,
        t_r: float,
        zeta_d: float,
        omega_d: float,
        r_omega: float,
        r_zeta: float,
        tau_p: float,
    ) -> None:
        if l_fa <= 0:
            raise ValueError("l_fa must be positive")
        if lambda_s == 0:
            raise ValueError("lambda_s must be non-zero")
        if t_r <= 0 or omega_d <= 0:
            raise ValueError("t_r and omega_d must be positive")
        if not 0 < zeta_d < 1:
            raise ValueError("zeta_d must be in (0, 1)")
        if r_omega <= 0 or r_zeta <= 0:
            raise ValueError("r_omega and r_zeta must be positive")
        if tau_p < 0:
            raise ValueError("tau_p must be non-negative")

        self.l_fa = float(l_fa)
        self.lambda_s = float(lambda_s)
        self.t_r = float(t_r)
        self.zeta_d = float(zeta_d)
        self.omega_d = float(omega_d)
        self.r_omega = float(r_omega)
        self.r_zeta = float(r_zeta)
        self.tau_p = float(tau_p)

        self.omega_phi = self.r_omega * self.omega_d
        self.zeta_phi = self.r_zeta * self.zeta_d

        self.numerator = self.l_fa * np.array(
            [
                1.0,
                2.0 * self.zeta_phi * self.omega_phi,
                self.omega_phi**2,
                0.0,
            ],
            dtype=float,
        )
        self.denominator = np.polymul(
            np.polymul(
                [1.0, -self.lambda_s],
                [1.0, 1.0 / self.t_r],
            ),
            [1.0, 2.0 * self.zeta_d * self.omega_d, self.omega_d**2],
        )
        self.transfer_function = signal.TransferFunction(self.numerator, self.denominator)

    def frequency_response(self, omega_rad_s: np.ndarray) -> np.ndarray:
        """Return the full frequency response, including pure delay."""

        omega = np.asarray(omega_rad_s, dtype=float)
        _, response = signal.freqresp(self.transfer_function, omega)
        return response * np.exp(-1j * omega * self.tau_p)

    def response(self, time_s: np.ndarray, force_n: np.ndarray) -> np.ndarray:
        """Return ``p(t)`` for an arbitrary sampled ``F_as(t)`` signal."""

        time = np.asarray(time_s, dtype=float)
        force = np.asarray(force_n, dtype=float)
        if time.ndim != 1 or force.ndim != 1 or len(time) != len(force):
            raise ValueError("time_s and force_n must be one-dimensional arrays of equal length")
        if len(time) < 2 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(force)):
            raise ValueError("response inputs must contain at least two finite samples")
        intervals = np.diff(time)
        if time[0] != 0.0 or np.any(intervals <= 0):
            raise ValueError("time_s must start at zero and increase strictly")
        if not np.allclose(intervals, intervals[0], rtol=1e-7, atol=1e-12):
            raise ValueError("time_s must use a constant sample interval")

        delayed_force = np.interp(time - self.tau_p, time, force, left=0.0)
        _, roll_rate, _ = signal.lsim(self.transfer_function, U=delayed_force, T=time)
        return np.asarray(roll_rate, dtype=float)

    def step_response(
        self,
        force_n: float,
        duration_s: float,
        dt_s: float,
        *,
        onset_s: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return time, force, and roll-rate arrays for a force step."""

        if not np.isfinite(force_n):
            raise ValueError("force_n must be finite")
        if duration_s <= 0 or dt_s <= 0:
            raise ValueError("duration_s and dt_s must be positive")
        if not 0 <= onset_s < duration_s:
            raise ValueError("onset_s must be inside the simulation interval")
        steps = int(round(duration_s / dt_s))
        if not np.isclose(steps * dt_s, duration_s):
            raise ValueError("duration_s must be an integer multiple of dt_s")

        time = np.arange(steps + 1, dtype=float) * dt_s
        force = np.zeros_like(time)
        force[time >= onset_s] = force_n
        return time, force, self.response(time, force)
