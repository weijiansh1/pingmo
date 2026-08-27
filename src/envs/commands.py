"""Deterministic 50 Hz pilot-command profiles shared by audit and training."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class CommandProfile:
    command_id: str
    kind: str
    amplitude: float = 1.0
    frequency_hz: float | None = None
    final_frequency_hz: float | None = None

    def samples(self, action_dt: float, duration_s: float, nominal_force_n: float) -> np.ndarray:
        if action_dt <= 0 or duration_s <= 0 or nominal_force_n <= 0:
            raise ValueError("action_dt, duration_s, and nominal_force_n must be positive")
        count = int(round(duration_s / action_dt))
        if count <= 0 or not np.isclose(count * action_dt, duration_s):
            raise ValueError("duration_s must be an integer multiple of action_dt")
        time_s = np.arange(count, dtype=float) * action_dt
        if self.kind == "step":
            normalized = np.full(count, self.amplitude, dtype=float)
        elif self.kind == "doublet":
            normalized = np.zeros(count, dtype=float)
            quarter = duration_s / 4.0
            normalized[time_s < quarter] = self.amplitude
            normalized[(time_s >= quarter) & (time_s < 2.0 * quarter)] = -self.amplitude
        elif self.kind == "sine":
            if self.frequency_hz is None or self.frequency_hz <= 0:
                raise ValueError("sine command requires a positive frequency_hz")
            normalized = self.amplitude * np.sin(2.0 * math.pi * self.frequency_hz * time_s)
        elif self.kind == "chirp":
            if self.frequency_hz is None or self.final_frequency_hz is None or self.frequency_hz <= 0 or self.final_frequency_hz <= 0:
                raise ValueError("chirp command requires positive start and final frequencies")
            slope = (self.final_frequency_hz - self.frequency_hz) / duration_s
            phase = 2.0 * math.pi * (self.frequency_hz * time_s + 0.5 * slope * time_s**2)
            normalized = self.amplitude * np.sin(phase)
        else:
            raise ValueError(f"unsupported command kind: {self.kind}")
        return (nominal_force_n * normalized).astype(float)


def default_command_suite() -> tuple[CommandProfile, ...]:
    steps = tuple(
        CommandProfile(f"step-{sign}{amplitude:.2f}", "step", amplitude=sign * amplitude)
        for sign in (1.0, -1.0)
        for amplitude in (0.25, 0.50, 1.00)
    )
    doublets = tuple(CommandProfile(f"doublet-{amplitude:.2f}", "doublet", amplitude=amplitude) for amplitude in (0.25, 0.50, 1.00))
    sines = tuple(CommandProfile(f"sine-{frequency:.2f}hz", "sine", amplitude=1.0, frequency_hz=frequency) for frequency in (0.25, 0.50, 1.00))
    chirp = CommandProfile("chirp-0.10-1.50hz", "chirp", amplitude=1.0, frequency_hz=0.10, final_frequency_hz=1.50)
    return steps + doublets + sines + (chirp,)
