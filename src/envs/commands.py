"""Deterministic pilot-force commands on the plant integration grid."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


DEFAULT_COMMAND_DURATION_S = 10.0


@dataclass(frozen=True, slots=True)
class CommandProfile:
    """A reproducible normalized pilot-force command.

    ``amplitude`` and ``levels`` are fractions of ``nominal_force_n``.  Timing is
    evaluated on the plant grid, independently of the slower policy period.
    """

    command_id: str
    kind: str
    amplitude: float = 1.0
    frequency_hz: float | None = None
    final_frequency_hz: float | None = None
    duration_s: float | None = None
    onset_s: float = 0.0
    segment_duration_s: float | None = None
    levels: tuple[float, ...] = ()
    seed: int | None = None

    def samples(self, plant_dt_s: float, duration_s: float, nominal_force_n: float) -> np.ndarray:
        if plant_dt_s <= 0 or duration_s <= 0 or nominal_force_n <= 0:
            raise ValueError("plant_dt_s, duration_s, and nominal_force_n must be positive")
        if self.duration_s is not None and not np.isclose(duration_s, self.duration_s):
            raise ValueError(
                f"command {self.command_id!r} is defined for {self.duration_s} s, "
                f"not {duration_s} s"
            )
        if not 0 <= self.onset_s < duration_s:
            raise ValueError("onset_s must be inside the command duration")
        count = int(round(duration_s / plant_dt_s))
        if count <= 0 or not np.isclose(count * plant_dt_s, duration_s):
            raise ValueError("duration_s must be an integer multiple of plant_dt_s")

        time_s = np.arange(count, dtype=float) * plant_dt_s
        relative_time_s = time_s - self.onset_s
        active = relative_time_s >= 0.0
        normalized = np.zeros(count, dtype=float)

        if self.kind == "step":
            normalized[active] = self.amplitude
        elif self.kind == "pulse":
            width = self.segment_duration_s or duration_s / 4.0
            normalized[active & (relative_time_s < width)] = self.amplitude
        elif self.kind == "doublet":
            width = self.segment_duration_s or duration_s / 4.0
            normalized[active & (relative_time_s < width)] = self.amplitude
            normalized[(relative_time_s >= width) & (relative_time_s < 2.0 * width)] = -self.amplitude
        elif self.kind == "square":
            frequency = self._positive_frequency("square")
            half_cycles = np.floor(2.0 * frequency * relative_time_s[active]).astype(int)
            normalized[active] = self.amplitude * np.where(half_cycles % 2 == 0, 1.0, -1.0)
        elif self.kind == "sine":
            frequency = self._positive_frequency("sine")
            normalized[active] = self.amplitude * np.sin(2.0 * math.pi * frequency * relative_time_s[active])
        elif self.kind == "chirp":
            start = self._positive_frequency("chirp")
            if self.final_frequency_hz is None or self.final_frequency_hz <= 0:
                raise ValueError("chirp command requires a positive final_frequency_hz")
            active_duration = duration_s - self.onset_s
            slope = (self.final_frequency_hz - start) / active_duration
            t_active = relative_time_s[active]
            phase = 2.0 * math.pi * (start * t_active + 0.5 * slope * t_active**2)
            normalized[active] = self.amplitude * np.sin(phase)
        elif self.kind == "staircase":
            if not self.levels:
                raise ValueError("staircase command requires non-empty levels")
            width = self._positive_segment_duration("staircase")
            indices = np.floor(relative_time_s[active] / width).astype(int)
            values = np.zeros(indices.shape, dtype=float)
            inside = indices < len(self.levels)
            values[inside] = np.asarray(self.levels, dtype=float)[indices[inside]]
            normalized[active] = values
        elif self.kind == "piecewise":
            width = self._positive_segment_duration("piecewise")
            if self.seed is None:
                raise ValueError("piecewise command requires a seed")
            segment_count = int(math.ceil((duration_s - self.onset_s) / width))
            rng = np.random.default_rng(self.seed)
            choices = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * abs(self.amplitude)
            values = rng.choice(choices, size=segment_count)
            indices = np.minimum(np.floor(relative_time_s[active] / width).astype(int), segment_count - 1)
            normalized[active] = values[indices]
        else:
            raise ValueError(f"unsupported command kind: {self.kind}")

        return (nominal_force_n * normalized).astype(float)

    def _positive_frequency(self, kind: str) -> float:
        if self.frequency_hz is None or self.frequency_hz <= 0:
            raise ValueError(f"{kind} command requires a positive frequency_hz")
        return self.frequency_hz

    def _positive_segment_duration(self, kind: str) -> float:
        if self.segment_duration_s is None or self.segment_duration_s <= 0:
            raise ValueError(f"{kind} command requires a positive segment_duration_s")
        return self.segment_duration_s


def training_command_suite() -> tuple[CommandProfile, ...]:
    """Balanced command families used to sample training episodes."""

    duration = DEFAULT_COMMAND_DURATION_S
    signed_amplitudes = tuple(sign * amplitude for sign in (1.0, -1.0) for amplitude in (0.25, 0.50, 1.00))
    onset = 0.2
    steps = tuple(CommandProfile(f"step-{'pos' if value > 0 else 'neg'}-{abs(value):.2f}", "step", amplitude=value, duration_s=duration, onset_s=onset) for value in signed_amplitudes)
    pulses = tuple(CommandProfile(f"pulse-{'pos' if value > 0 else 'neg'}-{abs(value):.2f}", "pulse", amplitude=value, segment_duration_s=0.8, duration_s=duration, onset_s=onset) for value in signed_amplitudes)
    doublets = tuple(CommandProfile(f"doublet-{'pos' if value > 0 else 'neg'}-{abs(value):.2f}", "doublet", amplitude=value, segment_duration_s=0.6, duration_s=duration, onset_s=onset) for value in signed_amplitudes)
    squares = tuple(CommandProfile(f"square-{frequency:.2f}hz", "square", amplitude=0.75, frequency_hz=frequency, duration_s=duration, onset_s=onset) for frequency in (0.20, 0.50, 1.00))
    sines = tuple(CommandProfile(f"sine-{frequency:.2f}hz", "sine", amplitude=0.75, frequency_hz=frequency, duration_s=duration, onset_s=onset) for frequency in (0.20, 0.50, 1.00, 1.50))
    chirps = (
        CommandProfile("chirp-0.10-2.00hz", "chirp", amplitude=0.75, frequency_hz=0.10, final_frequency_hz=2.00, duration_s=duration, onset_s=onset),
        CommandProfile("chirp-0.20-4.00hz", "chirp", amplitude=0.50, frequency_hz=0.20, final_frequency_hz=4.00, duration_s=duration, onset_s=onset),
    )
    staircases = (
        CommandProfile("staircase-positive", "staircase", levels=(0.25, 0.50, 0.75, 1.00, 0.50, 0.0), segment_duration_s=1.2, duration_s=duration, onset_s=onset),
        CommandProfile("staircase-signed", "staircase", levels=(0.25, 0.75, 0.0, -0.50, -1.00, 0.0), segment_duration_s=1.2, duration_s=duration, onset_s=onset),
    )
    piecewise = tuple(CommandProfile(f"piecewise-seed-{seed}", "piecewise", amplitude=1.0, segment_duration_s=0.5, duration_s=duration, onset_s=onset, seed=seed) for seed in (11, 29, 47, 83))
    return steps + pulses + doublets + squares + sines + chirps + staircases + piecewise


def evaluation_command_suite() -> tuple[CommandProfile, ...]:
    """Fixed held-out amplitudes, timings, and frequencies for reporting."""

    duration = DEFAULT_COMMAND_DURATION_S
    return (
        CommandProfile("eval-step-pos-0.75", "step", amplitude=0.75, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-step-neg-0.75", "step", amplitude=-0.75, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-pulse-0.75", "pulse", amplitude=0.75, segment_duration_s=1.1, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-doublet-0.75", "doublet", amplitude=0.75, segment_duration_s=0.75, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-square-0.35hz", "square", amplitude=0.75, frequency_hz=0.35, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-sine-0.35hz", "sine", amplitude=0.75, frequency_hz=0.35, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-sine-1.25hz", "sine", amplitude=0.75, frequency_hz=1.25, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-chirp-0.15-2.50hz", "chirp", amplitude=0.625, frequency_hz=0.15, final_frequency_hz=2.50, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-staircase", "staircase", levels=(0.375, 0.75, 0.25, -0.375, -0.75, 0.0), segment_duration_s=1.2, duration_s=duration, onset_s=0.4),
        CommandProfile("eval-piecewise-seed-101", "piecewise", amplitude=0.875, segment_duration_s=0.65, duration_s=duration, onset_s=0.4, seed=101),
    )


def default_command_suite() -> tuple[CommandProfile, ...]:
    """Compatibility alias for the active Stage-1 training distribution."""

    return training_command_suite()
