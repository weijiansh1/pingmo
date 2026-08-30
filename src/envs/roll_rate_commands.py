"""Deterministic roll-rate commands for specialist training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _bounded_onset(duration_s: float, preferred_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("command duration must be positive")
    return min(preferred_s, 0.2 * duration_s)


def _bounded_segment(duration_s: float, preferred_s: float) -> float:
    return min(preferred_s, 0.2 * duration_s)


@dataclass(frozen=True, slots=True)
class RollRateCommandProfile:
    command_id: str
    kind: str
    amplitude_deg_s: float = 0.0
    onset_s: float = 0.2
    segment_duration_s: float | None = None
    frequency_hz: float | None = None
    multisine_components: tuple[tuple[float, float], ...] = ()
    duration_s: float = 5.0

    def samples(self, dt_s: float) -> np.ndarray:
        if dt_s <= 0 or self.duration_s <= 0:
            raise ValueError("command dt and duration must be positive")
        count = int(round(self.duration_s / dt_s))
        if count <= 0 or not np.isclose(count * dt_s, self.duration_s):
            raise ValueError("command duration must be an integer multiple of dt_s")
        if not 0 <= self.onset_s < self.duration_s:
            raise ValueError("command onset must be inside the episode")

        time_s = np.arange(count, dtype=float) * dt_s
        relative = time_s - self.onset_s
        active = relative >= 0.0
        values_deg_s = np.zeros(count, dtype=float)
        if self.kind == "step":
            values_deg_s[active] = self.amplitude_deg_s
        elif self.kind == "doublet":
            width = self._positive_segment_duration()
            values_deg_s[active & (relative < width)] = self.amplitude_deg_s
            values_deg_s[(relative >= width) & (relative < 2.0 * width)] = -self.amplitude_deg_s
        elif self.kind == "sine":
            frequency = self._positive_frequency()
            values_deg_s[active] = self.amplitude_deg_s * np.sin(2.0 * math.pi * frequency * relative[active])
        elif self.kind == "multisine":
            if not self.multisine_components:
                raise ValueError("multisine command requires amplitude/frequency components")
            for amplitude_deg_s, frequency_hz in self.multisine_components:
                if frequency_hz <= 0:
                    raise ValueError("multisine frequencies must be positive")
                values_deg_s[active] += amplitude_deg_s * np.sin(
                    2.0 * math.pi * frequency_hz * relative[active]
                )
        else:
            raise ValueError(f"unsupported roll-rate command kind: {self.kind}")
        return np.deg2rad(values_deg_s)

    def _positive_segment_duration(self) -> float:
        if self.segment_duration_s is None or self.segment_duration_s <= 0:
            raise ValueError("doublet command requires a positive segment duration")
        return self.segment_duration_s

    def _positive_frequency(self) -> float:
        if self.frequency_hz is None or self.frequency_hz <= 0:
            raise ValueError("sine command requires a positive frequency")
        return self.frequency_hz


def specialist_step_commands(duration_s: float = 5.0) -> tuple[RollRateCommandProfile, ...]:
    onset_s = _bounded_onset(duration_s, 0.2)
    return tuple(
        RollRateCommandProfile(
            f"train-step-{'pos' if amplitude > 0 else 'neg'}-{abs(amplitude):02.0f}deg-s",
            "step",
            amplitude_deg_s=amplitude,
            onset_s=onset_s,
            duration_s=duration_s,
        )
        for amplitude in (10.0, 20.0, 30.0, -10.0, -20.0, -30.0)
    )


def specialist_extended_commands(duration_s: float = 5.0) -> tuple[RollRateCommandProfile, ...]:
    onset_s = _bounded_onset(duration_s, 0.2)
    segment_duration_s = _bounded_segment(duration_s, 0.6)
    return specialist_step_commands(duration_s) + (
        RollRateCommandProfile("train-doublet-pos-20deg-s", "doublet", 20.0, onset_s=onset_s, segment_duration_s=segment_duration_s, duration_s=duration_s),
        RollRateCommandProfile("train-doublet-neg-20deg-s", "doublet", -20.0, onset_s=onset_s, segment_duration_s=segment_duration_s, duration_s=duration_s),
        RollRateCommandProfile("train-sine-0.50hz", "sine", 20.0, onset_s=onset_s, frequency_hz=0.50, duration_s=duration_s),
        RollRateCommandProfile("train-sine-1.00hz", "sine", 15.0, onset_s=onset_s, frequency_hz=1.00, duration_s=duration_s),
        RollRateCommandProfile(
            "train-multisine",
            "multisine",
            onset_s=onset_s,
            multisine_components=((8.0, 0.25), (6.0, 0.70), (4.0, 1.30)),
            duration_s=duration_s,
        ),
    )


def specialist_evaluation_commands(duration_s: float = 5.0) -> tuple[RollRateCommandProfile, ...]:
    """Commands held out from the default step-only specialist training set."""

    onset_s = _bounded_onset(duration_s, 0.35)
    segment_duration_s = _bounded_segment(duration_s, 0.75)
    return (
        RollRateCommandProfile("eval-step-pos-15deg-s", "step", 15.0, onset_s=onset_s, duration_s=duration_s),
        RollRateCommandProfile("eval-step-neg-15deg-s", "step", -15.0, onset_s=onset_s, duration_s=duration_s),
        RollRateCommandProfile("eval-step-pos-25deg-s", "step", 25.0, onset_s=onset_s, duration_s=duration_s),
        RollRateCommandProfile("eval-doublet-15deg-s", "doublet", 15.0, onset_s=onset_s, segment_duration_s=segment_duration_s, duration_s=duration_s),
        RollRateCommandProfile("eval-sine-0.75hz", "sine", 12.0, onset_s=onset_s, frequency_hz=0.75, duration_s=duration_s),
        RollRateCommandProfile(
            "eval-multisine",
            "multisine",
            onset_s=onset_s,
            multisine_components=((7.0, 0.30), (5.0, 0.85), (3.0, 1.40)),
            duration_s=duration_s,
        ),
    )
