"""Deterministic roll-rate commands for specialist training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION = "specialist-independent-test-v1"


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
            values_deg_s[
                (relative >= width) & (relative < 2.0 * width)
            ] = -self.amplitude_deg_s
        elif self.kind == "sine":
            frequency = self._positive_frequency()
            values_deg_s[active] = self.amplitude_deg_s * np.sin(
                2.0 * math.pi * frequency * relative[active]
            )
        elif self.kind == "multisine":
            if not self.multisine_components:
                raise ValueError(
                    "multisine command requires amplitude/frequency components"
                )
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


@dataclass(frozen=True, slots=True)
class RollRateCommandSequence:
    """Several command profiles concatenated into one continuous episode."""

    command_id: str
    segments: tuple[RollRateCommandProfile, ...]

    def __post_init__(self) -> None:
        if not self.command_id or not self.segments:
            raise ValueError("command sequence requires an id and at least one segment")

    @property
    def kind(self) -> str:
        return "sequence"

    @property
    def duration_s(self) -> float:
        return float(sum(segment.duration_s for segment in self.segments))

    @property
    def multisine_components(self) -> tuple[tuple[float, float], ...]:
        return ()

    def samples(self, dt_s: float) -> np.ndarray:
        if dt_s <= 0:
            raise ValueError("command dt must be positive")
        values = tuple(segment.samples(dt_s) for segment in self.segments)
        return np.concatenate(values)


RollRateCommand = RollRateCommandProfile | RollRateCommandSequence


@dataclass(frozen=True, slots=True)
class RandomCommandDistribution:
    """Continuous command distribution used by reward-only Teachers."""

    kind_probabilities: tuple[float, float, float, float] = (0.35, 0.20, 0.25, 0.20)
    duration_range_s: tuple[float, float] = (4.0, 8.0)
    onset_range_s: tuple[float, float] = (0.10, 0.60)
    step_amplitude_range_deg_s: tuple[float, float] = (5.0, 30.0)
    doublet_amplitude_range_deg_s: tuple[float, float] = (5.0, 25.0)
    doublet_segment_range_s: tuple[float, float] = (0.25, 1.00)
    sine_amplitude_range_deg_s: tuple[float, float] = (5.0, 20.0)
    frequency_range_hz: tuple[float, float] = (0.20, 1.50)
    multisine_component_amplitude_range_deg_s: tuple[float, float] = (2.0, 8.0)
    multisine_total_amplitude_limit_deg_s: float = 25.0
    multisine_component_count_range: tuple[int, int] = (2, 3)

    def __post_init__(self) -> None:
        ranges = (
            self.duration_range_s,
            self.onset_range_s,
            self.step_amplitude_range_deg_s,
            self.doublet_amplitude_range_deg_s,
            self.doublet_segment_range_s,
            self.sine_amplitude_range_deg_s,
            self.frequency_range_hz,
            self.multisine_component_amplitude_range_deg_s,
        )
        if any(low <= 0 or low > high for low, high in ranges):
            raise ValueError("random command ranges must be positive and ordered")
        probabilities = np.asarray(self.kind_probabilities, dtype=float)
        if (
            probabilities.shape != (4,)
            or np.any(probabilities <= 0)
            or not np.isclose(probabilities.sum(), 1.0)
        ):
            raise ValueError(
                "random command kind probabilities must be four positive values summing to one"
            )
        component_low, component_high = self.multisine_component_count_range
        if not 1 <= component_low <= component_high <= 3:
            raise ValueError(
                "random multisine component count must be between one and three"
            )
        if self.multisine_total_amplitude_limit_deg_s <= 0:
            raise ValueError("random multisine total amplitude limit must be positive")


def sample_random_training_command(
    rng: np.random.Generator,
    episode_index: int,
    *,
    policy_dt_s: float,
    config: RandomCommandDistribution = RandomCommandDistribution(),
) -> RollRateCommandProfile:
    """Draw one reproducible command without enumerating a finite command bank."""

    if episode_index < 0 or policy_dt_s <= 0:
        raise ValueError("random command episode index and policy dt are invalid")
    duration_s = float(rng.uniform(*config.duration_range_s))
    duration_s = round(duration_s / policy_dt_s) * policy_dt_s
    onset_high = min(config.onset_range_s[1], duration_s - policy_dt_s)
    onset_s = float(rng.uniform(config.onset_range_s[0], onset_high))
    kind = str(
        rng.choice(
            ("step", "doublet", "sine", "multisine"),
            p=config.kind_probabilities,
        )
    )
    sign = float(rng.choice((-1.0, 1.0)))
    command_id = f"random-{episode_index:08d}-{kind}"
    if kind == "step":
        amplitude = sign * float(rng.uniform(*config.step_amplitude_range_deg_s))
        return RollRateCommandProfile(
            command_id,
            kind,
            amplitude_deg_s=amplitude,
            onset_s=onset_s,
            duration_s=duration_s,
        )
    if kind == "doublet":
        amplitude = sign * float(rng.uniform(*config.doublet_amplitude_range_deg_s))
        maximum_segment_s = min(
            config.doublet_segment_range_s[1],
            0.45 * (duration_s - onset_s),
        )
        segment_s = float(
            rng.uniform(config.doublet_segment_range_s[0], maximum_segment_s)
        )
        return RollRateCommandProfile(
            command_id,
            kind,
            amplitude_deg_s=amplitude,
            onset_s=onset_s,
            segment_duration_s=segment_s,
            duration_s=duration_s,
        )
    if kind == "sine":
        amplitude = sign * float(rng.uniform(*config.sine_amplitude_range_deg_s))
        frequency_hz = float(rng.uniform(*config.frequency_range_hz))
        return RollRateCommandProfile(
            command_id,
            kind,
            amplitude_deg_s=amplitude,
            onset_s=onset_s,
            frequency_hz=frequency_hz,
            duration_s=duration_s,
        )

    component_low, component_high = config.multisine_component_count_range
    component_count = int(rng.integers(component_low, component_high + 1))
    frequencies = np.sort(rng.uniform(*config.frequency_range_hz, size=component_count))
    amplitudes = rng.uniform(
        *config.multisine_component_amplitude_range_deg_s,
        size=component_count,
    )
    amplitudes *= rng.choice((-1.0, 1.0), size=component_count)
    total_amplitude = float(np.abs(amplitudes).sum())
    if total_amplitude > config.multisine_total_amplitude_limit_deg_s:
        amplitudes *= config.multisine_total_amplitude_limit_deg_s / total_amplitude
    components = tuple(
        (float(amplitude), float(frequency))
        for amplitude, frequency in zip(amplitudes, frequencies, strict=True)
    )
    return RollRateCommandProfile(
        command_id,
        kind,
        onset_s=onset_s,
        multisine_components=components,
        duration_s=duration_s,
    )


def sample_random_training_sequence(
    rng: np.random.Generator,
    episode_index: int,
    *,
    policy_dt_s: float,
    duration_s: float,
    segment_duration_range_s: tuple[float, float] = (2.0, 5.0),
    config: RandomCommandDistribution = RandomCommandDistribution(),
) -> RollRateCommandSequence:
    """Draw random command segments without resetting the plant between them."""

    if episode_index < 0 or policy_dt_s <= 0 or duration_s <= 0:
        raise ValueError("random sequence episode index, dt, and duration are invalid")
    segment_min_s, segment_max_s = segment_duration_range_s
    if segment_min_s <= 0 or segment_min_s > segment_max_s:
        raise ValueError("random sequence segment duration range is invalid")

    total_steps = int(round(duration_s / policy_dt_s))
    minimum_steps = int(math.ceil(segment_min_s / policy_dt_s))
    maximum_steps = int(math.floor(segment_max_s / policy_dt_s))
    if (
        total_steps <= 0
        or not np.isclose(total_steps * policy_dt_s, duration_s)
        or minimum_steps > maximum_steps
        or total_steps < minimum_steps
    ):
        raise ValueError(
            "sequence and segment durations must fit positive integer policy steps"
        )

    segment_steps: list[int] = []
    remaining_steps = total_steps
    while remaining_steps:
        if remaining_steps <= maximum_steps:
            current_steps = remaining_steps
        else:
            upper_steps = min(maximum_steps, remaining_steps - minimum_steps)
            current_steps = int(rng.integers(minimum_steps, upper_steps + 1))
        segment_steps.append(current_steps)
        remaining_steps -= current_steps

    sequence_id = f"random-sequence-{episode_index:08d}"
    segments: list[RollRateCommandProfile] = []
    for segment_index, current_steps in enumerate(segment_steps):
        segment_duration_s = current_steps * policy_dt_s
        segment_config = replace(
            config,
            duration_range_s=(segment_duration_s, segment_duration_s),
        )
        profile = sample_random_training_command(
            rng,
            episode_index * 10_000 + segment_index,
            policy_dt_s=policy_dt_s,
            config=segment_config,
        )
        segments.append(
            replace(
                profile,
                command_id=(
                    f"{sequence_id}-segment-{segment_index:03d}-{profile.kind}"
                ),
            )
        )
    return RollRateCommandSequence(sequence_id, tuple(segments))


def sample_random_long_dwell_step(
    rng: np.random.Generator,
    episode_index: int,
    *,
    policy_dt_s: float,
    duration_s: float,
    dwell_duration_range_s: tuple[float, float] = (15.0, 30.0),
    config: RandomCommandDistribution = RandomCommandDistribution(),
) -> RollRateCommandProfile:
    """Draw one held step whose active duration exposes slow plant modes."""

    if episode_index < 0 or policy_dt_s <= 0 or duration_s <= 0:
        raise ValueError("long-dwell step episode index, dt, and duration are invalid")
    dwell_min_s, dwell_max_s = dwell_duration_range_s
    if dwell_min_s <= 0 or dwell_min_s > dwell_max_s or dwell_max_s > duration_s:
        raise ValueError("long-dwell duration range must fit inside the episode")
    episode_steps = int(round(duration_s / policy_dt_s))
    minimum_dwell_steps = int(math.ceil(dwell_min_s / policy_dt_s))
    maximum_dwell_steps = int(math.floor(dwell_max_s / policy_dt_s))
    if (
        episode_steps <= 0
        or not np.isclose(episode_steps * policy_dt_s, duration_s)
        or minimum_dwell_steps > maximum_dwell_steps
    ):
        raise ValueError("long-dwell durations must fit positive integer policy steps")

    dwell_steps = int(
        rng.integers(minimum_dwell_steps, maximum_dwell_steps + 1)
    )
    onset_s = (episode_steps - dwell_steps) * policy_dt_s
    sign = float(rng.choice((-1.0, 1.0)))
    amplitude = sign * float(rng.uniform(*config.step_amplitude_range_deg_s))
    return RollRateCommandProfile(
        f"random-long-dwell-{episode_index:08d}-step",
        "step",
        amplitude_deg_s=amplitude,
        onset_s=onset_s,
        duration_s=duration_s,
    )


def sample_mixed_duration_training_episode(
    rng: np.random.Generator,
    episode_index: int,
    *,
    policy_dt_s: float,
    duration_s: float,
    long_dwell_step_probability: float,
    short_segment_duration_range_s: tuple[float, float] = (2.0, 5.0),
    long_dwell_duration_range_s: tuple[float, float] = (15.0, 30.0),
    config: RandomCommandDistribution = RandomCommandDistribution(),
) -> RollRateCommand:
    """Mix short continuous random segments with long held step episodes."""

    if not 0.0 <= long_dwell_step_probability <= 1.0:
        raise ValueError("long-dwell step probability must be in [0, 1]")
    if rng.random() < long_dwell_step_probability:
        return sample_random_long_dwell_step(
            rng,
            episode_index,
            policy_dt_s=policy_dt_s,
            duration_s=duration_s,
            dwell_duration_range_s=long_dwell_duration_range_s,
            config=config,
        )
    return sample_random_training_sequence(
        rng,
        episode_index,
        policy_dt_s=policy_dt_s,
        duration_s=duration_s,
        segment_duration_range_s=short_segment_duration_range_s,
        config=config,
    )


def specialist_step_commands(
    duration_s: float = 5.0,
) -> tuple[RollRateCommandProfile, ...]:
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


def specialist_extended_commands(
    duration_s: float = 5.0,
) -> tuple[RollRateCommandProfile, ...]:
    onset_s = _bounded_onset(duration_s, 0.2)
    segment_duration_s = _bounded_segment(duration_s, 0.6)
    return specialist_step_commands(duration_s) + (
        RollRateCommandProfile(
            "train-doublet-pos-20deg-s",
            "doublet",
            20.0,
            onset_s=onset_s,
            segment_duration_s=segment_duration_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "train-doublet-neg-20deg-s",
            "doublet",
            -20.0,
            onset_s=onset_s,
            segment_duration_s=segment_duration_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "train-sine-0.50hz",
            "sine",
            20.0,
            onset_s=onset_s,
            frequency_hz=0.50,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "train-sine-1.00hz",
            "sine",
            15.0,
            onset_s=onset_s,
            frequency_hz=1.00,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "train-multisine",
            "multisine",
            onset_s=onset_s,
            multisine_components=((8.0, 0.25), (6.0, 0.70), (4.0, 1.30)),
            duration_s=duration_s,
        ),
    )


def specialist_evaluation_commands(
    duration_s: float = 5.0,
) -> tuple[RollRateCommandProfile, ...]:
    """Commands held out from the default step-only specialist training set."""

    onset_s = _bounded_onset(duration_s, 0.35)
    segment_duration_s = _bounded_segment(duration_s, 0.75)
    return (
        RollRateCommandProfile(
            "eval-step-pos-15deg-s",
            "step",
            15.0,
            onset_s=onset_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "eval-step-neg-15deg-s",
            "step",
            -15.0,
            onset_s=onset_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "eval-step-pos-25deg-s",
            "step",
            25.0,
            onset_s=onset_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "eval-doublet-15deg-s",
            "doublet",
            15.0,
            onset_s=onset_s,
            segment_duration_s=segment_duration_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "eval-sine-0.75hz",
            "sine",
            12.0,
            onset_s=onset_s,
            frequency_hz=0.75,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "eval-multisine",
            "multisine",
            onset_s=onset_s,
            multisine_components=((7.0, 0.30), (5.0, 0.85), (3.0, 1.40)),
            duration_s=duration_s,
        ),
    )


def specialist_independent_test_commands(
    duration_s: float = 5.0,
) -> tuple[RollRateCommandProfile, ...]:
    """Frozen test-only commands that must not select or train a controller."""

    onset_s = _bounded_onset(duration_s, 0.47)
    segment_duration_s = _bounded_segment(duration_s, 0.90)
    return (
        RollRateCommandProfile(
            "test-v1-step-pos-18deg-s",
            "step",
            18.0,
            onset_s=onset_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "test-v1-step-neg-18deg-s",
            "step",
            -18.0,
            onset_s=onset_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "test-v1-step-pos-27deg-s",
            "step",
            27.0,
            onset_s=onset_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "test-v1-doublet-neg-18deg-s",
            "doublet",
            -18.0,
            onset_s=onset_s,
            segment_duration_s=segment_duration_s,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "test-v1-sine-0.43hz",
            "sine",
            11.0,
            onset_s=onset_s,
            frequency_hz=0.43,
            duration_s=duration_s,
        ),
        RollRateCommandProfile(
            "test-v1-multisine",
            "multisine",
            onset_s=onset_s,
            multisine_components=((6.0, 0.18), (4.5, 0.62), (3.0, 1.17)),
            duration_s=duration_s,
        ),
    )
