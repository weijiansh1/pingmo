"""Fixed-aircraft model-reference environment for specialist SAC Teachers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np

from src.aircraft.p_channel import PChannel
from src.aircraft.sampler import PlantRecord
from src.envs.reference_model import (
    SecondOrderReferenceConfig,
    SecondOrderRollRateReference,
)
from src.envs.roll_rate_commands import (
    RollRateCommand,
    RollRateCommandProfile,
    RollRateCommandSequence,
    specialist_step_commands,
)


@dataclass(frozen=True, slots=True)
class TrackingRewardWeights:
    tracking_error: float = 1.0
    force_energy: float = 0.02
    force_delta: float = 0.02

    def __post_init__(self) -> None:
        if min(self.tracking_error, self.force_energy, self.force_delta) < 0:
            raise ValueError("tracking reward weights must be non-negative")


class SpecialistRollRateEnv(gym.Env[np.ndarray, np.ndarray]):
    """One fixed P-channel, one direct F_as action, and no theta in Actor input."""

    metadata = {"render_modes": []}
    instantaneous_observation_names = (
        "p_command_normalized",
        "p_reference_normalized",
        "p_normalized",
        "tracking_error_normalized",
    )
    controller_state_observation_names = (
        "integrated_tracking_error_normalized",
        "p_dot_normalized",
        "previous_force_normalized",
    )
    current_observation_names = (
        instantaneous_observation_names + controller_state_observation_names
    )
    history_observation_names = (
        "p_command_normalized",
        "p_reference_normalized",
        "p_normalized",
        "tracking_error_normalized",
        "force_normalized",
    )
    command_kind_names = ("step", "doublet", "sine", "multisine")
    max_multisine_components = 3
    command_frequency_scale_hz = 2.0

    def __init__(
        self,
        plant: PlantRecord,
        *,
        command_profiles: tuple[RollRateCommand, ...] | None = None,
        plant_dt_s: float = 0.001,
        policy_dt_s: float = 0.020,
        history_steps: int = 0,
        requested_action_history_steps: int = 0,
        include_actor_actuator_state: bool = False,
        include_reference_derivative: bool = False,
        critic_include_episode_progress: bool = True,
        critic_include_command_context: bool = True,
        command_scale_deg_s: float = 30.0,
        force_limit_n: float = 22.0,
        force_rate_limit_n_s: float = 88.0,
        actuator_time_constant_s: float = 0.0,
        roll_acceleration_scale_rad_s2: float = 5.0,
        integral_error_time_scale_s: float = 1.0,
        reference_config: SecondOrderReferenceConfig = SecondOrderReferenceConfig(),
        reference_delay_s: float | None = None,
        reward_weights: TrackingRewardWeights = TrackingRewardWeights(),
        reward_scale: float = 1.0,
    ) -> None:
        profiles = command_profiles or specialist_step_commands()
        if not profiles:
            raise ValueError("specialist environment requires at least one command")
        if any(
            len(profile.multisine_components) > self.max_multisine_components
            for profile in profiles
        ):
            raise ValueError(
                "specialist command exceeds the fixed multisine context width"
            )
        if critic_include_command_context and any(
            isinstance(profile, RollRateCommandSequence) for profile in profiles
        ):
            raise ValueError(
                "sequence commands require critic_include_command_context=False"
            )
        durations = {profile.duration_s for profile in profiles}
        if len(durations) != 1:
            raise ValueError("all specialist commands must use one episode duration")
        if (
            min(
                plant_dt_s,
                policy_dt_s,
                command_scale_deg_s,
                force_limit_n,
                force_rate_limit_n_s,
                roll_acceleration_scale_rad_s2,
                integral_error_time_scale_s,
                reward_scale,
            )
            <= 0
        ):
            raise ValueError("specialist environment scales must be positive")
        if history_steps < 0:
            raise ValueError("specialist history_steps cannot be negative")
        if requested_action_history_steps < 0:
            raise ValueError("requested_action_history_steps cannot be negative")
        if actuator_time_constant_s < 0:
            raise ValueError("actuator time constant cannot be negative")
        if reference_delay_s is not None and reference_delay_s < 0:
            raise ValueError("reference delay cannot be negative")

        self.record = plant
        self.command_profiles = tuple(profiles)
        ratio = policy_dt_s / plant_dt_s
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("policy_dt_s must be an integer multiple of plant_dt_s")
        self.plant_dt_s = plant_dt_s
        self.policy_dt_s = policy_dt_s
        self._plant_substeps = int(round(ratio))
        self.history_steps = history_steps
        self.requested_action_history_steps = requested_action_history_steps
        self.include_actor_actuator_state = include_actor_actuator_state
        self.include_reference_derivative = include_reference_derivative
        self.critic_include_episode_progress = critic_include_episode_progress
        self.critic_include_command_context = critic_include_command_context
        self.command_scale_rad_s = math.radians(command_scale_deg_s)
        self.force_limit_n = force_limit_n
        self.force_rate_limit_n_s = force_rate_limit_n_s
        self.max_force_increment_n = force_rate_limit_n_s * policy_dt_s
        self._max_plant_force_increment_n = force_rate_limit_n_s * plant_dt_s
        self.actuator_time_constant_s = actuator_time_constant_s
        self.actuator_alpha = (
            1.0
            if actuator_time_constant_s == 0
            else 1.0 - math.exp(-plant_dt_s / actuator_time_constant_s)
        )
        self.roll_acceleration_scale_rad_s2 = roll_acceleration_scale_rad_s2
        self.integral_error_scale_rad = (
            self.command_scale_rad_s * integral_error_time_scale_s
        )
        self.reference_config = reference_config
        self.reference_delay_s = (
            plant.parameters.tau_p if reference_delay_s is None else reference_delay_s
        )
        self.reward_weights = reward_weights
        self.reward_scale = reward_scale
        self.episode_duration_s = next(iter(durations))
        self.horizon_steps = self._horizon_steps(self.episode_duration_s)

        self._current_observation_names = self.current_observation_names + (
            ("p_reference_dot_normalized",)
            if include_reference_derivative
            else ()
        )
        self._current_width = len(self._current_observation_names)
        self._history_width = len(self.history_observation_names)
        actor_actuator_width = 2 if include_actor_actuator_state else 0
        older_action_width = max(requested_action_history_steps - 1, 0)
        observation_dim = (
            self._current_width
            + actor_actuator_width
            + older_action_width
            + history_steps * self._history_width
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (observation_dim,), np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)
        template_plant = PChannel(plant.parameters, dt=plant_dt_s)
        self._plant_state_width = template_plant.state.size
        # FractionalDelay keeps floor(tau / dt) + 3 causal samples.  Each
        # specialist has one fixed aircraft, so its critic can use the exact
        # FIFO width while the deployable Actor uses a library-wide fixed-width
        # requested-action history.
        self._plant_delay_width = (
            int(math.floor(plant.parameters.tau_p / plant_dt_s)) + 3
        )
        self._command_context_width = (
            len(self.command_kind_names) + 4 + 2 * self.max_multisine_components
        )
        critic_actuator_width = 0 if include_actor_actuator_state else 2
        critic_progress_width = 1 if critic_include_episode_progress else 0
        critic_command_width = (
            self._command_context_width if critic_include_command_context else 0
        )
        self.critic_state_dim = (
            observation_dim
            + self._plant_state_width
            + self._plant_delay_width
            + critic_actuator_width
            + critic_progress_width
            + critic_command_width
        )

        self._rng = np.random.default_rng()
        self._history = np.zeros((history_steps, self._history_width), dtype=np.float32)
        self._requested_action_history = np.zeros(
            requested_action_history_steps, dtype=np.float32
        )

    def _horizon_steps(self, duration_s: float) -> int:
        horizon_steps = int(round(duration_s / self.policy_dt_s))
        if horizon_steps <= 0 or not np.isclose(
            horizon_steps * self.policy_dt_s, duration_s
        ):
            raise ValueError(
                "episode duration must be a positive integer multiple of policy_dt_s"
            )
        return horizon_steps

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        command_profile = None if options is None else options.get("command_profile")
        if command_profile is not None and not isinstance(
            command_profile, (RollRateCommandProfile, RollRateCommandSequence)
        ):
            raise ValueError("command_profile reset option has the wrong type")
        if (
            command_profile is not None
            and options is not None
            and "command_index" in options
        ):
            raise ValueError("reset accepts command_profile or command_index, not both")
        if command_profile is not None:
            if self.critic_include_command_context and isinstance(
                command_profile, RollRateCommandSequence
            ):
                raise ValueError(
                    "sequence commands require critic_include_command_context=False"
                )
            if (
                len(command_profile.multisine_components)
                > self.max_multisine_components
            ):
                raise ValueError("reset command exceeds the multisine component limit")
            self._command_index = -1
            self._profile = command_profile
        elif options is not None and "command_index" in options:
            command_index = int(options["command_index"])
            if not 0 <= command_index < len(self.command_profiles):
                raise ValueError("specialist command_index is outside the command bank")
            self._command_index = command_index
            self._profile = self.command_profiles[command_index]
        else:
            command_index = int(self._rng.integers(len(self.command_profiles)))
            self._command_index = command_index
            self._profile = self.command_profiles[command_index]
        self.episode_duration_s = self._profile.duration_s
        self.horizon_steps = self._horizon_steps(self.episode_duration_s)
        self._command = self._profile.samples(self.plant_dt_s)
        reference = SecondOrderRollRateReference(
            self.reference_config,
            dt_s=self.plant_dt_s,
            delay_s=self.reference_delay_s,
        )
        self._reference = reference.rollout(self._command)
        self._reference_dot = np.zeros_like(self._reference)
        self._reference_dot[1:] = np.diff(self._reference) / self.plant_dt_s
        self._plant = PChannel(self.record.parameters, dt=self.plant_dt_s)
        self._episode_step = 0
        self._plant_step = 0
        self._p = 0.0
        self._p_dot = 0.0
        self._integral_error_rad = 0.0
        self._requested_force_n = 0.0
        self._commanded_force_n = 0.0
        self._applied_force_n = 0.0
        self._requested_action_history[:] = 0.0
        self._history[:] = self._history_row()
        self._saturation_count = 0
        self._trace: dict[str, list[float]] = {
            "time_s": [0.0],
            "p_command_rad_s": [float(self._command[0])],
            "p_reference_rad_s": [0.0],
            "p_reference_dot_rad_s2": [0.0],
            "p_rad_s": [0.0],
            "f_as_n": [0.0],
            "requested_f_as_n": [0.0],
            "commanded_f_as_n": [0.0],
            "reward": [0.0],
            "tracking_error_cost": [0.0],
            "force_energy_cost": [0.0],
            "force_delta_cost": [0.0],
        }
        return self._observation(), self._info()

    def _current_command(self) -> float:
        return float(self._command[min(self._plant_step, len(self._command) - 1)])

    def _current_reference(self) -> float:
        return float(self._reference[min(self._plant_step, len(self._reference) - 1)])

    def _current_reference_derivative(self) -> float:
        return float(
            self._reference_dot[
                min(self._plant_step, len(self._reference_dot) - 1)
            ]
        )

    def _normalized_current(self) -> np.ndarray:
        reference = self._current_reference()
        error = reference - self._p
        values = [
            self._current_command() / self.command_scale_rad_s,
            reference / self.command_scale_rad_s,
            self._p / self.command_scale_rad_s,
            error / self.command_scale_rad_s,
            self._integral_error_rad / self.integral_error_scale_rad,
            self._p_dot / self.roll_acceleration_scale_rad_s2,
            self._requested_force_n / self.force_limit_n,
        ]
        if self.include_reference_derivative:
            values.append(
                self._current_reference_derivative()
                / self.roll_acceleration_scale_rad_s2
            )
        return np.asarray(values, dtype=np.float32)

    def _history_row(self) -> np.ndarray:
        reference = self._current_reference()
        error = reference - self._p
        return np.asarray(
            [
                self._current_command() / self.command_scale_rad_s,
                reference / self.command_scale_rad_s,
                self._p / self.command_scale_rad_s,
                error / self.command_scale_rad_s,
                self._requested_force_n / self.force_limit_n,
            ],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        parts = [self._normalized_current()]
        if self.include_actor_actuator_state:
            parts.append(
                np.asarray(
                    [
                        self._commanded_force_n / self.force_limit_n,
                        self._applied_force_n / self.force_limit_n,
                    ],
                    dtype=np.float32,
                )
            )
        if self.requested_action_history_steps > 1:
            parts.append(self._requested_action_history[1:])
        if self.history_steps:
            parts.append(self._history.ravel())
        return np.concatenate(parts).astype(np.float32)

    def actor_observation_contract(self) -> dict[str, object]:
        history_names = [
            f"raw_history_oldest_to_newest[{step}].{name}"
            for step in range(self.history_steps)
            for name in self.history_observation_names
        ]
        actuator_names = (
            ["commanded_force_normalized", "applied_force_normalized"]
            if self.include_actor_actuator_state
            else []
        )
        older_action_names = [
            f"requested_force_lag_{lag}_normalized"
            for lag in range(2, self.requested_action_history_steps + 1)
        ]
        controller_state_names = list(self.controller_state_observation_names)
        if self.include_reference_derivative:
            controller_state_names.append("p_reference_dot_normalized")
        return {
            "names": (
                list(self._current_observation_names)
                + actuator_names
                + older_action_names
                + history_names
            ),
            "instantaneous_signal_names": list(self.instantaneous_observation_names),
            "controller_state_names": controller_state_names,
            "raw_history_steps": self.history_steps,
            "raw_history_signal_names": list(self.history_observation_names),
            "uses_raw_history_window": self.history_steps > 0,
            "includes_actor_actuator_state": self.include_actor_actuator_state,
            "includes_reference_derivative": self.include_reference_derivative,
            "reference_derivative_scale_rad_s2": (
                self.roll_acceleration_scale_rad_s2
            ),
            "requested_action_history_steps": self.requested_action_history_steps,
            "requested_action_history_order": "newest_in_previous_force_then_lag_2_to_lag_k",
            "delay_coverage_s": (
                self.requested_action_history_steps * self.policy_dt_s
            ),
        }

    def critic_observation_contract(self) -> dict[str, object]:
        actor_names = list(self.actor_observation_contract()["names"])
        plant_state_names = [
            f"plant_continuous_state[{index}]"
            for index in range(self._plant_state_width)
        ]
        delay_names = [
            f"plant_force_delay_fifo_oldest_to_newest[{index}]_normalized"
            for index in range(self._plant_delay_width)
        ]
        actuator_names = (
            []
            if self.include_actor_actuator_state
            else ["commanded_force_normalized", "applied_force_normalized"]
        )
        progress_names = (
            ["episode_progress"] if self.critic_include_episode_progress else []
        )
        command_names = (
            [
                *(f"command_kind_one_hot.{kind}" for kind in self.command_kind_names),
                "command_amplitude_normalized",
                "command_onset_fraction",
                "command_segment_duration_fraction",
                "command_frequency_normalized",
                *(
                    name
                    for index in range(self.max_multisine_components)
                    for name in (
                        f"command_multisine[{index}].amplitude_normalized",
                        f"command_multisine[{index}].frequency_normalized",
                    )
                ),
            ]
            if self.critic_include_command_context
            else []
        )
        return {
            "names": (
                actor_names
                + plant_state_names
                + delay_names
                + actuator_names
                + progress_names
                + command_names
            ),
            "deployment_input": False,
            "includes_actor_observation": True,
            "plant_continuous_state_width": self._plant_state_width,
            "transport_delay_fifo_width": self._plant_delay_width,
            "transport_delay_fifo_normalized_by_force_limit": True,
            "includes_actuator_state": True,
            "includes_episode_progress": self.critic_include_episode_progress,
            "includes_command_context": self.critic_include_command_context,
            "command_context_encoding": (
                "fixed_parametric_profile_v1"
                if self.critic_include_command_context
                else "none"
            ),
            "command_context_width": len(command_names),
            "command_frequency_scale_hz": self.command_frequency_scale_hz,
            "maximum_multisine_components": self.max_multisine_components,
            "future_reference_determined_by_profile_and_progress": (
                self.critic_include_episode_progress
                and self.critic_include_command_context
            ),
        }

    def _command_context(self) -> np.ndarray:
        profile = self._profile
        if not isinstance(profile, RollRateCommandProfile):
            raise ValueError("sequence command has no fixed parametric critic context")
        try:
            kind_index = self.command_kind_names.index(profile.kind)
        except ValueError as error:
            raise ValueError(
                f"unsupported command kind in critic context: {profile.kind}"
            ) from error
        values = np.zeros(self._command_context_width, dtype=np.float32)
        values[kind_index] = 1.0
        offset = len(self.command_kind_names)
        values[offset] = (
            math.radians(profile.amplitude_deg_s) / self.command_scale_rad_s
        )
        values[offset + 1] = profile.onset_s / self.episode_duration_s
        values[offset + 2] = (
            0.0
            if profile.segment_duration_s is None
            else profile.segment_duration_s / self.episode_duration_s
        )
        values[offset + 3] = (
            0.0
            if profile.frequency_hz is None
            else profile.frequency_hz / self.command_frequency_scale_hz
        )
        component_offset = offset + 4
        for index, (amplitude_deg_s, frequency_hz) in enumerate(
            profile.multisine_components
        ):
            values[component_offset + 2 * index] = (
                math.radians(amplitude_deg_s) / self.command_scale_rad_s
            )
            values[component_offset + 2 * index + 1] = (
                frequency_hz / self.command_frequency_scale_hz
            )
        return values

    def _critic_state(self) -> np.ndarray:
        privileged_plant = self._plant.privileged_state(self._plant_delay_width)
        continuous_state = privileged_plant[: self._plant_state_width]
        delay_state = privileged_plant[self._plant_state_width :] / self.force_limit_n
        parts = [self._observation(), continuous_state, delay_state]
        if not self.include_actor_actuator_state:
            parts.append(
                np.asarray(
                    [
                        self._commanded_force_n / self.force_limit_n,
                        self._applied_force_n / self.force_limit_n,
                    ],
                    dtype=np.float32,
                )
            )
        if self.critic_include_episode_progress:
            parts.append(
                np.asarray([self._episode_step / self.horizon_steps], dtype=np.float32)
            )
        if self.critic_include_command_context:
            parts.append(self._command_context())
        return np.concatenate(parts).astype(np.float32)

    def _info(self) -> dict[str, object]:
        return {
            "plant_id": self.record.plant_id,
            "command_id": self._profile.command_id,
            "command_kind": self._profile.kind,
            "critic_state": self._critic_state(),
            "f_as_n": self._applied_force_n,
            "requested_f_as_n": self._requested_force_n,
            "commanded_f_as_n": self._commanded_force_n,
            "action_saturation_fraction": self._saturation_count
            / max(self._episode_step, 1),
            "actor_receives_theta": False,
            "reference_delay_s": self.reference_delay_s,
            "episode_duration_s": self.episode_duration_s,
            "requested_action_history_steps": self.requested_action_history_steps,
            "actor_includes_reference_derivative": (
                self.include_reference_derivative
            ),
            "critic_includes_command_context": self.critic_include_command_context,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_values = np.asarray(action, dtype=float).reshape(-1)
        if action_values.size != 1 or not np.isfinite(action_values[0]):
            raise ValueError(
                "specialist action must contain one finite normalized force"
            )
        requested_normalized = float(np.clip(action_values[0], -1.0, 1.0))
        requested_force_n = requested_normalized * self.force_limit_n
        previous_requested_normalized = self._requested_force_n / self.force_limit_n
        self._requested_force_n = requested_force_n
        if self.requested_action_history_steps:
            self._requested_action_history[1:] = self._requested_action_history[
                :-1
            ].copy()
            self._requested_action_history[0] = requested_normalized
        previous_p = self._p
        for _ in range(self._plant_substeps):
            force_error = requested_force_n - self._commanded_force_n
            self._commanded_force_n += float(
                np.clip(
                    force_error,
                    -self._max_plant_force_increment_n,
                    self._max_plant_force_increment_n,
                )
            )
            self._applied_force_n += self.actuator_alpha * (
                self._commanded_force_n - self._applied_force_n
            )
            self._p, _ = self._plant.step(self._applied_force_n)
            self._plant_step += 1
            self._integral_error_rad += (
                self._current_reference() - self._p
            ) * self.plant_dt_s
        self._p_dot = (self._p - previous_p) / self.policy_dt_s
        self._episode_step += 1

        reference = self._current_reference()
        normalized_error = (reference - self._p) / self.command_scale_rad_s
        normalized_force = self._applied_force_n / self.force_limit_n
        normalized_delta = requested_normalized - previous_requested_normalized
        costs = {
            "tracking_error": self.reward_weights.tracking_error
            * normalized_error**2
            * self.policy_dt_s,
            "force_energy": self.reward_weights.force_energy
            * normalized_force**2
            * self.policy_dt_s,
            # This is a per-decision jump cost, not a time integral. Multiplying
            # by policy_dt_s would weaken the requested-action penalty by 50x at
            # the default 20 ms policy period.
            "force_delta": self.reward_weights.force_delta * normalized_delta**2,
        }
        costs = {name: self.reward_scale * value for name, value in costs.items()}
        reward = -sum(costs.values())
        self._saturation_count += int(
            abs(self._commanded_force_n) >= self.force_limit_n - 1e-9
        )

        if self.history_steps:
            self._history[:-1] = self._history[1:]
            self._history[-1] = self._history_row()
        time_s = self._episode_step * self.policy_dt_s
        self._trace["time_s"].append(time_s)
        self._trace["p_command_rad_s"].append(self._current_command())
        self._trace["p_reference_rad_s"].append(reference)
        self._trace["p_reference_dot_rad_s2"].append(
            self._current_reference_derivative()
        )
        self._trace["p_rad_s"].append(self._p)
        self._trace["f_as_n"].append(self._applied_force_n)
        self._trace["requested_f_as_n"].append(self._requested_force_n)
        self._trace["commanded_f_as_n"].append(self._commanded_force_n)
        self._trace["reward"].append(reward)
        for name, value in costs.items():
            self._trace[f"{name}_cost"].append(value)

        truncated = self._episode_step >= self.horizon_steps
        return self._observation(), float(reward), False, truncated, self._info()

    def trajectory(self) -> dict[str, np.ndarray]:
        return {
            name: np.asarray(values, dtype=float).copy()
            for name, values in self._trace.items()
        }


class CommandForceBaseline:
    """Map normalized p_c directly to normalized full force for an open-loop baseline."""

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        values = np.asarray(observation, dtype=float)
        if values.ndim == 1:
            return np.asarray([np.clip(values[0], -1.0, 1.0)], dtype=np.float32)
        return np.clip(values[:, :1], -1.0, 1.0).astype(np.float32)
