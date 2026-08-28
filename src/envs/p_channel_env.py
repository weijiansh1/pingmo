"""Gymnasium environment with 1 kHz dynamics and a 1 kHz SAC policy."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from src.aircraft.p_channel import PChannel
from src.aircraft.sampler import PlantRecord
from src.envs.commands import CommandProfile, training_command_suite
from src.envs.response_cost import OnlineResponseCostTracker
from src.envs.reward import RewardWeights, roll_quality_reward


class RollQualityEnv(gym.Env[np.ndarray, np.ndarray]):
    """Reference-free roll-quality shaping over a sampled P-channel family."""

    metadata = {"render_modes": []}
    _THETA_LOW = np.array([0.04, -0.15, 0.18, 0.005, 0.40, 0.65, 0.149, 0.001], dtype=float)
    _THETA_HIGH = np.array([0.47083875, math.log(2.0) / 4.0, 10.0, 0.70, 6.0, 1.35, 128.0, 0.25], dtype=float)
    _THETA_LOG_COLUMNS = np.array([True, False, True, True, True, True, True, False])

    def __init__(
        self,
        plants: list[PlantRecord],
        horizon_steps: int = 10_000,
        action_limit: float = 1.0,
        correction_ratio: float = 0.3,
        pilot_signal: str | None = None,
        pilot_force_scale_n: float = 22.0,
        normalized_rate_limit_s_inv: float = 4.0,
        history_steps: int = 50,
        plant_dt_s: float = 0.001,
        policy_dt_s: float = 0.001,
        actuator_time_constant_s: float = 0.0,
        actor_privileged: bool = True,
        reward_weights: RewardWeights = RewardWeights(),
        command_profiles: tuple[CommandProfile, ...] | None = None,
    ) -> None:
        if not plants:
            raise ValueError("at least one plant is required")
        if horizon_steps <= 0 or action_limit <= 0 or not 0 < correction_ratio <= 1:
            raise ValueError("invalid horizon, action limit, or correction ratio")
        if min(pilot_force_scale_n, normalized_rate_limit_s_inv, history_steps, plant_dt_s, policy_dt_s) <= 0:
            raise ValueError("force, rate, history, and time-step values must be positive")
        if actuator_time_constant_s < 0:
            raise ValueError("actuator_time_constant_s cannot be negative")
        if not np.isclose(policy_dt_s, plant_dt_s):
            raise ValueError("Stage-1 requires one SAC decision per plant sample")
        if pilot_signal not in {None, "step", "sine"}:
            raise ValueError("pilot_signal must be None, 'step', or 'sine'")
        if command_profiles is not None and not command_profiles:
            raise ValueError("command_profiles must be non-empty when supplied")

        self.plants = plants
        self.horizon_steps = horizon_steps
        self.action_limit = action_limit
        self.correction_ratio = correction_ratio
        self.pilot_signal = pilot_signal
        self.pilot_force_scale_n = pilot_force_scale_n
        self.normalized_rate_limit_s_inv = normalized_rate_limit_s_inv
        self.history_steps = history_steps
        self.plant_dt_s = plant_dt_s
        self.policy_dt_s = policy_dt_s
        self.plant_substeps = 1
        self.actuator_time_constant_s = actuator_time_constant_s
        self.actor_privileged = actor_privileged
        self.reward_weights = reward_weights
        self._episode_duration_s = horizon_steps * policy_dt_s
        self._plant_horizon_steps = horizon_steps
        self._command_profiles = tuple(command_profiles) if command_profiles is not None else None
        self._training_profiles = training_command_suite() if command_profiles is None and pilot_signal is None else ()
        self._action_force_scale_n = correction_ratio * pilot_force_scale_n
        self._max_action_increment = normalized_rate_limit_s_inv * policy_dt_s
        self._actuator_alpha = 1.0 if actuator_time_constant_s == 0 else 1.0 - math.exp(-plant_dt_s / actuator_time_constant_s)

        theta_width = 8 if actor_privileged else 0
        self._response_feedback_width = 4
        self._current_width = 6 + self._response_feedback_width
        self._history_width = 5
        observation_dim = self._current_width + history_steps * self._history_width + theta_width
        self.action_space = gym.spaces.Box(-action_limit, action_limit, (1,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (observation_dim,), np.float32)
        plant_state_width = PChannel(plants[0].parameters, dt=plant_dt_s).state.size
        critic_theta_width = 0 if actor_privileged else 8
        self.critic_state_dim = observation_dim + critic_theta_width + plant_state_width + 1

        self._rng = np.random.default_rng()
        self._history = np.zeros((history_steps, self._history_width), dtype=np.float32)
        self._episode_step = 0
        self._plant_step = 0
        self._previous_commanded_action = 0.0
        self._applied_action = 0.0
        self._response_feedback = np.zeros(self._response_feedback_width, dtype=np.float32)

    def _theta_raw(self) -> np.ndarray:
        p = self._record.parameters
        return np.array([p.l_fa, p.lambda_s, p.t_r, p.zeta_d, p.omega_d, p.r_omega, p.r_zeta, p.tau_p], dtype=np.float32)

    def _theta_normalized(self) -> np.ndarray:
        values = self._theta_raw().astype(float)
        low = self._THETA_LOW.copy()
        high = self._THETA_HIGH.copy()
        values[self._THETA_LOG_COLUMNS] = np.log(values[self._THETA_LOG_COLUMNS])
        low[self._THETA_LOG_COLUMNS] = np.log(low[self._THETA_LOG_COLUMNS])
        high[self._THETA_LOG_COLUMNS] = np.log(high[self._THETA_LOG_COLUMNS])
        return (2.0 * (values - low) / (high - low) - 1.0).astype(np.float32)

    def _legacy_commands(self) -> np.ndarray:
        time_s = np.arange(self._plant_horizon_steps, dtype=float) * self.plant_dt_s
        if self.pilot_signal == "step":
            return np.full(self._plant_horizon_steps, self.pilot_force_scale_n, dtype=float)
        return self.pilot_force_scale_n * np.sin(2.0 * math.pi * 0.5 * time_s)

    def _select_commands(self) -> None:
        profiles = self._command_profiles or self._training_profiles
        if profiles:
            self._command_profile = profiles[int(self._rng.integers(len(profiles)))]
            self._pilot_commands = self._command_profile.samples(
                self.plant_dt_s,
                self._episode_duration_s,
                self.pilot_force_scale_n,
            )
        else:
            self._command_profile = None
            self._pilot_commands = self._legacy_commands()

    def _precompute_raw_response(self) -> np.ndarray:
        raw = PChannel(self._record.parameters, dt=self.plant_dt_s)
        response = np.empty(self._plant_horizon_steps, dtype=float)
        for index, force in enumerate(self._pilot_commands):
            response[index] = raw.step(float(force))[0]
        return response

    def _normalized_current(self) -> np.ndarray:
        physical = np.array([
            self._f_pilot / self.pilot_force_scale_n,
            self._p,
            self._p_dot / 5.0,
            self._phi,
            self._previous_commanded_action,
            self._applied_action,
        ], dtype=np.float32)
        return np.concatenate((physical, self._response_feedback))

    def _history_row(self) -> np.ndarray:
        return np.array([
            self._f_pilot / self.pilot_force_scale_n,
            self._p,
            self._p_dot / 5.0,
            self._phi,
            self._applied_action,
        ], dtype=np.float32)

    def _append_history(self) -> None:
        self._history[:-1] = self._history[1:]
        self._history[-1] = self._history_row()

    def _observation(self) -> np.ndarray:
        parts = [self._normalized_current(), self._history.ravel()]
        if self.actor_privileged:
            parts.append(self._theta_normalized())
        return np.concatenate(parts).astype(np.float32)

    def _critic_state(self) -> np.ndarray:
        parts = [self._observation()]
        if not self.actor_privileged:
            parts.append(self._theta_normalized())
        parts.extend((
            self._plant.state.astype(np.float32),
            np.array([self._episode_step / self.horizon_steps], dtype=np.float32),
        ))
        return np.concatenate(parts).astype(np.float32)

    def _command_id(self) -> str:
        return self._command_profile.command_id if self._command_profile is not None else f"legacy-{self.pilot_signal}"

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._record = self.plants[int(self._rng.integers(len(self.plants)))]
        self._select_commands()
        self._raw_roll_rate = self._precompute_raw_response()
        self._roll_rate_scale = max(float(np.max(np.abs(self._raw_roll_rate))), 1e-4)
        self._plant = PChannel(self._record.parameters, dt=self.plant_dt_s)
        self._response_cost = OnlineResponseCostTracker(
            plant_dt_s=self.plant_dt_s,
            force_scale_n=self.pilot_force_scale_n,
            roll_rate_scale_rad_s=self._roll_rate_scale,
            transport_delay_s=self._record.parameters.tau_p,
            command_kind=self._command_profile.kind if self._command_profile is not None else f"legacy-{self.pilot_signal}",
        )
        self._episode_step = 0
        self._plant_step = 0
        self._previous_commanded_action = 0.0
        self._applied_action = 0.0
        self._response_feedback.fill(0.0)
        self._p = self._p_dot = self._phi = 0.0
        self._f_pilot = float(self._pilot_commands[0])
        self._history[:] = self._history_row()
        # A legacy command can begin at t=0; use the pre-episode zero force as
        # its edge baseline so timing and sensitivity remain assessable.
        self._response_cost.reset(initial_force_n=0.0)
        self._diagnostic_count = self._saturation_count = 0
        self._sum_delta = self._sum_pilot = self._sum_delta_pilot = 0.0
        self._sum_delta_squared = self._sum_pilot_squared = 0.0
        self._trace: dict[str, list[float]] = {
            "time_s": [0.0], "f_pilot_n": [self._f_pilot], "p_rad_s": [0.0],
            "phi_rad": [0.0], "raw_p_rad_s": [0.0], "delta_f_n": [0.0],
            "commanded_delta_f_n": [0.0], "reward": [0.0],
            "wrong_way_cost": [0.0], "added_delay_cost": [0.0],
            "sensitivity_cost": [0.0], "oscillation_cost": [0.0],
            "spiral_recovery_cost": [0.0], "action_energy_cost": [0.0],
            "action_delta_cost": [0.0],
        }
        return self._observation(), {
            "plant_id": self._record.plant_id,
            "quality_region": self._record.quality_region,
            "command_id": self._command_id(),
            "transport_delay_s": self._record.parameters.tau_p,
            "critic_state": self._critic_state(),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_action = float(np.clip(np.asarray(action, dtype=float)[0], -self.action_limit, self.action_limit))
        previous_command = self._previous_commanded_action
        previous_applied = self._applied_action
        commanded_action = float(np.clip(
            requested_action,
            previous_command - self._max_action_increment,
            previous_command + self._max_action_increment,
        ))
        command_delta = commanded_action - previous_command
        substep_costs = {
            "wrong_way": 0.0,
            "added_delay": 0.0,
            "sensitivity": 0.0,
            "oscillation": 0.0,
            "spiral_recovery": 0.0,
        }
        first_pilot_force = float(self._pilot_commands[self._plant_step])

        last_pilot_force = first_pilot_force
        last_delta_f = self._applied_action * self._action_force_scale_n
        last_equivalent_force = first_pilot_force + last_delta_f
        for _ in range(self.plant_substeps):
            index = self._plant_step
            pilot_force = float(self._pilot_commands[index])
            self._applied_action += self._actuator_alpha * (commanded_action - self._applied_action)
            delta_f = self._applied_action * self._action_force_scale_n
            equivalent_force = pilot_force + delta_f
            previous_p = self._p
            self._p, self._p_dot = self._plant.step(equivalent_force)
            self._phi += 0.5 * (previous_p + self._p) * self.plant_dt_s
            time_s = (index + 1) * self.plant_dt_s
            sample = self._response_cost.update(
                time_s=time_s,
                force_n=pilot_force,
                roll_rate_rad_s=self._p,
                raw_roll_rate_rad_s=float(self._raw_roll_rate[index]),
                bank_angle_rad=self._phi,
            )
            substep_costs["wrong_way"] += sample.wrong_way
            substep_costs["added_delay"] += sample.added_delay
            substep_costs["sensitivity"] += sample.sensitivity
            substep_costs["oscillation"] += sample.oscillation
            substep_costs["spiral_recovery"] += sample.spiral_recovery
            self._response_feedback[:] = (
                math.log1p(sample.wrong_way + sample.sensitivity),
                math.log1p(sample.oscillation),
                math.log1p(sample.spiral_recovery),
                math.log1p(self._response_cost.current_response_wait_s / self._response_cost.delay_scale_s),
            )
            self._diagnostic_count += 1
            self._sum_delta += delta_f
            self._sum_pilot += pilot_force
            self._sum_delta_pilot += delta_f * pilot_force
            self._sum_delta_squared += delta_f * delta_f
            self._sum_pilot_squared += pilot_force * pilot_force
            self._trace["time_s"].append(time_s)
            self._trace["f_pilot_n"].append(pilot_force)
            self._trace["p_rad_s"].append(self._p)
            self._trace["phi_rad"].append(self._phi)
            self._trace["raw_p_rad_s"].append(float(self._raw_roll_rate[index]))
            self._trace["delta_f_n"].append(delta_f)
            self._trace["commanded_delta_f_n"].append(commanded_action * self._action_force_scale_n)
            last_pilot_force = pilot_force
            last_delta_f = delta_f
            last_equivalent_force = equivalent_force
            self._plant_step += 1

        normalized_command_delta = command_delta / self._max_action_increment
        reward, reward_costs = roll_quality_reward(
            wrong_way_cost=substep_costs["wrong_way"],
            added_delay_cost=substep_costs["added_delay"],
            sensitivity_cost=substep_costs["sensitivity"],
            oscillation_cost=substep_costs["oscillation"],
            spiral_recovery_cost=substep_costs["spiral_recovery"],
            applied_action=self._applied_action,
            normalized_command_delta=normalized_command_delta,
            step_duration_s=self.policy_dt_s,
            weights=self.reward_weights,
        )
        self._trace["reward"].append(reward)
        for name, value in reward_costs.items():
            self._trace[f"{name}_cost"].append(value)
        self._saturation_count += int(abs(commanded_action) >= self.action_limit - 1e-9)
        self._previous_commanded_action = commanded_action
        self._episode_step += 1
        self._f_pilot = float(self._pilot_commands[min(self._plant_step, self._plant_horizon_steps - 1)])
        self._append_history()

        denominator_count = max(self._diagnostic_count, 1)
        cancel_index = -self._sum_delta_pilot / self._sum_pilot_squared if self._sum_pilot_squared > 0 else 0.0
        covariance = self._sum_delta_pilot - self._sum_delta * self._sum_pilot / denominator_count
        delta_variance = self._sum_delta_squared - self._sum_delta**2 / denominator_count
        pilot_variance = self._sum_pilot_squared - self._sum_pilot**2 / denominator_count
        cancel_correlation = covariance / math.sqrt(delta_variance * pilot_variance) if delta_variance > 0 and pilot_variance > 0 else 0.0
        applied_delta = self._applied_action - previous_applied
        truncated = self._episode_step >= self.horizon_steps
        info = {
            "plant_substeps": self.plant_substeps,
            "plant_dt_s": self.plant_dt_s,
            "policy_dt_s": self.policy_dt_s,
            "f_pilot": first_pilot_force,
            "f_pilot_end": self._f_pilot,
            "delta_f": last_delta_f,
            "f_eq": last_equivalent_force,
            "f_pilot_last_substep": last_pilot_force,
            "commanded_action": commanded_action,
            "applied_action": self._applied_action,
            "command_delta": command_delta,
            "applied_action_delta": applied_delta,
            "action_rate_n_per_s": applied_delta * self._action_force_scale_n / self.policy_dt_s,
            "saturation_fraction": self._saturation_count / self._episode_step,
            "cancel_index": cancel_index,
            "cancel_correlation": cancel_correlation,
            "max_added_onset_delay_s": self._response_cost.max_added_onset_delay_s,
            "current_added_onset_delay_s": self._response_cost.current_added_onset_delay_s,
            "current_response_wait_s": self._response_cost.current_response_wait_s,
            "actor_response_feedback": {
                "roll_cost": float(self._response_feedback[0]),
                "oscillation_cost": float(self._response_feedback[1]),
                "spiral_recovery_cost": float(self._response_feedback[2]),
                "delay_response_cost": float(self._response_feedback[3]),
            },
            "sensitivity_1s_deg_per_n": self._response_cost.latest_sensitivity_deg_per_n,
            "reward_costs": reward_costs,
            "critic_state": self._critic_state(),
            "plant_id": self._record.plant_id,
            "command_id": self._command_id(),
            "transport_delay_s": self._record.parameters.tau_p,
        }
        return self._observation(), float(reward), False, truncated, info

    def trajectory(self) -> dict[str, np.ndarray]:
        """Return the complete 1 ms trace collected so far."""

        return {name: np.asarray(values, dtype=float).copy() for name, values in self._trace.items()}
