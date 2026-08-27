"""50 Hz Gymnasium environment over a 200 Hz P-channel simulation."""

import gymnasium as gym
import numpy as np
import math

from src.aircraft.p_channel import PChannel
from src.aircraft.reference import ReferenceRollModel
from src.aircraft.sampler import PlantRecord
from src.envs.reward import RewardWeights, roll_quality_reward


class RollQualityEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, plants: list[PlantRecord], horizon_steps: int = 250, action_limit: float = 1.0, correction_ratio: float = 0.3, pilot_signal: str = "sine", pilot_force_scale_n: float = 22.0, normalized_rate_limit_s_inv: float = 4.0, history_steps: int = 32, actuator_time_constant_s: float = 0.08, reward_weights: RewardWeights = RewardWeights()) -> None:
        if not plants:
            raise ValueError("at least one plant is required")
        if not 0 < correction_ratio <= 1:
            raise ValueError("correction_ratio must be in (0, 1]")
        if pilot_signal not in {"sine", "step"}:
            raise ValueError("pilot_signal must be 'sine' or 'step'")
        if pilot_force_scale_n <= 0 or normalized_rate_limit_s_inv <= 0 or history_steps <= 0 or actuator_time_constant_s <= 0:
            raise ValueError("force scale, rate limit, history steps, and actuator time constant must be positive")
        self.plants, self.horizon_steps, self.action_limit, self.correction_ratio, self.pilot_signal = plants, horizon_steps, action_limit, correction_ratio, pilot_signal
        self.pilot_force_scale_n = pilot_force_scale_n
        self.normalized_rate_limit_s_inv = normalized_rate_limit_s_inv
        self.history_steps = history_steps
        self._action_dt = 0.02
        self.actuator_time_constant_s = actuator_time_constant_s
        self.reward_weights = reward_weights
        self._actuator_alpha = 1.0 - math.exp(-self._action_dt / actuator_time_constant_s)
        self.action_space = gym.spaces.Box(-action_limit, action_limit, (1,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (6 + history_steps * 4 + 8,), np.float32)
        self._delay_state_width = max(int(math.floor(record.parameters.tau_p / 0.005)) + 3 for record in plants)
        self.critic_state_dim = self.observation_space.shape[0] + 6 + 2 * self._delay_state_width
        self._rng = np.random.default_rng()
        self._episode_step = 0
        self._previous_commanded_action = 0.0
        self._applied_action = 0.0
        self._history = np.zeros((history_steps, 4), dtype=np.float32)
        self._diagnostic_count = 0
        self._saturation_count = 0
        self._sum_delta = self._sum_pilot = self._sum_delta_pilot = 0.0
        self._sum_delta_squared = self._sum_pilot_squared = 0.0

    def _theta(self) -> np.ndarray:
        p = self._record.parameters
        return np.array([p.l_fa, p.lambda_s, p.t_r, p.zeta_d, p.omega_d, p.r_omega, p.r_zeta, p.tau_p], dtype=np.float32)

    def _pilot_command(self) -> float:
        return self.pilot_force_scale_n if self.pilot_signal == "step" else float(self.pilot_force_scale_n * np.sin(0.04 * self._episode_step))

    def _append_history(self) -> None:
        self._history[:-1] = self._history[1:]
        self._history[-1] = np.array([self._f_pilot, self._p, self._p_ref, self._applied_action * self.correction_ratio * self.pilot_force_scale_n], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        error = self._p - self._p_ref
        current = np.array([
            self._f_pilot,
            self._p,
            self._p_ref,
            error,
            self._previous_commanded_action * self.correction_ratio * self.pilot_force_scale_n,
            self._applied_action * self.correction_ratio * self.pilot_force_scale_n,
        ], dtype=np.float32)
        return np.concatenate((current, self._history.ravel(), self._theta())).astype(np.float32)

    def _critic_state(self) -> np.ndarray:
        return np.concatenate((
            self._observation(),
            self._plant.privileged_state(self._delay_state_width),
            self._reference.privileged_state(self._delay_state_width),
        )).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self._record = self.plants[int(self._rng.integers(len(self.plants)))]
        self._plant = PChannel(self._record.parameters, dt=0.005)
        self._reference = ReferenceRollModel(self._record.parameters, dt=0.005)
        self._episode_step = 0
        self._previous_commanded_action = 0.0
        self._applied_action = 0.0
        self._diagnostic_count = self._saturation_count = 0
        self._sum_delta = self._sum_pilot = self._sum_delta_pilot = 0.0
        self._sum_delta_squared = self._sum_pilot_squared = 0.0
        self._f_pilot, self._p, self._p_dot, self._p_ref = self._pilot_command(), 0.0, 0.0, 0.0
        self._history[:] = np.array([self._f_pilot, 0.0, 0.0, 0.0], dtype=np.float32)
        return self._observation(), {
            "plant_id": self._record.plant_id,
            "quality_region": self._record.quality_region,
            "critic_state": self._critic_state(),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        requested_action = float(np.clip(np.asarray(action, dtype=float)[0], -self.action_limit, self.action_limit))
        previous_command = self._previous_commanded_action
        previous_applied = self._applied_action
        max_action_increment = self.normalized_rate_limit_s_inv * self._action_dt
        commanded_action = float(np.clip(requested_action, previous_command - max_action_increment, previous_command + max_action_increment))
        applied_action = previous_applied + self._actuator_alpha * (commanded_action - previous_applied)
        command_delta = commanded_action - previous_command
        applied_action_delta = applied_action - previous_applied
        delta = applied_action * self.correction_ratio * self.pilot_force_scale_n
        previous_delta = previous_applied * self.correction_ratio * self.pilot_force_scale_n
        pilot_force = self._f_pilot
        f_eq = pilot_force + delta
        for _ in range(4):
            self._p, self._p_dot = self._plant.step(f_eq)
            self._p_ref, _ = self._reference.step(pilot_force)
        p_scale = max(abs(self._record.parameters.l_fa * self.pilot_force_scale_n), 1.0)
        normalized_error = (self._p - self._p_ref) / p_scale
        late_error = normalized_error if self.pilot_signal == "step" and self._episode_step * self._action_dt >= 1.0 else 0.0
        reward = roll_quality_reward(
            normalized_error,
            commanded_action,
            command_delta,
            applied_action_delta=applied_action_delta,
            late_error=late_error,
            weights=self.reward_weights,
        )
        self._diagnostic_count += 1
        self._saturation_count += int(abs(commanded_action) >= self.action_limit - 1e-9)
        self._sum_delta += delta
        self._sum_pilot += pilot_force
        self._sum_delta_pilot += delta * pilot_force
        self._sum_delta_squared += delta * delta
        self._sum_pilot_squared += pilot_force * pilot_force
        cancel_index = -self._sum_delta_pilot / self._sum_pilot_squared if self._sum_pilot_squared > 0 else 0.0
        covariance = self._sum_delta_pilot - self._sum_delta * self._sum_pilot / self._diagnostic_count
        delta_variance = self._sum_delta_squared - self._sum_delta**2 / self._diagnostic_count
        pilot_variance = self._sum_pilot_squared - self._sum_pilot**2 / self._diagnostic_count
        cancel_correlation = covariance / math.sqrt(delta_variance * pilot_variance) if delta_variance > 0 and pilot_variance > 0 else 0.0
        self._previous_commanded_action = commanded_action
        self._applied_action = applied_action
        self._episode_step += 1
        self._f_pilot = self._pilot_command()
        self._append_history()
        truncated = self._episode_step >= self.horizon_steps
        return self._observation(), float(reward), False, truncated, {
            "plant_substeps": 4, "f_pilot": pilot_force, "delta_f": delta, "f_eq": f_eq,
            "commanded_action": commanded_action, "applied_action": applied_action,
            "command_delta": command_delta, "applied_action_delta": applied_action_delta,
            "action_rate_n_per_s": (delta - previous_delta) / self._action_dt,
            "saturation_fraction": self._saturation_count / self._diagnostic_count,
            "cancel_index": cancel_index, "cancel_correlation": cancel_correlation,
            "critic_state": self._critic_state(), "plant_id": self._record.plant_id, "p_ref": self._p_ref,
        }
