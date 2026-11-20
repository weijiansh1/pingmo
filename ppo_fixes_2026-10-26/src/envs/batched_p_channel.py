"""Vectorised torch P-channel environment for PPO Teachers.

One plant per environment batch (a Teacher is per-aircraft): the batch dimension
holds independent episodes that share the same parameters but draw different
commands.  Everything runs on the GPU in a single tensor program:

* plant     4-state ZOH discretisation of the P-channel at 1 kHz
* actuator  force-rate action -> 88 N/s ramp -> 20 ms first-order lag
* command   step / doublet / sine / multisine, drawn per episode
* reference 2nd-order 2 rad/s / 0.7 model, shifted by the plant transport delay

The observation contract is the deployment contract: it is what the Student will
later have to reproduce without theta.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy import signal

from src.aircraft.gain_calibration import sensitivity_1s_deg_per_n
from src.aircraft.p_channel import p_channel_polynomials
from src.aircraft.parameters import PChannelParameters
from src.context.aircraft_parameters import normalize_aircraft_parameters
from src.envs.roll_rate_commands import (
    RollRateCommand,
    specialist_evaluation_commands,
    specialist_step_commands,
)

DT_PLANT = 0.001
DT_POLICY = 0.02
SUBSTEPS = int(round(DT_POLICY / DT_PLANT))
FORCE_LIMIT_N = 22.0
FORCE_RATE_LIMIT_N_S = 88.0
COMMAND_SCALE_RAD_S = math.radians(30.0)
P_DOT_SCALE = 5.0
INTEGRAL_TIME_SCALE_S = 1.0
HISTORY_STEPS = 32
OLDER_ACTIONS = 7
REFERENCE_OMEGA = 2.0
REFERENCE_ZETA = 0.7
ACTUATOR_TAU_S = 0.02
EPISODE_STEPS = 400          # 8 s at 50 Hz
COMMAND_ONSET_S = 1.0

REWARD_TRACKING = 1.0
REWARD_ENERGY = 0.02
REWARD_DELTA = 0.02
REWARD_SCALE = 50.0

CURRENT_DIM = 9
HISTORY_CHANNELS = 5
THETA_DIM = 8
OBS_DIM = CURRENT_DIM + HISTORY_STEPS * HISTORY_CHANNELS + OLDER_ACTIONS


def build_command_set() -> list[RollRateCommand]:
    """Four command families weighted 4:3:2:1 by repetition, stretched to 8 s."""
    def stretch(profiles: list[RollRateCommand]) -> list[RollRateCommand]:
        return [dataclasses.replace(p, duration_s=EPISODE_STEPS * DT_POLICY,
                                    onset_s=COMMAND_ONSET_S) for p in profiles]

    steps = stretch(specialist_step_commands())
    extra = stretch([c for c in specialist_evaluation_commands()
                     if c.command_id not in {s.command_id for s in steps}])
    doublets = [c for c in extra if c.command_id.startswith("eval-doublet")]
    sines = [c for c in extra if c.command_id.startswith("eval-sine")]
    multis = [c for c in extra if c.command_id.startswith("eval-multisine")]
    if not (doublets and sines and multis):
        raise RuntimeError("command set is missing doublet/sine/multisine profiles")
    return steps * 4 + doublets * 3 + sines * 2 + multis


@dataclass
class PlantBatch:
    """Discretised plant plus the reference tables needed by the batch."""

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    commands: torch.Tensor        # (n_cmd, T) rad/s
    reference: torch.Tensor       # (n_cmd, T) rad/s
    reference_dot: torch.Tensor   # (n_cmd, T) rad/s^2
    delay_len: int
    transport_delay_s: float
    sensitivity_deg_per_n: float
    theta: torch.Tensor           # (8,) normalised aircraft parameters
    state_scale: torch.Tensor     # (4,) peak |state| under a 1 N step


def _discretise(parameters: PChannelParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numerator, denominator = p_channel_polynomials(parameters)
    a, b, c, d = signal.tf2ss(numerator, denominator)
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), DT_PLANT, method="zoh")
    if abs(float(dd.squeeze())) > 1e-9:
        raise ValueError("P-channel must be strictly proper")
    # The fractional transport delay is modelled by the FIFO, not by the state space.
    return np.asarray(ad), np.asarray(bd), np.asarray(cd)


def _state_scale(ad: np.ndarray, bd: np.ndarray) -> np.ndarray:
    """Peak |state| under a 1 N step, used to normalise the privileged state."""
    x = np.zeros(ad.shape[0])
    peak = np.zeros(ad.shape[0])
    for _ in range(int(8.0 / DT_PLANT)):
        peak = np.maximum(peak, np.abs(x))
        x = ad @ x + bd[:, 0]
    return np.maximum(peak, 1e-9)


def _second_order_reference(command: np.ndarray, delay_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """2 rad/s / 0.7 reference at plant rate, delayed like the plant."""
    omega, zeta = REFERENCE_OMEGA, REFERENCE_ZETA
    a = np.array([[0.0, 1.0], [-omega ** 2, -2 * zeta * omega]])
    b = np.array([[0.0], [omega ** 2]])
    c = np.array([[1.0, 0.0]])
    d = np.zeros((1, 1))
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), DT_PLANT, method="zoh")
    x = np.zeros(2)
    out = np.empty(len(command))
    for k, u in enumerate(command):
        out[k] = float((cd @ x + dd.squeeze() * u).item())
        x = ad @ x + bd[:, 0] * u
    if delay_samples > 0:
        out = np.concatenate([np.full(delay_samples, out[0]), out[:-delay_samples]])
    dot = np.gradient(out, DT_PLANT)
    return out, dot


def build_plant_batch(parameters: PChannelParameters, commands: list[RollRateCommand]) -> PlantBatch:
    ad, bd, cd = _discretise(parameters)
    steps = EPISODE_STEPS
    plant_index = np.arange(steps) * SUBSTEPS
    delay_samples = int(round(parameters.tau_p / DT_PLANT))
    cmd_rows, ref_rows, dot_rows = [], [], []
    for profile in commands:
        samples = profile.samples(DT_PLANT)            # plant rate, onset applied
        ref_plant, dot_plant = _second_order_reference(samples, delay_samples)
        cmd_rows.append(samples[plant_index])
        ref_rows.append(ref_plant[plant_index])
        dot_rows.append(dot_plant[plant_index])
        if len(samples) < (steps - 1) * SUBSTEPS + 1:
            raise ValueError(f"command {profile.command_id} shorter than the episode")
    return PlantBatch(
        a=torch.as_tensor(ad, dtype=torch.float32),
        b=torch.as_tensor(bd, dtype=torch.float32),
        c=torch.as_tensor(cd, dtype=torch.float32),
        theta=torch.as_tensor(normalize_aircraft_parameters(parameters),
                              dtype=torch.float32),
        state_scale=torch.as_tensor(_state_scale(ad, bd), dtype=torch.float32),
        commands=torch.as_tensor(np.stack(cmd_rows), dtype=torch.float32),
        reference=torch.as_tensor(np.stack(ref_rows), dtype=torch.float32),
        reference_dot=torch.as_tensor(np.stack(dot_rows), dtype=torch.float32),
        delay_len=max(delay_samples + 1, 2),
        transport_delay_s=float(parameters.tau_p),
        sensitivity_deg_per_n=sensitivity_1s_deg_per_n(parameters, dt=DT_PLANT),
    )


class BatchedRollRateEnv:
    """Vectorised environment; ``step`` takes normalised force-rate actions."""

    def __init__(self, parameters: PChannelParameters, n_envs: int,
                 commands: list[RollRateCommand] | None = None,
                 device: str = "cpu", seed: int = 0) -> None:
        self.n_envs = n_envs
        self.device = torch.device(device)
        self.plant = build_plant_batch(parameters, commands or build_command_set())
        self.n_cmd = self.plant.commands.shape[0]
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        for name in ("a", "b", "c", "commands", "reference", "reference_dot",
                     "theta", "state_scale"):
            setattr(self.plant, name, getattr(self.plant, name).to(self.device))
        self.alpha_lag = 1.0 - math.exp(-DT_PLANT / ACTUATOR_TAU_S)
        self.max_step_n = FORCE_RATE_LIMIT_N_S * DT_POLICY
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        if not hasattr(self, "state") or mask is None:
            self.state = torch.zeros(self.n_envs, 4, device=self.device)
            self.delay = torch.zeros(self.n_envs, self.plant.delay_len, device=self.device)
            self.delay_index = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
            self.applied = torch.zeros(self.n_envs, device=self.device)
            self.commanded = torch.zeros(self.n_envs, device=self.device)
            self.requested = torch.zeros(self.n_envs, device=self.device)
            self.integral = torch.zeros(self.n_envs, device=self.device)
            self.history = torch.zeros(self.n_envs, HISTORY_STEPS, HISTORY_CHANNELS,
                                       device=self.device)
            self.older = torch.zeros(self.n_envs, OLDER_ACTIONS, device=self.device)
            self.command_index = torch.randint(0, self.n_cmd, (self.n_envs,),
                                               generator=self.generator).to(self.device)
            self.t = torch.zeros(self.n_envs, dtype=torch.long, device=self.device)
            self._output_dot = torch.zeros(self.n_envs, device=self.device)
            self.reset_mask = torch.ones(self.n_envs, dtype=torch.bool, device=self.device)
            return self.observation()

        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        self.state[idx] = 0.0
        self.delay[idx] = 0.0
        self.delay_index[idx] = 0
        self.applied[idx] = 0.0
        self.commanded[idx] = 0.0
        self.requested[idx] = 0.0
        self.integral[idx] = 0.0
        self._output_dot[idx] = 0.0
        self.history[idx] = 0.0
        self.older[idx] = 0.0
        self.command_index[idx] = torch.randint(0, self.n_cmd, (idx.numel(),),
                                                generator=self.generator).to(self.device)
        self.t[idx] = 0
        return self.observation()

    # -------------------------------------------------------------- reference
    def _lookup(self, table: torch.Tensor) -> torch.Tensor:
        step = self.t.clamp(max=EPISODE_STEPS - 1)
        return table[self.command_index, step]

    @property
    def reference(self) -> torch.Tensor:
        return self._lookup(self.plant.reference)

    @property
    def reference_dot(self) -> torch.Tensor:
        return self._lookup(self.plant.reference_dot)

    @property
    def command(self) -> torch.Tensor:
        return self._lookup(self.plant.commands)

    # ------------------------------------------------------------- observation
    def observation(self) -> torch.Tensor:
        ref, cmd = self.reference, self.command
        output = (self.state @ self.plant.c.squeeze(0))
        error = ref - output
        dot = getattr(self, "_output_dot", torch.zeros_like(output))
        current = torch.stack([
            cmd / COMMAND_SCALE_RAD_S,
            ref / COMMAND_SCALE_RAD_S,
            output / COMMAND_SCALE_RAD_S,
            error / COMMAND_SCALE_RAD_S,
            self.integral / (COMMAND_SCALE_RAD_S * INTEGRAL_TIME_SCALE_S),
            dot / P_DOT_SCALE,
            self.reference_dot / P_DOT_SCALE,
            self.requested / FORCE_LIMIT_N,
            self.applied / FORCE_LIMIT_N,
        ], dim=-1)
        return torch.cat([current, self.history.reshape(self.n_envs, -1), self.older], dim=-1)

    def actor_privileged(self) -> torch.Tensor:
        """theta and the plant internal state: inputs the deployed Student never gets."""
        state = self.state / self.plant.state_scale.unsqueeze(0)
        theta = self.plant.theta.unsqueeze(0).expand(self.n_envs, -1)
        return torch.cat([theta, state], dim=-1)

    def privileged(self) -> torch.Tensor:
        """theta + plant internals + pending delay queue + progress (critic only)."""
        progress = (self.t.float() / EPISODE_STEPS).unsqueeze(-1)
        theta = self.plant.theta.unsqueeze(0).expand(self.n_envs, -1)
        return torch.cat([
            theta,
            self.state,
            self.delay / FORCE_LIMIT_N,
            progress,
            self.commanded.unsqueeze(-1) / FORCE_LIMIT_N,
        ], dim=-1)

    # -------------------------------------------------------------------- step
    def step(self, action_normalized: torch.Tensor, *, auto_reset: bool = True
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance one policy step, optionally retaining the final state.

        PPO uses ``auto_reset=False`` through its full rollout so value
        bootstrapping sees the state before a time-limit reset. Existing
        evaluation callers retain the default automatic reset behavior.
        """
        action = action_normalized.clamp(-1.0, 1.0).squeeze(-1)
        previous_output = (self.state @ self.plant.c.squeeze(0))
        self.requested = (self.requested + action * self.max_step_n).clamp(
            -FORCE_LIMIT_N, FORCE_LIMIT_N)
        max_increment = FORCE_RATE_LIMIT_N_S * DT_PLANT
        ref_step = self.t.clamp(max=EPISODE_STEPS - 1)
        reference = self.plant.reference[self.command_index, ref_step]
        for _ in range(SUBSTEPS):
            self.commanded = self.commanded + (self.requested - self.commanded).clamp(
                -max_increment, max_increment)
            self.applied = self.applied + self.alpha_lag * (self.commanded - self.applied)
            self.delay.scatter_(1, self.delay_index.unsqueeze(-1), self.applied.unsqueeze(-1))
            self.delay_index = (self.delay_index + 1) % self.plant.delay_len
            delayed = self.delay.gather(1, self.delay_index.unsqueeze(-1)).squeeze(-1)
            self.state = (self.state @ self.plant.a.transpose(0, 1)
                          + delayed.unsqueeze(-1) * self.plant.b.squeeze(-1).unsqueeze(0))
            output = (self.state @ self.plant.c.squeeze(0))
            self.integral = self.integral + (reference - output) * DT_PLANT
        output = (self.state @ self.plant.c.squeeze(0))
        self._output_dot = (output - previous_output) / DT_POLICY
        error = reference - output
        tracking = REWARD_TRACKING * (error / COMMAND_SCALE_RAD_S).pow(2) * DT_POLICY
        energy = REWARD_ENERGY * (self.applied / FORCE_LIMIT_N).pow(2) * DT_POLICY
        delta = REWARD_DELTA * action.pow(2)
        reward = -REWARD_SCALE * (tracking + energy + delta)
        self._advance_history(action, reference, output)
        self.t = self.t + 1
        truncated = self.t >= EPISODE_STEPS
        if auto_reset and bool(truncated.any()):
            self.reset(truncated)
        return self.observation(), reward, output

    def _advance_history(self, action: torch.Tensor, reference: torch.Tensor,
                         output: torch.Tensor) -> None:
        error = reference - output
        row = torch.stack([
            self.command / COMMAND_SCALE_RAD_S,
            reference / COMMAND_SCALE_RAD_S,
            output / COMMAND_SCALE_RAD_S,
            error / COMMAND_SCALE_RAD_S,
            self.requested / FORCE_LIMIT_N,
        ], dim=-1)
        self.history = torch.cat([self.history[:, 1:], row.unsqueeze(1)], dim=1)
        self.older = torch.cat([self.older[:, 1:], action.unsqueeze(-1)], dim=1)





