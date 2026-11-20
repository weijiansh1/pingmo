"""Shared pieces for the delay-representation experiment (2026-10-23).

The experiment asks whether a sparse autoencoder (SAE) over the 32-step
closed-loop history contains a usable delay representation, and whether a
policy that reads that representation handles held-out transport delays better
than a policy that reads the raw history.

Everything here is CPU-only and uses the 3,856-parameter tiny Student that is
currently reported on the project page.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

WS = Path(__file__).resolve().parents[3]
PINGMO = WS / "pingmo"
if str(PINGMO) not in sys.path:
    sys.path.insert(0, str(PINGMO))

from src.aircraft.parameters import PChannelParameters  # noqa: E402
from src.envs.batched_p_channel import (  # noqa: E402
    BatchedRollRateEnv,
    CURRENT_DIM,
    DT_POLICY,
    EPISODE_STEPS,
    HISTORY_CHANNELS,
    HISTORY_STEPS,
    OLDER_ACTIONS,
)
from src.envs.roll_rate_commands import RollRateCommandProfile  # noqa: E402
from src.teacher.ppo.networks import HistoryConv  # noqa: E402
from src.teacher.residual import ResidualMLPTrunk  # noqa: E402

KEYS = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")
DEG = 180.0 / math.pi
LIB = PINGMO / "data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"
TINY_CKPT = WS / "_report_review/weights/student_tiny.pt"

HISTORY_DIM = HISTORY_STEPS * HISTORY_CHANNELS
OBS_DIM = CURRENT_DIM + HISTORY_DIM + OLDER_ACTIONS
RECORD_START = 40          # 0.8 s: history is populated and the step is active
METRIC_START = 50          # 1.0 s: matches the reported evaluation window
RECORD_STRIDE = 3          # 16.7 Hz is enough for the representation experiments

# Eight representative commands: six steps covering the useful amplitude range,
# one doublet and one multisine.  The full library has 26 profiles; this subset
# keeps collection and policy training inside a CPU budget.
COMMANDS: list[RollRateCommandProfile] = [
    RollRateCommandProfile("s1", "step", amplitude_deg_s=1.0, onset_s=1.0, duration_s=8.0),
    RollRateCommandProfile("s2", "step", amplitude_deg_s=2.0, onset_s=1.0, duration_s=8.0),
    RollRateCommandProfile("s5", "step", amplitude_deg_s=5.0, onset_s=1.0, duration_s=8.0),
    RollRateCommandProfile("s10", "step", amplitude_deg_s=10.0, onset_s=1.0, duration_s=8.0),
    RollRateCommandProfile("s21", "step", amplitude_deg_s=21.0, onset_s=1.0, duration_s=8.0),
    RollRateCommandProfile("s31", "step", amplitude_deg_s=31.0, onset_s=1.0, duration_s=8.0),
    RollRateCommandProfile("d10", "doublet", amplitude_deg_s=10.0, onset_s=1.0,
                           duration_s=8.0, segment_duration_s=0.6),
    RollRateCommandProfile("m4", "multisine", amplitude_deg_s=4.0, onset_s=1.0,
                           duration_s=8.0,
                           multisine_components=((2.0, 0.5), (1.5, 1.0), (1.0, 1.6))),
]
COMMAND_IDS = [c.command_id for c in COMMANDS]


def delay_bin(tau_s: float) -> str:
    """Operational split used in the paper: short / medium / long transport delay."""
    if tau_s <= 0.05:
        return "short"
    if tau_s <= 0.15:
        return "medium"
    return "long"


def delay_bin_index(tau_s: np.ndarray) -> np.ndarray:
    out = np.full(np.shape(tau_s), 2, dtype=np.int64)
    out[np.asarray(tau_s) <= 0.15] = 1
    out[np.asarray(tau_s) <= 0.05] = 0
    return out


class TinyStudent(nn.Module):
    """Exact architecture of `_report_review/weights/student_tiny.pt`."""

    def __init__(self, conv_width: int = 8, width: int = 32, blocks: int = 1) -> None:
        super().__init__()
        self.encoder = HistoryConv(width=conv_width)
        self.body = ResidualMLPTrunk(2 * conv_width + CURRENT_DIM + OLDER_ACTIONS,
                                     width, blocks)
        self.head = nn.Linear(width, 1, bias=False)

    def _logits(self, obs: torch.Tensor) -> torch.Tensor:
        cur = obs[..., :CURRENT_DIM]
        he = CURRENT_DIM + HISTORY_DIM
        hist = obs[..., CURRENT_DIM:he].reshape(-1, HISTORY_STEPS, HISTORY_CHANNELS)
        old = obs[..., he:he + OLDER_ACTIONS]
        return self.head(self.body(torch.cat([self.encoder(hist), cur, old], dim=-1)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.5 * (self._logits(obs) - self._logits(-obs)).float())


class HistoryOnlyPolicy(nn.Module):
    """Policy body shared by all representation variants.

    The front end maps the 32x5 history window to a 16-D feature vector; the
    remaining 9 current signals and 7 older actions are appended unchanged.  The
    body and head are exactly the tiny Student's body and head, so variants
    differ only in the history representation.
    """

    def __init__(self, feature_dim: int = 16, width: int = 32, blocks: int = 1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.body = ResidualMLPTrunk(feature_dim + CURRENT_DIM + OLDER_ACTIONS,
                                     width, blocks)
        self.head = nn.Linear(width, 1, bias=False)

    def _logits_from_features(self, features: torch.Tensor, current: torch.Tensor,
                              older: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([features, current, older], dim=-1)))

    def forward_features(self, features: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        cur = obs[..., :CURRENT_DIM]
        he = CURRENT_DIM + HISTORY_DIM
        old = obs[..., he:he + OLDER_ACTIONS]
        return self._logits_from_features(features, cur, old)


def load_tiny(device: str = "cpu") -> TinyStudent:
    payload = torch.load(TINY_CKPT, map_location="cpu", weights_only=False)
    model = TinyStudent()
    model.load_state_dict(payload["student"])
    return model.to(device).eval()


def load_library() -> list[dict]:
    return [json.loads(line) for line in LIB.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def plant_parameters(row: dict) -> PChannelParameters:
    return PChannelParameters(**{k: row["parameters"][k] for k in KEYS})


def sample_plants(rows: list[dict], split: str, n_per_bin: int, seed: int,
                  bins: tuple[str, ...] = ("short", "medium", "long")) -> list[dict]:
    """Plant-level sample stratified by transport delay, disjoint by split."""
    rng = np.random.default_rng(seed)
    pool = [r for r in rows
            if r["split"] == split and r["gjb_level"] in (1, 2, 3)]
    picked: list[dict] = []
    for name in bins:
        group = [r for r in pool if delay_bin(float(r["parameters"]["tau_p"])) == name]
        if len(group) < n_per_bin:
            raise ValueError(f"{split}/{name}: need {n_per_bin}, have {len(group)}")
        idx = rng.choice(len(group), size=n_per_bin, replace=False)
        picked.extend(group[int(i)] for i in idx)
    return picked


def split_observation(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cur = obs[..., :CURRENT_DIM]
    he = CURRENT_DIM + HISTORY_DIM
    hist = obs[..., CURRENT_DIM:he].reshape(*obs.shape[:-1], HISTORY_STEPS, HISTORY_CHANNELS)
    old = obs[..., he:he + OLDER_ACTIONS]
    return cur, hist, old


def run_rollout(model: nn.Module, row: dict,
                commands: list[RollRateCommandProfile] | None = None,
                seed: int = 0, record: bool = True,
                label_model: nn.Module | None = None) -> dict:
    """Closed-loop rollout of one plant against a batch of commands.

    ``label_model`` is used by DAgger: ``model`` drives the environment, while
    ``label_model`` supplies the recorded action labels.
    """
    commands = commands or COMMANDS
    params = plant_parameters(row)
    env = BatchedRollRateEnv(params, n_envs=len(commands), commands=commands,
                             seed=seed)
    env.command_index[:] = torch.arange(len(commands), dtype=torch.long)
    obs = env.observation()

    n_cmd = len(commands)
    history, current, older, actions, steps, cmds = [], [], [], [], [], []
    se = torch.zeros(n_cmd)
    peak = torch.zeros(n_cmd)
    tv = torch.zeros(n_cmd)
    sat = torch.zeros(n_cmd)
    prev = torch.zeros(n_cmd)
    active = 0

    with torch.no_grad():
        for t in range(EPISODE_STEPS - 1):
            ref = env.reference.clone()
            record_now = bool(record and t >= RECORD_START
                              and (t - RECORD_START) % RECORD_STRIDE == 0)
            if record_now:
                cur, hist, old = split_observation(obs)
                history.append(hist.cpu().numpy().astype(np.float32))
                current.append(cur.cpu().numpy().astype(np.float32))
                older.append(old.cpu().numpy().astype(np.float32))
                steps.append(np.full(n_cmd, t, dtype=np.int16))
                cmds.append(np.arange(n_cmd, dtype=np.int16))
            action = model(obs)
            if record_now:
                label = label_model(obs) if label_model is not None else action
                actions.append(label.squeeze(-1).cpu().numpy().astype(np.float32))
            obs, _, output = env.step(action)
            if t >= METRIC_START:
                err = ref - output
                se += err.pow(2)
                peak = torch.maximum(peak, err.abs())
                tv += (env.requested - prev).abs()
                sat += (env.requested.abs() >= 21.9).float()
                active += 1
            prev = env.requested.clone()

    metrics = {
        "rmse": (se / max(active, 1)).sqrt().numpy() * DEG,
        "peak": peak.numpy() * DEG,
        "tv": (tv / max(active, 1) / DT_POLICY).numpy(),
        "sat": (sat / max(active, 1)).numpy(),
    }
    out = {"metrics": metrics}
    if record:
        out.update({
            "history": np.concatenate(history, axis=0) if history else np.zeros((0, HISTORY_STEPS, HISTORY_CHANNELS), np.float32),
            "current": np.concatenate(current, axis=0) if current else np.zeros((0, CURRENT_DIM), np.float32),
            "older": np.concatenate(older, axis=0) if older else np.zeros((0, OLDER_ACTIONS), np.float32),
            "action": np.concatenate(actions, axis=0) if actions else np.zeros((0,), np.float32),
            "step": np.concatenate(steps, axis=0) if steps else np.zeros((0,), np.int16),
            "command": np.concatenate(cmds, axis=0) if cmds else np.zeros((0,), np.int16),
        })
    return out


def summarize_metrics(metrics: dict) -> dict:
    return {
        "rmse_median": float(np.median(metrics["rmse"])),
        "rmse_mean": float(np.mean(metrics["rmse"])),
        "peak_median": float(np.median(metrics["peak"])),
        "tv_median": float(np.median(metrics["tv"])),
        "sat_mean": float(np.mean(metrics["sat"])),
        "rmse_by_command": [float(v) for v in metrics["rmse"]],
        "peak_by_command": [float(v) for v in metrics["peak"]],
        "tv_by_command": [float(v) for v in metrics["tv"]],
        "sat_by_command": [float(v) for v in metrics["sat"]],
    }
