"""Model-assisted few-shot **weight** fine-tuning of the distilled Student on DHC-6.

Pilot protocol (one target state only; see ``dhc6_finetune/PROTOCOL.md``)
-----------------------------------------------------------------------
target record   ``run_V050_H2500_c01_prbs_A6.mat``, window ``[0, 30) s``
                (= 1500 samples at 50 Hz) used for identification, in-window
                order selection, theta estimation and L_Fa calibration only.
input           ``da_act`` (deg) -- the actuator output, so the separate
                actuator/rate-limit dynamics are *not* re-modelled.
output          ``X_meas[:, 3]`` (deg/s) -- the realistically available
                measurement (32.7 dB SNR, r = 0.9997 against ``X_true``).
theta           T_R / zeta_d / omega_d from the fitted poles; L_Fa by least
                squares against the ARX frequency response (0.2-3 Hz);
                lambda_s = -0.05, R_omega = R_zeta = 1, tau_p = 0.02 s are the
                same frozen assumptions as the previous frozen-baseline run.
plant           ARX -> continuous zero-pole map -> ZOH state space at 1 kHz,
                the same contract as ``dhc6_transfer.ContinuousPlant``
                (integer-sample delay ``nk * 0.02 s`` on a 1 ms grid).
loop            50 Hz policy / 1 kHz plant, 20 substeps, +/-22 N, 88 N/s
                (0.088 N per 1 ms), bridge 10 deg / 22 N with the minus sign.
loss            ``w_e e^2 dt + w_u u_applied^2 dt + w_delta d(da_requested)^2``
                (the original reward's dt placement: dt on tracking and energy,
                none on the per-decision action jump).

No import of ``dhc6_arx_all.py`` (that script has no main guard and would refit
all 12 states and rewrite its JSON); the ARX maths below is a pure, side-effect
free re-implementation of it, cited at each function.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from scipy import signal
from scipy.io import loadmat

from src.aircraft.delay import FractionalDelay
from src.aircraft.parameters import PChannelParameters
from src.context.aircraft_parameters import normalize_aircraft_parameters
from src.envs.reference_model import SecondOrderReferenceConfig, SecondOrderRollRateReference
from src.envs.roll_rate_commands import RollRateCommandProfile, specialist_evaluation_commands
from src.student.dense.network import DenseConditionalStudent
from src.student.dense.policy import load_dense_student

# --------------------------------------------------------------------- constants
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "results/pure_reward_teacher_bank_coverage_v4/student_driven_dense_stability"
    / "final/dense_student.pt"
)
EXPECTED_W0_SHA256 = (
    "3c7e545fe7273363c65185355b5ab63f2fd29f7eedb02f5fc91c0b913520e432"
)

DT = 0.02
PLANT_DT = 0.001
SUBSTEPS = int(round(DT / PLANT_DT))
FORCE_LIMIT_N = 22.0
FORCE_RATE_LIMIT_N_S = 88.0
MAX_PLANT_INCREMENT_N = FORCE_RATE_LIMIT_N_S * PLANT_DT
COMMAND_SCALE_RAD_S = math.radians(30.0)
ACC_SCALE_RAD_S2 = 5.0
INTEGRAL_SCALE_RAD = COMMAND_SCALE_RAD_S
HISTORY_STEPS = 26
DA_FULL_SCALE_DEG = 10.0
BRIDGE_DEG_PER_N = DA_FULL_SCALE_DEG / FORCE_LIMIT_N
SIGN = -1.0
TAU_P_REFERENCE_S = DT

DEFAULT_RECORD = "run_V050_H2500_c01_prbs_A6.mat"
DEFAULT_WINDOW_S = (0.0, 30.0)
DEFAULT_SEED = 20260904
DEFAULT_LR = 1e-4
DEFAULT_WD = 1e-5
DEFAULT_CLIP = 10.0
WEIGHT_TRACKING = 1.0
WEIGHT_ENERGY = 0.02
WEIGHT_ACTION_DELTA = 0.02

HELDOUT_PREDICTION_RECORDS = (
    "run_V050_H2500_c01_chirp_A6.mat",
    "run_V050_H2500_c01_step3211_A10.mat",
)


# --------------------------------------------------------------------- helpers
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------- data
@dataclass(frozen=True)
class AdaptationWindow:
    record: str
    path: str
    t0_s: float
    t1_s: float
    n_samples: int
    input_source: str
    output_source: str
    file_sha256: str
    file_bytes: int
    dt_s: float
    units: str
    pD_units: str
    meta_amp_deg: float
    meta_type: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "record": self.record,
            "path": self.path,
            "window_s": [self.t0_s, self.t1_s],
            "n_samples": self.n_samples,
            "input_source": self.input_source,
            "output_source": self.output_source,
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "dt_s": self.dt_s,
            "units": self.units,
            "pD_units": self.pD_units,
            "meta_amp_deg": self.meta_amp_deg,
            "meta_type": self.meta_type,
            "adaptation_seconds": float(self.n_samples) * self.dt_s,
        }


def load_adaptation_window(
    data_dir: str | Path,
    record: str = DEFAULT_RECORD,
    window_s: tuple[float, float] = DEFAULT_WINDOW_S,
    *,
    output_source: str = "X_meas",
) -> tuple[AdaptationWindow, np.ndarray, np.ndarray]:
    """Read exactly one adaptation window.

    ``loadmat`` unavoidably reads the whole file; the window is sliced and
    **copied** immediately, and the caller only receives the copy, so no sample
    outside ``window_s`` can reach a fit or a selection step.
    """

    data_dir = Path(data_dir)
    path = data_dir / record
    if not path.is_file():
        raise FileNotFoundError(path)
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    meta = mat["meta"]
    dt_s = float(meta.dt)
    tsim = np.asarray(mat["tsim"]).ravel()
    i0 = int(round(window_s[0] / dt_s))
    i1 = int(round(window_s[1] / dt_s))
    if not 0 <= i0 < i1 <= tsim.size:
        raise ValueError(f"window {window_s} outside record of {tsim.size} samples")
    if not np.isclose(tsim[i0], window_s[0]) or not np.isclose(tsim[i1 - 1], window_s[1] - dt_s):
        raise ValueError("window is not aligned to the record time base")

    da_full = np.asarray(mat["da_act"]).ravel()
    if output_source == "X_meas":
        p_full = np.asarray(mat["X_meas"])[:, 3]
    elif output_source == "X_true":
        p_full = np.asarray(mat["X_true"])[:, 3]
    else:
        raise ValueError(f"unsupported output source {output_source!r}")

    da = np.array(da_full[i0:i1], dtype=np.float64, copy=True)
    p = np.array(p_full[i0:i1], dtype=np.float64, copy=True)
    window = AdaptationWindow(
        record=record,
        path=str(path),
        t0_s=float(window_s[0]),
        t1_s=float(window_s[1]),
        n_samples=int(i1 - i0),
        input_source="da_act",
        output_source=output_source,
        file_sha256=sha256_file(path),
        file_bytes=path.stat().st_size,
        dt_s=dt_s,
        units=str(meta.units),
        pD_units=str(meta.pD_units),
        meta_amp_deg=float(meta.amp_deg),
        meta_type=str(meta.type),
    )
    del mat, da_full, p_full, tsim
    if not np.isfinite(da).all() or not np.isfinite(p).all():
        raise ValueError("adaptation window contains non-finite samples")
    return window, da, p


# --------------------------------------------------------------------- ARX maths
# Pure re-implementation of dhc6_arx_all.py:38-75 (that script must not be
# imported: no main guard -> importing it would refit all 12 states and rewrite
# dhc6_arx_all.json).
def fit_arx(da: np.ndarray, y: np.ndarray, na: int, nb: int, nk: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    start = max(na, nk + nb - 1)
    rows = np.empty((n - start, na + nb))
    for i in range(na):
        rows[:, i] = -y[start - 1 - i : n - 1 - i]
    for j in range(nb):
        rows[:, na + j] = da[start - nk - j : n - nk - j]
    coef, *_ = np.linalg.lstsq(rows, y[start:], rcond=None)
    return coef[:na], coef[na:]


def simulate_arx(da: np.ndarray, a: Sequence[float], b: Sequence[float], nk: int) -> np.ndarray:
    den = np.concatenate([[1.0], np.asarray(a, dtype=float)])
    u = np.concatenate([np.zeros(int(nk)), np.asarray(da, dtype=float)])
    return signal.lfilter(np.asarray(b, dtype=float), den, u)[: len(da)]


def vaf(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(100.0 * (1.0 - np.var(y - y_hat) / np.var(y)))


@dataclass(frozen=True)
class ArxFit:
    na: int
    nb: int
    nk: int
    a: tuple[float, ...]
    b: tuple[float, ...]
    vaf_percent: float
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "na": self.na, "nb": self.nb, "nk": self.nk,
            "a": list(self.a), "b": list(self.b),
            "vaf_percent": self.vaf_percent, "n_samples": self.n_samples,
        }


def search_arx(
    da: np.ndarray,
    y: np.ndarray,
    na_values: Sequence[int] = (2, 3, 4, 5),
    nb_values: Sequence[int] = (2, 3, 4),
    nk_values: Sequence[int] = (1, 2, 3),
) -> ArxFit:
    best: ArxFit | None = None
    for na in na_values:
        for nb in nb_values:
            for nk in nk_values:
                a, b = fit_arx(da, y, na, nb, nk)
                poles = np.roots(np.concatenate([[1.0], a]))
                if np.any(np.abs(poles) > 1.001):
                    continue
                score = vaf(y, simulate_arx(da, a, b, nk))
                if not np.isfinite(score):
                    continue
                if best is None or score > best.vaf_percent:
                    best = ArxFit(na, nb, nk, tuple(a.tolist()), tuple(b.tolist()),
                                  float(score), int(len(y)))
    if best is None:
        raise RuntimeError("no stable ARX candidate on this window")
    return best


def arx_frequency_response(
    a: Sequence[float], b: Sequence[float], nk: int, freqs_hz: np.ndarray, dt: float = DT
) -> np.ndarray:
    num = np.concatenate([np.zeros(int(nk)), np.asarray(b, dtype=float)])
    den = np.concatenate([[1.0], np.asarray(a, dtype=float)])
    _, response = signal.freqz(num, den, worN=2.0 * np.pi * np.asarray(freqs_hz) * dt)
    return response


def arx_continuous(
    a: Sequence[float], b: Sequence[float], nk: int, dt: float = DT
) -> tuple[np.ndarray, np.ndarray, float]:
    """Mirror of dhc6_transfer.py:80-95 (zero-pole map z -> s, DC matched)."""

    num = np.concatenate([np.zeros(int(nk)), np.asarray(b, dtype=float)])
    den = np.concatenate([[1.0], np.asarray(a, dtype=float)])
    z_poles = np.roots(den)
    z_zeros = np.roots(num)
    s_poles = np.log(z_poles.astype(complex)) / dt
    s_zeros = np.log(z_zeros.astype(complex)) / dt
    dc_disc = float(np.sum(num) / np.sum(den))
    gain = dc_disc * np.prod(-s_poles) / np.prod(-s_zeros)
    num_c = (gain * np.poly(s_zeros)).real
    den_c = np.poly(s_poles).real
    return num_c, den_c, int(nk) * dt


def arx_dc_gain(a: Sequence[float], b: Sequence[float], nk: int) -> float:
    """Steady-state p (deg/s) per 1 deg of held aileron."""

    num = np.concatenate([np.zeros(int(nk)), np.asarray(b, dtype=float)])
    den = np.concatenate([[1.0], np.asarray(a, dtype=float)])
    return float(np.sum(num) / np.sum(den))


def pingmo_response(freqs_hz: np.ndarray, params: PChannelParameters) -> np.ndarray:
    s = 1j * 2.0 * np.pi * np.asarray(freqs_hz, dtype=float)
    num = params.l_fa * s * (
        s**2 + 2.0 * params.zeta_phi * params.omega_phi * s + params.omega_phi**2
    )
    den = (
        (s - params.lambda_s)
        * (s + 1.0 / params.t_r)
        * (s**2 + 2.0 * params.zeta_d * params.omega_d * s + params.omega_d**2)
    )
    return num / den


def parameters_from_fit(
    fit: ArxFit, freqs_hz: np.ndarray | None = None
) -> tuple[PChannelParameters, dict[str, Any]]:
    """theta estimate following dhc6_states.py:35-47 + dhc6_transfer.py:131-136."""

    freqs = np.linspace(0.2, 3.0, 40) if freqs_hz is None else np.asarray(freqs_hz)
    poles = np.roots(np.concatenate([[1.0], np.asarray(fit.a)]))
    s_poles = np.log(poles.astype(complex)) / DT
    real_fast = [z.real for z in s_poles if abs(z.imag) < 1e-6 and -50.0 < z.real < -1.0]
    t_r = 1.0 / abs(max(real_fast)) if real_fast else 0.2
    slow = [z for z in s_poles if abs(z) < 20.0 and z.imag > 1e-6]
    omega_d = abs(slow[0]) if slow else 3.0
    zeta_d = -slow[0].real / omega_d if slow else 0.5
    base = {
        "lambda_s": -0.05,
        "t_r": float(t_r),
        "zeta_d": float(zeta_d),
        "omega_d": float(omega_d),
        "r_omega": 1.0,
        "r_zeta": 1.0,
        "tau_p": float(DT),
    }
    h_data = np.abs(arx_frequency_response(fit.a, fit.b, fit.nk, freqs))
    target = np.radians(h_data) * BRIDGE_DEG_PER_N
    unit = PChannelParameters(l_fa=1.0, **base)
    h_unit = np.abs(pingmo_response(freqs, unit))
    l_fa = float(np.sum(h_unit * target) / np.sum(h_unit * h_unit))
    params = PChannelParameters(l_fa=l_fa, **base)
    residual = float(np.max(np.abs(h_unit * l_fa - target) / np.maximum(target, 1e-12)))
    return params, {
        "t_r": float(t_r),
        "zeta_d": float(zeta_d),
        "omega_d": float(omega_d),
        "l_fa": l_fa,
        "assumptions": {"lambda_s": -0.05, "r_omega": 1.0, "r_zeta": 1.0, "tau_p": DT},
        "calibration_freqs_hz": [float(freqs.min()), float(freqs.max())],
        "calibration_max_rel_residual": residual,
        "slow_poles_s": [[float(z.real), float(z.imag)] for z in slow],
        "fast_real_poles_s": [float(z) for z in sorted(real_fast)],
    }


# --------------------------------------------------------------------- plants
class PlantNumpy:
    """ZOH state-space plant at 1 kHz with an integer-sample transport delay."""

    def __init__(self, num_c: np.ndarray, den_c: np.ndarray, delay_s: float, dt: float = PLANT_DT):
        a, b, c, d = signal.tf2ss(np.asarray(num_c, float), np.asarray(den_c, float))
        ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), dt, method="zoh")
        self.ad, self.bd, self.cd, self.dd = ad, bd, cd, dd
        self.dt = dt
        self.delay_s = delay_s
        self._delay = FractionalDelay(dt, delay_s)
        self.x = np.zeros(ad.shape[0])

    def reset(self) -> None:
        self.x[:] = 0.0
        self._delay.reset()

    def step(self, u: float) -> float:
        delayed = self._delay.push(float(u))
        y = float((self.cd @ self.x + self.dd.squeeze() * delayed).item())
        self.x = self.ad @ self.x + self.bd[:, 0] * delayed
        return y


def plant_torch_parameters(num_c, den_c, delay_s, dtype=torch.float64) -> dict[str, Any]:
    a, b, c, d = signal.tf2ss(np.asarray(num_c, float), np.asarray(den_c, float))
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), PLANT_DT, method="zoh")
    return {
        "ad": torch.as_tensor(ad, dtype=dtype),
        "bd": torch.as_tensor(bd[:, 0], dtype=dtype),
        "cd": torch.as_tensor(np.atleast_1d(np.squeeze(cd)), dtype=dtype),
        "dd": torch.as_tensor(float(np.squeeze(dd)), dtype=dtype),
        "delay_samples": int(round(delay_s / PLANT_DT)),
        "state_dim": int(ad.shape[0]),
        "dtype": dtype,
    }


# --------------------------------------------------------------------- observation
OBSERVATION_LAYOUT = [
    "cmd_over_30dps", "ref_over_30dps", "p_over_30dps", "error_over_30dps",
    "integral_over_30dps_s", "p_dot_over_5", "requested_force_over_22",
    "ref_dot_over_5", "commanded_force_over_22", "applied_force_over_22",
] + [f"requested_history_lag_{lag}" for lag in range(2, 27)]


def build_observation(
    cmd: float, ref: float, ref_dot: float, p: float, p_dot: float, integral_err: float,
    requested_n: float, commanded_n: float, applied_n: float, history: np.ndarray,
) -> np.ndarray:
    """Mirror of dhc6_transfer.py:139-154 (35 = 10 + history[1:])."""

    values = [
        cmd / COMMAND_SCALE_RAD_S,
        ref / COMMAND_SCALE_RAD_S,
        p / COMMAND_SCALE_RAD_S,
        (ref - p) / COMMAND_SCALE_RAD_S,
        integral_err / INTEGRAL_SCALE_RAD,
        p_dot / ACC_SCALE_RAD_S2,
        requested_n / FORCE_LIMIT_N,
        ref_dot / ACC_SCALE_RAD_S2,
        commanded_n / FORCE_LIMIT_N,
        applied_n / FORCE_LIMIT_N,
    ]
    values.extend(history[1:])
    return np.asarray(values, dtype=np.float32)


# --------------------------------------------------------------------- commands
def regression_commands(duration_s: float = 5.0) -> tuple[RollRateCommandProfile, ...]:
    """The six pre-existing regression commands (already explored -> not blind)."""

    return specialist_evaluation_commands(duration_s=duration_s)


def training_commands(duration_s: float = 5.0) -> tuple[RollRateCommandProfile, ...]:
    """New training profiles, disjoint from the regression and held-out sets."""

    onset = 0.35
    return (
        RollRateCommandProfile("train-step-pos-18deg-s", "step", 18.0,
                               onset_s=onset, duration_s=duration_s),
        RollRateCommandProfile("train-step-neg-18deg-s", "step", -18.0,
                               onset_s=onset, duration_s=duration_s),
        RollRateCommandProfile("train-doublet-18deg-s", "doublet", 18.0,
                               onset_s=onset, segment_duration_s=0.55, duration_s=duration_s),
        RollRateCommandProfile("train-sine-0.45hz", "sine", 14.0,
                               onset_s=onset, frequency_hz=0.45, duration_s=duration_s),
        RollRateCommandProfile("train-sine-1.10hz", "sine", 9.0,
                               onset_s=onset, frequency_hz=1.10, duration_s=duration_s),
        RollRateCommandProfile("train-multisine-a", "multisine", 0.0,
                               onset_s=onset, duration_s=duration_s,
                               multisine_components=((6.0, 0.25), (4.0, 0.70), (2.5, 1.20))),
    )


def heldout_commands(duration_s: float = 5.0) -> tuple[RollRateCommandProfile, ...]:
    """Newly declared held-out profiles: never used for training or selection."""

    onset = 0.35
    return (
        RollRateCommandProfile("held-step-pos-22deg-s", "step", 22.0,
                               onset_s=onset, duration_s=duration_s),
        RollRateCommandProfile("held-step-neg-22deg-s", "step", -22.0,
                               onset_s=onset, duration_s=duration_s),
        RollRateCommandProfile("held-doublet-22deg-s", "doublet", 22.0,
                               onset_s=onset, segment_duration_s=0.60, duration_s=duration_s),
        RollRateCommandProfile("held-sine-0.62hz", "sine", 11.0,
                               onset_s=onset, frequency_hz=0.62, duration_s=duration_s),
        RollRateCommandProfile("held-sine-1.35hz", "sine", 7.0,
                               onset_s=onset, frequency_hz=1.35, duration_s=duration_s),
        RollRateCommandProfile("held-multisine-b", "multisine", 0.0,
                               onset_s=onset, duration_s=duration_s,
                               multisine_components=((6.5, 0.20), (4.5, 0.55), (2.0, 1.05))),
    )


def command_batch(
    profiles: Sequence[RollRateCommandProfile], plant_dt: float = PLANT_DT
) -> dict[str, Any]:
    """Build the 1 kHz command / reference arrays shared by both loops."""

    horizon = int(round(profiles[0].duration_s / DT))
    plant_steps = horizon * SUBSTEPS
    command = np.stack([
        np.asarray(profile.samples(plant_dt), dtype=np.float64).ravel() for profile in profiles
    ])
    if command.shape[1] < plant_steps + 1:
        command = np.concatenate(
            [command, np.repeat(command[:, -1:], plant_steps + 1 - command.shape[1], axis=1)],
            axis=1,
        )
    else:
        command = command[:, : plant_steps + 1]
    reference = SecondOrderRollRateReference(
        SecondOrderReferenceConfig(), dt_s=plant_dt, delay_s=TAU_P_REFERENCE_S
    )
    ref_seq = np.stack([reference.rollout(command[i, :plant_steps]) for i in range(len(profiles))])
    ref_dot = np.zeros_like(ref_seq)
    ref_dot[:, 1:] = np.diff(ref_seq, axis=1) / plant_dt
    return {
        "ids": [profile.command_id for profile in profiles],
        "duration_s": profiles[0].duration_s,
        "horizon": horizon,
        "plant_steps": plant_steps,
        "command": command,
        "reference": ref_seq,
        "reference_dot": ref_dot,
        "peak_demand_deg_s": np.degrees(np.max(np.abs(command), axis=1)),
    }


# --------------------------------------------------------------------- loops
def _trace_metrics(trace: dict[str, np.ndarray], saturation: int, horizon: int) -> dict[str, float]:
    error_deg = np.degrees(trace["error"])
    return {
        "rmse_deg_s": float(np.sqrt(np.mean(error_deg**2))),
        "peak_err_deg_s": float(np.max(np.abs(error_deg))),
        "tv_n_per_s": float(np.sum(np.abs(np.diff(trace["requested_n"]))) / trace["t"][-1]),
        "sat_ratio": float(saturation / horizon),
        "mean_abs_force_n": float(np.mean(np.abs(trace["requested_n"]))),
        "max_force_n": float(np.max(np.abs(trace["requested_n"]))),
        "nonfinite": bool(not np.isfinite(trace["p"]).all()),
    }


def rollout_numpy(
    controller: Callable[[np.ndarray], np.ndarray],
    batch: dict[str, Any],
    plant: PlantNumpy,
    *,
    command_index: int = 0,
    bridge: float = BRIDGE_DEG_PER_N,
    sign: float = SIGN,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Independent numpy mirror of dhc6_transfer.rollout (one command at a time)."""

    horizon = batch["horizon"]
    command = batch["command"][command_index]
    ref_seq = batch["reference"][command_index]
    ref_dot = batch["reference_dot"][command_index]

    plant.reset()
    p = 0.0
    p_dot = 0.0
    integral_err = 0.0
    requested_n = commanded_n = applied_n = 0.0
    history = np.zeros(HISTORY_STEPS, dtype=np.float64)
    plant_step = 0

    obs = build_observation(command[0], ref_seq[0], ref_dot[0], p, p_dot,
                            integral_err, requested_n, commanded_n, applied_n, history)
    trace = {k: [] for k in ("t", "cmd", "ref", "p", "requested_n", "applied_n", "error")}
    saturation = 0
    for _ in range(horizon):
        action = np.asarray(controller(obs), dtype=float).reshape(-1)
        requested_norm = float(np.clip(action[0], -1.0, 1.0))
        requested_n = requested_norm * FORCE_LIMIT_N
        previous_p = p
        for _ in range(SUBSTEPS):
            delta = float(np.clip(requested_n - commanded_n,
                                  -MAX_PLANT_INCREMENT_N, MAX_PLANT_INCREMENT_N))
            commanded_n += delta
            applied_n = commanded_n
            p_deg = plant.step(sign * bridge * applied_n)
            p = math.radians(p_deg)
            plant_step += 1
            integral_err += (ref_seq[plant_step] - p) * PLANT_DT
        p_dot = (p - previous_p) / DT
        if abs(commanded_n) >= FORCE_LIMIT_N - 1e-9:
            saturation += 1
        history[1:] = history[:-1]
        history[0] = requested_norm
        trace["t"].append(plant_step * PLANT_DT)
        trace["cmd"].append(command[plant_step])
        trace["ref"].append(ref_seq[plant_step])
        trace["p"].append(p)
        trace["requested_n"].append(requested_n)
        trace["applied_n"].append(applied_n)
        trace["error"].append(ref_seq[plant_step] - p)
        obs = build_observation(command[plant_step], ref_seq[plant_step], ref_dot[plant_step],
                                p, p_dot, integral_err, requested_n, commanded_n, applied_n, history)

    trace = {k: np.asarray(v, dtype=float) for k, v in trace.items()}
    return trace, _trace_metrics(trace, saturation, horizon)


def _gather(sequence: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Index a (B, T) tensor by a per-sample float step counter."""

    idx = index.to(torch.long).clamp(0, sequence.shape[1] - 1)
    return sequence.gather(1, idx.unsqueeze(1)).squeeze(1)


def rollout_torch(
    model: DenseConditionalStudent,
    theta: torch.Tensor,
    batch: dict[str, Any],
    plant_params: dict[str, Any],
    *,
    bridge: float = BRIDGE_DEG_PER_N,
    sign: float = SIGN,
    record_trace: bool = True,
) -> tuple[dict[str, Any], torch.Tensor]:
    """Differentiable mirror of the numpy loop; batch dimension = commands."""

    horizon = batch["horizon"]
    dtype = plant_params["dtype"]
    device = theta.device

    command = torch.as_tensor(batch["command"], dtype=dtype, device=device)
    ref_seq = torch.as_tensor(batch["reference"], dtype=dtype, device=device)
    ref_dot = torch.as_tensor(batch["reference_dot"], dtype=dtype, device=device)
    n_batch = command.shape[0]

    ad, bd = plant_params["ad"].to(device), plant_params["bd"].to(device)
    cd, dd = plant_params["cd"].to(device), plant_params["dd"].to(device)
    delay = int(plant_params["delay_samples"])
    state = torch.zeros(n_batch, plant_params["state_dim"], dtype=dtype, device=device)
    buffer = torch.zeros(n_batch, delay + 1, dtype=dtype, device=device)

    p = torch.zeros(n_batch, dtype=dtype, device=device)
    p_dot = torch.zeros(n_batch, dtype=dtype, device=device)
    integral_err = torch.zeros(n_batch, dtype=dtype, device=device)
    requested_n = torch.zeros(n_batch, dtype=dtype, device=device)
    commanded_n = torch.zeros(n_batch, dtype=dtype, device=device)
    applied_n = torch.zeros(n_batch, dtype=dtype, device=device)
    history = torch.zeros(n_batch, HISTORY_STEPS, dtype=dtype, device=device)
    cached_theta = theta.unsqueeze(0).expand(n_batch, -1).to(torch.float32)

    def make_obs(step: int) -> torch.Tensor:
        base = torch.stack([
            command[:, step] / COMMAND_SCALE_RAD_S,
            ref_seq[:, step] / COMMAND_SCALE_RAD_S,
            p / COMMAND_SCALE_RAD_S,
            (ref_seq[:, step] - p) / COMMAND_SCALE_RAD_S,
            integral_err / INTEGRAL_SCALE_RAD,
            p_dot / ACC_SCALE_RAD_S2,
            requested_n / FORCE_LIMIT_N,
            ref_dot[:, step] / ACC_SCALE_RAD_S2,
            commanded_n / FORCE_LIMIT_N,
            applied_n / FORCE_LIMIT_N,
        ], dim=-1)
        return torch.cat([base, history[:, 1:]], dim=-1)

    n_steps = torch.zeros(n_batch, dtype=dtype, device=device)
    trace = {"t": [], "cmd": [], "ref": [], "p": [], "requested_n": [], "applied_n": [], "error": []}
    saturation = torch.zeros(n_batch, dtype=dtype, device=device)
    obs = make_obs(0)

    for _ in range(horizon):
        actions = model(obs.to(torch.float32), cached_theta).to(dtype).reshape(n_batch)
        requested_norm = actions.clamp(-1.0, 1.0)
        requested_n = requested_norm * FORCE_LIMIT_N
        previous_p = p
        for _ in range(SUBSTEPS):
            delta = (requested_n - commanded_n).clamp(-MAX_PLANT_INCREMENT_N, MAX_PLANT_INCREMENT_N)
            commanded_n = commanded_n + delta
            applied_n = commanded_n
            u = sign * bridge * applied_n
            buffer = torch.cat([u.unsqueeze(1), buffer[:, :-1]], dim=1)
            delayed = buffer[:, delay]
            y = state @ cd + dd * delayed
            state = state @ ad.T + bd.unsqueeze(0) * delayed.unsqueeze(1)
            p = y * (math.pi / 180.0)
            n_steps = n_steps + 1.0
            integral_err = integral_err + (_gather(ref_seq, n_steps) - p) * PLANT_DT
        p_dot = (p - previous_p) / DT
        saturation = saturation + (commanded_n.abs() >= FORCE_LIMIT_N - 1e-9).to(dtype)
        history = torch.cat([requested_norm.unsqueeze(1), history[:, :-1]], dim=1)
        step_index = int(n_steps[0].item())
        if record_trace:
            trace["t"].append(step_index * PLANT_DT)
            trace["cmd"].append(command[:, step_index])
            trace["ref"].append(ref_seq[:, step_index])
            trace["p"].append(p)
            trace["requested_n"].append(requested_n)
            trace["applied_n"].append(applied_n)
            trace["error"].append(ref_seq[:, step_index] - p)
        obs = make_obs(step_index)

    trace["saturation"] = saturation
    return trace, saturation


def surrogate_loss(trace: dict[str, Any]) -> torch.Tensor:
    """w_e e^2 dt + w_u u^2 dt + w_delta (delta a)^2 (reward dt placement kept)."""

    p = torch.stack(trace["p"], dim=1)
    requested = torch.stack(trace["requested_n"], dim=1) / FORCE_LIMIT_N
    applied = torch.stack(trace["applied_n"], dim=1) / FORCE_LIMIT_N
    ref = torch.stack(trace["ref"], dim=1)
    error = (ref - p) / COMMAND_SCALE_RAD_S
    delta = torch.diff(requested, dim=1)
    tracking = WEIGHT_TRACKING * (error[:, 1:] ** 2).mean() * DT
    energy = WEIGHT_ENERGY * (applied[:, 1:] ** 2).mean() * DT
    jump = WEIGHT_ACTION_DELTA * (delta**2).mean()
    return tracking + energy + jump


# --------------------------------------------------------------------- parameter evidence
def state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().to(torch.float64).cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_report(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> dict[str, Any]:
    l2_before = float(sum(t.detach().double().pow(2).sum() for t in before.values()).sqrt())
    deltas = {k: (after[k].detach().double() - before[k].detach().double()) for k in before}
    l2_delta = float(sum(d.pow(2).sum() for d in deltas.values()).sqrt())
    changed_tensors, changed_params = [], 0
    for key, delta in deltas.items():
        n = int((delta != 0).sum())
        if n:
            changed_tensors.append({"name": key, "changed_values": n, "shape": list(delta.shape)})
            changed_params += n
    return {
        "l2_norm_before": l2_before,
        "l2_delta": l2_delta,
        "l2_relative": l2_delta / max(l2_before, 1e-30),
        "n_changed_tensors": len(changed_tensors),
        "n_changed_values": changed_params,
        "changed_tensors": changed_tensors,
        "hash_before": state_dict_hash(before),
        "hash_after": state_dict_hash(after),
    }


# --------------------------------------------------------------------- training
def build_model_from_checkpoint(
    checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str = "cpu"
) -> tuple[DenseConditionalStudent, dict[str, Any]]:
    model, payload = load_dense_student(checkpoint, device=device)
    if not isinstance(model, DenseConditionalStudent):
        raise TypeError(f"expected DenseConditionalStudent, got {type(model).__name__}")
    return model, payload


def fresh_scratch_model(payload: dict[str, Any], seed: int) -> DenseConditionalStudent:
    torch.manual_seed(seed)
    return DenseConditionalStudent(
        int(payload["observation_dim"]),
        int(payload["aircraft_parameter_dim"]),
        int(payload["action_dim"]),
        width=int(payload["network_width"]),
        residual_blocks=int(payload["residual_blocks"]),
        residual_scale=float(payload["residual_scale"]),
        enforce_odd_policy=bool(payload["enforce_odd_policy"]),
    )


def train_arm(
    arm: str,
    model: DenseConditionalStudent,
    theta: torch.Tensor,
    batch: dict[str, Any],
    plant_params: dict[str, Any],
    *,
    iterations: int,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WD,
    clip: float = DEFAULT_CLIP,
    seed: int = DEFAULT_SEED,
) -> tuple[DenseConditionalStudent, dict[str, Any]]:
    if arm == "frozen":
        return model, {"curve": [], "optimizer_steps": 0, "trainable_params": 0, "wall_clock_s": 0.0}

    if arm == "head_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_head."))
    elif arm in {"full", "scratch"}:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    torch.manual_seed(seed)
    model.train()
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        trace, _ = rollout_torch(model, theta, batch, plant_params)
        loss = surrogate_loss(trace)
        finite = bool(torch.isfinite(loss).item())
        curve.append({"iteration": iteration, "loss": float(loss.item()), "finite": finite})
        if not finite:
            raise RuntimeError(f"non-finite loss at iteration {iteration} for arm {arm}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip).item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at iteration {iteration} for arm {arm}")
        optimizer.step()
    elapsed = time.perf_counter() - started
    return model, {
        "curve": curve,
        "optimizer_steps": iterations,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "wall_clock_s": elapsed,
        "seconds_per_iteration": elapsed / max(iterations, 1),
        "final_loss": curve[-1]["loss"] if curve else None,
    }


# --------------------------------------------------------------------- evaluation
def evaluate_commands(
    model: DenseConditionalStudent,
    theta: torch.Tensor,
    batch: dict[str, Any],
    plant: PlantNumpy,
    *,
    authority_deg_s: float,
) -> list[dict[str, Any]]:
    model.eval()
    theta_cpu = theta.detach().cpu().to(torch.float32)
    rows = []

    def controller(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                          theta_cpu.unsqueeze(0))
        return value.numpy().reshape(-1)

    for index, command_id in enumerate(batch["ids"]):
        plant.reset()
        _, metrics = rollout_numpy(controller, batch, plant, command_index=index)
        peak = float(batch["peak_demand_deg_s"][index])
        rows.append({
            "command_id": command_id,
            "peak_demand_deg_s": peak,
            "estimated_steady_authority_deg_s": authority_deg_s,
            "demand_exceeds_authority": bool(peak > authority_deg_s),
            "fraction_demand_above_authority": float(
                np.mean(np.abs(np.degrees(batch["command"][index])) > authority_deg_s)
            ),
            **metrics,
        })
    return rows


def model_prediction_vaf(fit: ArxFit, data_dir: str | Path, record: str) -> dict[str, Any]:
    """E1: free-run ARX prediction against a held-out record (model locked first)."""

    path = Path(data_dir) / record
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    da = np.asarray(mat["da_act"]).ravel().astype(float)
    p = np.asarray(mat["X_meas"])[:, 3].astype(float)
    predicted = simulate_arx(da, fit.a, fit.b, fit.nk)
    return {
        "record": record,
        "file_sha256": sha256_file(path),
        "n_samples": int(da.size),
        "vaf_percent": vaf(p, predicted),
        "rmse_deg_s": float(np.sqrt(np.mean((p - predicted) ** 2))),
    }


# ============================================================ v2_discrete50
# Plant = the ORIGINAL 50 Hz ARX recursion fitted on the adaptation window.
# No matched-z conversion: the continuous object reproduces only ~69 % of the
# held-out chirp variance while the discrete ARX reproduces 98.89 %, so only the
# discrete object has independent data support.  This is a NEW protocol and is
# not claimed to be equivalent to the previous 1 kHz continuous harness.

MAX_POLICY_INCREMENT_N = FORCE_RATE_LIMIT_N_S * DT  # 1.76 N per 20 ms policy step


class ArxPlantNumpy:
    """Direct 50 Hz ARX recursion.

    ``push(u_k)`` returns ``y[k+1]`` where

        y[t] = -sum_i a_i y[t-1-i] + sum_j b_j u[t-nk-j],

    exactly the recursion of ``dhc6_arx_all.simulate`` (``lfilter`` of
    ``b / [1, a]`` over ``[zeros(nk), u]``).  ``u`` is aileron in deg, ``y`` is
    roll rate in deg/s, both at 50 Hz.  No extra delay is added.
    """

    def __init__(self, a: Sequence[float], b: Sequence[float], nk: int) -> None:
        self.a = np.asarray(a, dtype=float).ravel()
        self.b = np.asarray(b, dtype=float).ravel()
        self.nk = int(nk)
        if self.nk < 1:
            raise ValueError("v2_discrete50 requires nk >= 1")
        self.y_hist = np.zeros(self.a.size + 1, dtype=float)
        self.u_hist = np.zeros(self.nk + self.b.size, dtype=float)

    def reset(self) -> None:
        self.y_hist[:] = 0.0
        self.u_hist[:] = 0.0

    def push(self, u_k: float) -> float:
        self.u_hist[1:] = self.u_hist[:-1]
        self.u_hist[0] = float(u_k)
        y_next = -float(np.dot(self.a, self.y_hist[: self.a.size]))
        y_next += float(np.dot(self.b, self.u_hist[self.nk - 1 : self.nk - 1 + self.b.size]))
        self.y_hist[1:] = self.y_hist[:-1]
        self.y_hist[0] = y_next
        return y_next


def arx_torch_parameters(a: Sequence[float], b: Sequence[float], nk: int,
                         dtype: torch.dtype = torch.float64) -> dict[str, Any]:
    return {
        "a": torch.as_tensor(np.asarray(a, dtype=np.float64).ravel(), dtype=dtype),
        "b": torch.as_tensor(np.asarray(b, dtype=np.float64).ravel(), dtype=dtype),
        "nk": int(nk),
        "dtype": dtype,
    }


def command_batch_50hz(profiles: Sequence[RollRateCommandProfile]) -> dict[str, Any]:
    """50 Hz command samples and the 50 Hz second-order reference (delay = 1 sample)."""

    horizon = int(round(profiles[0].duration_s / DT))
    command = np.stack([
        np.asarray(profile.samples(DT), dtype=np.float64).ravel() for profile in profiles
    ])
    if command.shape[1] < horizon:
        command = np.concatenate(
            [command, np.repeat(command[:, -1:], horizon - command.shape[1], axis=1)], axis=1
        )
    command = command[:, :horizon]
    reference = SecondOrderRollRateReference(
        SecondOrderReferenceConfig(), dt_s=DT, delay_s=TAU_P_REFERENCE_S
    )
    ref_seq = np.stack([reference.rollout(command[i]) for i in range(len(profiles))])
    ref_dot = np.zeros_like(ref_seq)
    ref_dot[:, 1:] = np.diff(ref_seq, axis=1) / DT
    return {
        "ids": [profile.command_id for profile in profiles],
        "duration_s": profiles[0].duration_s,
        "horizon": horizon,
        "command": command,
        "reference": ref_seq,
        "reference_dot": ref_dot,
        "peak_demand_deg_s": np.degrees(np.max(np.abs(command), axis=1)),
    }


def arx_rollout_numpy(
    controller: Callable[[np.ndarray], np.ndarray],
    batch: dict[str, Any],
    plant: ArxPlantNumpy,
    *,
    command_index: int = 0,
    bridge: float = BRIDGE_DEG_PER_N,
    sign: float = SIGN,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Independent numpy digital closed loop at 50 Hz.

    Time bookkeeping: at policy step ``k`` the observation carries the state at
    ``t_k``; applying ``u_k`` advances the plant by exactly one 20 ms sample to
    ``y[k+1]``; the record row is written at ``t_{k+1}`` with ``ref[k+1]`` and
    ``p[k+1]``; the trapezoidal integral covers ``[t_k, t_{k+1}]``; ``p_dot`` is
    ``(p[k+1]-p[k])/dt`` and enters the next observation.
    """

    horizon = batch["horizon"]
    command = batch["command"][command_index]
    ref_seq = batch["reference"][command_index]
    ref_dot = batch["reference_dot"][command_index]

    plant.reset()
    p = 0.0
    p_dot = 0.0
    integral = 0.0
    requested_prev = commanded_prev = applied_prev = 0.0
    history = np.zeros(HISTORY_STEPS, dtype=np.float64)

    obs = build_observation(command[0], ref_seq[0], ref_dot[0], p, p_dot,
                            integral, requested_prev, commanded_prev, applied_prev, history)
    trace = {k: [] for k in ("t", "cmd", "ref", "p", "requested_n", "applied_n", "error")}
    saturation = 0
    for k in range(horizon):
        action = np.asarray(controller(obs), dtype=float).reshape(-1)
        requested_norm = float(np.clip(action[0], -1.0, 1.0))
        requested_n = requested_norm * FORCE_LIMIT_N
        delta = float(np.clip(requested_n - commanded_prev,
                              -MAX_POLICY_INCREMENT_N, MAX_POLICY_INCREMENT_N))
        commanded_n = commanded_prev + delta
        applied_n = commanded_n
        p_next = math.radians(plant.push(sign * bridge * applied_n))
        if abs(commanded_n) >= FORCE_LIMIT_N - 1e-9:
            saturation += 1
        p_dot_next = (p_next - p) / DT
        integral_next = integral + 0.5 * ((ref_seq[k] - p) + (ref_seq[k + 1] - p_next)) * DT
        history[1:] = history[:-1]
        history[0] = requested_norm
        trace["t"].append((k + 1) * DT)
        trace["cmd"].append(command[k])
        trace["ref"].append(ref_seq[k + 1])
        trace["p"].append(p_next)
        trace["requested_n"].append(requested_n)
        trace["applied_n"].append(applied_n)
        trace["error"].append(ref_seq[k + 1] - p_next)
        p, p_dot, integral = p_next, p_dot_next, integral_next
        requested_prev, commanded_prev, applied_prev = requested_n, commanded_n, applied_n
        next_index = min(k + 1, horizon - 1)
        obs = build_observation(command[next_index], ref_seq[next_index], ref_dot[next_index],
                                p, p_dot, integral, requested_prev, commanded_prev,
                                applied_prev, history)

    trace = {k: np.asarray(v, dtype=float) for k, v in trace.items()}
    requested = np.concatenate([[0.0], trace["requested_n"]])
    metrics = {
        "rmse_deg_s": float(np.sqrt(np.mean(np.degrees(trace["error"]) ** 2))),
        "peak_err_deg_s": float(np.max(np.abs(np.degrees(trace["error"])))),
        "tv_n_per_s": float(np.sum(np.abs(np.diff(requested))) / (horizon * DT)),
        "sat_ratio": float(saturation / horizon),
        "mean_abs_force_n": float(np.mean(np.abs(trace["requested_n"]))),
        "max_force_n": float(np.max(np.abs(trace["requested_n"]))),
        "nonfinite": bool(not np.isfinite(trace["p"]).all()),
    }
    return trace, metrics


def arx_rollout_torch(
    model: DenseConditionalStudent,
    theta: torch.Tensor,
    batch: dict[str, Any],
    params: dict[str, Any],
    *,
    bridge: float = BRIDGE_DEG_PER_N,
    sign: float = SIGN,
) -> dict[str, Any]:
    """Differentiable mirror of :func:`arx_rollout_numpy` (batch = commands)."""

    horizon = batch["horizon"]
    dtype = params["dtype"]
    device = theta.device
    a = params["a"].to(device)
    b = params["b"].to(device)
    nk = int(params["nk"])

    command = torch.as_tensor(batch["command"], dtype=dtype, device=device)
    ref_seq = torch.as_tensor(batch["reference"], dtype=dtype, device=device)
    ref_dot = torch.as_tensor(batch["reference_dot"], dtype=dtype, device=device)
    n_batch = command.shape[0]

    p = torch.zeros(n_batch, dtype=dtype, device=device)
    p_dot = torch.zeros(n_batch, dtype=dtype, device=device)
    integral = torch.zeros(n_batch, dtype=dtype, device=device)
    requested_prev = torch.zeros(n_batch, dtype=dtype, device=device)
    commanded_prev = torch.zeros(n_batch, dtype=dtype, device=device)
    applied_prev = torch.zeros(n_batch, dtype=dtype, device=device)
    y_hist = torch.zeros(n_batch, a.numel() + 1, dtype=dtype, device=device)
    u_hist = torch.zeros(n_batch, nk + b.numel(), dtype=dtype, device=device)
    req_hist = torch.zeros(n_batch, HISTORY_STEPS, dtype=dtype, device=device)
    cached_theta = theta.unsqueeze(0).expand(n_batch, -1).to(torch.float32)

    requested_list, error_list, applied_list, p_list, ref_list = [], [], [], [], []

    def obs_vector(step: int) -> torch.Tensor:
        base = torch.stack([
            command[:, step] / COMMAND_SCALE_RAD_S,
            ref_seq[:, step] / COMMAND_SCALE_RAD_S,
            p / COMMAND_SCALE_RAD_S,
            (ref_seq[:, step] - p) / COMMAND_SCALE_RAD_S,
            integral / INTEGRAL_SCALE_RAD,
            p_dot / ACC_SCALE_RAD_S2,
            requested_prev / FORCE_LIMIT_N,
            ref_dot[:, step] / ACC_SCALE_RAD_S2,
            commanded_prev / FORCE_LIMIT_N,
            applied_prev / FORCE_LIMIT_N,
        ], dim=-1)
        return torch.cat([base, req_hist[:, 1:]], dim=-1)

    obs = obs_vector(0)
    for k in range(horizon):
        actions = model(obs.to(torch.float32), cached_theta).to(dtype).reshape(n_batch)
        requested_norm = actions.clamp(-1.0, 1.0)
        requested_n = requested_norm * FORCE_LIMIT_N
        delta = (requested_n - commanded_prev).clamp(
            -MAX_POLICY_INCREMENT_N, MAX_POLICY_INCREMENT_N)
        commanded_n = commanded_prev + delta
        applied_n = commanded_n
        u = sign * bridge * applied_n
        u_hist = torch.cat([u.unsqueeze(1), u_hist[:, :-1]], dim=1)
        y_next = -(y_hist[:, : a.numel()] * a).sum(dim=1)
        y_next = y_next + (u_hist[:, nk - 1 : nk - 1 + b.numel()] * b).sum(dim=1)
        y_hist = torch.cat([y_next.unsqueeze(1), y_hist[:, :-1]], dim=1)
        p_next = y_next * (math.pi / 180.0)
        previous_p = p
        p_dot_next = (p_next - previous_p) / DT
        integral_next = integral + 0.5 * ((ref_seq[:, k] - p)
                                          + (ref_seq[:, k + 1] - p_next)) * DT
        requested_list.append(requested_norm)
        error_list.append(ref_seq[:, k + 1] - p_next)
        applied_list.append(applied_n)
        p_list.append(p_next)
        ref_list.append(ref_seq[:, k + 1])
        req_hist = torch.cat([requested_norm.unsqueeze(1), req_hist[:, :-1]], dim=1)
        p, p_dot, integral = p_next, p_dot_next, integral_next
        requested_prev, commanded_prev, applied_prev = requested_n, commanded_n, applied_n
        obs = obs_vector(min(k + 1, horizon - 1))

    return {
        "requested_norm": torch.stack(requested_list, dim=1),
        "error": torch.stack(error_list, dim=1),
        "applied_n": torch.stack(applied_list, dim=1),
        "p": torch.stack(p_list, dim=1),
        "ref": torch.stack(ref_list, dim=1),
    }


def arx_surrogate_loss(trace: dict[str, Any]) -> torch.Tensor:
    """w_e e^2 dt + w_u u^2 dt + w_delta (delta a)^2 over ALL steps.

    ``Delta a_0 = a_0 - 0`` is included; the previous version differenced only
    the recorded trace, silently dropping the first action jump.  tracking and
    energy carry dt, the jump term does not.
    """

    error = trace["error"] / COMMAND_SCALE_RAD_S
    applied = trace["applied_n"] / FORCE_LIMIT_N
    requested = trace["requested_norm"]
    zero = torch.zeros_like(requested[:, :1])
    delta = torch.diff(torch.cat([zero, requested], dim=1), dim=1)
    tracking = WEIGHT_TRACKING * (error**2).mean() * DT
    energy = WEIGHT_ENERGY * (applied**2).mean() * DT
    jump = WEIGHT_ACTION_DELTA * (delta**2).mean()
    return tracking + energy + jump


def train_arm_v2(
    arm: str,
    model: DenseConditionalStudent,
    theta: torch.Tensor,
    batch: dict[str, Any],
    params: dict[str, Any],
    *,
    iterations: int,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WD,
    clip: float = DEFAULT_CLIP,
    seed: int = DEFAULT_SEED,
) -> tuple[DenseConditionalStudent, dict[str, Any]]:
    """Same arms/optimizer as :func:`train_arm`, but on the v2 discrete plant."""

    if arm == "frozen":
        return model, {"curve": [], "optimizer_steps": 0, "trainable_params": 0,
                       "wall_clock_s": 0.0}
    if arm == "head_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_head."))
    elif arm in {"full", "scratch"}:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    torch.manual_seed(seed)
    model.train()
    curve: list[dict[str, Any]] = []
    started = time.perf_counter()
    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        trace = arx_rollout_torch(model, theta, batch, params)
        loss = arx_surrogate_loss(trace)
        finite = bool(torch.isfinite(loss).item())
        curve.append({"iteration": iteration, "loss": float(loss.item()), "finite": finite})
        if not finite:
            raise RuntimeError(f"non-finite loss at iteration {iteration} for arm {arm}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip).item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at iteration {iteration} for arm {arm}")
        optimizer.step()
    elapsed = time.perf_counter() - started
    return model, {
        "curve": curve,
        "optimizer_steps": iterations,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "wall_clock_s": elapsed,
        "seconds_per_iteration": elapsed / max(iterations, 1),
        "final_loss": curve[-1]["loss"] if curve else None,
    }


def make_controller(model: DenseConditionalStudent, theta: torch.Tensor):
    theta_cpu = theta.detach().cpu().to(torch.float32)

    def controller(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                          theta_cpu.unsqueeze(0))
        return value.numpy().reshape(-1)

    return controller


def evaluate_commands_v2(
    model: DenseConditionalStudent,
    theta: torch.Tensor,
    batch: dict[str, Any],
    fit: ArxFit,
    authority_deg_s: float,
) -> list[dict[str, Any]]:
    """Closed-loop metrics on the 50 Hz discrete ARX (v2 protocol)."""

    model.eval()
    controller = make_controller(model, theta)
    rows = []
    for index, command_id in enumerate(batch["ids"]):
        plant = ArxPlantNumpy(fit.a, fit.b, fit.nk)
        trace, metrics = arx_rollout_numpy(controller, batch, plant, command_index=index)
        peak = float(batch["peak_demand_deg_s"][index])
        requested = np.concatenate([[0.0], trace["requested_n"]])
        rows.append({
            "command_id": command_id,
            "peak_demand_deg_s": peak,
            "estimated_steady_authority_deg_s": authority_deg_s,
            "demand_exceeds_authority": bool(peak > authority_deg_s),
            "fraction_demand_above_authority": float(
                np.mean(np.abs(np.degrees(batch["command"][index])) > authority_deg_s)
            ),
            "initial_action_jump_n": float(requested[1]),
            **metrics,
        })
    return rows


def continuous_conversion_prediction(
    fit: ArxFit, data_dir: str | Path, record: str, *, substeps: int = SUBSTEPS
) -> dict[str, Any]:
    """Diagnose the matched-z path (the previous pilot's plant).

    The 50 Hz aileron is held for ``substeps`` 1 kHz samples and the output is
    sampled at 50 Hz -- i.e. the same 1 kHz re-discretisation the previous pilot
    trained on.  Used only to quantify conversion distortion.
    """

    path = Path(data_dir) / record
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    da = np.asarray(mat["da_act"]).ravel().astype(float)
    p = np.asarray(mat["X_meas"])[:, 3].astype(float)
    num_c, den_c, delay_s = arx_continuous(fit.a, fit.b, fit.nk)
    plant = PlantNumpy(num_c, den_c, delay_s)
    plant.reset()
    predicted = np.empty_like(da)
    for k in range(da.size):
        value = 0.0
        for _ in range(substeps):
            value = plant.step(da[k])
        predicted[k] = value
    discrete = simulate_arx(da, fit.a, fit.b, fit.nk)
    alignment = {}
    for shift in (-2, -1, 0, 1, 2):
        # positive shift = continuous prediction is late by <shift> 50 Hz samples
        if shift >= 0:
            a_slice = predicted[: predicted.size - shift] if shift else predicted
            b_slice = p[shift:]
        else:
            a_slice = predicted[-shift:]
            b_slice = p[: p.size + shift]
        alignment[str(shift)] = vaf(b_slice, a_slice)
    return {
        "record": record,
        "file_sha256": sha256_file(path),
        "n_samples": int(da.size),
        "discrete_arx_vaf_percent": vaf(p, discrete),
        "continuous_conversion_vaf_percent": vaf(p, predicted),
        "continuous_conversion_rmse_deg_s": float(np.sqrt(np.mean((p - predicted) ** 2))),
        "alignment_vaf_by_shift": alignment,
        "best_alignment_shift": max(alignment, key=alignment.get),
        "best_alignment_vaf_percent": max(alignment.values()),
    }
