"""Unified DHC-6 conditional student: 8-state adaptation, 12-state evaluation.

One shared Student ``pi(o, theta)`` is fine-tuned on eight trim conditions and
evaluated on all twelve.  The four held-out conditions never contribute to any
fit, to the theta map, or to model selection.

Design decisions (frozen before the run, see ``dhc6_unified/UNIFIED_PROTOCOL.md``)
-----------------------------------------------------------------------------------
* Plant for training: the original 50 Hz discrete ARX of each adaptation
  condition, fitted on that condition's ``prbs_A6`` prefix ``[0, B)`` only.
* Budgets: B = 15 / 30 / 60 s per state.  Real target seconds are 8*B, plus the
  15 s of ``V050_H2500`` that the previous pilot's hyper-parameter probe already
  saw (only for B=15, and only as *prior exposure*, never re-used for fitting).
* Theta: adaptation states use their own prefix fit.  The four held-out states
  use a fixed linear least-squares map on ``[1, V_s, H_s, V_s*H_s]`` built from
  the eight adaptation thetas only (``clip=False``).
* Evaluation: for every one of the twelve states a *separate* 50 Hz ARX is
  fitted on the whole ``chirp_A10`` record.  That object is shared by all
  budgets and is validated on ``prbs_A10`` and ``step3211_A10`` (E1).  It never
  touches training, the theta map, or any arm choice.
* Training: 64 optimizer steps = eight conditions x eight rounds, one fixed
  order from seed 20260904, the same six training profiles every step.
"""

from __future__ import annotations

import json
import math
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.io import loadmat

from src.transfer import fine_tune as ft

# --------------------------------------------------------------------- protocol
BUDGETS_S = (15.0, 30.0, 60.0)
ADAPTATION_CONDITIONS: tuple[tuple[int, int], ...] = (
    (50, 2500), (50, 3000), (50, 3500),
    (60, 2500), (60, 3000), (60, 3500),
    (70, 2500), (70, 3000),
)
HELDOUT_CONDITIONS: tuple[tuple[int, int], ...] = (
    (70, 3500), (80, 2500), (80, 3000), (80, 3500),
)
ALL_CONDITIONS = ADAPTATION_CONDITIONS + HELDOUT_CONDITIONS
ADAPT_STEPS = 64
ROUNDS = 8
SEED = ft.DEFAULT_SEED
LR = 3e-4
WD = ft.DEFAULT_WD
CLIP = ft.DEFAULT_CLIP
EVAL_VALIDATION_VAF_THRESHOLD = 85.0
PRIOR_PROBE_EXPOSURE_S = 15.0          # pilot probe saw V050_H2500 [0,30) s
PRIOR_PROBE_STATE = (50, 2500)
TRAIN_RECORD = ("prbs", 6)
EVAL_FIT_RECORD = ("chirp", 10)
EVAL_VALIDATION_RECORDS = (("prbs", 10), ("step3211", 10))


def state_name(condition: Sequence[int]) -> str:
    return f"V{int(condition[0]):03d}_H{int(condition[1])}"


_RECORD_RE = re.compile(r"run_V(\d+)_H(\d+)_c(\d+)_([A-Za-z0-9]+)_A(\d+)\.mat")


@lru_cache(maxsize=8)
def record_index(data_dir: str | Path) -> dict[str, Any]:
    """Map (V, H, kind, amplitude) -> actual file name (the c-id varies per state)."""

    by_key: dict[tuple[int, int, str, int], str] = {}
    for path in sorted(Path(data_dir).glob("*.mat")):
        match = _RECORD_RE.match(path.name)
        if not match:
            continue
        v, h, _cid, kind, amplitude = match.groups()
        by_key[(int(v), int(h), kind.lower(), int(amplitude))] = path.name
    if not by_key:
        raise FileNotFoundError(f"no run_*.mat records under {data_dir}")
    return by_key


def record_name(condition: Sequence[int], kind: str, amplitude: int,
                data_dir: str | Path = None) -> str:
    if data_dir is None:
        raise ValueError("data_dir is required to resolve the record name")
    index = record_index(str(data_dir))
    key = (int(condition[0]), int(condition[1]), kind.lower(), int(amplitude))
    try:
        return index[key]
    except KeyError as exc:
        raise FileNotFoundError(f"record {key} not found under {data_dir}") from exc


# --------------------------------------------------------------------- data
def load_full_record(data_dir: str | Path, name: str) -> dict[str, Any]:
    """Load every sample of one record (da_act and X_meas p), units already checked."""

    path = Path(data_dir) / name
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    da = np.array(np.asarray(mat["da_act"]).ravel(), dtype=np.float64, copy=True)
    p = np.array(np.asarray(mat["X_meas"])[:, 3], dtype=np.float64, copy=True)
    meta = mat["meta"]
    info = {
        "record": name,
        "file_sha256": ft.sha256_file(path),
        "n_samples": int(da.size),
        "dt_s": float(meta.dt),
        "units": str(meta.units),
        "pD_units": str(meta.pD_units),
    }
    return {"da": da, "p": p, "info": info}


def adaptation_window(condition: Sequence[int], budget_s: float) -> tuple[float, float]:
    if budget_s not in BUDGETS_S:
        raise ValueError(f"budget {budget_s} not in {BUDGETS_S}")
    return (0.0, float(budget_s))


def budget_table() -> list[dict[str, Any]]:
    """Real target seconds: 8*B adaptation, plus the pilot probe's extra 15 s at B=15."""

    rows = []
    for budget in BUDGETS_S:
        independent = len(ADAPTATION_CONDITIONS) * budget
        extra = PRIOR_PROBE_EXPOSURE_S if budget <= PRIOR_PROBE_EXPOSURE_S else 0.0
        rows.append({
            "budget_s": budget,
            "per_state_window_s": budget,
            "n_adaptation_states": len(ADAPTATION_CONDITIONS),
            "independent_adaptation_seconds": independent,
            "prior_probe_exposure_seconds": extra,
            "prior_probe_state": state_name(PRIOR_PROBE_STATE) if extra else None,
            "total_target_seconds": independent + extra,
            "note": ("pilot lr probe already saw V050_H2500 [0,30) s; at B=15 only "
                     "[0,15) is fitted and the other 15 s is counted as prior exposure"
                     if extra else "no extra prior exposure at or above 30 s"),
        })
    return rows


# --------------------------------------------------------------------- theta
def fit_condition_theta(data_dir: str | Path, condition: Sequence[int], budget_s: float,
                        *, output_source: str = "X_meas") -> dict[str, Any]:
    """Fit the 50 Hz ARX on that condition's own prefix and derive its theta."""

    window, da, p = ft.load_adaptation_window(
        data_dir, record_name(condition, *TRAIN_RECORD, data_dir), adaptation_window(condition, budget_s),
        output_source=output_source)
    fit = ft.search_arx(da, p)
    params, meta = ft.parameters_from_fit(fit)
    normalized = np.asarray(ft.normalize_aircraft_parameters(params), dtype=np.float32)
    return {
        "condition": state_name(condition),
        "group": "adaptation",
        "theta_source": "own_prefix_fit",
        "window": window.as_dict(),
        "fit": fit.as_dict(),
        "params": {k: float(getattr(params, k)) for k in
                   ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")},
        "normalized_theta": [float(x) for x in normalized],
        "theta_meta": meta,
        "fallback_flags": {
            "zeta_d_omega_d_default": bool(abs(params.zeta_d - 0.5) < 1e-12
                                           and abs(params.omega_d - 3.0) < 1e-12),
            "t_r_from_max_real_fast_helper": len(meta["fast_real_poles_s"]) > 0,
            "slow_pole_pairs_found": len(meta["slow_poles_s"]),
        },
        "prefix_samples": int(da.size),
        "prefix_seconds": float(da.size) * ft.DT,
    }


def theta_features(v: float, h: float, v_mean: float, v_std: float,
                   h_mean: float, h_std: float) -> list[float]:
    vs = (float(v) - v_mean) / v_std
    hs = (float(h) - h_mean) / h_std
    return [1.0, vs, hs, vs * hs]


def build_theta_model(train_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fixed linear map from trim metadata to 8-dim normalized theta.

    Only the eight adaptation records may be passed in; the function asserts it.
    No held-out response, no evaluation-model value, no clip.
    """

    names = [record["condition"] for record in train_records]
    expected = [state_name(c) for c in ADAPTATION_CONDITIONS]
    if sorted(names) != sorted(expected):
        raise ValueError(f"theta map must be built from the 8 adaptation states, got {names}")
    v = np.asarray([int(name.split("_")[0][1:]) for name in names], dtype=float)
    h = np.asarray([int(name.split("_")[1][1:]) for name in names], dtype=float)
    y = np.asarray([record["normalized_theta"] for record in train_records], dtype=float)
    v_mean, v_std = float(v.mean()), float(v.std())
    h_mean, h_std = float(h.mean()), float(h.std())
    if v_std == 0 or h_std == 0:
        raise ValueError("degenerate trim metadata spread")
    x = np.asarray([theta_features(vi, hi, v_mean, v_std, h_mean, h_std)
                    for vi, hi in zip(v, h)], dtype=float)
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)          # (4, 8)
    residual = x @ coef - y
    return {
        "form": "theta_norm = [1, V_s, H_s, V_s*H_s] @ C",
        "features": ["1", "V_scaled", "H_scaled", "V_scaled*H_scaled"],
        "clip": False,
        "V_mean": v_mean, "V_std": v_std, "H_mean": h_mean, "H_std": h_std,
        "coefficients": coef.tolist(),
        "train_states": names,
        "train_fit_rmse_per_dim": np.sqrt((residual**2).mean(axis=0)).tolist(),
        "train_fit_max_abs": float(np.abs(residual).max()),
    }


def predict_theta(model: dict[str, Any], v: float, h: float) -> list[float]:
    x = theta_features(v, h, model["V_mean"], model["V_std"],
                       model["H_mean"], model["H_std"])
    coef = np.asarray(model["coefficients"], dtype=float)
    return [float(xi) for xi in (np.asarray(x) @ coef)]


# --------------------------------------------------------------------- evaluation models
def fit_evaluation_models(data_dir: str | Path) -> dict[str, Any]:
    """One 50 Hz ARX per state on the whole chirp_A10 record, validated on prbs/step A10."""

    rows = {}
    for condition in ALL_CONDITIONS:
        name = state_name(condition)
        record = load_full_record(data_dir, record_name(condition, *EVAL_FIT_RECORD, data_dir))
        fit = ft.search_arx(record["da"], record["p"])
        entry = {
            "condition": name,
            "group": "adaptation" if condition in ADAPTATION_CONDITIONS else "heldout",
            "fit_record": record["info"],
            "fit": fit.as_dict(),
            "dc_gain_deg_s_per_deg": ft.arx_dc_gain(fit.a, fit.b, fit.nk),
            "validation": [],
        }
        entry["estimated_steady_authority_deg_s"] = (
            abs(entry["dc_gain_deg_s_per_deg"]) * ft.DA_FULL_SCALE_DEG)
        for kind, amplitude in EVAL_VALIDATION_RECORDS:
            validation = load_full_record(data_dir, record_name(condition, kind, amplitude, data_dir))
            predicted = ft.simulate_arx(validation["da"], fit.a, fit.b, fit.nk)
            entry["validation"].append({
                "record": validation["info"]["record"],
                "file_sha256": validation["info"]["file_sha256"],
                "n_samples": validation["info"]["n_samples"],
                "vaf_percent": ft.vaf(validation["p"], predicted),
                "rmse_deg_s": float(np.sqrt(np.mean((validation["p"] - predicted) ** 2))),
            })
        vafs = [row["vaf_percent"] for row in entry["validation"]]
        entry["min_validation_vaf_percent"] = float(min(vafs))
        entry["validation_ok"] = bool(min(vafs) >= EVAL_VALIDATION_VAF_THRESHOLD)
        rows[name] = entry
    return {
        "fit_record_format": "run_<state>_c01_chirp_A10.mat (whole record)",
        "order_search": {"na": [2, 3, 4, 5], "nb": [2, 3, 4], "nk": [1, 2, 3],
                         "criterion": "in-record VAF, stable poles only"},
        "validation_records": [f"{k}_A{a}" for k, a in EVAL_VALIDATION_RECORDS],
        "validation_vaf_threshold": EVAL_VALIDATION_VAF_THRESHOLD,
        "budget_independent": True,
        "boards": rows,
    }


# --------------------------------------------------------------------- training
def training_order(seed: int = SEED) -> list[int]:
    base = np.random.default_rng(seed).permutation(len(ADAPTATION_CONDITIONS))
    return [int(i) for i in np.tile(base, ROUNDS)]


def train_shared_student(
    arm: str,
    model: ft.DenseConditionalStudent,
    thetas: Sequence[torch.Tensor],
    batches: Sequence[dict[str, Any]],
    params: Sequence[dict[str, Any]],
    order: Sequence[int],
    *,
    steps: int,
    lr: float = LR,
    weight_decay: float = WD,
    clip: float = CLIP,
    seed: int = SEED,
) -> tuple[ft.DenseConditionalStudent, dict[str, Any]]:
    """One shared model, ``steps`` updates cycling through the 8 conditions."""

    if arm == "frozen":
        return model, {"curve": [], "optimizer_steps": 0, "trainable_params": 0,
                       "visits": {}, "order": list(order[:0]), "independent_models": 1,
                       "wall_clock_s": 0.0}
    if arm == "head_only":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("action_head."))
    elif arm in {"full", "scratch"}:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if len(order) < steps:
        raise ValueError("order shorter than the step budget")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    torch.manual_seed(seed)
    model.train()
    curve: list[dict[str, Any]] = []
    visits = {state_name(c): 0 for c in ADAPTATION_CONDITIONS}
    started = time.perf_counter()
    for step in range(steps):
        index = order[step]
        visits[state_name(ADAPTATION_CONDITIONS[index])] += 1
        optimizer.zero_grad(set_to_none=True)
        trace = ft.arx_rollout_torch(model, thetas[index], batches[index], params[index])
        loss = ft.arx_surrogate_loss(trace)
        finite = bool(torch.isfinite(loss).item())
        curve.append({"step": step, "condition": state_name(ADAPTATION_CONDITIONS[index]),
                      "loss": float(loss.item()), "finite": finite})
        if not finite:
            raise RuntimeError(f"non-finite loss at step {step} arm {arm}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, clip).item())
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient at step {step} arm {arm}")
        optimizer.step()
    elapsed = time.perf_counter() - started
    return model, {
        "curve": curve,
        "optimizer_steps": steps,
        "trainable_params": int(sum(p.numel() for p in trainable)),
        "visits": visits,
        "order": [state_name(ADAPTATION_CONDITIONS[i]) for i in order[:steps]],
        "independent_models": 1,
        "wall_clock_s": elapsed,
        "seconds_per_step": elapsed / max(steps, 1),
        "final_loss": curve[-1]["loss"] if curve else None,
    }


# --------------------------------------------------------------------- evaluation
def demand_fraction(command_rad: np.ndarray, authority: float) -> float:
    """Fraction of the episode whose command demand exceeds the steady authority.

    Both sides are in deg/s: ``command_rad`` is the reference roll rate, the
    authority is ``|DC gain| * 10 deg`` of the evaluation model.  Strict ``>``,
    so a command exactly at the limit does not count.
    """

    if not np.isfinite(authority) or authority <= 0:
        raise ValueError(f"authority must be positive and finite, got {authority}")
    command_deg_s = np.degrees(np.asarray(command_rad, dtype=float))
    return float(np.mean(np.abs(command_deg_s) > authority))


def batch_metrics(trace: dict[str, torch.Tensor], ids: Sequence[str],
                  peak_demand: np.ndarray, authority: float,
                  command_rad: np.ndarray) -> list[dict[str, Any]]:
    command_deg = np.degrees(np.asarray(command_rad, dtype=float))
    error_deg = np.degrees(trace["error"].detach().numpy())
    requested = trace["requested_norm"].detach().numpy() * ft.FORCE_LIMIT_N
    applied = trace["applied_n"].detach().numpy()
    p = trace["p"].detach().numpy()
    horizon = error_deg.shape[1]
    requested_padded = np.concatenate([np.zeros((requested.shape[0], 1)), requested], axis=1)
    rows = []
    for i, command_id in enumerate(ids):
        rows.append({
            "command_id": command_id,
            "rmse_deg_s": float(np.sqrt(np.mean(error_deg[i] ** 2))),
            "peak_err_deg_s": float(np.max(np.abs(error_deg[i]))),
            "tv_n_per_s": float(np.sum(np.abs(np.diff(requested_padded[i]))) / (horizon * ft.DT)),
            "sat_ratio": float(np.mean(np.abs(applied[i]) >= ft.FORCE_LIMIT_N - 1e-9)),
            "mean_abs_force_n": float(np.mean(np.abs(requested[i]))),
            "max_force_n": float(np.max(np.abs(requested[i]))),
            "demand_exceeds_authority": bool(peak_demand[i] > authority),
            "fraction_demand_above_authority": demand_fraction(command_rad[i], authority),
            "finite": bool(np.isfinite(p[i]).all() and np.isfinite(requested[i]).all()),
        })
    return rows


def evaluate_all(model: ft.DenseConditionalStudent,
                 eval_models: dict[str, Any],
                 eval_batch: dict[str, Any],
                 thetas: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Batched torch evaluation: one call per condition, 12 commands per call."""

    model.eval()
    peak_demand = np.asarray(eval_batch["peak_demand_deg_s"], dtype=float)
    out = {}
    with torch.no_grad():
        for state, theta in thetas.items():
            entry = eval_models["boards"][state]
            fit = ft.ArxFit(entry["fit"]["na"], entry["fit"]["nb"], entry["fit"]["nk"],
                            tuple(entry["fit"]["a"]), tuple(entry["fit"]["b"]),
                            entry["fit"]["vaf_percent"], entry["fit"]["n_samples"])
            params = ft.arx_torch_parameters(fit.a, fit.b, fit.nk)
            trace = ft.arx_rollout_torch(model, theta, eval_batch, params)
            out[state] = {
                "group": entry["group"],
                "theta_source": None,
                "commands": batch_metrics(trace, eval_batch["ids"], peak_demand,
                                          entry["estimated_steady_authority_deg_s"],
                                          eval_batch["command"]),
            }
    return out


def audit_batched_against_numpy(model: ft.DenseConditionalStudent,
                                eval_models: dict[str, Any],
                                eval_batch: dict[str, Any],
                                theta: torch.Tensor,
                                state: str) -> dict[str, Any]:
    """Full-trajectory check of the batched torch path against the numpy loop."""

    entry = eval_models["boards"][state]
    fit = ft.ArxFit(entry["fit"]["na"], entry["fit"]["nb"], entry["fit"]["nk"],
                    tuple(entry["fit"]["a"]), tuple(entry["fit"]["b"]),
                    entry["fit"]["vaf_percent"], entry["fit"]["n_samples"])
    controller = ft.make_controller(model, theta)
    max_dp = 0.0
    max_dforce = 0.0
    with torch.no_grad():
        batched = ft.arx_rollout_torch(model, theta, eval_batch, ft.arx_torch_parameters(fit.a, fit.b, fit.nk))
    for index in range(len(eval_batch["ids"])):
        plant = ft.ArxPlantNumpy(fit.a, fit.b, fit.nk)
        numpy_trace, _ = ft.arx_rollout_numpy(controller, eval_batch, plant, command_index=index)
        max_dp = max(max_dp, float(np.max(np.abs(numpy_trace["p"] - batched["p"][index].numpy()))))
        max_dforce = max(max_dforce, float(np.max(
            np.abs(numpy_trace["requested_n"] - batched["requested_norm"][index].numpy() * ft.FORCE_LIMIT_N))))
    return {"condition": state, "command": "all 12 profiles", "max_abs_dp_rad": max_dp,
            "max_abs_dforce_n": max_dforce}


# --------------------------------------------------------------------- summaries
def summarize_arm(conditions: dict[str, Any], ids: Sequence[str]) -> dict[str, Any]:
    groups = {"adaptation": [], "heldout": []}
    for state, payload in conditions.items():
        for row in payload["commands"]:
            groups[payload["group"]].append(row)
    out = {}
    for group, rows in groups.items():
        out[group] = {
            "n_cells": len(rows),
            "mean_rmse_deg_s": float(np.mean([r["rmse_deg_s"] for r in rows])) if rows else float("nan"),
            "mean_peak_err_deg_s": float(np.mean([r["peak_err_deg_s"] for r in rows])) if rows else float("nan"),
            "mean_tv_n_per_s": float(np.mean([r["tv_n_per_s"] for r in rows])) if rows else float("nan"),
            "sum_tv_n_per_s": float(np.sum([r["tv_n_per_s"] for r in rows])) if rows else float("nan"),
            "n_nonfinite": int(sum(0 if r["finite"] else 1 for r in rows)),
            "n_unreachable_demand": int(sum(1 for r in rows if r["demand_exceeds_authority"])),
        }
    return out


def compare_arms(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    better = worse = equal = 0
    worst = []
    for state in candidate:
        for i, row in enumerate(candidate[state]["commands"]):
            base = reference[state]["commands"][i]
            if row["command_id"] != base["command_id"]:
                raise ValueError("command order mismatch")
            delta = row["rmse_deg_s"] - base["rmse_deg_s"]
            if delta < -1e-9:
                better += 1
            elif delta > 1e-9:
                worse += 1
                worst.append({"state": state, "command": row["command_id"],
                              "candidate": row["rmse_deg_s"], "reference": base["rmse_deg_s"],
                              "delta": delta})
            else:
                equal += 1
    worst.sort(key=lambda item: -item["delta"])
    return {"better": better, "worse": worse, "equal": equal, "n_cells": better + worse + equal,
            "worst_regressions": worst[:5]}
