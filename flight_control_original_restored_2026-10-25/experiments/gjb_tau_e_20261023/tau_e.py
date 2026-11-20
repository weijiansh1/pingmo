"""GJB tau_e evaluation prototype for the roll-rate loop.

    cd pingmo
    python experiments/gjb_tau_e_20261023/tau_e.py synthetic
    python experiments/gjb_tau_e_20261023/tau_e.py pilot
    python experiments/gjb_tau_e_20261023/tau_e.py batch

The evaluator injects a small multisine roll-rate command, records the
closed-loop response at 50 Hz, estimates H(jw) at 0.1-10 rad/s, and fits the
4th-order roll LOES

    P/F_as = K s(s^2+2 zeta_phi w_phi s + w_phi^2)
             / [(s-lambda_s)(s+1/T_R)(s^2+2 zeta_d w_d s + w_d^2)]
             * exp(-tau_e s).

Only fits that pass the residual gate receive an A21 level.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WS = HERE.parents[2]
PINGMO = WS / "pingmo"
SAE = PINGMO / "experiments/sae_delay_20261023"
sys.path.insert(0, str(PINGMO))
sys.path.insert(0, str(SAE))

import src.envs.batched_p_channel as bpc  # noqa: E402
from src.aircraft.parameters import PChannelParameters  # noqa: E402
from src.envs.roll_rate_commands import RollRateCommandProfile  # noqa: E402
import common  # noqa: E402
import train_policy as TP  # noqa: E402

TP.DATA = SAE / "results_large"
TP.FEAT = TP.DATA / "features"

FS = 50.0
DT = 1.0 / FS
DURATION_S = 180.0
STEPS = int(round(DURATION_S / DT))
POLICIES = ("raw_ft", "topk4", "aux_topk4")
POLICY_DIR = TP.DATA / "policies_all_dagger2"
RESULT_DIR = HERE / "results"
KEYS = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")


# ------------------------------------------------------------------ excitation
def multisine_profile(n_components: int = 18, amplitude_deg_s: float = 1.0):
    """Log-spaced integer DFT bins between 0.1 and 10 rad/s over DURATION_S."""
    f_lo, f_hi = 0.1 / (2 * math.pi), 10.0 / (2 * math.pi)
    k_lo = max(2, int(math.ceil(f_lo * DURATION_S)))
    k_hi = int(math.floor(f_hi * DURATION_S))
    bins = np.unique(np.round(np.geomspace(k_lo, k_hi, n_components)).astype(int))
    freqs = bins / DURATION_S
    comps = tuple((amplitude_deg_s, float(f)) for f in freqs)
    return RollRateCommandProfile("gjb-tau-e-multisine", "multisine", onset_s=0.0,
                                  multisine_components=comps,
                                  duration_s=DURATION_S), freqs


# ---------------------------------------------------------------- long rollout
def run_case(policy_name: str, row: dict, duration_s: float = DURATION_S,
             n_components: int = 18, amplitude_deg_s: float = 1.0,
             return_signals: bool = False) -> dict:
    torch.set_num_threads(1)
    policy = load_policy(policy_name)
    params = PChannelParameters(**{k: row["parameters"][k] for k in KEYS})
    profile, freqs = multisine_profile(n_components, amplitude_deg_s)
    steps = int(round(duration_s / DT))
    old_steps = bpc.EPISODE_STEPS
    bpc.EPISODE_STEPS = steps
    try:
        env = bpc.BatchedRollRateEnv(params, n_envs=1, commands=[profile], seed=0)
        env.command_index[:] = 0
        obs = env.observation()
        pc, pref, p, req = [], [], [], []
        with torch.no_grad():
            for _ in range(steps - 1):
                pc.append(float(env.command.item()))
                pref.append(float(env.reference.item()))
                action = policy(obs)
                obs, _, out = env.step(action)
                p.append(float(out.item()))
                req.append(float(env.requested.item()))
    finally:
        bpc.EPISODE_STEPS = old_steps
    pc = np.asarray(pc)
    p = np.asarray(p)
    H, fit_res = estimate_frf(pc, p, freqs)
    out = {"plant_id": row["plant_id"], "policy": policy_name,
           "tau_p": float(row["parameters"]["tau_p"]),
           "split": row["split"], "level": int(row["gjb_level"]),
           "duration_s": duration_s, "freqs_hz": freqs.tolist(),
           "H_real": H.real.tolist(), "H_imag": H.imag.tolist(),
           "p_rms": float(np.sqrt(np.mean(p ** 2))),
           "pc_rms": float(np.sqrt(np.mean(pc ** 2))),
           "residual_rms": float(fit_res)}
    if return_signals:
        out["pc"] = pc.tolist()
        out["p"] = p.tolist()
    return out


def estimate_frf(pc: np.ndarray, p: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, float]:
    n = len(pc)
    t = np.arange(n) / FS
    H = np.empty(len(freqs), dtype=complex)
    resid = []
    for i, f in enumerate(freqs):
        w = 2 * math.pi * f
        X = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones(n)])
        cu, *_ = np.linalg.lstsq(X, pc, rcond=None)
        cy, *_ = np.linalg.lstsq(X, p, rcond=None)
        u = cu[0] - 1j * cu[1]
        y = cy[0] - 1j * cy[1]
        H[i] = y / u if abs(u) > 1e-12 else np.nan
        resid.append(float(np.sqrt(np.mean((X @ cy - p) ** 2))))
    return H, float(np.mean(resid))


# --------------------------------------------------------------------- fitting
def loes_H(params: np.ndarray, w: np.ndarray) -> np.ndarray:
    logk, zphi, logwphi, loglam, logtr, zd, logwd, tau = params
    s = 1j * w
    k = math.exp(logk)
    wphi = math.exp(logwphi)
    lam = -math.exp(loglam)
    tr = math.exp(logtr)
    wd = math.exp(logwd)
    num = k * s * (s ** 2 + 2 * zphi * wphi * s + wphi ** 2)
    den = (s - lam) * (s + 1.0 / tr) * (s ** 2 + 2 * zd * wd * s + wd ** 2)
    return (num / den) * np.exp(-tau * s)


def fit_loes(freqs: np.ndarray, H: np.ndarray, n_starts: int = 16,
             seed: int = 0, w_phase: float = 2.0) -> dict:
    from scipy.optimize import least_squares
    w = 2 * math.pi * freqs
    good = np.isfinite(H) & (np.abs(H) > 0)
    w, H = w[good], H[good]
    logmag = np.log(np.abs(H) + 1e-12)
    phase = np.angle(H)

    lb = np.array([-8.0, 0.01, math.log(0.5), math.log(0.005), math.log(0.03),
                   0.01, math.log(0.2), 0.0])
    ub = np.array([8.0, 1.5, math.log(30.0), math.log(3.0), math.log(30.0),
                   1.5, math.log(40.0), 0.5])
    rng = np.random.default_rng(seed)

    def residual(x):
        Hf = loes_H(x, w)
        dm = np.log(np.abs(Hf) + 1e-12) - logmag
        dp = np.angle(Hf / H)
        return np.concatenate([2.0 * dm, w_phase * dp])

    best = None
    for i in range(n_starts):
        if i == 0:
            x0 = np.array([0.0, 0.5, math.log(3.0), math.log(0.1),
                           math.log(0.5), 0.3, math.log(2.0), 0.05])
        else:
            x0 = lb + rng.random(len(lb)) * (ub - lb)
            x0[7] = rng.uniform(0.0, 0.15)
        try:
            sol = least_squares(residual, x0, bounds=(lb, ub), max_nfev=4000,
                                xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                x_scale="jac")
        except Exception:
            continue
        if not np.all(np.isfinite(sol.x)):
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return {"ok": False, "reason": "all starts failed"}
    x = best.x
    Hf = loes_H(x, w)
    dm = np.log(np.abs(Hf) + 1e-12) - logmag
    dp = np.angle(Hf / H)
    ss_res = float(np.sum(dm ** 2))
    ss_tot = float(np.sum((logmag - logmag.mean()) ** 2))
    vaf = 1.0 - ss_res / max(ss_tot, 1e-12)
    phase_rmse = float(np.degrees(np.sqrt(np.mean(dp ** 2))))
    return {
        "ok": True,
        "params": {
            "K": float(math.exp(x[0])), "zeta_phi": float(x[1]),
            "omega_phi": float(math.exp(x[2])), "lambda_s": float(-math.exp(x[3])),
            "T_R": float(math.exp(x[4])), "zeta_d": float(x[5]),
            "omega_d": float(math.exp(x[6])), "tau_e": float(x[7]),
        },
        "vaf_mag": vaf, "phase_rmse_deg": phase_rmse,
        "cost": float(best.cost),
    }


def a21_level(tau_e: float) -> str:
    if tau_e <= 0.10:
        return "level1"
    if tau_e <= 0.20:
        return "level2"
    if tau_e <= 0.25:
        return "level3"
    return "exceeds_level3"


def gate(fit: dict, vaf_min: float = 0.95, phase_max: float = 10.0) -> bool:
    return bool(fit.get("ok") and fit["vaf_mag"] >= vaf_min
                and fit["phase_rmse_deg"] <= phase_max)


# ---------------------------------------------------------------------- policy
_MODEL = None
_POLICY_NAME = None
_AMPLITUDE = 1.0


def load_policy(name: str):
    global _MODEL, _POLICY_NAME
    if _MODEL is not None and _POLICY_NAME == name:
        return _MODEL
    path = POLICY_DIR / f"{name}.pt"
    config = torch.load(path, map_location="cpu", weights_only=False)
    model = TP.build_model(config)
    _MODEL, _POLICY_NAME = model, name
    return model


# ------------------------------------------------------------------------ batch
def _init_worker(name: str, amplitude: float = 1.0,
                 policy_dir: str | None = None) -> None:
    global _AMPLITUDE, POLICY_DIR
    _AMPLITUDE = float(amplitude)
    if policy_dir:
        POLICY_DIR = Path(policy_dir)
    load_policy(name)


def _work(args: tuple[str, dict]) -> dict:
    name, row = args
    return run_case(name, row, amplitude_deg_s=_AMPLITUDE)


def select_plants(n_per_bin: int | None) -> list[dict]:
    metas = json.loads((SAE / "results_large/plants.json").read_text(encoding="utf-8"))
    test = [m for m in metas if m["split"] == "id_test"]
    rows = {r["plant_id"]: r for r in common.load_library()}
    picked = []
    for bin_name in ("short", "medium", "long"):
        group = [m for m in test if m["delay_bin"] == bin_name]
        if n_per_bin is not None:
            group = group[:n_per_bin]
        picked += [rows[m["plant_id"]] for m in group]
    return picked


def run_batch(tag: str, n_per_bin: int | None, policies=POLICIES,
              workers: int = 12, amplitude: float = 1.0,
              policy_dir: Path | None = None) -> None:
    rows = select_plants(n_per_bin)
    out = RESULT_DIR / tag
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pdir = str(policy_dir or POLICY_DIR)
    for name in policies:
        print(f"policy {name}: {len(rows)} plants @ {amplitude:.2f} deg/s", flush=True)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(name, amplitude, pdir)) as pool:
            for i, result in enumerate(pool.map(_work, [(name, r) for r in rows]), 1):
                (out / f"{name}__{result['plant_id']}.json").write_text(
                    json.dumps(result), encoding="utf-8")
                if i % 10 == 0 or i == len(rows):
                    print(f"  {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {out} in {time.time() - t0:.0f}s", flush=True)


def fit_saved(tag: str, vaf_min: float = 0.95, phase_max: float = 10.0) -> dict:
    out = RESULT_DIR / tag
    files = sorted(out.glob("*.json"))
    results = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        H = np.asarray(data["H_real"]) + 1j * np.asarray(data["H_imag"])
        fit = fit_loes(np.asarray(data["freqs_hz"]), H, w_phase=2.0)
        ok = gate(fit, vaf_min, phase_max)
        results[path.stem] = {
            "plant_id": data["plant_id"], "policy": data["policy"],
            "tau_p": data["tau_p"], "delay_bin": common.delay_bin(data["tau_p"]),
            "fit": fit, "gate": ok,
            "tau_e": fit["params"]["tau_e"] if fit.get("ok") else None,
            "a21": a21_level(fit["params"]["tau_e"]) if ok else "not_assessable",
        }
    (out / "fits.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    return results


def summarize(fits: dict) -> dict:
    out = {}
    for policy in POLICIES:
        group = [v for v in fits.values() if v["policy"] == policy]
        if not group:
            continue
        taus = [v["tau_e"] for v in group if v["tau_e"] is not None]
        ok = [v for v in group if v["gate"]]
        out[policy] = {
            "n": len(group), "gate_pass": len(ok),
            "tau_e_median": float(np.median(taus)) if taus else None,
            "tau_e_p90": float(np.percentile(taus, 90)) if taus else None,
            "levels": {k: sum(1 for v in ok if v["a21"] == k)
                       for k in ("level1", "level2", "level3", "exceeds_level3")},
            "vaf_median": float(np.median([v["fit"]["vaf_mag"] for v in group
                                          if v["fit"].get("ok")])),
            "phase_rmse_median": float(np.median([v["fit"]["phase_rmse_deg"]
                                                  for v in group if v["fit"].get("ok")])),
        }
    return out


# --------------------------------------------------------------------- synthetic
def synthetic() -> None:
    freqs = np.geomspace(0.1, 10.0, 24) / (2 * math.pi)
    true = np.array([0.0, 0.4, math.log(2.5), math.log(0.08), math.log(0.6),
                     0.25, math.log(2.0), 0.06])
    H = loes_H(true, 2 * math.pi * freqs)
    rng = np.random.default_rng(1)
    Hn = H * (1 + 0.01 * rng.normal(size=len(H)))
    fit = fit_loes(freqs, Hn, n_starts=8)
    print(json.dumps(fit, indent=1))


# -------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("synthetic", "pilot", "batch", "fit"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-per-bin", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--vaf-min", type=float, default=0.95)
    ap.add_argument("--phase-max", type=float, default=10.0)
    ap.add_argument("--amplitude", type=float, default=1.0)
    ap.add_argument("--policy-dir", default=None)
    args = ap.parse_args()
    if args.policy_dir:
        global POLICY_DIR
        POLICY_DIR = Path(args.policy_dir)
    if args.mode == "synthetic":
        synthetic()
    elif args.mode == "pilot":
        tag = args.tag or "pilot"
        run_batch(tag, args.n_per_bin, workers=args.workers,
                  amplitude=args.amplitude, policy_dir=POLICY_DIR)
        fits = fit_saved(tag, args.vaf_min, args.phase_max)
        print(json.dumps(summarize(fits), ensure_ascii=False, indent=1))
    elif args.mode == "batch":
        tag = args.tag or ("batch" if args.amplitude == 1.0
                           else f"batch_amp{int(round(args.amplitude * 100)):03d}")
        run_batch(tag, None, workers=args.workers, amplitude=args.amplitude,
                  policy_dir=POLICY_DIR)
        fits = fit_saved(tag, args.vaf_min, args.phase_max)
        print(json.dumps(summarize(fits), ensure_ascii=False, indent=1))
    elif args.mode == "fit":
        fits = fit_saved(args.tag or "batch", args.vaf_min, args.phase_max)
        print(json.dumps(summarize(fits), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
