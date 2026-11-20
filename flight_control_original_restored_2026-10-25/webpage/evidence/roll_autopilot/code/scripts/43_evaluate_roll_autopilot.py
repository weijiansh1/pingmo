"""Evaluate conventional roll-loop periphery, without training any network.

Writes a new, separate evidence bundle. Historical rate-loop experiments remain
unchanged. This is a reduced-roll integration experiment, not a six-DOF test.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.controllers.roll_autopilot import PingmoRateController, RollAutopilot, RollStateEstimate
from src.student.controller import load_student_controller
from src.transfer import fine_tune as ft

WORKSPACE = REPO.parent


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def physical_rollout(policy, fit, command_rad: np.ndarray, *, bank_mode: bool):
    """At t_k sample state; apply u_k; record state at t_(k+1).

    Plant I/O is da_act [deg] -> p [deg/s]. The state provider converts p to
    radians and integrates phi with trapezoidal kinematics, ASSUMING pitch=q=r=0.
    In general phi_dot != p: RollOuterLoop explicitly supports q/r compensation,
    but this SISO plant cannot validate the coupled aircraft response.
    """
    autopilot = RollAutopilot(PingmoRateController(policy))
    plant = ft.ArxPlantNumpy(fit["a"], fit["b"], fit["nk"])
    trace = {key: [] for key in (
        "time_s", "command_rad", "bank_rad", "p_rad_s", "p_sp_rad_s", "p_ref_rad_s",
        "aileron_request_rad", "aileron_command_rad", "aileron_applied_rad",
        "rate_limited", "saturated")}
    bank = p = 0.0
    for k, command in enumerate(command_rad):
        state = RollStateEstimate(k * ft.DT, bank, p)
        kwargs = {"roll_setpoint_rad" if bank_mode else "p_setpoint_rad_s": float(command)}
        output = autopilot.step(state, **kwargs)
        p_next = math.radians(plant.push(math.degrees(output.surface.applied_rad)))
        bank += 0.5 * (p + p_next) * ft.DT
        p = p_next
        if not math.isfinite(bank) or not math.isfinite(p):
            raise FloatingPointError(f"nonfinite plant at sample {k}")
        values = (
            (k + 1) * ft.DT, command, bank, p, output.p_setpoint_rad_s,
            output.inner.next_reference_rad_s, output.surface.requested_rad,
            output.surface.commanded_rad, output.surface.applied_rad,
            output.surface.rate_limited,
            abs(output.surface.applied_rad) >= math.radians(10) - 1e-10)
        for key, value in zip(trace, values, strict=True):
            trace[key].append(value)
    return {key: np.asarray(values) for key, values in trace.items()}


def bank_commands() -> dict[str, np.ndarray]:
    t = np.arange(1000) * ft.DT
    step = np.where(t >= 1, 20.0, 0.0)
    reversal = np.select([t < 1, t < 8, t < 15], [0.0, 15.0, -15.0], default=0.0)
    sine = np.where(t >= 1, 10 * np.sin(2 * np.pi * 0.08 * (t - 1)), 0.0)
    return {"bank_step_20": np.radians(step), "bank_reversal_15": np.radians(reversal),
            "bank_sine_10_008Hz": np.radians(sine)}


def bank_metrics(trace) -> dict:
    error = np.degrees(trace["command_rad"] - trace["bank_rad"])
    da = np.degrees(trace["aileron_applied_rad"])
    increments = np.diff(np.r_[0.0, da])
    bank = np.degrees(trace["bank_rad"])
    return {
        "bank_rmse_deg": float(np.sqrt(np.mean(error ** 2))),
        "bank_peak_error_deg": float(np.max(np.abs(error))),
        "tail_mean_abs_error_deg": float(np.mean(np.abs(error[-50:]))),
        "max_abs_bank_deg": float(np.max(np.abs(bank))),
        "max_abs_p_deg_s": float(np.max(np.abs(np.degrees(trace["p_rad_s"])))),
        "aileron_max_deg": float(np.max(np.abs(da))),
        "aileron_max_slew_deg_s": float(np.max(np.abs(increments)) / ft.DT),
        "aileron_tv_deg_s": float(np.sum(np.abs(increments)) / (len(da) * ft.DT)),
        "saturation_ratio": float(np.mean(trace["saturated"])),
        "slew_limited_ratio": float(np.mean(trace["rate_limited"])),
        "within_30deg_bank": bool(np.max(np.abs(bank)) <= 30.0),
        "finite": bool(all(np.isfinite(v).all() for v in trace.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=WORKSPACE / "roll_autopilot" / "20261019")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"evidence already exists: {args.out}; select a new --out")
    torch.set_num_threads(args.threads)
    base = WORKSPACE / "dhc6_finetune" / "unified"
    theta_path = base / "b30" / "theta.json"
    eval_path = base / "shared" / "eval_models.json"
    checkpoints = {"frozen": ft.DEFAULT_CHECKPOINT, "full_b30": base / "b30" / "weights" / "full.pt"}
    inputs = list(checkpoints.values()) + [theta_path, eval_path]
    inputs += [base / f"b{b}" / "manifest.json" for b in (15, 30, 60)]
    before = {str(path): ft.sha256_file(path) for path in inputs}
    theta_data = json.loads(theta_path.read_text(encoding="utf-8"))
    thetas = {row["condition"]: np.asarray(row["normalized_theta"], dtype=np.float32)
              for group in ("adaptation", "heldout") for row in theta_data[group]}
    boards = json.loads(eval_path.read_text(encoding="utf-8"))["boards"]
    commands = bank_commands()
    rate_profiles = ft.regression_commands() + ft.heldout_commands()
    rate_batch = ft.command_batch_50hz(rate_profiles)
    args.out.mkdir(parents=True)
    (args.out / "trajectories").mkdir()
    rows, parity_rows = [], []
    started = time.perf_counter()
    for arm, path in checkpoints.items():
        for state_name, board in boards.items():
            policy, _ = load_student_controller(path, thetas[state_name])
            for command_name, command in commands.items():
                try:
                    trace = physical_rollout(policy.predict, board["fit"], command, bank_mode=True)
                    row = {"arm": arm, "condition": state_name, "group": board["group"],
                           "command_id": command_name, **bank_metrics(trace)}
                    np.savez_compressed(args.out / "trajectories" / f"{arm}__{state_name}__{command_name}.npz",
                                        **trace)
                except (ValueError, FloatingPointError, OverflowError) as exc:
                    # A failed episode is retained; never replace it with the best successful one.
                    row = {"arm": arm, "condition": state_name, "group": board["group"],
                           "command_id": command_name, "finite": False, "failure": str(exc)}
                rows.append(row)
            for index, profile in enumerate(rate_profiles):
                fit = board["fit"]
                old, _ = ft.arx_rollout_numpy(policy.predict, rate_batch,
                    ft.ArxPlantNumpy(fit["a"], fit["b"], fit["nk"]), command_index=index)
                new = physical_rollout(policy.predict, fit, rate_batch["command"][index], bank_mode=False)
                parity_rows.append({"arm": arm, "condition": state_name, "command_id": profile.command_id,
                    "max_p_difference_deg_s": float(np.max(np.abs(np.degrees(new["p_rad_s"] - old["p"])))),
                    "max_reference_difference_rad_s": float(np.max(np.abs(new["p_ref_rad_s"] - old["ref"]))),
                    "max_surface_difference_deg": float(np.max(np.abs(
                        np.degrees(new["aileron_applied_rad"]) + 10 / 22 * old["applied_n"])))})
            print(f"{arm} {state_name}: attitude 3 / rate parity {len(rate_profiles)}", flush=True)
    after = {str(path): ft.sha256_file(path) for path in inputs}
    if before != after:
        raise RuntimeError("an archived input changed during the evaluation")
    summary = {}
    for arm in checkpoints:
        summary[arm] = {}
        for group in ("adaptation", "heldout", "all"):
            selected = [r for r in rows if r["arm"] == arm and (group == "all" or r["group"] == group)]
            finite = [r for r in selected if r["finite"]]
            summary[arm][group] = {
                "n_cases": len(selected), "n_finite": len(finite),
                "n_within_30deg_bank": sum(r.get("within_30deg_bank", False) for r in selected),
                "mean_bank_rmse_deg": float(np.mean([r["bank_rmse_deg"] for r in finite])) if finite else None,
                "max_abs_bank_deg": max((r["max_abs_bank_deg"] for r in finite), default=None),
                "mean_aileron_tv_deg_s": float(np.mean([r["aileron_tv_deg_s"] for r in finite])) if finite else None,
            }
    parity = {"n_cases": len(parity_rows), "tolerance_deg_s": 0.001,
              "max_p_difference_deg_s": max(r["max_p_difference_deg_s"] for r in parity_rows),
              "max_surface_difference_deg": max(r["max_surface_difference_deg"] for r in parity_rows),
              "max_reference_difference_rad_s": max(r["max_reference_difference_rad_s"] for r in parity_rows)}
    parity["passed"] = parity["max_p_difference_deg_s"] < parity["tolerance_deg_s"]
    manifest = {
        "schema_version": "pingmo_roll_periphery_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "training": "none; all central networks and theta frozen", "inputs_sha256": before,
        "archived_inputs_unchanged": before == after,
        "implementation_sha256": {str(path.relative_to(REPO)): ft.sha256_file(path) for path in (
            Path(__file__), REPO / "src/controllers/roll_autopilot.py", REPO / "src/aircraft/actuator.py",
            REPO / "src/student/dense/policy.py")},
        "protocol": {"dt_s": ft.DT, "duration_s": 20, "outer_time_constant_s": 2.0,
                     "bank_command_limit_deg": 30, "body_rate_command_limit_deg_s": 30,
                     "aileron_limit_deg": 10, "aileron_rate_limit_deg_s": 40,
                     "limits_source": "experiment assumptions, derived from historical mapping; not DHC-6 hardware specs",
                     "aileron_action_polarity": -1, "theta_budget_s_per_state": 30,
                     "state_feedback": "simulated p; trapezoid phi integration with pitch=q=r=0; no IMU/EKF",
                     "plant": "fixed independent 50 Hz da_act->p ARX; no additional fitted servo lag",
                     "metric_time": "post-step state t[k+1] against ZOH command over [t[k],t[k+1])",
                     "bank_commands": {"step": "+20 deg at 1s", "reversal": "+15 at 1s, -15 at 8s, 0 at 15s",
                                       "sine": "10 deg, 0.08 Hz, onset 1s"},
                     "rate_regression_commands": [p.command_id for p in rate_profiles],
                     "outer_gains_tuned_on_results": False,
                     "quality_scope": "integration evidence only; not GJB handling-quality certification"},
        "n_attitude_cases": len(rows), "rate_parity": parity, "summary": summary,
        "failures": [r for r in rows if not r["finite"]],
        "elapsed_s": time.perf_counter() - started,
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__,
                    "torch_threads": args.threads},
    }
    write_json(args.out / "manifest.json", manifest)
    write_json(args.out / "metrics.json", rows)
    write_json(args.out / "rate_parity.json", parity_rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (args.out / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"summary": summary, "rate_parity": parity, "elapsed_s": manifest["elapsed_s"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
