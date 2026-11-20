"""Library statistics, fractional-delay transfer-function check and two figures.

CPU only; no training.  Writes verification.json and figures/.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pinmo6 import PINGMO, ROLL_SOURCE  # noqa: E402
from pinmo6.aircraft import simulate  # noqa: E402
from pinmo6.lateral import directional_polynomials  # noqa: E402
from pinmo6.longitudinal import COMPONENTS, longitudinal_polynomials  # noqa: E402
from pinmo6.sampler import SOURCE_LIBRARY, load_joint_library  # noqa: E402
from src.aircraft.p_channel import p_channel_polynomials  # noqa: E402

LIBRARY = HERE / "library/joint_six_channel_5000_v1/plants.jsonl"
FIGURES = HERE / "figures"
DT = 0.001
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
DASHES = ("-", "--", ":")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_roll_source() -> dict:
    out = {}
    for name in ("p_channel.py", "delay.py", "parameters.py"):
        current, original = ROLL_SOURCE / "src/aircraft" / name, PINGMO / "FQControl_Model/src/aircraft" / name
        out[name] = {"sha256": sha256(current), "identical_to_FQControl_Model": sha256(current) == sha256(original)}
    return out


def library_statistics(entries) -> dict:
    records = [record for record, _ in entries]
    lon = [parameters.longitudinal for _, parameters in entries]
    directional = [parameters.directional for _, parameters in entries]
    span = lambda values: [float(np.min(values)), float(np.max(values))]  # noqa: E731
    pairs = Counter((r["sampling_bucket"], r["longitudinal"]["gjb_level"]) for r in records)
    return {
        "plants": len(records),
        "joint_level_counts": dict(Counter(str(r["joint_level"]) for r in records)),
        "sampling_bucket_by_longitudinal_level": {f"{bucket}/lon_L{level}": count for (bucket, level), count in sorted(pairs.items())},
        "longitudinal_component_level_counts": {
            name: dict(Counter(r["longitudinal"]["gjb_component_levels"][name] for r in records)) for name in COMPONENTS},
        "longitudinal_ranges": {
            "omega_sp_rad_s": span([p.omega_sp for p in lon]), "zeta_sp": span([p.zeta_sp for p in lon]),
            "n_alpha_g_rad": span([p.n_alpha for p in lon]), "cap": span([p.cap for p in lon]),
            "t_theta2_s": span([p.t_theta2 for p in lon]), "fs_per_g_n": span([p.fs_per_g for p in lon]),
            "tau_theta_s": span([p.tau_theta for p in lon]), "v0_m_s": span([p.v0 for p in lon])},
        "directional_ranges": {
            "sideslip_per_force_deg_per_n": span([d.sideslip_per_force_deg_per_n for d in directional]),
            "r_omega_r": span([d.r_omega_r for d in directional]), "zeta_r": span([d.zeta_r for d in directional]),
            "tau_rp_s": span([d.tau_rp for d in directional])},
    }


def _delayed_step(numerator, denominator, time, tau):
    """Exact continuous unit-step response at time - tau, sampled on the plant grid.

    The state is propagated over the partial interval from tau to the next grid
    point, then by the exact one-step map; no interpolation is involved.
    """
    a, b, c, _ = signal.tf2ss(numerator, denominator)
    first = math.floor(tau / DT) + 1
    partial = first * DT - tau
    ad1, bd1, *_ = signal.cont2discrete((a, b, c, np.zeros((1, 1))), partial, method="zoh")
    ad, bd, *_ = signal.cont2discrete((a, b, c, np.zeros((1, 1))), DT, method="zoh")
    out = np.zeros(len(time))
    state = bd1[:, 0].copy()
    for index in range(first, len(time)):
        out[index] = float(c @ state)
        state = ad @ state + bd[:, 0]
    return out


def fractional_delay_check(entries, per_bucket: int = 10, seconds: float = 6.0) -> dict:
    """Library delays are off the 1 ms grid; FractionalDelay interpolates the input.

    The reference is the exact continuous response with a pure delay, so the
    error measures the interpolated delay, not the transfer-function realization.
    """
    time = np.arange(int(seconds / DT)) * DT
    chosen = []
    for bucket in ("L1", "L2", "L3", "OOD"):
        members = [entry for entry in entries if entry[0]["sampling_bucket"] == bucket]
        chosen += members[:: max(1, len(members) // per_bucket)][:per_bucket]
    errors = {name: [] for name in ("q", "alpha", "nz", "p", "beta", "r")}
    for _, parameters in chosen:
        lon, roll, directional = parameters.longitudinal, parameters.roll, parameters.directional
        runs = [simulate(parameters, np.eye(3)[[channel]].repeat(len(time), 0), inertial_coupling=False)
                for channel in range(3)]
        lon_tf, dir_tf = longitudinal_polynomials(lon), directional_polynomials(roll, directional)
        cases = (("q", runs[0]["states"][:, 2], lon_tf["q"], lon.tau_theta),
                 ("alpha", runs[0]["states"][:, 1] - lon.alpha_trim, lon_tf["alpha"], lon.tau_theta),
                 ("nz", runs[0]["nz_g"], lon_tf["nz"], lon.tau_theta),
                 ("p", runs[1]["states"][:, 5], p_channel_polynomials(roll), roll.tau_p),
                 ("beta", runs[2]["states"][:, 4], dir_tf["beta"], directional.tau_rp),
                 ("r", runs[2]["states"][:, 6], dir_tf["r"], directional.tau_rp))
        for name, simulated, (num, den), tau in cases:
            reference = _delayed_step(num, den, time, tau)
            errors[name].append(float(np.max(np.abs(simulated - reference)) / max(np.max(np.abs(reference)), 1e-12)))
    return {"plants": len(chosen), "seconds": seconds,
            "max_relative_error": {name: max(values) for name, values in errors.items()},
            "median_relative_error": {name: float(np.median(values)) for name, values in errors.items()}}


def _style(axis, title: str, ylabel: str) -> None:
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", fontsize=10, color=INK)
    axis.set_ylabel(ylabel, fontsize=9, color=INK_2)
    axis.grid(True, color=GRID, linewidth=0.6)
    axis.tick_params(colors=INK_2, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(GRID)


def representatives(entries) -> list:
    out = []
    for level in (1, 2, 3):
        out.append(next(entry for entry in entries
                        if entry[0]["split"] == "train_core" and entry[0]["joint_level"] == level
                        and entry[0]["longitudinal"]["gjb_level"] == level))
    return out


def step_figure(chosen, source_rows) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seconds = 6.0
    time = np.arange(int(seconds / DT)) * DT
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.2), facecolor=SURFACE, sharex=True)
    panels = (("A  q, pitch stick for 0.5 g", "deg/s"), ("B  alpha, same input", "deg"),
              ("C  n_z, same input", "g"), ("D  p, lateral stick 5 N", "deg/s"),
              ("E  beta, pedal for 5 deg sideslip", "deg"), ("F  r, same pedal input", "deg/s"))
    summary = []
    for index, (record, parameters) in enumerate(chosen):
        lon, d = parameters.longitudinal, parameters.directional
        fs, fas, frp = 0.5 * lon.fs_per_g, 5.0, 5.0 / d.sideslip_per_force_deg_per_n
        runs = [simulate(parameters, np.tile(vector, (len(time), 1)), inertial_coupling=False)
                for vector in ((fs, 0, 0), (0, fas, 0), (0, 0, frp))]
        curves = (np.degrees(runs[0]["states"][:, 2]), np.degrees(runs[0]["states"][:, 1] - lon.alpha_trim),
                  runs[0]["nz_g"], np.degrees(runs[1]["states"][:, 5]),
                  np.degrees(runs[2]["states"][:, 4]), np.degrees(runs[2]["states"][:, 6]))
        label = f"L{record['joint_level']} · {record['plant_id']}"
        for axis, curve in zip(axes.flat, curves):
            axis.plot(time, curve, color=SERIES[index], linestyle=DASHES[index], linewidth=1.5, label=label)
        axes.flat[0].annotate(f"L{record['joint_level']}", (time[-1], curves[0][-1]), xytext=(4, 0),
                              textcoords="offset points", fontsize=8, color=INK_2, va="center")
        summary.append({"plant_id": record["plant_id"], "joint_level": record["joint_level"],
                        "pitch_stick_n": fs, "pedal_n": frp,
                        "nz_at_6s_g": float(runs[0]["nz_g"][-1]), "beta_at_6s_deg": float(curves[4][-1]),
                        "omega_sp": lon.omega_sp, "zeta_sp": lon.zeta_sp, "cap": lon.cap,
                        "roll_sensitivity_deg_per_n": source_rows[record["plant_id"]]["sensitivity_measured_deg_per_n"]})
    for axis, (title, ylabel) in zip(axes.flat, panels):
        _style(axis, title, ylabel)
    for axis in axes[1]:
        axis.set_xlabel("time (s)", fontsize=9, color=INK_2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"step_responses.{suffix}", dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return {"figure": "figures/step_responses.png", "aircraft": summary}


def maneuver_figure(chosen, source_rows) -> dict:
    """Bank to about 30 deg with the lateral stick, then pull 0.3 g: the 12 states via kinematics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    seconds = 20.0
    time = np.arange(int(seconds / DT)) * DT
    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), facecolor=SURFACE)
    summary = []
    for index, (record, parameters) in enumerate(chosen):
        lon = parameters.longitudinal
        inputs = np.zeros((len(time), 3))
        inputs[time < 1.0, 1] = 30.0 / source_rows[record["plant_id"]]["sensitivity_measured_deg_per_n"]
        inputs[time >= 3.0, 0] = 0.3 * lon.fs_per_g
        run = simulate(parameters, inputs)
        states = run["states"]
        style = dict(color=SERIES[index], linestyle=DASHES[index], linewidth=1.5, label=f"L{record['joint_level']} · {record['plant_id']}")
        axes[0].plot(time, np.degrees(states[:, 7]), **style)
        axes[1].plot(states[:, 10] / 1000, states[:, 9] / 1000, **style)
        axes[2].plot(time, -states[:, 11], **style)
        summary.append({"plant_id": record["plant_id"], "phi_at_3s_deg": float(np.degrees(states[3000, 7])),
                        "heading_change_deg": float(np.degrees(states[-1, 8])),
                        "altitude_change_m": float(-states[-1, 11])})
    _style(axes[0], "A  bank angle phi", "deg")
    _style(axes[1], "B  ground track", "north X (km)")
    _style(axes[2], "C  altitude change", "m")
    axes[0].set_xlabel("time (s)", fontsize=9, color=INK_2)
    axes[1].set_xlabel("east Y (km)", fontsize=9, color=INK_2)
    axes[2].set_xlabel("time (s)", fontsize=9, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"maneuver.{suffix}", dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return {"figure": "figures/maneuver.png", "inputs": "lateral stick for 30 deg at 1 s, released; 0.3 g pull from 3 s",
            "aircraft": summary}


def main() -> None:
    entries = load_joint_library(LIBRARY)
    source_rows = {row["plant_id"]: row for row in map(json.loads, SOURCE_LIBRARY.read_text(encoding="utf-8").splitlines())}
    chosen = representatives(entries)
    result = {
        "roll_source": frozen_roll_source(),
        "library_sha256": sha256(LIBRARY),
        "library": library_statistics(entries),
        "fractional_delay_transfer_function_check": fractional_delay_check(entries),
        "step_responses": step_figure(chosen, source_rows),
        "maneuver": maneuver_figure(chosen, source_rows),
    }
    (HERE / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("roll_source", "fractional_delay_transfer_function_check")}, indent=1))


if __name__ == "__main__":
    main()
