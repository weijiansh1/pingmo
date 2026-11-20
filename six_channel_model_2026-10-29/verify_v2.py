"""Coupled v2: library statistics, cross-channel and large-maneuver figures.

CPU only; no training.  Writes verification_v2.json and figures/*_v2.*.
"""
from collections import Counter
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pinmo6.coupled import CoupledAircraft, InertiaMoments, simulate_coupled  # noqa: E402
from pinmo6.derivatives import LateralDerivatives, LongitudinalDerivatives  # noqa: E402
from pinmo6.lateral import directional_polynomials  # noqa: E402
from pinmo6.sampler import load_joint_library  # noqa: E402
from pinmo6.turbulence import DrydenGust, high_altitude_scales  # noqa: E402
from verify import DASHES, FIGURES, INK, INK_2, SERIES, SURFACE, _style  # noqa: E402

V1 = HERE / "library/joint_six_channel_5000_v1/plants.jsonl"
V2 = HERE / "library/coupled_six_channel_5000_v2/plants.jsonl"
DT = 0.001


def load() -> list[tuple[dict, dict, object]]:
    v2 = {json.loads(line)["plant_id"]: json.loads(line) for line in V2.read_text(encoding="utf-8").splitlines()}
    return [(record, v2[record["plant_id"]], base) for record, base in load_joint_library(V1)]


def aircraft_of(extra: dict, base) -> CoupledAircraft:
    lon = {k: v for k, v in extra["longitudinal"].items()
           if k in LongitudinalDerivatives.__dataclass_fields__}
    lon["long_period"] = tuple(lon["long_period"])
    return CoupledAircraft(base.longitudinal, base.roll, base.directional.tau_rp,
                           LateralDerivatives(**extra["lateral"]), LongitudinalDerivatives(**lon),
                           InertiaMoments(**extra["inertia"]), dt=DT)


def statistics(entries) -> dict:
    stats: dict[str, Counter] = {name: Counter() for name in (
        "selection", "dutch_roll", "roll_oscillation", "sideslip_M", "sideslip_L", "cross_worst",
        "joint_v2", "long_period", "dihedral_sense", "pedal_sense", "violated")}
    for record, extra, _ in entries:
        bucket = record["sampling_bucket"]
        cross = extra["cross_axis"]
        stats["selection"][f"{bucket}/{extra['lateral_selection']}"] += 1
        for key, value in (("dutch_roll", cross["dutch_roll_corrected"]["level"]),
                           ("roll_oscillation", cross["roll_oscillation"]["level"]),
                           ("sideslip_M", cross["sideslip_excursion"].get("level")),
                           ("sideslip_L", cross["sideslip_excursion_low_speed"].get("level")),
                           ("cross_worst", extra["cross_axis_worst_level"]),
                           ("long_period", extra["long_period"]["level"])):
            stats[key][f"{bucket}/L{value}"] += 1
        levels = [record["roll"]["gjb_level"], record["longitudinal"]["gjb_level"],
                  extra["cross_axis_worst_level"], extra["long_period"]["level"]]
        joint = None if bucket == "OOD" or any(level is None for level in levels) else max(levels)
        stats["joint_v2"][f"{bucket}/L{joint}"] += 1
        stats["dihedral_sense"][f"{bucket}/{cross['steady_sideslip']['dihedral_sense_ok']}"] += 1
        stats["pedal_sense"][f"{bucket}/{cross['steady_sideslip']['pedal_sense_ok']}"] += 1
        for name in extra["lateral_violated"]:
            stats["violated"][name] += 1
    return {name: dict(sorted(counter.items())) for name, counter in stats.items()}


def representatives(entries) -> list:
    chosen = []
    for level in (1, 2, 3):
        chosen.append(next(entry for entry in entries
                           if entry[0]["split"] == "train_core" and entry[0]["sampling_bucket"] == f"L{level}"
                           and entry[1]["lateral_selection"] != "closest_to_envelope"))
    return chosen


def cross_channel_figure(chosen) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = 6000
    time = np.arange(steps) * DT
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.2), facecolor=SURFACE, sharex=True)
    titles = (("A  p, lateral stick", "deg/s"), ("B  beta, lateral stick", "deg"), ("C  r, lateral stick", "deg/s"),
              ("D  beta, pedal", "deg"), ("E  r, pedal", "deg/s"), ("F  p, pedal", "deg/s"))
    summary = []
    for index, (record, extra, base) in enumerate(chosen):
        aircraft = aircraft_of(extra, base)
        sensitivity = 1.0 / base.roll.l_fa                       # scale so all aircraft get comparable inputs
        f_as = 0.02 * sensitivity
        f_rp = 1.0 / base.directional.sideslip_per_force_deg_per_n   # 1 deg of second-order steady sideslip
        stick = simulate_coupled(aircraft, np.tile([0.0, f_as, 0.0], (steps, 1)))["states"]
        pedal = simulate_coupled(aircraft, np.tile([0.0, 0.0, f_rp], (steps, 1)))["states"]
        curves = (stick[:, 5], stick[:, 4], stick[:, 6], pedal[:, 4], pedal[:, 6], pedal[:, 5])
        label = f"L{record['roll']['gjb_level']} · {record['plant_id']}"
        style = dict(color=SERIES[index], linestyle=DASHES[index], linewidth=1.5)
        for axis, curve in zip(axes.flat, curves):
            axis.plot(time, np.degrees(curve), label=label, **style)
        # the mind-map decoupled pedal forms, for comparison
        polynomials = directional_polynomials(base.roll, base.directional)
        delayed = time - base.directional.tau_rp
        for axis, key in ((axes[1, 0], "beta"), (axes[1, 1], "r")):
            _, response = signal.step(polynomials[key], T=time)
            reference = np.interp(delayed, time, response, left=0.0) * f_rp
            axis.plot(time, np.degrees(reference), color=SERIES[index], linewidth=0.8, alpha=0.55)
        summary.append({"plant_id": record["plant_id"], "eta": extra["lateral"]["n_a"] / extra["lateral"]["l_a"],
                        "phi_over_beta": extra["cross_axis"]["dutch_roll_shape"]["phi_over_beta"],
                        "psi_beta_deg": extra["cross_axis"]["psi_beta_deg"],
                        "sideslip_excursion_M": extra["cross_axis"]["sideslip_excursion"],
                        "lateral_selection": extra["lateral_selection"]})
    for axis, (title, unit) in zip(axes.flat, titles):
        _style(axis, title, unit)
    for axis in axes[1]:
        axis.set_xlabel("time (s)", fontsize=9, color=INK_2)
    axes[1, 0].annotate("thin: mind-map decoupled form", (0.02, 0.04), xycoords="axes fraction", fontsize=8, color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"cross_channels_v2.{suffix}", dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return {"figure": "figures/cross_channels_v2.png", "aircraft": summary}


def maneuver_figure(record, extra, base) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aircraft = aircraft_of(extra, base)
    lon = base.longitudinal
    out: dict[str, object] = {"plant_id": record["plant_id"]}
    figure, axes = plt.subplots(2, 2, figsize=(11, 6.8), facecolor=SURFACE)

    # A: roll to about 60 deg, then hold with or without a 1 g pull (level turn needs n = 2)
    steps = 15000
    time = np.arange(steps) * DT
    pulse = np.zeros((steps, 3))
    unit = simulate_coupled(aircraft, np.tile([0.0, 1.0, 0.0], (1500, 1)))["states"]
    bank_per_n = math.degrees(unit[-1, 7])
    pulse[:1500, 1] = 60.0 / bank_per_n
    pull = pulse.copy()
    pull[1500:, 0] = 1.0 * lon.fs_per_g
    for index, (name, inputs) in enumerate((("bank, no pull", pulse), ("bank, 1 g pull", pull))):
        states = simulate_coupled(aircraft, inputs)["states"]
        axes[0, 0].plot(time, -states[:, 11], color=SERIES[index], linestyle=DASHES[index], linewidth=1.5, label=name)
        out[f"altitude_change_m_{name}"] = float(-states[-1, 11])
        out[f"heading_change_deg_{name}"] = float(math.degrees(states[-1, 8]))
    _style(axes[0, 0], "A  altitude after a 60 deg bank", "m")
    axes[0, 0].legend(frameon=False, fontsize=8)

    # B: 360 deg roll, inertial coupling on and off
    steps = 6000
    time = np.arange(steps) * DT
    stick = np.zeros((steps, 3))
    stick[:, 1] = 5.0 * 60.0 / bank_per_n                 # about 200 deg/s, a full-performance roll
    for index, (name, inertia) in enumerate((("with inertial coupling", aircraft.inertia),
                                             ("without", InertiaMoments(1.0, 1.0, 0.0)))):
        rolling = CoupledAircraft(base.longitudinal, base.roll, base.directional.tau_rp,
                                  aircraft.lateral, aircraft.longitudinal, inertia, dt=DT)
        states = simulate_coupled(rolling, stick)["states"]
        stop = int(np.argmax(states[:, 7] >= 2 * math.pi)) or steps
        axes[0, 1].plot(time[:stop], np.degrees(states[:stop, 1] - lon.alpha_trim), color=SERIES[index],
                        linestyle=DASHES[index], linewidth=1.5, label=f"alpha, {name}")
        out[f"roll_360_{name}"] = {"time_s": float(time[stop - 1]),
                                   "max_abs_d_alpha_deg": float(np.max(np.abs(np.degrees(states[:stop, 1] - lon.alpha_trim)))),
                                   "max_abs_beta_deg": float(np.max(np.abs(np.degrees(states[:stop, 4]))))}
    _style(axes[0, 1], "B  angle of attack during a 360 deg roll", "deg")
    axes[0, 1].legend(frameon=False, fontsize=8)

    # C: throttle step of 5 % weight, long period
    steps = 120000
    time = np.arange(steps) * DT
    throttle = np.zeros((steps, 4))
    throttle[:, 3] = 0.05
    states = simulate_coupled(aircraft, throttle)["states"]
    axes[1, 0].plot(time, states[:, 0] - lon.v0, color=SERIES[0], linewidth=1.5, label="V - V0 (m/s)")
    axes[1, 0].plot(time, -states[:, 11] / 10.0, color=SERIES[1], linestyle="--", linewidth=1.5, label="altitude / 10 (m)")
    _style(axes[1, 0], "C  thrust +5 % of weight", "")
    axes[1, 0].legend(frameon=False, fontsize=8)
    out["throttle_step"] = {"phugoid": extra["longitudinal"]["phugoid"], "long_period_level": extra["long_period"]["level"],
                            "speed_change_at_120s": float(states[-1, 0] - lon.v0), "altitude_change_at_120s": float(-states[-1, 11])}

    # D: moderate turbulence at 10,000 ft
    steps = 60000
    time = np.arange(steps) * DT
    scales = high_altitude_scales("moderate", extra["span_m"])
    generator = DrydenGust(scales, lon.v0, DT, seed=11)
    gusts = np.array([generator.step() for _ in range(steps)])
    run = simulate_coupled(aircraft, np.zeros((steps, 3)), gusts=gusts)
    axes[1, 1].plot(time, run["nz_g"], color=SERIES[0], linewidth=1.0, label="n_z (g)")
    axes[1, 1].plot(time, np.degrees(run["states"][:, 4]), color=SERIES[1], linewidth=1.0, linestyle="--", label="beta (deg)")
    _style(axes[1, 1], "D  moderate Dryden turbulence, 10,000 ft", "")
    axes[1, 1].legend(frameon=False, fontsize=8)
    out["turbulence"] = {"sigma_m_s": scales.sigma_w, "rms_nz_g": float(np.std(run["nz_g"])),
                         "rms_beta_deg": float(np.degrees(np.std(run["states"][:, 4])))}
    for axis in axes[1]:
        axis.set_xlabel("time (s)", fontsize=9, color=INK_2)
    axes[0, 0].set_xlabel("time (s)", fontsize=9, color=INK_2)
    axes[0, 1].set_xlabel("time (s)", fontsize=9, color=INK_2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"large_maneuvers_v2.{suffix}", dpi=160, facecolor=SURFACE)
    plt.close(figure)
    out["figure"] = "figures/large_maneuvers_v2.png"
    return out


def main() -> None:
    entries = load()
    chosen = representatives(entries)
    result = {
        "library_v2": statistics(entries),
        "cross_channels": cross_channel_figure(chosen),
        "large_maneuvers": maneuver_figure(*chosen[0]),
    }
    (HERE / "verification_v2.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
                                               encoding="utf-8")
    print(json.dumps(result["library_v2"], indent=1))


if __name__ == "__main__":
    main()
