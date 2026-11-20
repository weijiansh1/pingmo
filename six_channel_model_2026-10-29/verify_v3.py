"""v3 library: statistics against the original roll library, and representative responses.

CPU only; no training.  Writes verification_v3.json and figures/*_v3.*.
"""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pinmo6.coupled import simulate_coupled  # noqa: E402
from pinmo6.criteria import dutch_roll_eigenvector  # noqa: E402
from pinmo6.derivatives import LateralDerivatives  # noqa: E402
from pinmo6.sampler import SOURCE_LIBRARY  # noqa: E402
from pinmo6.sampler_v3 import load_v3  # noqa: E402
from verify import DASHES, FIGURES, INK, INK_2, SERIES, SURFACE, _style  # noqa: E402

LIBRARY = HERE / "library/six_channel_5000_v3/plants.jsonl"


def statistics(entries) -> dict:
    binding: dict[str, Counter] = defaultdict(Counter)
    components: dict[str, Counter] = defaultdict(Counter)
    for record, _ in entries:
        bucket = record["sampling_bucket"]
        levels = record["component_levels"]
        if bucket != "OOD":
            top = max(levels.values())
            for name, value in levels.items():
                components[name][f"L{value}"] += 1
                if value == top and top > 1:
                    binding[bucket][name] += 1
    return {
        "joint_level_counts": dict(Counter(r["sampling_bucket"] for r, _ in entries)),
        "speed_ranges": dict(Counter(r["envelope"]["speed_range"] for r, _ in entries)),
        "binding_components_by_level": {k: dict(v.most_common()) for k, v in sorted(binding.items())},
        "component_levels_in_distribution": {k: dict(sorted(v.items())) for k, v in sorted(components.items())},
        "ood_reasons": dict(Counter(x for r, _ in entries for x in r["ood_reasons"]).most_common()),
        "long_period_forms": dict(Counter(r["criteria"]["long_period"]["form"] for r, _ in entries)),
    }


def distribution_figure(entries) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    original = [json.loads(line) for line in SOURCE_LIBRARY.read_text(encoding="utf-8").splitlines()]
    v3 = [r for r, _ in entries]
    quantities = (
        ("T_R (s)", lambda r: r["parameters"]["t_r"], lambda r: r["roll"]["t_r"], True),
        ("omega_d (rad/s)", lambda r: r["parameters"]["omega_d"], lambda r: r["roll"]["omega_d"], True),
        ("zeta_d", lambda r: r["parameters"]["zeta_d"], lambda r: r["roll"]["zeta_d"], False),
        ("lambda_s (1/s)", lambda r: r["parameters"]["lambda_s"], lambda r: r["roll"]["lambda_s"], False),
        ("omega_phi / omega_d", lambda r: r["parameters"]["r_omega"], lambda r: r["roll"]["r_omega"], False),
        ("zeta_phi", lambda r: r["parameters"]["zeta_phi"], lambda r: r["roll"]["zeta_phi"], False),
        ("1 s sensitivity (deg/N)", lambda r: r["sensitivity_measured_deg_per_n"],
         lambda r: r["roll"]["sensitivity_1s_deg_per_n"], True),
        ("v3 aileron yaw N'a/L'a", None, lambda r: r["eta"], False),
    )
    figure, axes = plt.subplots(2, 4, figsize=(12, 5.6), facecolor=SURFACE)
    summary = {}
    for axis, (title, old, new, log) in zip(axes.flat, quantities):
        new_values = np.array([new(r) for r in v3])
        series = [("v3 library", new_values, SERIES[0])]
        if old is not None:
            series.insert(0, ("original roll library", np.array([old(r) for r in original]), SERIES[1]))
        everything = np.concatenate([values for _, values, _ in series])
        low, high = np.percentile(everything, [0.5, 99.5])
        bins = np.geomspace(low, high, 40) if log else np.linspace(low, high, 40)
        for index, (label, values, color) in enumerate(series):
            axis.hist(values, bins=bins, histtype="step", linewidth=1.5, color=color,
                      linestyle=DASHES[index if old is not None else 0], label=label)
        if log:
            axis.set_xscale("log")
        _style(axis, title, "objects")
        summary[title] = {label: [float(np.percentile(v, 5)), float(np.median(v)), float(np.percentile(v, 95))]
                          for label, v, _ in series}
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"distributions_v3.{suffix}", dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return {"figure": "figures/distributions_v3.png", "p5_median_p95": summary}


def response_figure(entries) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chosen = [next(e for e in entries if e[0]["split"] == "train_core" and e[0]["sampling_bucket"] == f"L{level}")
              for level in (1, 2, 3)]
    steps = 6000
    time = np.arange(steps) * 0.001
    figure, axes = plt.subplots(2, 3, figsize=(11, 6.2), facecolor=SURFACE, sharex=True)
    titles = (("A  p, lateral stick", "deg/s"), ("B  beta, lateral stick", "deg"), ("C  r, lateral stick", "deg/s"),
              ("D  beta, pedal", "deg"), ("E  p, pedal", "deg/s"), ("F  n_z, pitch stick for 0.5 g", "g"))
    summary = []
    for index, (record, parts) in enumerate(chosen):
        aircraft = parts["aircraft"]()
        stick = simulate_coupled(aircraft, np.tile([0.0, 0.02 / parts["roll"].l_fa, 0.0], (steps, 1)))["states"]
        pedal = simulate_coupled(aircraft, np.tile([0.0, 0.0, 1.0 / record["sideslip_per_force_deg_per_n"]],
                                                   (steps, 1)))["states"]
        pitch = simulate_coupled(aircraft, np.tile([0.5 * parts["loes"].fs_per_g, 0.0, 0.0], (steps, 1)))
        curves = (np.degrees(stick[:, 5]), np.degrees(stick[:, 4]), np.degrees(stick[:, 6]),
                  np.degrees(pedal[:, 4]), np.degrees(pedal[:, 5]), pitch["nz_g"])
        binding = [k for k, v in record["component_levels"].items() if v == record["joint_level"] and v > 1]
        label = f"L{record['joint_level']} · {record['plant_id']}"
        for axis, curve in zip(axes.flat, curves):
            axis.plot(time, curve, color=SERIES[index], linestyle=DASHES[index], linewidth=1.5, label=label)
        summary.append({"plant_id": record["plant_id"], "joint_level": record["joint_level"],
                        "binding_components": binding, "speed_range": record["envelope"]["speed_range"],
                        "eta": record["eta"], "phi_over_beta": record["criteria"]["dutch_roll_shape"]["phi_over_beta"],
                        "delta_beta_over_k": record["criteria"]["sideslip_excursion"].get("delta_beta_over_k_deg")})
    for axis, (title, unit) in zip(axes.flat, titles):
        _style(axis, title, unit)
    for axis in axes[1]:
        axis.set_xlabel("time (s)", fontsize=9, color=INK_2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"responses_v3.{suffix}", dpi=160, facecolor=SURFACE)
    plt.close(figure)
    return {"figure": "figures/responses_v3.png", "aircraft": summary}


def main() -> None:
    entries = load_v3(LIBRARY)
    result = {"library": statistics(entries), "distributions": distribution_figure(entries),
              "responses": response_figure(entries)}
    (HERE / "verification_v3.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n",
                                               encoding="utf-8")
    print(json.dumps(result["library"], ensure_ascii=False, indent=1))
    print(json.dumps(result["responses"]["aircraft"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
