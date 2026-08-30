"""Plot long-horizon time responses for representative GJB Dutch-roll levels."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from basic.moe_td3 import (
    CausalOscillationFilterBank,
    DampingConfig,
    MoETD3,
    Plant,
    PlantCase,
    ResidualDampingEnv,
    TemporalObservation,
)


PLANT_BANK = (
    ROOT
    / "data"
    / "aircraft"
    / "generated"
    / "p_channel_library_iv_a_manual_v1"
    / "plants.jsonl"
)
REPRESENTATIVES = {
    1: ("validation-2011", 20.0),
    2: ("validation-2065", 50.0),
    3: ("validation-1967", 50.0),
}


def _case_map() -> dict[str, PlantCase]:
    requested_ids = {plant_id for plant_id, _ in REPRESENTATIVES.values()}
    cases: dict[str, PlantCase] = {}
    with PLANT_BANK.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            plant_id = str(row["plant_id"])
            if plant_id not in requested_ids:
                continue
            parameters = row["parameters"]
            levels = row["gjb_component_levels"]
            cases[plant_id] = PlantCase(
                plant_id=plant_id,
                gjb_level=int(row["gjb_level"]),
                dutch_roll_level=int(levels["dutch_roll"]),
                plant=Plant(
                    parameters["l_fa"],
                    parameters["lambda_s"],
                    parameters["t_r"],
                    parameters["zeta_d"],
                    parameters["omega_d"],
                    parameters["r_omega"],
                    parameters["r_zeta"],
                    parameters["tau_p"],
                ),
            )
    missing = requested_ids.difference(cases)
    if missing:
        raise ValueError(f"representative plants not found: {sorted(missing)}")
    return cases


def _policy(controller: MoETD3):
    history = TemporalObservation()
    first_step = True

    def policy(observation: np.ndarray) -> np.ndarray:
        nonlocal first_step
        if first_step:
            temporal = history.reset(observation)
            first_step = False
        else:
            temporal = history.append(observation)
        return controller.act(temporal)

    return policy


def _oscillatory_component(
    trace: dict[str, np.ndarray], case: PlantCase, config: DampingConfig
) -> np.ndarray:
    detector = CausalOscillationFilterBank(
        config.control_dt_s,
        (case.plant.omega_d,),
        config.filter_damping,
        config.energy_time_constant_s,
    )
    output = np.zeros_like(trace["p_rad_s"])
    for index, roll_rate in enumerate(trace["p_rad_s"]):
        filtered, _ = detector.update(float(roll_rate), config.p_scale_rad_s)
        output[index] = filtered[0]
    return output


def _prepend_step(trace: dict[str, np.ndarray], key: str, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    time = np.concatenate(([-dt_s], trace["time_s"]))
    values = np.concatenate(([0.0], trace[key]))
    return time, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller = MoETD3(device=args.device)
    controller.load(args.checkpoint)
    controller.actor.eval()
    cases = _case_map()
    figure, axes = plt.subplots(3, 3, figsize=(16, 11))
    report: dict[str, object] = {}

    for row, level in enumerate((1, 2, 3)):
        plant_id, duration_s = REPRESENTATIVES[level]
        case = cases[plant_id]
        config = DampingConfig(duration_s=duration_s)
        raw = ResidualDampingEnv(case.plant, config).rollout()
        controlled = ResidualDampingEnv(case.plant, config).rollout(_policy(controller))
        raw_osc = _oscillatory_component(raw, case, config)
        controlled_osc = _oscillatory_component(controlled, case, config)
        damped_frequency = case.plant.omega_d * np.sqrt(1.0 - case.plant.zeta_d**2)
        period_s = 2.0 * np.pi / damped_frequency

        raw_time, raw_p = _prepend_step(raw, "p_rad_s", config.control_dt_s)
        controlled_time, controlled_p = _prepend_step(
            controlled, "p_rad_s", config.control_dt_s
        )
        axes[row, 0].plot(raw_time, np.rad2deg(raw_p), label="Raw", lw=1.8)
        axes[row, 0].plot(
            controlled_time,
            np.rad2deg(controlled_p),
            label="MoE closed loop",
            lw=1.8,
        )
        axes[row, 0].set_ylabel(f"L{level} p (deg/s)")

        axes[row, 1].plot(
            raw["time_s"], np.rad2deg(raw_osc), label="Raw", lw=1.6
        )
        axes[row, 1].plot(
            controlled["time_s"],
            np.rad2deg(controlled_osc),
            label="MoE closed loop",
            lw=1.6,
        )
        oscillation_limit = max(
            0.25,
            1.08
            * float(
                np.max(
                    np.abs(
                        np.concatenate(
                            (np.rad2deg(raw_osc), np.rad2deg(controlled_osc))
                        )
                    )
                )
            ),
        )
        axes[row, 1].set_ylim(-oscillation_limit, oscillation_limit)
        axes[row, 1].set_ylabel(f"L{level} p_osc (deg/s)")

        raw_force_time, raw_force = _prepend_step(
            raw, "total_force_n", config.control_dt_s
        )
        controlled_force_time, controlled_force = _prepend_step(
            controlled, "total_force_n", config.control_dt_s
        )
        axes[row, 2].plot(raw_force_time, raw_force, label="Raw: 3 N", lw=1.8)
        axes[row, 2].plot(
            controlled_force_time,
            controlled_force,
            label="Controlled input",
            lw=1.8,
        )
        axes[row, 2].set_ylabel(f"L{level} F_as (N)")

        title = (
            f"{plant_id} | zeta_d={case.plant.zeta_d:.3f}, "
            f"omega_d={case.plant.omega_d:.3f} rad/s, T={period_s:.1f} s"
        )
        axes[row, 0].set_title(title)
        axes[row, 1].set_title(f"Band-pass at omega_d={case.plant.omega_d:.3f} rad/s")
        axes[row, 2].set_title(f"Actual input | horizon={duration_s:.0f} s")
        for axis in axes[row]:
            axis.axvline(0.0, color="black", ls=":", lw=0.8)
            axis.axhline(0.0, color="black", lw=0.7)
            axis.grid(alpha=0.25)
            axis.set_xlim(-config.control_dt_s, duration_s)
            axis.set_xlabel("Time (s)")

        report[f"level_{level}"] = {
            "plant_id": plant_id,
            "duration_s": duration_s,
            "zeta_d": case.plant.zeta_d,
            "omega_d_rad_s": case.plant.omega_d,
            "damped_period_s": float(period_s),
            "raw_completed_s": float(raw["time_s"][-1]),
            "controlled_completed_s": float(controlled["time_s"][-1]),
            "raw_peak_abs_p_deg_s": float(np.max(np.abs(np.rad2deg(raw["p_rad_s"])))),
            "controlled_peak_abs_p_deg_s": float(
                np.max(np.abs(np.rad2deg(controlled["p_rad_s"])))
            ),
        }

    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[0, 2].legend(loc="best")
    figure.suptitle("Long-horizon time-domain response: F_as(t) -> p(t)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
