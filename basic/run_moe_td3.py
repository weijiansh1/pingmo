"""Run the basic no-reference MoE-TD3 damping experiment."""

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
import torch

from basic.moe_td3 import (
    DampingConfig,
    MoETD3,
    PlantCase,
    ResidualDampingEnv,
    TemporalObservation,
    load_balanced_plant_cases,
    train_moe_td3,
)

PLANT_BANK = (
    ROOT
    / "data"
    / "aircraft"
    / "generated"
    / "p_channel_library_iv_a_manual_v1"
    / "plants.jsonl"
)
DEFAULT_CASE_ID = "train_boundary-1477"


def _device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def _evaluation_case(cases: list[PlantCase]) -> PlantCase:
    for case in cases:
        if case.plant_id == DEFAULT_CASE_ID:
            return case
    return next(case for case in cases if case.dutch_roll_level == 2)


def _temporal_policy(controller: MoETD3):
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


def _trace_metrics(trace: dict[str, np.ndarray]) -> dict[str, float]:
    def integral(values: np.ndarray) -> float:
        intervals = np.diff(trace["time_s"])
        return float(np.sum(0.5 * (values[:-1] + values[1:]) * intervals))

    return {
        "peak_abs_p_deg_s": float(np.max(np.abs(np.rad2deg(trace["p_rad_s"])))),
        "oscillation_energy_integral": integral(trace["oscillation_energy"]),
        "primary_energy_integral": integral(trace["primary_energy"]),
        "high_frequency_energy_integral": integral(trace["high_frequency_energy"]),
        "residual_force_rms_n": float(
            np.sqrt(np.mean(np.square(trace["residual_force_n"])))
        ),
        "episode_return": float(np.sum(trace["reward"])),
    }


def _plot(
    raw: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray] | None,
    output_path: Path,
    case: PlantCase,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(raw["time_s"], np.rad2deg(raw["p_rad_s"]), label="Raw: 3 N", lw=2)
    if controlled is not None:
        axes[0].plot(
            controlled["time_s"],
            np.rad2deg(controlled["p_rad_s"]),
            label="MoE-TD3 residual control",
            lw=2,
        )
    axes[0].set_ylabel("p (deg/s)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    axes[1].plot(raw["time_s"], raw["total_force_n"], label="Raw force", lw=2)
    if controlled is not None:
        axes[1].plot(
            controlled["time_s"],
            controlled["total_force_n"],
            label="Controlled total force",
            lw=2,
        )
        axes[1].plot(
            controlled["time_s"],
            controlled["residual_force_n"],
            label="Learned residual force",
            lw=1.3,
        )
    axes[1].set_ylabel("Force (N)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(
        raw["time_s"], raw["oscillation_energy"], label="Raw detected energy", lw=2
    )
    if controlled is not None:
        axes[2].plot(
            controlled["time_s"],
            controlled["oscillation_energy"],
            label="Controlled detected energy",
            lw=2,
        )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Oscillation energy")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.25)
    figure.suptitle(
        f"TCN MoE-TD3 damping | {case.plant_id} | Dutch-roll level {case.dutch_roll_level}"
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="TD3 environment steps; zero only verifies the raw signal detector",
    )
    parser.add_argument("--plants-per-level", type=int, default=64)
    parser.add_argument("--random-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--router-noise-std", type=float, default=0.0)
    parser.add_argument("--balance-coefficient", type=float, default=0.0)
    parser.add_argument("--simbal-coefficient", type=float, default=0.1)
    parser.add_argument("--router-z-coefficient", type=float, default=0.001)
    parser.add_argument("--routing-bias-update-rate", type=float, default=0.001)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "basic" / "moe_td3_response.png"
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "basic" / "moe_td3.pt"
    )
    args = parser.parse_args()
    if (
        args.steps < 0
        or args.plants_per_level <= 0
        or args.batch_size <= 0
        or args.replay_capacity < args.batch_size
        or args.router_noise_std < 0
        or args.balance_coefficient < 0
        or args.simbal_coefficient < 0
        or args.router_z_coefficient < 0
        or args.routing_bias_update_rate < 0
    ):
        parser.error("invalid step, plant, batch, or replay setting")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    all_cases = load_balanced_plant_cases(PLANT_BANK, seed=args.seed)
    case = _evaluation_case(all_cases)
    config = DampingConfig()
    environment = ResidualDampingEnv(case.plant, config)
    raw = environment.rollout()

    controlled = None
    report: dict[str, object] = {
        "case": case.plant_id,
        "gjb_level": case.gjb_level,
        "dutch_roll_level": case.dutch_roll_level,
        "raw": _trace_metrics(raw),
    }
    if args.steps > 0:
        selected_cases = load_balanced_plant_cases(
            PLANT_BANK,
            per_level=args.plants_per_level,
            seed=args.seed,
        )
        controller = MoETD3(
            device=_device(args.device),
            router_noise_std=args.router_noise_std,
            balance_coefficient=args.balance_coefficient,
            simbal_coefficient=args.simbal_coefficient,
            router_z_coefficient=args.router_z_coefficient,
            routing_bias_update_rate=args.routing_bias_update_rate,
        )
        counts = controller.parameter_counts()
        print(json.dumps(counts, indent=2))
        history = train_moe_td3(
            controller,
            selected_cases,
            total_steps=args.steps,
            config=config,
            replay_capacity=args.replay_capacity,
            random_steps=min(args.random_steps, max(0, args.steps - 1)),
            batch_size=args.batch_size,
            seed=args.seed,
        )
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        controller.save(args.checkpoint)
        controlled = environment.rollout(_temporal_policy(controller))
        report["controlled"] = _trace_metrics(controlled)
        report["train_episodes"] = len(history)
        report["parameters"] = counts
        report["router_training"] = {
            "method": "loss_free_logit_bias_top2_simbal_z_loss",
            "noise_std": args.router_noise_std,
            "balance_coefficient": args.balance_coefficient,
            "simbal_coefficient": args.simbal_coefficient,
            "z_coefficient": args.router_z_coefficient,
            "bias_update_rate": args.routing_bias_update_rate,
            "final_routing_bias": controller.actor.routing_bias.cpu().tolist(),
        }

    _plot(raw, controlled, args.output, case)
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"plot={args.output}")


if __name__ == "__main__":
    main()
