"""Evaluate a trained TCN MoE-TD3 Teacher on held-out aircraft."""

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
    MoETD3,
    PlantCase,
    ResidualDampingEnv,
    TemporalObservation,
    load_balanced_plant_cases,
)


PLANT_BANK = (
    ROOT
    / "data"
    / "aircraft"
    / "generated"
    / "p_channel_library_iv_a_manual_v1"
    / "plants.jsonl"
)


def _controlled_rollout(
    controller: MoETD3, case: PlantCase
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    history = TemporalObservation()
    first_step = True
    router_weights: list[np.ndarray] = []
    sparse_router_weights: list[np.ndarray] = []

    def policy(observation: np.ndarray) -> np.ndarray:
        nonlocal first_step
        if first_step:
            temporal = history.reset(observation)
            first_step = False
        else:
            temporal = history.append(observation)
        tensor = torch.as_tensor(temporal[None, :], device=controller.device)
        with torch.no_grad():
            action, sparse_weights, dense_weights, _ = controller.actor(
                tensor, noisy_routing=False
            )
        router_weights.append(dense_weights[0].cpu().numpy())
        sparse_router_weights.append(sparse_weights[0].cpu().numpy())
        return action[0].cpu().numpy()

    trace = ResidualDampingEnv(case.plant).rollout(policy)
    return trace, np.asarray(router_weights), np.asarray(sparse_router_weights)


def _case_metrics(
    case: PlantCase,
    raw: dict[str, np.ndarray],
    controlled: dict[str, np.ndarray],
    router_weights: np.ndarray,
    sparse_router_weights: np.ndarray,
) -> dict[str, object]:
    def integral(trace: dict[str, np.ndarray], key: str) -> float:
        values = trace[key]
        intervals = np.diff(trace["time_s"])
        return float(np.sum(0.5 * (values[:-1] + values[1:]) * intervals))

    raw_energy = integral(raw, "oscillation_energy")
    controlled_energy = integral(controlled, "oscillation_energy")
    raw_peak = float(np.max(np.abs(np.rad2deg(raw["p_rad_s"]))))
    controlled_peak = float(np.max(np.abs(np.rad2deg(controlled["p_rad_s"]))))
    raw_high_frequency_energy = integral(raw, "high_frequency_energy")
    controlled_high_frequency_energy = integral(controlled, "high_frequency_energy")
    top_experts = np.argmax(sparse_router_weights, axis=1)
    expert_count = router_weights.shape[1]
    top1_fraction = np.bincount(top_experts, minlength=expert_count) / len(top_experts)
    selected = sparse_router_weights > 0
    selected_share = selected.sum(axis=0) / max(float(selected.sum()), 1.0)
    activation_fraction = selected.mean(axis=0)
    top1_switch_fraction = float(np.mean(top_experts[1:] != top_experts[:-1]))
    top2_switch_fraction = float(np.mean(np.any(selected[1:] != selected[:-1], axis=1)))
    top1_run_count = int(np.count_nonzero(top_experts[1:] != top_experts[:-1])) + 1
    top1_mean_dwell_s = (
        len(top_experts)
        * ResidualDampingEnv(case.plant).config.control_dt_s
        / top1_run_count
    )
    selected_entropy = -float(
        np.sum(selected_share * np.log(np.clip(selected_share, 1e-12, None)))
    )
    sequence_expert_utilization = float(np.mean(np.any(selected, axis=0)))
    return {
        "plant_id": case.plant_id,
        "gjb_level": case.gjb_level,
        "dutch_roll_level": case.dutch_roll_level,
        "raw_energy": raw_energy,
        "controlled_energy": controlled_energy,
        "energy_ratio": controlled_energy / max(raw_energy, 1e-12),
        "raw_high_frequency_energy": raw_high_frequency_energy,
        "controlled_high_frequency_energy": controlled_high_frequency_energy,
        "high_frequency_energy_ratio": controlled_high_frequency_energy
        / max(raw_high_frequency_energy, 1e-12),
        "raw_peak_deg_s": raw_peak,
        "controlled_peak_deg_s": controlled_peak,
        "peak_ratio": controlled_peak / max(raw_peak, 1e-12),
        "raw_return": float(np.sum(raw["reward"])),
        "controlled_return": float(np.sum(controlled["reward"])),
        "residual_force_rms_n": float(
            np.sqrt(np.mean(np.square(controlled["residual_force_n"])))
        ),
        "residual_force_max_abs_n": float(
            np.max(np.abs(controlled["residual_force_n"]))
        ),
        "router_mean_weights": np.mean(router_weights, axis=0).tolist(),
        "router_top_expert_fraction": top1_fraction.tolist(),
        "router_selected_expert_share": selected_share.tolist(),
        "router_expert_activation_fraction": activation_fraction.tolist(),
        "router_top1_switch_fraction": top1_switch_fraction,
        "router_top2_switch_fraction": top2_switch_fraction,
        "router_top1_mean_dwell_s": float(top1_mean_dwell_s),
        "router_effective_expert_count": float(np.exp(selected_entropy)),
        "router_sequence_expert_utilization": sequence_expert_utilization,
    }


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    energy_ratios = np.asarray([row["energy_ratio"] for row in rows], dtype=float)
    peak_ratios = np.asarray([row["peak_ratio"] for row in rows], dtype=float)
    force_rms = np.asarray([row["residual_force_rms_n"] for row in rows], dtype=float)
    mean_router_weights = np.mean(
        np.asarray([row["router_mean_weights"] for row in rows], dtype=float), axis=0
    )
    top1_fraction = np.mean(
        np.asarray([row["router_top_expert_fraction"] for row in rows], dtype=float),
        axis=0,
    )
    selected_share = np.mean(
        np.asarray([row["router_selected_expert_share"] for row in rows], dtype=float),
        axis=0,
    )
    uniform_share = 1.0 / len(selected_share)
    return {
        "aircraft": float(len(rows)),
        "mean_energy_reduction_percent": float(100.0 * (1.0 - energy_ratios.mean())),
        "median_energy_reduction_percent": float(
            100.0 * (1.0 - np.median(energy_ratios))
        ),
        "energy_improved_fraction": float(np.mean(energy_ratios < 1.0)),
        "mean_peak_reduction_percent": float(100.0 * (1.0 - peak_ratios.mean())),
        "peak_improved_fraction": float(np.mean(peak_ratios < 1.0)),
        "mean_residual_force_rms_n": float(force_rms.mean()),
        "router_mean_weights": mean_router_weights.tolist(),
        "router_top1_fraction": top1_fraction.tolist(),
        "router_selected_expert_share": selected_share.tolist(),
        "router_max_load_violation": float(
            (selected_share.max() - uniform_share) / uniform_share
        ),
        "mean_router_top1_switch_fraction": float(
            np.mean([row["router_top1_switch_fraction"] for row in rows])
        ),
        "mean_router_top2_switch_fraction": float(
            np.mean([row["router_top2_switch_fraction"] for row in rows])
        ),
        "mean_router_top1_dwell_s": float(
            np.mean([row["router_top1_mean_dwell_s"] for row in rows])
        ),
        "mean_router_effective_expert_count": float(
            np.mean([row["router_effective_expert_count"] for row in rows])
        ),
        "mean_router_sequence_expert_utilization": float(
            np.mean([row["router_sequence_expert_utilization"] for row in rows])
        ),
    }


def _plot_representatives(
    representatives: dict[
        int, tuple[PlantCase, dict[str, np.ndarray], dict[str, np.ndarray]]
    ],
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for row_index, level in enumerate((1, 2, 3)):
        case, raw, controlled = representatives[level]
        pre_step_time = -ResidualDampingEnv(case.plant).config.control_dt_s

        def with_pre_step(values: np.ndarray) -> np.ndarray:
            return np.concatenate(([0.0], values))

        raw_time = np.concatenate(([pre_step_time], raw["time_s"]))
        controlled_time = np.concatenate(([pre_step_time], controlled["time_s"]))
        axes[row_index, 0].plot(
            raw_time,
            np.rad2deg(with_pre_step(raw["p_rad_s"])),
            label="Raw: 3 N step",
            lw=2,
        )
        axes[row_index, 0].plot(
            controlled_time,
            np.rad2deg(with_pre_step(controlled["p_rad_s"])),
            label="MoE closed loop",
            lw=2,
        )
        axes[row_index, 0].set_ylabel(f"L{level} p (deg/s)")
        axes[row_index, 0].set_title(case.plant_id)
        axes[row_index, 0].axvline(0.0, color="black", ls=":", lw=0.9)
        axes[row_index, 0].grid(alpha=0.25)
        axes[row_index, 1].plot(
            raw_time,
            with_pre_step(raw["total_force_n"]),
            label="Raw input: 3 N step",
            lw=2,
        )
        axes[row_index, 1].plot(
            controlled_time,
            with_pre_step(controlled["total_force_n"]),
            label="Controlled input",
            lw=2,
        )
        axes[row_index, 1].axhline(0.0, color="black", lw=0.8)
        axes[row_index, 1].axvline(0.0, color="black", ls=":", lw=0.9)
        axes[row_index, 1].set_ylabel("F_as (N)")
        axes[row_index, 1].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle("Time-domain response: F_as(t) -> p(t)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _plot_routing(report: dict[str, object], output: Path) -> Path:
    """Plot the dispatch statistics that reveal expert concentration."""

    routing_output = output.with_name(f"{output.stem}_routing{output.suffix}")
    aggregates = [report["by_dutch_roll_level"][str(level)] for level in (1, 2, 3)]
    expert_indices = np.arange(4)
    level_offsets = (-0.24, 0.0, 0.24)
    colors = ("tab:green", "tab:orange", "tab:red")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for level, aggregate, offset, color in zip(
        (1, 2, 3), aggregates, level_offsets, colors
    ):
        axes[0, 0].bar(
            expert_indices + offset,
            aggregate["router_top1_fraction"],
            width=0.22,
            label=f"L{level}",
            color=color,
        )
        axes[0, 1].bar(
            expert_indices + offset,
            aggregate["router_selected_expert_share"],
            width=0.22,
            label=f"L{level}",
            color=color,
        )
        axes[1, 0].bar(
            expert_indices + offset,
            aggregate["router_mean_weights"],
            width=0.22,
            label=f"L{level}",
            color=color,
        )

    for axis, title, ylabel in (
        (axes[0, 0], "Top-1 expert usage", "Fraction of control steps"),
        (axes[0, 1], "Top-2 dispatch share", "Share of selected slots"),
        (axes[1, 0], "Mean clean router probability", "Probability"),
    ):
        axis.axhline(0.25, color="black", ls="--", lw=1, label="Uniform")
        axis.set_xticks(expert_indices, [f"E{index}" for index in expert_indices])
        axis.set_ylim(0.0, 1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="upper right")

    x = np.arange(3)
    top1_switch = [
        aggregate["mean_router_top1_switch_fraction"] for aggregate in aggregates
    ]
    top2_switch = [
        aggregate["mean_router_top2_switch_fraction"] for aggregate in aggregates
    ]
    axes[1, 1].bar(x - 0.18, top1_switch, 0.36, label="Top-1 switch")
    axes[1, 1].bar(x + 0.18, top2_switch, 0.36, label="Top-2 set switch")
    axes[1, 1].set_xticks(x, ["L1", "L2", "L3"])
    axes[1, 1].set_ylim(0.0, 1.0)
    axes[1, 1].set_title("Route switching between adjacent steps")
    axes[1, 1].set_ylabel("Switch fraction")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend(loc="upper right")

    figure.suptitle("MoE router utilization on held-out aircraft")
    figure.tight_layout()
    figure.savefig(routing_output, dpi=180)
    plt.close(figure)
    return routing_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--per-level", type=int, default=10)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller = MoETD3(device=args.device)
    controller.load(args.checkpoint)
    controller.actor.eval()
    cases = load_balanced_plant_cases(
        PLANT_BANK,
        per_level=args.per_level,
        seed=args.seed,
        split_prefix=args.split,
    )
    rows: list[dict[str, object]] = []
    representatives: dict[
        int, tuple[PlantCase, dict[str, np.ndarray], dict[str, np.ndarray]]
    ] = {}
    for case in cases:
        raw = ResidualDampingEnv(case.plant).rollout()
        controlled, router_weights, sparse_router_weights = _controlled_rollout(
            controller, case
        )
        rows.append(
            _case_metrics(
                case,
                raw,
                controlled,
                router_weights,
                sparse_router_weights,
            )
        )
        representatives.setdefault(case.dutch_roll_level, (case, raw, controlled))

    report = {
        "split": args.split,
        "per_level": args.per_level,
        "routing_bias": controller.actor.routing_bias.cpu().tolist(),
        "overall": _aggregate(rows),
        "by_dutch_roll_level": {
            str(level): _aggregate(
                [row for row in rows if row["dutch_roll_level"] == level]
            )
            for level in (1, 2, 3)
        },
        "cases": rows,
    }
    json_path = args.output.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot_representatives(representatives, args.output)
    routing_output = _plot_routing(report, args.output)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, indent=2
        )
    )
    print(f"plot={args.output}")
    print(f"routing_plot={routing_output}")


if __name__ == "__main__":
    main()
