"""Summarize multi-seed reward-only RL comparisons against tuned PID."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.provenance import git_source_revision, sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=ROOT / "results/pure_reward_sac_pilot",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _reward_only_contract_passes(contract: object) -> bool:
    if not isinstance(contract, dict):
        return False
    algorithm = contract.get("algorithm")
    common_contract = {
        "supervision": "environment_reward_only",
        "uses_pid_demonstrations": False,
        "uses_behavior_cloning": False,
        "uses_embedded_control_prior": False,
        "uses_pid_regularization": False,
        "actor_action": "normalized_direct_full_F_as",
    }
    if algorithm not in {
        "soft_actor_critic",
        "twin_delayed_deep_deterministic_policy_gradient",
    }:
        return False
    if any(contract.get(key) != value for key, value in common_contract.items()):
        return False
    return not (
        algorithm == "twin_delayed_deep_deterministic_policy_gradient"
        and contract.get("uses_entropy_regularization") is not False
    )


def _algorithm_metadata(contract: object) -> tuple[str, str]:
    if not isinstance(contract, dict):
        raise ValueError("training report is missing its reward-only contract")
    algorithm = contract.get("algorithm")
    if algorithm == "soft_actor_critic":
        return "sac", "Reward-only SAC"
    if algorithm == "twin_delayed_deep_deterministic_policy_gradient":
        return "td3", "Reward-only TD3"
    raise ValueError(f"unsupported reward-only algorithm: {algorithm}")


def _ratio(value: float, baseline: float) -> float:
    return value / baseline if baseline > 0 else float("inf")


def _write_summary_plot(
    rows: list[dict[str, object]], path: Path, controller_display_label: str
) -> None:
    plants = sorted({str(row["plant_id"]) for row in rows})
    specifications = (
        ("mean_episode_cost", "Mean episode cost"),
        ("mean_tracking_rmse_deg_s", "Tracking RMSE (deg/s)"),
        ("maximum_peak_error_deg_s", "Peak error (deg/s)"),
        ("mean_requested_force_total_variation_n", "Requested-force TV (N)"),
    )
    figure, axes = plt.subplots(1, len(specifications), figsize=(16, 4.5))
    colors = ("#d1495b", "#00798c")
    for axis, (key, title) in zip(axes, specifications, strict=True):
        for plant_index, plant_id in enumerate(plants):
            plant_rows = [row for row in rows if row["plant_id"] == plant_id]
            offsets = np.linspace(-0.12, 0.12, len(plant_rows))
            values = [float(row[f"rl_{key}"]) for row in plant_rows]
            axis.scatter(
                plant_index + offsets,
                values,
                color=colors[plant_index % len(colors)],
                s=42,
                label=controller_display_label if plant_index == 0 else None,
                zorder=3,
            )
            pid_value = float(plant_rows[0][f"pid_{key}"])
            axis.scatter(
                [plant_index],
                [pid_value],
                color="#222222",
                marker="_",
                linewidths=3,
                s=650,
                label="Tuned PID" if plant_index == 0 else None,
                zorder=4,
            )
        axis.set_title(title)
        axis.set_xticks(
            range(len(plants)), [value.replace("train_core-", "") for value in plants]
        )
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    figure.suptitle(
        f"{controller_display_label}: held-out commands, {len(rows)} independent runs"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    root = _parse_args().pilot_root.resolve()
    comparison_paths = sorted(root.glob("*/*/*/comparison_vs_pid/comparison.json"))
    if not comparison_paths:
        raise FileNotFoundError(f"no comparison reports under {root}")

    rows: list[dict[str, object]] = []
    artifacts: list[dict[str, object]] = []
    algorithm_keys: set[str] = set()
    controller_labels: set[str] = set()
    for comparison_path in comparison_paths:
        comparison = _load_json(comparison_path)
        run_dir = comparison_path.parent.parent
        training_report_path = run_dir / "report.json"
        training_report = _load_json(training_report_path)
        algorithm_key, controller_display_label = _algorithm_metadata(
            training_report.get("training_contract")
        )
        algorithm_keys.add(algorithm_key)
        controller_labels.add(controller_display_label)
        plant_id = str(comparison["plant"]["plant_id"])
        seed = int(training_report["config"]["seed"])
        pid = comparison["summary"]["PID"]
        sac = comparison["summary"]["RL Teacher"]
        row: dict[str, object] = {
            "plant_id": plant_id,
            "seed": seed,
            "algorithm": algorithm_key,
            "training_steps": int(training_report["steps"]),
            "actor_parameter_count": int(training_report["parameter_counts"]["actor"]),
            "network_width": int(training_report["config"]["network_width"]),
            "residual_blocks": int(training_report["config"]["residual_blocks"]),
            "reward_only_contract_passed": _reward_only_contract_passes(
                training_report.get("training_contract")
            ),
            "comparison": str(comparison_path),
            "time_domain_plot": str(
                comparison_path.parent / "controller_comparison.png"
            ),
        }
        for key in (
            "mean_episode_cost",
            "mean_tracking_rmse_deg_s",
            "maximum_peak_error_deg_s",
            "mean_requested_force_total_variation_n",
            "mean_force_rate_limit_active_fraction",
            "mean_abs_force_rate_limit_gap_n",
            "maximum_abs_force_rate_limit_gap_n",
            "mean_force_saturation_fraction",
        ):
            row[f"pid_{key}"] = float(pid[key])
            row[f"rl_{key}"] = float(sac[key])
        row["episode_cost_ratio_rl_over_pid"] = _ratio(
            float(sac["mean_episode_cost"]), float(pid["mean_episode_cost"])
        )
        row["rmse_ratio_rl_over_pid"] = _ratio(
            float(sac["mean_tracking_rmse_deg_s"]),
            float(pid["mean_tracking_rmse_deg_s"]),
        )
        row["reward_objective_better_than_pid"] = float(
            sac["mean_episode_cost"]
        ) < float(pid["mean_episode_cost"])
        row["pareto_beats_pid"] = (
            float(sac["mean_tracking_rmse_deg_s"])
            <= float(pid["mean_tracking_rmse_deg_s"])
            and float(sac["maximum_peak_error_deg_s"])
            <= float(pid["maximum_peak_error_deg_s"])
            and float(sac["mean_requested_force_total_variation_n"])
            <= float(pid["mean_requested_force_total_variation_n"])
        )
        rows.append(row)
        artifacts.extend(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                training_report_path,
                comparison_path,
                comparison_path.parent / "controller_comparison.png",
            )
        )

    if len(algorithm_keys) != 1 or len(controller_labels) != 1:
        raise ValueError("one pilot root must contain exactly one RL algorithm")
    algorithm_key = next(iter(algorithm_keys))
    controller_display_label = next(iter(controller_labels))

    expected_seed_count = max(
        len([row for row in rows if row["plant_id"] == plant_id])
        for plant_id in {row["plant_id"] for row in rows}
    )
    plant_summaries: list[dict[str, object]] = []
    for plant_id in sorted({str(row["plant_id"]) for row in rows}):
        plant_rows = [row for row in rows if row["plant_id"] == plant_id]
        best = min(plant_rows, key=lambda row: float(row["rl_mean_episode_cost"]))
        plant_summaries.append(
            {
                "plant_id": plant_id,
                "seed_count": len(plant_rows),
                "best_seed_by_reward_objective": best["seed"],
                "best_comparison": best["comparison"],
                "best_time_domain_plot": best["time_domain_plot"],
                "median_episode_cost_ratio_rl_over_pid": float(
                    np.median(
                        [row["episode_cost_ratio_rl_over_pid"] for row in plant_rows]
                    )
                ),
                "median_rmse_ratio_rl_over_pid": float(
                    np.median([row["rmse_ratio_rl_over_pid"] for row in plant_rows])
                ),
                "reward_objective_win_rate": float(
                    np.mean(
                        [row["reward_objective_better_than_pid"] for row in plant_rows]
                    )
                ),
                "pareto_win_rate": float(
                    np.mean([row["pareto_beats_pid"] for row in plant_rows])
                ),
            }
        )

    fieldnames = list(rows[0])
    csv_path = root / "pilot_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    plot_path = root / "pure_reward_vs_pid_summary.png"
    _write_summary_plot(rows, plot_path, controller_display_label)
    passed = (
        all(bool(row["reward_only_contract_passed"]) for row in rows)
        and all(
            summary["seed_count"] == expected_seed_count for summary in plant_summaries
        )
        and len(plant_summaries) >= 2
    )
    report = {
        "schema_version": "pure_reward_rl_pilot_v2",
        "status": "complete" if passed else "self_check_failed",
        "source": git_source_revision(),
        "scope": {
            "algorithm": algorithm_key,
            "controller_display_label": controller_display_label,
            "training": (
                f"{controller_display_label} from random initialization using "
                "environment reward only"
            ),
            "pid_used_during_training": False,
            "evaluation": "paired held-out commands against separately tuned PID",
            "aircraft_count": len(plant_summaries),
            "seed_count_per_aircraft": expected_seed_count,
        },
        "plant_summaries": plant_summaries,
        "rows": rows,
        "artifacts": artifacts,
        "summary_plot": str(plot_path),
        "metrics_csv": str(csv_path),
    }
    report_path = root / "pilot_report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "runs": len(rows),
                "plant_summaries": plant_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
