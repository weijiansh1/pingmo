"""Compare the original and revised reward-only TD3 training contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import PIDGains, RollRatePIDPolicy  # noqa: E402
from src.envs.roll_rate_commands import specialist_evaluation_commands  # noqa: E402
from src.teacher.specialist.trainer import (  # noqa: E402
    CommandForceBaseline,
    build_specialist_env,
    load_specialist_actor,
    rollout_policy,
)
from src.utils.plotting import save_controller_comparison_grid  # noqa: E402
from src.utils.provenance import git_source_revision, sha256_file  # noqa: E402


METRICS = (
    ("mean_episode_cost", "Mean episode cost"),
    ("mean_tracking_rmse_deg_s", "Tracking RMSE (deg/s)"),
    ("maximum_peak_error_deg_s", "Peak error (deg/s)"),
    ("mean_requested_force_total_variation_n", "Requested-force TV (N)"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-report",
        type=Path,
        default=ROOT / "results/pure_reward_td3_pilot/pilot_report.json",
    )
    parser.add_argument(
        "--revised-report",
        type=Path,
        default=(ROOT / "results/pure_reward_td3_delay_random_pilot/pilot_report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/pure_reward_td3_revised_comparison",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("pilot report contains no runs")
    return [dict(row) for row in rows]


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _actor_path(row: dict[str, object]) -> Path:
    return Path(str(row["comparison"])).parent.parent / "teacher_actor.pt"


def _best_row(rows: list[dict[str, object]], plant_id: str) -> dict[str, object]:
    candidates = [row for row in rows if row["plant_id"] == plant_id]
    return min(candidates, key=lambda row: float(row["rl_mean_episode_cost"]))


def _pid_policy(gains: PIDGains, environment: object) -> RollRatePIDPolicy:
    return RollRatePIDPolicy(
        gains,
        policy_dt_s=environment.policy_dt_s,
        command_scale_rad_s=environment.command_scale_rad_s,
        integral_error_scale_rad=environment.integral_error_scale_rad,
        roll_acceleration_scale_rad_s2=environment.roll_acceleration_scale_rad_s2,
        force_limit_n=environment.force_limit_n,
    )


def _write_metric_plot(
    original_rows: list[dict[str, object]],
    delay_rows: list[dict[str, object]],
    path: Path,
) -> None:
    plants = sorted({str(row["plant_id"]) for row in original_rows})
    figure, axes = plt.subplots(1, len(METRICS), figsize=(18, 4.8))
    colors = {"PID": "#222222", "Original": "#2878b5", "Revised": "#c82423"}
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        ticks: list[float] = []
        labels: list[str] = []
        for plant_index, plant_id in enumerate(plants):
            base = plant_index * 4.0
            old = [row for row in original_rows if row["plant_id"] == plant_id]
            new = [row for row in delay_rows if row["plant_id"] == plant_id]
            axis.scatter(
                base,
                float(new[0][f"pid_{metric}"]),
                color=colors["PID"],
                marker="D",
                s=42,
            )
            for offset, label, rows in (
                (1.0, "Original", old),
                (2.0, "Revised", new),
            ):
                values = [float(row[f"rl_{metric}"]) for row in rows]
                jitter = np.linspace(-0.12, 0.12, len(values))
                axis.scatter(
                    base + offset + jitter,
                    values,
                    color=colors[label],
                    s=38,
                    alpha=0.9,
                )
                axis.plot(
                    [base + offset - 0.24, base + offset + 0.24],
                    [float(np.median(values))] * 2,
                    color=colors[label],
                    linewidth=2.4,
                )
            short_id = plant_id.replace("train_core-", "")
            ticks.extend((base, base + 1.0, base + 2.0))
            labels.extend((f"{short_id}\nPID", f"{short_id}\nold", f"{short_id}\nnew"))
        axis.set_title(title)
        axis.set_xticks(ticks, labels)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Reward-only TD3 ablation: three seeds; bars mark medians")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_controlled_zoom(
    traces: list[tuple[str, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
    path: Path,
    *,
    title: str,
) -> Path:
    colors = {
        "Tuned PID": "#2878b5",
        "Original TD3": "#7a5195",
        "Revised TD3": "#c82423",
    }
    figure, axes = plt.subplots(
        len(traces),
        2,
        figsize=(14, 3.0 * len(traces)),
        squeeze=False,
        layout="constrained",
    )
    for row, (command_id, raw, controllers) in enumerate(traces):
        response_axis, error_axis = axes[row]
        time_s = np.asarray(raw["time_s"], dtype=float)
        reference = np.rad2deg(raw["p_reference_rad_s"])
        response_axis.plot(
            time_s,
            np.rad2deg(raw["p_command_rad_s"]),
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            label="p_c",
        )
        response_axis.plot(
            time_s,
            reference,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="p_ref",
        )
        for label, trace in controllers.items():
            controlled_time = np.asarray(trace["time_s"], dtype=float)
            response = np.rad2deg(trace["p_rad_s"])
            response_axis.plot(
                controlled_time,
                response,
                color=colors[label],
                linewidth=1.6,
                label=label,
            )
            error_axis.plot(
                controlled_time,
                response - reference,
                color=colors[label],
                linewidth=1.4,
                label=label,
            )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        response_axis.grid(alpha=0.25)
        response_axis.legend(loc="best", ncol=3)
        error_axis.axhline(0.0, color="black", linewidth=0.8)
        error_axis.set_ylabel("p - p_ref (deg/s)")
        error_axis.grid(alpha=0.25)
        error_axis.legend(loc="best", ncol=2)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _write_time_domain_plots(
    plant_id: str,
    original_row: dict[str, object],
    delay_row: dict[str, object],
    destination: Path,
    device: str,
) -> dict[str, object]:
    original_path = _actor_path(original_row)
    delay_path = _actor_path(delay_row)
    original_policy, original_record, original_config, _ = load_specialist_actor(
        original_path, device=device
    )
    delay_policy, delay_record, delay_config, _ = load_specialist_actor(
        delay_path, device=device
    )
    if original_record != delay_record or original_record.plant_id != plant_id:
        raise ValueError("ablation checkpoints refer to different aircraft")
    comparison = _load_json(Path(str(delay_row["comparison"])))
    gains = PIDGains(**comparison["pid_report"]["gains"])
    traces: list[
        tuple[str, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ] = []
    profiles = specialist_evaluation_commands(delay_config.episode_duration_s)
    for index, profile in enumerate(profiles):
        raw_env = build_specialist_env(delay_record, delay_config, (profile,))
        pid_env = build_specialist_env(delay_record, delay_config, (profile,))
        original_env = build_specialist_env(
            original_record, original_config, (profile,)
        )
        delay_env = build_specialist_env(delay_record, delay_config, (profile,))
        seed = 20260828 + index
        raw = rollout_policy(CommandForceBaseline(), raw_env, seed=seed)
        pid = rollout_policy(_pid_policy(gains, pid_env), pid_env, seed=seed)
        original = rollout_policy(original_policy, original_env, seed=seed)
        delay = rollout_policy(delay_policy, delay_env, seed=seed)
        traces.append(
            (
                profile.command_id,
                raw,
                {
                    "Tuned PID": pid,
                    "Revised TD3": delay,
                    "Original TD3": original,
                },
            )
        )
    full_path = save_controller_comparison_grid(
        traces,
        destination / f"{plant_id}_time_domain.png",
        title=f"{plant_id}: original vs revised reward-only TD3",
    )
    zoom_path = _save_controlled_zoom(
        traces,
        destination / f"{plant_id}_controlled_zoom.png",
        title=f"{plant_id}: controlled response and tracking error",
    )
    return {
        "plant_id": plant_id,
        "original_seed": int(original_row["seed"]),
        "revised_seed": int(delay_row["seed"]),
        "pid_gains": asdict(gains),
        "full_plot": str(full_path),
        "controlled_zoom_plot": str(zoom_path),
    }


def main() -> None:
    args = _parse_args()
    original_rows = _rows(_load_json(args.original_report.resolve()))
    delay_rows = _rows(_load_json(args.revised_report.resolve()))
    original_keys = {(str(row["plant_id"]), int(row["seed"])) for row in original_rows}
    delay_keys = {(str(row["plant_id"]), int(row["seed"])) for row in delay_rows}
    if original_keys != delay_keys:
        raise ValueError("ablation pilots must use identical aircraft and seeds")

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metric_plot = destination / "delay_ablation_metrics.png"
    _write_metric_plot(original_rows, delay_rows, metric_plot)
    summaries: list[dict[str, object]] = []
    time_domain: list[dict[str, object]] = []
    for plant_id in sorted({str(row["plant_id"]) for row in original_rows}):
        old = [row for row in original_rows if row["plant_id"] == plant_id]
        new = [row for row in delay_rows if row["plant_id"] == plant_id]
        summary: dict[str, object] = {"plant_id": plant_id}
        for metric, _ in METRICS:
            pid = float(new[0][f"pid_{metric}"])
            old_median = float(np.median([row[f"rl_{metric}"] for row in old]))
            new_median = float(np.median([row[f"rl_{metric}"] for row in new]))
            summary[f"pid_{metric}"] = pid
            summary[f"original_median_{metric}"] = old_median
            summary[f"revised_median_{metric}"] = new_median
            summary[f"revised_over_original_{metric}"] = new_median / old_median
            summary[f"revised_over_pid_{metric}"] = new_median / pid
        summaries.append(summary)
        time_domain.append(
            _write_time_domain_plots(
                plant_id,
                _best_row(original_rows, plant_id),
                _best_row(delay_rows, plant_id),
                destination,
                args.device,
            )
        )

    artifacts = [metric_plot]
    for row in time_domain:
        artifacts.extend((Path(row["full_plot"]), Path(row["controlled_zoom_plot"])))
    report = {
        "schema_version": "pure_reward_td3_revised_comparison_v1",
        "status": "complete",
        "source": git_source_revision(),
        "selection_note": (
            "Metric conclusions use three-seed medians. Time-domain plots use "
            "each variant's lowest held-out episode-cost seed."
        ),
        "change_bundle": [
            "actor_observes_fixed-width action delay memory and actuator state",
            "Bellman targets bootstrap across time-limit truncation",
            "critic excludes artificial episode progress and gamma is 0.9995",
            "training commands use a continuous random parameter distribution",
        ],
        "plant_summaries": summaries,
        "time_domain": time_domain,
        "artifacts": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }
    report_path = destination / "comparison_report.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "plant_summaries": summaries,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
