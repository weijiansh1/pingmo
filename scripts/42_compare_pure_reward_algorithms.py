"""Compare reward-only SAC and TD3 pilots against the same tuned PID."""

from __future__ import annotations

import argparse
import csv
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
        "--sac-report",
        type=Path,
        default=ROOT / "results/pure_reward_sac_pilot/pilot_report.json",
    )
    parser.add_argument(
        "--td3-report",
        type=Path,
        default=ROOT / "results/pure_reward_td3_pilot/pilot_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/pure_reward_algorithm_comparison",
    )
    parser.add_argument("--device", default="cpu")
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


def _pid_policy(gains: PIDGains, environment: object) -> RollRatePIDPolicy:
    return RollRatePIDPolicy(
        gains,
        policy_dt_s=environment.policy_dt_s,
        command_scale_rad_s=environment.command_scale_rad_s,
        integral_error_scale_rad=environment.integral_error_scale_rad,
        roll_acceleration_scale_rad_s2=environment.roll_acceleration_scale_rad_s2,
        force_limit_n=environment.force_limit_n,
    )


def _rows(report: dict[str, object], algorithm: str) -> list[dict[str, object]]:
    scope = report.get("scope")
    if not isinstance(scope, dict) or scope.get("algorithm") != algorithm:
        raise ValueError(f"expected a {algorithm} pilot report")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{algorithm} pilot has no runs")
    return [dict(row) for row in rows]


def _write_metric_plot(
    sac_rows: list[dict[str, object]],
    td3_rows: list[dict[str, object]],
    path: Path,
) -> None:
    plants = sorted({str(row["plant_id"]) for row in sac_rows + td3_rows})
    figure, axes = plt.subplots(1, len(METRICS), figsize=(17, 4.8))
    colors = {"PID": "#222222", "SAC": "#2878b5", "TD3": "#c82423"}
    for axis, (metric, title) in zip(axes, METRICS, strict=True):
        tick_positions: list[float] = []
        tick_labels: list[str] = []
        for plant_index, plant_id in enumerate(plants):
            base = plant_index * 4.0
            plant_sac = [row for row in sac_rows if row["plant_id"] == plant_id]
            plant_td3 = [row for row in td3_rows if row["plant_id"] == plant_id]
            pid_value = float(plant_sac[0][f"pid_{metric}"])
            axis.scatter(base, pid_value, color=colors["PID"], marker="D", s=42)
            for offset, label, rows in (
                (1.0, "SAC", plant_sac),
                (2.0, "TD3", plant_td3),
            ):
                values = [float(row[f"rl_{metric}"]) for row in rows]
                jitter = np.linspace(-0.12, 0.12, len(values))
                axis.scatter(
                    offset + base + jitter,
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
            tick_positions.extend((base, base + 1.0, base + 2.0))
            short_id = plant_id.replace("train_core-", "")
            tick_labels.extend(
                (f"{short_id}\nPID", f"{short_id}\nSAC", f"{short_id}\nTD3")
            )
        axis.set_title(title)
        axis.set_xticks(tick_positions, tick_labels)
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Pure reward RL: three seeds per aircraft; horizontal marks show medians"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _best_row(rows: list[dict[str, object]], plant_id: str) -> dict[str, object]:
    candidates = [row for row in rows if row["plant_id"] == plant_id]
    return min(candidates, key=lambda row: float(row["rl_mean_episode_cost"]))


def _actor_path(row: dict[str, object]) -> Path:
    return Path(str(row["comparison"])).parent.parent / "teacher_actor.pt"


def _save_controlled_response_zoom(
    traces: list[tuple[str, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
    destination: Path,
    *,
    title: str,
) -> Path:
    colors = {
        "Tuned PID": "#2878b5",
        "Reward-only SAC": "#c82423",
        "Reward-only TD3": "#7a5195",
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
        reference_deg_s = np.rad2deg(raw["p_reference_rad_s"])
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
            reference_deg_s,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="p_ref",
        )
        for label, controlled in controllers.items():
            controlled_time = np.asarray(controlled["time_s"], dtype=float)
            response_deg_s = np.rad2deg(controlled["p_rad_s"])
            response_axis.plot(
                controlled_time,
                response_deg_s,
                color=colors[label],
                linewidth=1.7,
                label=label,
            )
            error_axis.plot(
                controlled_time,
                response_deg_s - reference_deg_s,
                color=colors[label],
                linewidth=1.5,
                label=label,
            )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        response_axis.grid(alpha=0.25)
        response_axis.legend(loc="best", ncol=3)
        error_axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        error_axis.set_ylabel("p - p_ref (deg/s)")
        error_axis.grid(alpha=0.25)
        error_axis.legend(loc="best", ncol=2)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def _write_time_domain_plot(
    plant_id: str,
    sac_row: dict[str, object],
    td3_row: dict[str, object],
    destination: Path,
    device: str,
) -> dict[str, object]:
    sac_path = _actor_path(sac_row)
    td3_path = _actor_path(td3_row)
    sac_policy, sac_record, sac_config, _ = load_specialist_actor(
        sac_path, device=device
    )
    td3_policy, td3_record, td3_config, _ = load_specialist_actor(
        td3_path, device=device
    )
    if sac_record != td3_record or sac_record.plant_id != plant_id:
        raise ValueError("SAC and TD3 checkpoints refer to different aircraft")
    if sac_config.episode_duration_s != td3_config.episode_duration_s:
        raise ValueError("SAC and TD3 evaluation durations differ")

    comparison = _load_json(Path(str(sac_row["comparison"])))
    gains = PIDGains(**comparison["pid_report"]["gains"])
    traces: list[
        tuple[str, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ] = []
    for index, profile in enumerate(
        specialist_evaluation_commands(sac_config.episode_duration_s)
    ):
        environments = [
            build_specialist_env(sac_record, sac_config, (profile,)) for _ in range(4)
        ]
        seed = 20260828 + index
        raw = rollout_policy(CommandForceBaseline(), environments[0], seed=seed)
        pid = rollout_policy(
            _pid_policy(gains, environments[1]), environments[1], seed=seed
        )
        sac = rollout_policy(sac_policy, environments[2], seed=seed)
        td3 = rollout_policy(td3_policy, environments[3], seed=seed)
        traces.append(
            (
                profile.command_id,
                raw,
                {
                    "Tuned PID": pid,
                    "Reward-only SAC": sac,
                    "Reward-only TD3": td3,
                },
            )
        )
    path = save_controller_comparison_grid(
        traces,
        destination / f"{plant_id}_time_domain.png",
        title=f"{plant_id}: best reward-objective seed per RL algorithm",
    )
    zoom_path = _save_controlled_response_zoom(
        traces,
        destination / f"{plant_id}_controlled_zoom.png",
        title=f"{plant_id}: controlled response and tracking-error zoom",
    )
    return {
        "plant_id": plant_id,
        "sac_seed": int(sac_row["seed"]),
        "td3_seed": int(td3_row["seed"]),
        "pid_gains": asdict(gains),
        "sac_actor": str(sac_path),
        "td3_actor": str(td3_path),
        "plot": str(path),
        "controlled_zoom_plot": str(zoom_path),
    }


def main() -> None:
    args = _parse_args()
    sac_report = _load_json(args.sac_report.resolve())
    td3_report = _load_json(args.td3_report.resolve())
    sac_rows = _rows(sac_report, "sac")
    td3_rows = _rows(td3_report, "td3")
    sac_keys = {(row["plant_id"], int(row["seed"])) for row in sac_rows}
    td3_keys = {(row["plant_id"], int(row["seed"])) for row in td3_rows}
    if sac_keys != td3_keys:
        raise ValueError("SAC and TD3 pilots must use the same aircraft and seeds")

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metric_plot = destination / "algorithm_metric_comparison.png"
    _write_metric_plot(sac_rows, td3_rows, metric_plot)
    plant_summaries: list[dict[str, object]] = []
    time_domain: list[dict[str, object]] = []
    for plant_id in sorted({str(row["plant_id"]) for row in sac_rows}):
        plant_sac = [row for row in sac_rows if row["plant_id"] == plant_id]
        plant_td3 = [row for row in td3_rows if row["plant_id"] == plant_id]
        summary: dict[str, object] = {"plant_id": plant_id}
        for metric, _ in METRICS:
            pid_value = float(plant_sac[0][f"pid_{metric}"])
            sac_median = float(np.median([row[f"rl_{metric}"] for row in plant_sac]))
            td3_median = float(np.median([row[f"rl_{metric}"] for row in plant_td3]))
            summary[f"pid_{metric}"] = pid_value
            summary[f"sac_median_{metric}"] = sac_median
            summary[f"td3_median_{metric}"] = td3_median
            summary[f"td3_over_sac_{metric}"] = td3_median / sac_median
            summary[f"td3_over_pid_{metric}"] = td3_median / pid_value
        sac_by_seed = {int(row["seed"]): row for row in plant_sac}
        td3_by_seed = {int(row["seed"]): row for row in plant_td3}
        summary["td3_reward_win_rate_vs_sac"] = float(
            np.mean(
                [
                    float(td3_by_seed[seed]["rl_mean_episode_cost"])
                    < float(sac_by_seed[seed]["rl_mean_episode_cost"])
                    for seed in sorted(sac_by_seed)
                ]
            )
        )
        plant_summaries.append(summary)
        time_domain.append(
            _write_time_domain_plot(
                plant_id,
                _best_row(sac_rows, plant_id),
                _best_row(td3_rows, plant_id),
                destination,
                args.device,
            )
        )

    csv_path = destination / "all_runs.csv"
    combined_rows = sac_rows + td3_rows
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined_rows[0]))
        writer.writeheader()
        writer.writerows(combined_rows)
    artifacts = [metric_plot, csv_path]
    for item in time_domain:
        artifacts.extend((Path(item["plot"]), Path(item["controlled_zoom_plot"])))
    report = {
        "schema_version": "pure_reward_algorithm_comparison_v1",
        "status": "complete",
        "source": git_source_revision(),
        "selection_note": (
            "Conclusions use three-seed medians; time-domain plots use each "
            "algorithm's lowest held-out reward-objective seed for visualization."
        ),
        "plant_summaries": plant_summaries,
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
                "plant_summaries": plant_summaries,
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
