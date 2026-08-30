"""Compare one RL Teacher with the tuned PID and raw response on identical commands."""

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
    tracking_metrics,
)
from src.utils.plotting import save_controller_comparison_grid  # noqa: E402
from src.utils.provenance import git_source_revision, sha256_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--pid-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _pid_policy(gains: PIDGains, environment: object) -> RollRatePIDPolicy:
    return RollRatePIDPolicy(
        gains,
        policy_dt_s=environment.policy_dt_s,
        command_scale_rad_s=environment.command_scale_rad_s,
        integral_error_scale_rad=environment.integral_error_scale_rad,
        roll_acceleration_scale_rad_s2=environment.roll_acceleration_scale_rad_s2,
        force_limit_n=environment.force_limit_n,
    )


def _summary(rows: list[dict[str, object]], label: str) -> dict[str, float]:
    metrics = [row[label] for row in rows]
    return {
        "mean_episode_cost": float(np.mean([row["episode_cost"] for row in metrics])),
        "mean_tracking_error_cost": float(
            np.mean([row["tracking_error_cost"] for row in metrics])
        ),
        "mean_force_energy_cost": float(
            np.mean([row["force_energy_cost"] for row in metrics])
        ),
        "mean_requested_force_delta_cost": float(
            np.mean([row["requested_force_delta_cost"] for row in metrics])
        ),
        "mean_tracking_rmse_deg_s": float(
            np.mean([row["tracking_rmse_deg_s"] for row in metrics])
        ),
        "maximum_peak_error_deg_s": float(
            np.max([row["tracking_peak_error_deg_s"] for row in metrics])
        ),
        "mean_requested_force_total_variation_n": float(
            np.mean([row["requested_force_total_variation_n"] for row in metrics])
        ),
        "maximum_requested_force_total_variation_n": float(
            np.max([row["requested_force_total_variation_n"] for row in metrics])
        ),
        "mean_force_rate_limit_active_fraction": float(
            np.mean([row["force_rate_limit_active_fraction"] for row in metrics])
        ),
        "mean_abs_force_rate_limit_gap_n": float(
            np.mean([row["mean_abs_force_rate_limit_gap_n"] for row in metrics])
        ),
        "maximum_abs_force_rate_limit_gap_n": float(
            np.max([row["maximum_abs_force_rate_limit_gap_n"] for row in metrics])
        ),
        "mean_force_saturation_fraction": float(
            np.mean([row["force_saturation_fraction"] for row in metrics])
        ),
    }


def _save_controlled_comparison(
    traces: list[tuple[str, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]],
    path: Path,
    *,
    title: str,
) -> Path:
    colors = {"PID": "#2878b5", "RL Teacher": "#c82423"}
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
        for label, trace in controllers.items():
            controlled_time = np.asarray(trace["time_s"], dtype=float)
            response_deg_s = np.rad2deg(trace["p_rad_s"])
            response_axis.plot(
                controlled_time,
                response_deg_s,
                color=colors[label],
                linewidth=1.5,
                label=label,
            )
            error_axis.plot(
                controlled_time,
                response_deg_s - reference_deg_s,
                color=colors[label],
                linewidth=1.4,
                label=label,
            )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        response_axis.grid(alpha=0.25)
        response_axis.legend(loc="best", ncol=4)
        error_axis.axhline(0.0, color="black", linewidth=0.8)
        error_axis.set_ylabel("p - p_ref (deg/s)")
        error_axis.grid(alpha=0.25)
        error_axis.legend(loc="best")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    figure.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = _parse_args()
    teacher, record, config, teacher_payload = load_specialist_actor(
        args.teacher_checkpoint,
        device=args.device,
    )
    pid_payload = json.loads(args.pid_report.read_text(encoding="utf-8"))
    if str(pid_payload.get("plant_id")) != record.plant_id:
        raise ValueError("PID report and Teacher checkpoint refer to different plants")
    gains = PIDGains(**pid_payload["gains"])
    traces: list[
        tuple[str, dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]
    ] = []
    rows: list[dict[str, object]] = []
    for index, profile in enumerate(
        specialist_evaluation_commands(config.episode_duration_s)
    ):
        seed = config.seed + index
        raw_environment = build_specialist_env(record, config, (profile,))
        pid_environment = build_specialist_env(record, config, (profile,))
        teacher_environment = build_specialist_env(record, config, (profile,))
        raw_trace = rollout_policy(CommandForceBaseline(), raw_environment, seed=seed)
        pid_trace = rollout_policy(
            _pid_policy(gains, pid_environment), pid_environment, seed=seed
        )
        teacher_trace = rollout_policy(teacher, teacher_environment, seed=seed)
        traces.append(
            (
                profile.command_id,
                raw_trace,
                {"PID": pid_trace, "RL Teacher": teacher_trace},
            )
        )
        rows.append(
            {
                "command_id": profile.command_id,
                "command_kind": profile.kind,
                "raw": tracking_metrics(raw_trace, config.force_limit_n),
                "PID": tracking_metrics(pid_trace, config.force_limit_n),
                "RL Teacher": tracking_metrics(teacher_trace, config.force_limit_n),
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    plot_path = save_controller_comparison_grid(
        traces,
        args.output / "controller_comparison.png",
        title=f"{record.plant_id}: Raw vs PID vs RL Teacher",
    )
    controlled_plot_path = _save_controlled_comparison(
        traces,
        args.output / "controlled_comparison.png",
        title=f"{record.plant_id}: PID vs RL Teacher (controlled-response zoom)",
    )
    report = {
        "schema_version": "teacher_pid_comparison_v1",
        "status": "complete",
        "source": git_source_revision(),
        "plant": {
            "plant_id": record.plant_id,
            "parameters": asdict(record.parameters),
        },
        "teacher_checkpoint": {
            "path": str(args.teacher_checkpoint.resolve()),
            "sha256": sha256_file(args.teacher_checkpoint),
            "source": teacher_payload.get("source"),
        },
        "pid_report": {
            "path": str(args.pid_report.resolve()),
            "sha256": sha256_file(args.pid_report),
            "gains": asdict(gains),
        },
        "commands": len(rows),
        "summary": {
            "raw": _summary(rows, "raw"),
            "PID": _summary(rows, "PID"),
            "RL Teacher": _summary(rows, "RL Teacher"),
        },
        "rows": rows,
        "artifacts": {
            "controller_comparison": str(plot_path),
            "controlled_comparison": str(controlled_plot_path),
        },
    }
    report_path = args.output / "comparison.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    print(f"plot={plot_path}")
    print(f"controlled_plot={controlled_plot_path}")


if __name__ == "__main__":
    main()
