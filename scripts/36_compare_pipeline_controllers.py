"""Create the final Raw/PID/RL-Teacher/Student closed-loop evidence package."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import PIDGains, RollRatePIDPolicy
from src.envs.roll_rate_commands import (
    SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION,
    specialist_evaluation_commands,
    specialist_independent_test_commands,
)
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import (
    CommandForceBaseline,
    build_specialist_env,
    load_specialist_actor,
    rollout_policy,
    tracking_metrics,
)
from src.utils.plotting import (
    save_controlled_response_error_grid,
    save_controller_comparison_grid,
)
from src.utils.provenance import git_source_revision, sha256_file


CONTROLLER_LABELS = ("raw", "PID", "RL Teacher", "Student")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pid-report-root",
        type=Path,
        help=(
            "Fallback directory containing <plant-id>/pid_report.json when the "
            "Teacher Bank does not embed PID-oracle references."
        ),
    )
    parser.add_argument(
        "--command-suite",
        choices=("evaluation", "independent-test-v1"),
        default="evaluation",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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


def _pid_report_path(
    entry: dict[str, object], plant_id: str, fallback_root: Path | None
) -> tuple[Path, str | None]:
    reference = entry.get("pid_oracle")
    if isinstance(reference, dict):
        return Path(str(reference["path"])), str(reference["sha256"])
    if fallback_root is None:
        raise ValueError(
            f"Teacher has no PID oracle reference and no fallback was given: {plant_id}"
        )
    return (fallback_root / plant_id / "pid_report.json").resolve(), None


def _command_profiles(suite: str, duration_s: float) -> tuple[object, ...]:
    if suite == "independent-test-v1":
        return specialist_independent_test_commands(duration_s)
    return specialist_evaluation_commands(duration_s)


def _dataset_split(student_payload: dict[str, object], plant_id: str) -> str:
    dataset = student_payload.get("dataset_manifest")
    if not isinstance(dataset, dict):
        return "unknown"
    if dataset.get("split_strategy") == "single_aircraft_command_holdout":
        return "single_aircraft_command_holdout"
    if dataset.get("split_strategy") == "all_aircraft_command_holdout":
        return "all_aircraft_command_holdout"
    if plant_id in dataset.get("validation_plant_ids", []):
        return "validation_aircraft"
    if plant_id in dataset.get("train_plant_ids", []):
        return "train_aircraft"
    return "unknown"


def _aggregate(rows: list[dict[str, object]], label: str) -> dict[str, float | int]:
    metrics = [row[label] for row in rows]
    return {
        "pair_count": len(metrics),
        "mean_tracking_rmse_deg_s": float(
            np.mean([metric["tracking_rmse_deg_s"] for metric in metrics])
        ),
        "maximum_peak_error_deg_s": float(
            np.max([metric["tracking_peak_error_deg_s"] for metric in metrics])
        ),
        "mean_requested_force_total_variation_n": float(
            np.mean([metric["requested_force_total_variation_n"] for metric in metrics])
        ),
        "mean_force_saturation_fraction": float(
            np.mean([metric["force_saturation_fraction"] for metric in metrics])
        ),
    }


def _distillation_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"pair_count": 0}
    teacher = np.asarray(
        [row["RL Teacher"]["tracking_rmse_deg_s"] for row in rows], dtype=float
    )
    student = np.asarray(
        [row["Student"]["tracking_rmse_deg_s"] for row in rows], dtype=float
    )
    return {
        "pair_count": len(rows),
        "mean_teacher_tracking_rmse_deg_s": float(np.mean(teacher)),
        "mean_student_tracking_rmse_deg_s": float(np.mean(student)),
        "median_student_minus_teacher_rmse_deg_s": float(np.median(student - teacher)),
        "student_within_10pct_teacher_rate": float(
            np.mean(student <= 1.1 * np.maximum(teacher, 1e-8))
        ),
        "student_better_or_equal_teacher_rate": float(np.mean(student <= teacher)),
    }


def _save_summary_plot(aircraft_rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["plant_id"]) for row in aircraft_rows]
    positions = np.arange(len(labels), dtype=float)
    width = 0.2
    colors = {
        "raw": "#64748b",
        "PID": "#16803c",
        "RL Teacher": "#2864b4",
        "Student": "#c82d2d",
    }
    figure, axes = plt.subplots(3, 1, figsize=(14, 12), layout="constrained")
    for offset, label in enumerate(CONTROLLER_LABELS):
        values = [
            row["summary"][label]["mean_tracking_rmse_deg_s"] for row in aircraft_rows
        ]
        axes[0].bar(
            positions + (offset - 1.5) * width,
            np.maximum(values, 1e-4),
            width,
            color=colors[label],
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking RMSE (deg/s, log scale)")
    axes[0].set_title("All aircraft: closed-loop tracking")
    axes[0].legend(ncol=4)

    detail_labels = CONTROLLER_LABELS[1:]
    detail_width = 0.25
    for offset, label in enumerate(detail_labels):
        values = [
            row["summary"][label]["mean_tracking_rmse_deg_s"] for row in aircraft_rows
        ]
        axes[1].bar(
            positions + (offset - 1) * detail_width,
            values,
            detail_width,
            color=colors[label],
            label=label,
        )
    axes[1].set_ylabel("Tracking RMSE (deg/s)")
    axes[1].set_title("Controller detail")
    axes[1].legend(ncol=3)

    for offset, label in enumerate(detail_labels):
        values = [
            row["summary"][label]["mean_requested_force_total_variation_n"]
            for row in aircraft_rows
        ]
        axes[2].bar(
            positions + (offset - 1) * detail_width,
            values,
            detail_width,
            color=colors[label],
            label=label,
        )
    axes[2].set_ylabel("Requested-force TV (N)")
    axes[2].set_title("Control smoothness")
    axes[2].legend(ncol=3)

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_metrics_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields = [
        "plant_id",
        "quality_region",
        "distillation_split",
        "controller",
        "mean_tracking_rmse_deg_s",
        "maximum_peak_error_deg_s",
        "mean_requested_force_total_variation_n",
        "mean_force_saturation_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for label in CONTROLLER_LABELS:
                metrics = row["summary"][label]
                writer.writerow(
                    {
                        "plant_id": row["plant_id"],
                        "quality_region": row["quality_region"],
                        "distillation_split": row["distillation_split"],
                        "controller": label,
                        "mean_tracking_rmse_deg_s": metrics["mean_tracking_rmse_deg_s"],
                        "maximum_peak_error_deg_s": metrics["maximum_peak_error_deg_s"],
                        "mean_requested_force_total_variation_n": metrics[
                            "mean_requested_force_total_variation_n"
                        ],
                        "mean_force_saturation_fraction": metrics[
                            "mean_force_saturation_fraction"
                        ],
                    }
                )


def main() -> None:
    args = _parse_args()
    bank_path = args.teacher_bank.resolve()
    student_checkpoint = args.student_checkpoint.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pid_report_root = (
        args.pid_report_root.resolve() if args.pid_report_root is not None else None
    )
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "specialist_teacher_bank_v1":
        raise ValueError("unsupported Teacher Bank schema")
    if bank.get("status") != "complete":
        raise ValueError("final comparison requires a complete Teacher Bank")
    student_model, student_payload = load_dense_student(
        student_checkpoint, device=args.device
    )

    aircraft_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    contract_checks: list[dict[str, object]] = []
    for entry in bank["teachers"]:
        teacher_checkpoint = bank_path.parent / str(entry["actor_checkpoint"])
        teacher, record, config, teacher_payload = load_specialist_actor(
            teacher_checkpoint, device=args.device
        )
        pid_path, expected_pid_hash = _pid_report_path(
            entry, record.plant_id, pid_report_root
        )
        if not pid_path.is_file():
            raise FileNotFoundError(pid_path)
        if expected_pid_hash is not None and sha256_file(pid_path) != expected_pid_hash:
            raise ValueError(f"PID oracle hash mismatch: {pid_path}")
        pid_payload = json.loads(pid_path.read_text(encoding="utf-8"))
        if pid_payload.get("plant_id") != record.plant_id:
            raise ValueError(f"PID report aircraft mismatch: {pid_path}")
        gains = PIDGains(**pid_payload["gains"])
        student = DenseStudentPolicy(
            student_model, record.parameters, device=args.device
        )
        traces = []
        command_rows: list[dict[str, object]] = []
        for command_index, profile in enumerate(
            _command_profiles(args.command_suite, config.episode_duration_s)
        ):
            seed = config.seed + command_index
            raw_environment = build_specialist_env(record, config, (profile,))
            pid_environment = build_specialist_env(record, config, (profile,))
            teacher_environment = build_specialist_env(record, config, (profile,))
            student_environment = build_specialist_env(record, config, (profile,))
            raw_trace = rollout_policy(
                CommandForceBaseline(), raw_environment, seed=seed
            )
            pid_trace = rollout_policy(
                _pid_policy(gains, pid_environment), pid_environment, seed=seed
            )
            teacher_trace = rollout_policy(teacher, teacher_environment, seed=seed)
            student_trace = rollout_policy(student, student_environment, seed=seed)
            traces.append(
                (
                    profile.command_id,
                    raw_trace,
                    {
                        "PID": pid_trace,
                        "RL Teacher": teacher_trace,
                        "Student": student_trace,
                    },
                )
            )
            command_row = {
                "plant_id": record.plant_id,
                "command_id": profile.command_id,
                "raw": tracking_metrics(raw_trace, config.force_limit_n),
                "PID": tracking_metrics(pid_trace, config.force_limit_n),
                "RL Teacher": tracking_metrics(teacher_trace, config.force_limit_n),
                "Student": tracking_metrics(student_trace, config.force_limit_n),
            }
            command_rows.append(command_row)
            flat_rows.append(
                {
                    **command_row,
                    "quality_region": record.quality_region,
                    "distillation_split": _dataset_split(
                        student_payload, record.plant_id
                    ),
                }
            )
        plant_dir = destination / "aircraft" / record.plant_id
        plot_path = save_controller_comparison_grid(
            traces,
            plant_dir / "raw_pid_teacher_student.png",
            title=f"{record.plant_id}: Raw vs PID vs RL Teacher vs Student",
        )
        zoom_plot_path = save_controlled_response_error_grid(
            traces,
            plant_dir / "pid_teacher_student_zoom.png",
            title=f"{record.plant_id}: controlled response and tracking error",
        )
        summary = {
            label: _aggregate(command_rows, label) for label in CONTROLLER_LABELS
        }
        aircraft_rows.append(
            {
                "plant_id": record.plant_id,
                "quality_region": record.quality_region,
                "distillation_split": _dataset_split(student_payload, record.plant_id),
                "parameters": asdict(record.parameters),
                "summary": summary,
                "commands": command_rows,
                "plot": str(plot_path),
                "zoom_plot": str(zoom_plot_path),
            }
        )
        teacher_contract = teacher_payload.get("actor_observation_contract", {})
        contract_checks.append(
            {
                "plant_id": record.plant_id,
                "teacher_raw_history_steps_zero": (
                    isinstance(teacher_contract, dict)
                    and int(teacher_contract.get("raw_history_steps", -1)) == 0
                    and not bool(teacher_contract.get("uses_raw_history_window", True))
                ),
                "teacher_observation_dim_matches_student": int(
                    teacher_payload["actor_observation_dim"]
                )
                == student_model.observation_dim,
                "pid_oracle_verified": True,
            }
        )

    rows_by_split = {
        split: [row for row in flat_rows if row["distillation_split"] == split]
        for split in sorted({str(row["distillation_split"]) for row in flat_rows})
    }
    summary_path = destination / "raw_pid_teacher_student_summary.png"
    _save_summary_plot(aircraft_rows, summary_path)
    csv_path = destination / "controller_metrics.csv"
    _write_metrics_csv(aircraft_rows, csv_path)
    temporal_contract = student_payload.get("temporal_contract", {})
    self_check = {
        "passed": all(
            check["teacher_raw_history_steps_zero"]
            and check["teacher_observation_dim_matches_student"]
            and check["pid_oracle_verified"]
            for check in contract_checks
        )
        and isinstance(temporal_contract, dict)
        and int(temporal_contract.get("raw_history_steps", -1)) == 0
        and not bool(temporal_contract.get("uses_raw_history_window", True)),
        "student_temporal_contract": temporal_contract,
        "aircraft_checks": contract_checks,
    }
    report = {
        "schema_version": "pipeline_controller_comparison_v1",
        "status": "complete" if self_check["passed"] else "self_check_failed",
        "source": git_source_revision(),
        "command_suite": {
            "name": args.command_suite,
            "version": (
                SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION
                if args.command_suite == "independent-test-v1"
                else "specialist-evaluation-commands"
            ),
            "selection_independent": args.command_suite == "independent-test-v1",
        },
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
            "teacher_count": len(bank["teachers"]),
        },
        "student_checkpoint": {
            "path": str(student_checkpoint),
            "sha256": sha256_file(student_checkpoint),
            "architecture": student_payload.get("student_architecture", "dense"),
            "parameter_count": student_payload["parameter_count"],
        },
        "self_check": self_check,
        "overall": {label: _aggregate(flat_rows, label) for label in CONTROLLER_LABELS},
        "distillation": {
            "overall": _distillation_summary(flat_rows),
            "by_distillation_split": {
                split: _distillation_summary(rows)
                for split, rows in rows_by_split.items()
            },
        },
        "aircraft": aircraft_rows,
        "artifacts": {
            "summary_plot": str(summary_path),
            "metrics_csv": str(csv_path),
            "aircraft_plots": [row["plot"] for row in aircraft_rows],
            "aircraft_zoom_plots": [row["zoom_plot"] for row in aircraft_rows],
        },
    }
    report_path = destination / "comparison_report.json"
    _write_json(report_path, report)
    print(json.dumps(report["distillation"], ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    print(f"plot={summary_path}")
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
