"""Evaluate a frozen conditional Student on aircraft absent from distillation."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
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

from src.context.aircraft_parameters import (
    AIRCRAFT_PARAMETER_NAMES,
    normalize_aircraft_parameters,
)
from src.controllers.pid import PIDGains, RollRatePIDPolicy
from src.controllers.policy_wrappers import ForceSlewLimitedPolicy
from src.envs.roll_rate_commands import (
    SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION,
    specialist_independent_test_commands,
)
from src.experiments.exploratory_sac import load_persisted_records
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


CONTROLLER_LABELS = ("raw", "PID", "Student")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--pid-report-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-id", action="append", required=True)
    parser.add_argument("--student-force-rate-limit", type=float)
    parser.add_argument("--slew-limit-scan", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _generalization_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    raw = np.asarray([row["raw"]["tracking_rmse_deg_s"] for row in rows])
    pid = np.asarray([row["PID"]["tracking_rmse_deg_s"] for row in rows])
    student = np.asarray([row["Student"]["tracking_rmse_deg_s"] for row in rows])
    ratios = student / np.maximum(pid, 1e-8)
    return {
        "pair_count": len(rows),
        "student_improves_raw_rate": float(np.mean(student < raw)),
        "student_beats_or_equals_pid_rate": float(np.mean(student <= pid)),
        "mean_student_to_pid_rmse_ratio": float(np.mean(ratios)),
        "median_student_to_pid_rmse_ratio": float(np.median(ratios)),
        "maximum_student_to_pid_rmse_ratio": float(np.max(ratios)),
    }


def _save_summary_plot(rows: list[dict[str, object]], path: Path) -> Path:
    labels = [str(row["plant_id"]) for row in rows]
    positions = np.arange(len(labels), dtype=float)
    width = 0.25
    colors = {"raw": "#64748b", "PID": "#16803c", "Student": "#c82d2d"}
    figure, axes = plt.subplots(3, 1, figsize=(16, 12), layout="constrained")
    for offset, label in enumerate(CONTROLLER_LABELS):
        values = [row["summary"][label]["mean_tracking_rmse_deg_s"] for row in rows]
        axes[0].bar(
            positions + (offset - 1) * width,
            values,
            width,
            color=colors[label],
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking RMSE (deg/s, log scale)")
    axes[0].set_title("Frozen Student on unseen aircraft")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for offset, label in enumerate(("PID", "Student")):
        values = [row["summary"][label]["mean_tracking_rmse_deg_s"] for row in rows]
        axes[1].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            color=colors[label],
            label=label,
        )
    axes[1].set_ylabel("Tracking RMSE (deg/s)")
    axes[1].set_title("Controller detail")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    for offset, label in enumerate(("PID", "Student")):
        values = [
            row["summary"][label]["mean_requested_force_total_variation_n"]
            for row in rows
        ]
        axes[2].bar(
            positions + (offset - 0.5) * width,
            values,
            width,
            color=colors[label],
            label=label,
        )
    axes[2].set_ylabel("Requested-force TV (N)")
    axes[2].set_title("Control smoothness")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_distance_plot(rows: list[dict[str, object]], path: Path) -> Path:
    colors = {"train_core": "#2878b5", "train_boundary": "#c82423"}
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    for split in sorted({str(row["split"]) for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        x = np.asarray([row["parameter_distance"]["nearest_distance"] for row in selected])
        y = np.asarray(
            [
                row["summary"]["Student"]["mean_tracking_rmse_deg_s"]
                / max(row["summary"]["PID"]["mean_tracking_rmse_deg_s"], 1e-8)
                for row in selected
            ]
        )
        axis.scatter(x, y, s=60, color=colors.get(split, "#666666"), label=split)
        for x_value, y_value, row in zip(x, y, selected, strict=True):
            axis.annotate(
                str(row["plant_id"]).split("-", 1)[-1],
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Nearest seen-aircraft distance in normalized theta")
    axis.set_ylabel("Student / dedicated PID mean RMSE")
    axis.set_title("Zero-shot generalization versus parameter distance")
    axis.grid(alpha=0.25)
    axis.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _validate_pid_contract(payload: dict[str, object], record: object, config: object) -> None:
    if payload.get("status") != "complete" or payload.get("plant_id") != record.plant_id:
        raise ValueError(f"invalid PID oracle for {record.plant_id}")
    expected = {
        "plant_dt_s": config.plant_dt_s,
        "policy_dt_s": config.policy_dt_s,
        "reference_natural_frequency_rad_s": config.reference_natural_frequency_rad_s,
        "reference_damping_ratio": config.reference_damping_ratio,
        "reference_delay_mode": config.reference_delay_mode,
    }
    for name, value in expected.items():
        observed = payload.get(name)
        if isinstance(value, float):
            if observed is None or not np.isclose(float(observed), value):
                raise ValueError(f"PID oracle {name} mismatch for {record.plant_id}")
        elif observed != value:
            raise ValueError(f"PID oracle {name} mismatch for {record.plant_id}")


def main() -> None:
    args = _parse_args()
    library_path = args.library.resolve()
    bank_path = args.teacher_bank.resolve()
    student_path = args.student_checkpoint.resolve()
    pid_root = args.pid_report_root.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    plant_ids = list(dict.fromkeys(args.plant_id))
    if len(plant_ids) != len(args.plant_id):
        raise ValueError("unseen plant IDs must be unique")

    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("status") != "complete" or not bank.get("teachers"):
        raise ValueError("a complete Teacher Bank is required for the environment contract")
    environment_actor = bank_path.parent / str(bank["teachers"][0]["actor_checkpoint"])
    _, _, config, teacher_payload = load_specialist_actor(
        environment_actor, device=args.device
    )
    student_model, student_payload = load_dense_student(student_path, device=args.device)
    student_hash_before = sha256_file(student_path)
    selection_report: dict[str, object] | None = None
    selection_verified = args.student_force_rate_limit is None
    if args.student_force_rate_limit is not None:
        if (
            not np.isfinite(args.student_force_rate_limit)
            or args.student_force_rate_limit <= 0
        ):
            raise ValueError("Student force-rate limit must be finite and positive")
        if args.slew_limit_scan is None:
            raise ValueError("a slew-limit scan is required for a limited Student")
        scan_path = args.slew_limit_scan.resolve()
        selection_report = json.loads(scan_path.read_text(encoding="utf-8"))
        selected = selection_report.get("selected")
        if not isinstance(selected, dict):
            raise ValueError("slew-limit scan has no selected candidate")
        selected_rate = float(selected["force_rate_limit_n_s"])
        selected_student = selection_report.get("student_checkpoint")
        selection_verified = bool(
            np.isclose(selected_rate, args.student_force_rate_limit)
            and isinstance(selected_student, dict)
            and selected_student.get("sha256") == student_hash_before
            and selection_report.get("selection_is_training_only") is True
        )
        if not selection_verified:
            raise ValueError("slew-limit selection does not match the frozen Student")
    dataset = student_payload.get("dataset_manifest")
    if not isinstance(dataset, dict):
        raise ValueError("Student checkpoint has no dataset manifest")
    seen_ids = sorted(
        set(dataset.get("train_plant_ids", []))
        | set(dataset.get("validation_plant_ids", []))
    )
    overlap = sorted(set(plant_ids) & set(seen_ids))
    if overlap:
        raise ValueError(f"requested aircraft are not unseen by the Student: {overlap}")
    if int(teacher_payload["actor_observation_dim"]) != student_model.observation_dim:
        raise ValueError("environment and Student observation contracts differ")

    records = load_persisted_records(library_path, seen_ids + plant_ids)
    lookup = {record.plant_id: record for record in records}
    seen_theta = np.stack(
        [normalize_aircraft_parameters(lookup[plant_id].parameters) for plant_id in seen_ids]
    )
    aircraft_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    pid_checks: list[dict[str, object]] = []
    profiles = specialist_independent_test_commands(config.episode_duration_s)

    for plant_index, plant_id in enumerate(plant_ids):
        record = lookup[plant_id]
        theta = normalize_aircraft_parameters(record.parameters)
        distances = np.linalg.norm(seen_theta - theta, axis=1)
        nearest_index = int(np.argmin(distances))
        pid_path = pid_root / plant_id / "pid_report.json"
        pid_payload = json.loads(pid_path.read_text(encoding="utf-8"))
        _validate_pid_contract(pid_payload, record, config)
        gains = PIDGains(**pid_payload["gains"])
        student = DenseStudentPolicy(student_model, record.parameters, device=args.device)
        if args.student_force_rate_limit is not None:
            student = ForceSlewLimitedPolicy(
                student,
                force_rate_limit_n_s=args.student_force_rate_limit,
                policy_dt_s=config.policy_dt_s,
                force_limit_n=config.force_limit_n,
            )
        traces = []
        command_rows: list[dict[str, object]] = []
        for command_index, profile in enumerate(profiles):
            seed = config.seed + plant_index * 1000 + command_index
            raw_environment = build_specialist_env(record, config, (profile,))
            pid_environment = build_specialist_env(record, config, (profile,))
            student_environment = build_specialist_env(record, config, (profile,))
            raw_trace = rollout_policy(CommandForceBaseline(), raw_environment, seed=seed)
            pid_trace = rollout_policy(
                _pid_policy(gains, pid_environment), pid_environment, seed=seed
            )
            student_trace = rollout_policy(student, student_environment, seed=seed)
            traces.append(
                (
                    profile.command_id,
                    raw_trace,
                    {"PID": pid_trace, "Student": student_trace},
                )
            )
            row = {
                "plant_id": plant_id,
                "command_id": profile.command_id,
                "raw": tracking_metrics(raw_trace, config.force_limit_n),
                "PID": tracking_metrics(pid_trace, config.force_limit_n),
                "Student": tracking_metrics(student_trace, config.force_limit_n),
            }
            command_rows.append(row)
            flat_rows.append(
                {
                    **row,
                    "split": record.split,
                    "quality_region": record.quality_region,
                }
            )

        plant_dir = destination / "aircraft" / plant_id
        full_plot = save_controller_comparison_grid(
            traces,
            plant_dir / "raw_pid_student.png",
            title=f"{plant_id}: unseen-aircraft zero-shot control",
        )
        zoom_plot = save_controlled_response_error_grid(
            traces,
            plant_dir / "pid_student_zoom.png",
            title=f"{plant_id}: frozen Student zero-shot response",
        )
        summary = {
            label: _aggregate(command_rows, label) for label in CONTROLLER_LABELS
        }
        aircraft_rows.append(
            {
                "plant_id": plant_id,
                "split": record.split,
                "quality_region": record.quality_region,
                "parameters": asdict(record.parameters),
                "normalized_parameters": {
                    name: float(value)
                    for name, value in zip(AIRCRAFT_PARAMETER_NAMES, theta, strict=True)
                },
                "parameter_distance": {
                    "nearest_seen_plant_id": seen_ids[nearest_index],
                    "nearest_distance": float(distances[nearest_index]),
                    "inside_seen_axis_aligned_envelope": bool(
                        np.all(
                            (theta >= np.min(seen_theta, axis=0))
                            & (theta <= np.max(seen_theta, axis=0))
                        )
                    ),
                    "outside_global_normalization_bounds": bool(
                        np.any(np.abs(theta) > 1.0 + 1e-6)
                    ),
                },
                "summary": summary,
                "commands": command_rows,
                "pid_report": {
                    "path": str(pid_path),
                    "sha256": sha256_file(pid_path),
                },
                "plots": {"full": str(full_plot), "zoom": str(zoom_plot)},
            }
        )
        pid_checks.append(
            {
                "plant_id": plant_id,
                "verified": True,
                "tuning_episode_duration_s": float(
                    pid_payload["episode_duration_s"]
                ),
                "test_episode_duration_s": config.episode_duration_s,
            }
        )
        print(
            json.dumps(
                {
                    "plant_id": plant_id,
                    "split": record.split,
                    "quality_region": record.quality_region,
                    "nearest_seen_distance": float(distances[nearest_index]),
                    "pid_rmse_deg_s": summary["PID"]["mean_tracking_rmse_deg_s"],
                    "student_rmse_deg_s": summary["Student"][
                        "mean_tracking_rmse_deg_s"
                    ],
                }
            ),
            flush=True,
        )

    summary_plot = _save_summary_plot(aircraft_rows, destination / "summary.png")
    distance_plot = _save_distance_plot(
        aircraft_rows, destination / "distance_vs_generalization.png"
    )
    by_split = {
        split: [row for row in flat_rows if row["split"] == split]
        for split in sorted({str(row["split"]) for row in flat_rows})
    }
    by_quality = {
        region: [row for row in flat_rows if row["quality_region"] == region]
        for region in sorted({str(row["quality_region"]) for row in flat_rows})
    }
    self_check = {
        "passed": (
            not overlap
            and all(row["verified"] for row in pid_checks)
            and sha256_file(student_path) == student_hash_before
            and len(flat_rows) == len(plant_ids) * len(profiles)
            and selection_verified
        ),
        "student_checkpoint_unchanged": sha256_file(student_path)
        == student_hash_before,
        "target_seen_overlap": overlap,
        "pid_oracles": pid_checks,
        "environment_observation_dim_matches_student": int(
            teacher_payload["actor_observation_dim"]
        )
        == student_model.observation_dim,
    }
    report = {
        "schema_version": "unseen_aircraft_student_evaluation_v1",
        "status": "complete" if self_check["passed"] else "self_check_failed",
        "source": git_source_revision(),
        "scope": {
            "zero_shot_unseen_aircraft": True,
            "student_training_or_adaptation": False,
            "target_aircraft_teacher_used": False,
            "target_specific_pid_used_for_comparison": True,
            "pid_tuning_window_matches_test_window": all(
                np.isclose(
                    row["tuning_episode_duration_s"],
                    row["test_episode_duration_s"],
                )
                for row in pid_checks
            ),
            "command_suite": {
                "name": "independent-test-v1",
                "version": SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION,
                "selection_independent": True,
            },
            "deployment_wrapper": {
                "kind": (
                    "requested_force_slew_limit"
                    if args.student_force_rate_limit is not None
                    else "none"
                ),
                "force_rate_limit_n_s": args.student_force_rate_limit,
                "selected_on_training_aircraft_only": selection_verified,
            },
            "seen_plant_ids": seen_ids,
            "target_plant_ids": plant_ids,
        },
        "library": {"path": str(library_path), "sha256": sha256_file(library_path)},
        "student_checkpoint": {
            "path": str(student_path),
            "sha256": student_hash_before,
            "parameter_count": student_payload["parameter_count"],
            "architecture": student_payload.get("student_architecture", "dense"),
        },
        "environment_contract_source": {
            "teacher_bank": str(bank_path),
            "actor_checkpoint": str(environment_actor),
        },
        "slew_limit_selection": (
            {
                "path": str(args.slew_limit_scan.resolve()),
                "sha256": sha256_file(args.slew_limit_scan.resolve()),
                "verified": selection_verified,
            }
            if args.slew_limit_scan is not None
            else None
        ),
        "self_check": self_check,
        "overall": {
            label: _aggregate(flat_rows, label) for label in CONTROLLER_LABELS
        },
        "generalization": _generalization_summary(flat_rows),
        "by_split": {
            split: {
                "controllers": {
                    label: _aggregate(rows, label) for label in CONTROLLER_LABELS
                },
                "generalization": _generalization_summary(rows),
            }
            for split, rows in by_split.items()
        },
        "by_quality_region": {
            region: {
                "controllers": {
                    label: _aggregate(rows, label) for label in CONTROLLER_LABELS
                },
                "generalization": _generalization_summary(rows),
            }
            for region, rows in by_quality.items()
        },
        "aircraft": aircraft_rows,
        "artifacts": {
            "summary_plot": str(summary_plot),
            "distance_plot": str(distance_plot),
            "aircraft_full_plots": [row["plots"]["full"] for row in aircraft_rows],
            "aircraft_zoom_plots": [row["plots"]["zoom"] for row in aircraft_rows],
        },
    }
    report_path = destination / "report.json"
    _write_json(report_path, report)
    print(json.dumps(report["generalization"], indent=2), flush=True)
    print(f"report={report_path}", flush=True)
    if report["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
