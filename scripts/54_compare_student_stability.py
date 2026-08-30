"""Compare baseline and candidate Students on matched Teacher-Bank trajectories."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
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

from src.envs.roll_rate_commands import specialist_evaluation_commands
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import (
    build_specialist_env,
    load_specialist_actor,
    rollout_policy,
    tracking_metrics,
)
from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--baseline-student", type=Path, required=True)
    parser.add_argument("--candidate-student", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-id", action="append", required=True)
    parser.add_argument("--command-id", default="eval-step-pos-25deg-s")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _first_threshold_crossing_s(
    trace: dict[str, np.ndarray], threshold_deg_s: float = 5.0
) -> float | None:
    error_deg_s = np.rad2deg(
        np.asarray(trace["p_rad_s"]) - np.asarray(trace["p_reference_rad_s"])
    )
    indices = np.flatnonzero(np.abs(error_deg_s) > threshold_deg_s)
    if not len(indices):
        return None
    return float(np.asarray(trace["time_s"])[indices[0]])


def _save_plot(
    traces: dict[str, dict[str, np.ndarray]],
    output_path: Path,
    title: str,
) -> None:
    teacher = traces["teacher"]
    figure, axes = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, layout="constrained"
    )
    axes[0].plot(
        teacher["time_s"],
        np.rad2deg(teacher["p_command_rad_s"]),
        color="black",
        linestyle=":",
        linewidth=1.4,
        label="p_c",
    )
    axes[0].plot(
        teacher["time_s"],
        np.rad2deg(teacher["p_reference_rad_s"]),
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="p_ref",
    )
    colors = {"teacher": "#2878b5", "v3": "#d18b19", "v4": "#c82423"}
    labels = {"teacher": "Teacher", "v3": "v3 Student", "v4": "v4 Student"}
    for name in ("teacher", "v3", "v4"):
        trace = traces[name]
        axes[0].plot(
            trace["time_s"],
            np.rad2deg(trace["p_rad_s"]),
            color=colors[name],
            linewidth=1.7,
            label=labels[name],
        )
        axes[1].plot(
            trace["time_s"],
            trace["requested_f_as_n"],
            color=colors[name],
            linewidth=1.3,
            label=labels[name],
        )
    axes[0].set_ylabel("Roll rate p (deg/s)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Requested F_as (N)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    bank_path = args.teacher_bank.resolve()
    baseline_path = args.baseline_student.resolve()
    candidate_path = args.candidate_student.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("status") != "complete":
        raise ValueError("a complete Teacher Bank is required")
    entries = {str(row["plant_id"]): row for row in bank["teachers"]}
    missing = sorted(set(args.plant_id) - set(entries))
    if missing:
        raise ValueError(f"aircraft are absent from the Teacher Bank: {missing}")
    baseline_model, baseline_payload = load_dense_student(
        baseline_path, device=args.device
    )
    candidate_model, candidate_payload = load_dense_student(
        candidate_path, device=args.device
    )
    if baseline_model.observation_dim != candidate_model.observation_dim:
        raise ValueError("baseline and candidate observation dimensions differ")

    rows: list[dict[str, object]] = []
    for plant_id in args.plant_id:
        entry = entries[plant_id]
        actor_path = bank_path.parent / str(entry["actor_checkpoint"])
        teacher, record, config, teacher_payload = load_specialist_actor(
            actor_path, device=args.device
        )
        if int(teacher_payload["actor_observation_dim"]) != baseline_model.observation_dim:
            raise ValueError(f"observation contract mismatch for {plant_id}")
        profiles = specialist_evaluation_commands(config.episode_duration_s)
        profile_index = next(
            (
                index
                for index, profile in enumerate(profiles)
                if profile.command_id == args.command_id
            ),
            None,
        )
        if profile_index is None:
            raise ValueError(f"unknown evaluation command: {args.command_id}")
        profile = profiles[profile_index]
        policies = {
            "teacher": teacher,
            "v3": DenseStudentPolicy(
                baseline_model, record.parameters, device=args.device
            ),
            "v4": DenseStudentPolicy(
                candidate_model, record.parameters, device=args.device
            ),
        }
        traces = {
            name: rollout_policy(
                policy,
                build_specialist_env(record, config, (profile,)),
                seed=config.seed + profile_index,
            )
            for name, policy in policies.items()
        }
        metrics = {
            name: tracking_metrics(trace, config.force_limit_n)
            for name, trace in traces.items()
        }
        row = {
            "plant_id": plant_id,
            "quality_region": record.quality_region,
            "command_id": profile.command_id,
            "teacher": metrics["teacher"],
            "baseline_student": metrics["v3"],
            "candidate_student": metrics["v4"],
            "baseline_first_5deg_s_error_crossing_s": _first_threshold_crossing_s(
                traces["v3"]
            ),
            "candidate_first_5deg_s_error_crossing_s": _first_threshold_crossing_s(
                traces["v4"]
            ),
        }
        rows.append(row)
        _save_plot(
            traces,
            destination / plant_id / "response_comparison.png",
            f"{plant_id}: {profile.command_id}",
        )

    baseline_rmse = np.asarray(
        [row["baseline_student"]["tracking_rmse_deg_s"] for row in rows],
        dtype=float,
    )
    candidate_rmse = np.asarray(
        [row["candidate_student"]["tracking_rmse_deg_s"] for row in rows],
        dtype=float,
    )
    baseline_tv = np.asarray(
        [
            row["baseline_student"]["requested_force_total_variation_n"]
            for row in rows
        ],
        dtype=float,
    )
    candidate_tv = np.asarray(
        [
            row["candidate_student"]["requested_force_total_variation_n"]
            for row in rows
        ],
        dtype=float,
    )
    report = {
        "schema_version": "student_stability_comparison_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "baseline_student": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
            "parameter_count": baseline_payload["parameter_count"],
        },
        "candidate_student": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
            "parameter_count": candidate_payload["parameter_count"],
        },
        "matched_contract": {
            "plant_ids": args.plant_id,
            "command_id": args.command_id,
            "same_environment_seed": True,
            "same_teacher_bank": True,
        },
        "summary": {
            "aircraft_count": len(rows),
            "candidate_improves_rmse_rate": float(
                np.mean(candidate_rmse < baseline_rmse)
            ),
            "mean_baseline_rmse_deg_s": float(np.mean(baseline_rmse)),
            "mean_candidate_rmse_deg_s": float(np.mean(candidate_rmse)),
            "mean_baseline_requested_force_tv_n": float(np.mean(baseline_tv)),
            "mean_candidate_requested_force_tv_n": float(np.mean(candidate_tv)),
        },
        "aircraft": rows,
    }
    _write_json(destination / "comparison.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
