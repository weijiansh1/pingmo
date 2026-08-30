"""Offline imitation and closed-loop validation for the dense Student."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.envs.roll_rate_commands import specialist_evaluation_commands
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import (
    build_specialist_env,
    evaluate_specialist,
    load_specialist_actor,
    rollout_policy,
)
from src.utils.provenance import git_source_revision, sha256_file


def imitation_metrics(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: str | torch.device,
) -> dict[str, float | int]:
    target_device = torch.device(device)
    squared_error = 0.0
    absolute_error = 0.0
    maximum_error = 0.0
    sample_count = 0
    delta_squared_error = 0.0
    delta_absolute_error = 0.0
    delta_maximum_error = 0.0
    temporal_count = 0
    model.eval()
    with torch.no_grad():
        for batch in batches:
            observation = batch["observation"].to(target_device)
            theta = batch["aircraft_parameters"].to(target_device)
            target = batch["teacher_action"].to(target_device)
            prediction = model(observation, theta)
            error = prediction - target
            squared_error += float(error.square().sum())
            absolute_error += float(error.abs().sum())
            maximum_error = max(maximum_error, float(error.abs().max()))
            sample_count += error.numel()
            if "previous_observation" in batch:
                previous_observation = batch["previous_observation"].to(target_device)
                previous_target = batch["previous_teacher_action"].to(target_device)
                temporal_mask = batch["temporal_mask"].to(target_device).bool()
                step_delta = batch["policy_step_delta"].to(target_device).clamp_min(1.0)
                previous_prediction = model(previous_observation, theta)
                prediction_rate = (prediction - previous_prediction) / step_delta.unsqueeze(-1)
                target_rate = (target - previous_target) / step_delta.unsqueeze(-1)
                delta_error = prediction_rate - target_rate
                valid_delta_error = delta_error[temporal_mask]
                if valid_delta_error.numel():
                    delta_squared_error += float(valid_delta_error.square().sum())
                    delta_absolute_error += float(valid_delta_error.abs().sum())
                    delta_maximum_error = max(
                        delta_maximum_error,
                        float(valid_delta_error.abs().max()),
                    )
                    temporal_count += valid_delta_error.numel()
    if sample_count == 0:
        raise ValueError("cannot validate an empty distillation loader")
    delta_mse = delta_squared_error / max(temporal_count, 1)
    return {
        "action_mse": squared_error / sample_count,
        "action_rmse": float(np.sqrt(squared_error / sample_count)),
        "action_mae": absolute_error / sample_count,
        "action_max_abs_error": maximum_error,
        "action_elements": sample_count,
        "action_delta_mse_per_policy_step": delta_mse,
        "action_delta_rmse_per_policy_step": float(np.sqrt(delta_mse)),
        "action_delta_mae_per_policy_step": delta_absolute_error
        / max(temporal_count, 1),
        "action_delta_max_abs_error_per_policy_step": delta_maximum_error,
        "temporal_action_elements": temporal_count,
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _closed_loop_summary(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {"pair_count": 0}
    teacher_rmse = np.asarray(
        [row["teacher"]["tracking_rmse_rad_s"] for row in rows], dtype=float
    )
    student_rmse = np.asarray(
        [row["student"]["tracking_rmse_rad_s"] for row in rows], dtype=float
    )
    raw_rmse = np.asarray([row["raw"]["tracking_rmse_rad_s"] for row in rows], dtype=float)
    teacher_cost = np.asarray([row["teacher"]["episode_cost"] for row in rows], dtype=float)
    student_cost = np.asarray([row["student"]["episode_cost"] for row in rows], dtype=float)
    raw_cost = np.asarray([row["raw"]["episode_cost"] for row in rows], dtype=float)
    teacher_requested_variation = np.asarray(
        [row["teacher"]["requested_force_total_variation_n"] for row in rows],
        dtype=float,
    )
    student_requested_variation = np.asarray(
        [row["student"]["requested_force_total_variation_n"] for row in rows],
        dtype=float,
    )
    student_peak_error = np.asarray(
        [row["student"]["tracking_peak_error_deg_s"] for row in rows], dtype=float
    )
    mean_teacher_variation = float(np.mean(teacher_requested_variation))
    mean_student_variation = float(np.mean(student_requested_variation))
    return {
        "pair_count": len(rows),
        "teacher_improvement_rate": float(np.mean(teacher_rmse < raw_rmse)),
        "student_improvement_rate": float(np.mean(student_rmse < raw_rmse)),
        "teacher_harm_rate": float(np.mean(teacher_cost > raw_cost)),
        "student_harm_rate": float(np.mean(student_cost > raw_cost)),
        "median_student_minus_teacher_rmse_rad_s": float(
            np.median(student_rmse - teacher_rmse)
        ),
        "mean_teacher_tracking_rmse_deg_s": float(np.rad2deg(np.mean(teacher_rmse))),
        "mean_student_tracking_rmse_deg_s": float(np.rad2deg(np.mean(student_rmse))),
        "maximum_student_peak_error_deg_s": float(np.max(student_peak_error)),
        "mean_teacher_requested_force_total_variation_n": mean_teacher_variation,
        "mean_student_requested_force_total_variation_n": mean_student_variation,
        "student_teacher_requested_force_variation_ratio": mean_student_variation
        / max(mean_teacher_variation, 1e-8),
        "student_within_10pct_teacher_rate": float(
            np.mean(student_rmse <= 1.1 * np.maximum(teacher_rmse, 1e-8))
        ),
    }


def _save_teacher_student_response_plot(
    teacher_trace: dict[str, np.ndarray],
    student_trace: dict[str, np.ndarray],
    output_path: Path,
    *,
    title: str,
) -> None:
    figure, (response_axis, force_axis) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, layout="constrained"
    )
    time_s = teacher_trace["time_s"]
    response_axis.plot(
        time_s,
        np.rad2deg(teacher_trace["p_command_rad_s"]),
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="p_c",
    )
    response_axis.plot(
        time_s,
        np.rad2deg(teacher_trace["p_reference_rad_s"]),
        color="black",
        linestyle="--",
        linewidth=2.0,
        label="p_ref",
    )
    response_axis.plot(
        time_s,
        np.rad2deg(teacher_trace["p_rad_s"]),
        color="#2878b5",
        linewidth=2.0,
        label="RL Teacher",
    )
    response_axis.plot(
        student_trace["time_s"],
        np.rad2deg(student_trace["p_rad_s"]),
        color="#c82423",
        linewidth=1.8,
        label="Student",
    )
    response_axis.set_title(title)
    response_axis.set_ylabel("p (deg/s)")
    response_axis.grid(alpha=0.25)
    response_axis.legend(loc="best")

    force_axis.plot(
        time_s,
        teacher_trace["requested_f_as_n"],
        color="#79add2",
        linestyle="--",
        linewidth=1.2,
        label="Teacher requested F_as",
    )
    force_axis.plot(
        time_s,
        teacher_trace["f_as_n"],
        color="#2878b5",
        linewidth=1.7,
        label="Teacher applied F_as",
    )
    force_axis.plot(
        student_trace["time_s"],
        student_trace["requested_f_as_n"],
        color="#ed8b89",
        linestyle="--",
        linewidth=1.2,
        label="Student requested F_as",
    )
    force_axis.plot(
        student_trace["time_s"],
        student_trace["f_as_n"],
        color="#c82423",
        linewidth=1.5,
        label="Student applied F_as",
    )
    force_axis.set_xlabel("Time (s)")
    force_axis.set_ylabel("Force (N)")
    force_axis.grid(alpha=0.25)
    force_axis.legend(loc="best")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_closed_loop_summary_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    teacher = np.rad2deg(
        np.asarray([row["teacher"]["tracking_rmse_rad_s"] for row in rows], dtype=float)
    )
    student = np.rad2deg(
        np.asarray([row["student"]["tracking_rmse_rad_s"] for row in rows], dtype=float)
    )
    splits = sorted({str(row["distillation_split"]) for row in rows})
    colors = {split: plt.cm.tab10(index) for index, split in enumerate(splits)}
    figure, axis = plt.subplots(figsize=(7.5, 7), layout="constrained")
    for split in splits:
        mask = np.asarray([str(row["distillation_split"]) == split for row in rows])
        axis.scatter(
            teacher[mask],
            student[mask],
            s=30,
            alpha=0.8,
            color=colors[split],
            label=split,
        )
    upper = max(float(np.max(teacher)), float(np.max(student)), 1e-3)
    axis.plot([0.0, upper], [0.0, upper], color="black", linestyle="--", label="Student = Teacher")
    axis.set_xlim(0.0, 1.05 * upper)
    axis.set_ylim(0.0, 1.05 * upper)
    axis.set_xlabel("Teacher tracking RMSE (deg/s)")
    axis.set_ylabel("Student tracking RMSE (deg/s)")
    axis.set_title("Closed-loop distillation quality")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_dense_student_bank(
    checkpoint_path: str | Path,
    teacher_bank_path: str | Path,
    output_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, object]:
    """Compare raw, specialist Teacher, and Student on every Teacher-Bank aircraft."""

    checkpoint = Path(checkpoint_path)
    bank_path = Path(teacher_bank_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model, student_payload = load_dense_student(checkpoint, device=device)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "specialist_teacher_bank_v1" or bank.get("status") != "complete":
        raise ValueError("a complete specialist Teacher Bank is required")

    dataset_reference = student_payload.get("dataset_manifest")
    if not isinstance(dataset_reference, dict):
        raise ValueError("Student checkpoint does not identify its distillation dataset")
    dataset_path = Path(str(dataset_reference["path"]))
    dataset_manifest_verified = False
    if dataset_path.is_file():
        if sha256_file(dataset_path) != str(dataset_reference["sha256"]):
            raise ValueError("Student distillation dataset manifest hash has changed")
        dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
        dataset_manifest_verified = True
    elif all(
        name in dataset_reference
        for name in ("split_strategy", "train_plant_ids", "validation_plant_ids")
    ):
        dataset_manifest = dataset_reference
    else:
        raise FileNotFoundError(
            "Student dataset manifest is unavailable and split metadata is not embedded"
        )
    split_strategy = str(dataset_manifest["split_strategy"])
    train_plants = set(dataset_manifest["train_plant_ids"])
    validation_plants = set(dataset_manifest["validation_plant_ids"])

    def split_for_plant(plant_id: str) -> str:
        if split_strategy == "single_aircraft_command_holdout":
            return "single_aircraft_command_holdout"
        if split_strategy == "all_aircraft_command_holdout":
            return "all_aircraft_command_holdout"
        if plant_id in validation_plants:
            return "validation_aircraft"
        if plant_id in train_plants:
            return "train_aircraft"
        return "unassigned_aircraft"

    aircraft_rows: list[dict[str, object]] = []
    flat_rows: list[dict[str, object]] = []
    for entry in bank["teachers"]:
        actor_path = bank_path.parent / str(entry["actor_checkpoint"])
        teacher_policy, record, training_config, teacher_payload = load_specialist_actor(
            actor_path,
            device=device,
        )
        if int(teacher_payload["actor_observation_dim"]) != model.observation_dim:
            raise ValueError("Student and specialist observation contracts differ")
        distillation_split = split_for_plant(record.plant_id)
        student_policy = DenseStudentPolicy(model, record.parameters, device=device)
        plant_dir = destination / record.plant_id
        teacher_result = evaluate_specialist(
            teacher_policy,
            record,
            training_config,
            output_dir=plant_dir / "teacher",
            controller_label="teacher",
        )
        student_result = evaluate_specialist(
            student_policy,
            record,
            training_config,
            output_dir=plant_dir / "student",
            controller_label="student",
        )
        comparison_profile = specialist_evaluation_commands(
            training_config.episode_duration_s
        )[0]
        teacher_trace = rollout_policy(
            teacher_policy,
            build_specialist_env(record, training_config, (comparison_profile,)),
            seed=training_config.seed + 50_000,
        )
        student_trace = rollout_policy(
            student_policy,
            build_specialist_env(record, training_config, (comparison_profile,)),
            seed=training_config.seed + 50_000,
        )
        _save_teacher_student_response_plot(
            teacher_trace,
            student_trace,
            plant_dir / "teacher_student_comparison.png",
            title=f"{record.plant_id}: {comparison_profile.command_id}",
        )
        for teacher_row, student_row in zip(teacher_result["rows"], student_result["rows"], strict=True):
            flat_rows.append(
                {
                    "plant_id": record.plant_id,
                    "command_id": teacher_row["command_id"],
                    "quality_region": record.quality_region,
                    "distillation_split": distillation_split,
                    "raw": teacher_row["raw"],
                    "teacher": teacher_row["teacher"],
                    "student": student_row["student"],
                }
            )
        aircraft_rows.append(
            {
                "plant_id": record.plant_id,
                "quality_region": record.quality_region,
                "distillation_split": distillation_split,
                "teacher": teacher_result,
                "student": student_result,
            }
        )

    overall = _closed_loop_summary(flat_rows)
    by_distillation_split = {
        split: _closed_loop_summary(
            [row for row in flat_rows if row["distillation_split"] == split]
        )
        for split in sorted({str(row["distillation_split"]) for row in flat_rows})
    }
    report: dict[str, object] = {
        "schema_version": "dense_student_closed_loop_evaluation_v1",
        "status": "complete",
        "source": git_source_revision(),
        "student_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "parameter_count": student_payload["parameter_count"],
        },
        "teacher_bank": {"path": str(bank_path.resolve()), "sha256": sha256_file(bank_path)},
        "distillation_split_strategy": split_strategy,
        "dataset_manifest_verified": dataset_manifest_verified,
        "aircraft_count": len(aircraft_rows),
        **overall,
        "by_distillation_split": by_distillation_split,
        "aircraft": aircraft_rows,
    }
    _write_json(destination / "evaluation.json", report)
    _save_closed_loop_summary_plot(flat_rows, destination / "closed_loop_summary.png")
    return report
