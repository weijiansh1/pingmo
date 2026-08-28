"""Offline imitation and closed-loop validation for the dense Student."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import evaluate_specialist, load_specialist_actor
from src.utils.provenance import git_source_revision, sha256_file


def imitation_metrics(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: str | torch.device,
) -> dict[str, float]:
    target_device = torch.device(device)
    squared_error = 0.0
    absolute_error = 0.0
    maximum_error = 0.0
    sample_count = 0
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
    if sample_count == 0:
        raise ValueError("cannot validate an empty distillation loader")
    return {
        "action_mse": squared_error / sample_count,
        "action_rmse": float(np.sqrt(squared_error / sample_count)),
        "action_mae": absolute_error / sample_count,
        "action_max_abs_error": maximum_error,
        "action_elements": sample_count,
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
    return {
        "pair_count": len(rows),
        "teacher_improvement_rate": float(np.mean(teacher_rmse < raw_rmse)),
        "student_improvement_rate": float(np.mean(student_rmse < raw_rmse)),
        "teacher_harm_rate": float(np.mean(teacher_cost > raw_cost)),
        "student_harm_rate": float(np.mean(student_cost > raw_cost)),
        "median_student_minus_teacher_rmse_rad_s": float(
            np.median(student_rmse - teacher_rmse)
        ),
    }


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
    if sha256_file(dataset_path) != str(dataset_reference["sha256"]):
        raise ValueError("Student distillation dataset manifest hash has changed")
    dataset_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    split_strategy = str(dataset_manifest["split_strategy"])
    train_plants = set(dataset_manifest["train_plant_ids"])
    validation_plants = set(dataset_manifest["validation_plant_ids"])

    def split_for_plant(plant_id: str) -> str:
        if split_strategy == "single_aircraft_command_holdout":
            return "single_aircraft_command_holdout"
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
        "aircraft_count": len(aircraft_rows),
        **overall,
        "by_distillation_split": by_distillation_split,
        "aircraft": aircraft_rows,
    }
    _write_json(destination / "evaluation.json", report)
    return report
