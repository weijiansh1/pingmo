"""Closed-loop student-driven aggregation for specialist-policy distillation."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.context.aircraft_parameters import (
    AIRCRAFT_PARAMETER_NAMES,
    AIRCRAFT_PARAMETER_NORMALIZATION,
    normalize_aircraft_parameters,
)
from src.distillation.collect_data import (
    DistillationCollectionConfig,
    collect_teacher_bank_data,
)
from src.distillation.dataset import (
    DISTILLATION_DATASET_SCHEMA_V2,
    SUPPORTED_DISTILLATION_DATASET_SCHEMAS,
    DistillationArrays,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    save_distillation_shard,
)
from src.distillation.distill import DenseStudentTrainingConfig, train_dense_student
from src.distillation.validate import evaluate_dense_student_bank
from src.envs.roll_rate_commands import (
    RollRateCommandProfile,
    specialist_evaluation_commands,
    specialist_extended_commands,
    specialist_step_commands,
)
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import build_specialist_env, load_specialist_actor
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class StudentDrivenDistillationConfig:
    dagger_rounds: int = 3
    initial_sample_stride: int = 2
    student_sample_stride: int = 1
    validation_aircraft_fraction: float = 0.2
    split_strategy: str = "aircraft_holdout"
    validation_plant_ids: tuple[str, ...] = ()
    student_training: DenseStudentTrainingConfig = field(
        default_factory=DenseStudentTrainingConfig
    )
    seed: int = 20260828
    device: str = "cpu"
    max_student_teacher_rmse_gap_deg_s: float = 1.0
    minimum_student_improvement_rate: float = 0.8
    maximum_student_harm_rate: float = 0.2
    maximum_student_peak_error_deg_s: float = 10.0
    maximum_mean_student_requested_force_variation_n: float = 120.0
    maximum_student_teacher_force_variation_ratio: float = 1.25

    def __post_init__(self) -> None:
        if self.dagger_rounds < 0:
            raise ValueError("dagger_rounds cannot be negative")
        if min(self.initial_sample_stride, self.student_sample_stride) <= 0:
            raise ValueError("distillation sample strides must be positive")
        if not 0 < self.validation_aircraft_fraction < 1:
            raise ValueError("validation_aircraft_fraction must be in (0, 1)")
        if self.split_strategy not in {
            "aircraft_holdout",
            "all_aircraft_command_holdout",
        }:
            raise ValueError("unsupported Student distillation split strategy")
        if len(set(self.validation_plant_ids)) != len(self.validation_plant_ids):
            raise ValueError("validation plant IDs must be unique")
        if self.validation_plant_ids and self.split_strategy != "aircraft_holdout":
            raise ValueError(
                "explicit validation plant IDs require the aircraft_holdout strategy"
            )
        if (
            self.max_student_teacher_rmse_gap_deg_s < 0
            or min(
                self.maximum_student_peak_error_deg_s,
                self.maximum_mean_student_requested_force_variation_n,
                self.maximum_student_teacher_force_variation_ratio,
            )
            <= 0
        ):
            raise ValueError("Student quality thresholds cannot be negative")
        if not 0 <= self.minimum_student_improvement_rate <= 1:
            raise ValueError("minimum Student improvement rate must be in [0, 1]")
        if not 0 <= self.maximum_student_harm_rate <= 1:
            raise ValueError("maximum Student harm rate must be in [0, 1]")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json_compatible(value: object) -> object:
    """Normalize tuples and other JSON containers before provenance comparisons."""

    return json.loads(json.dumps(value, ensure_ascii=False))


def _load_complete_artifact(
    path: Path,
    *,
    schema_version: str | tuple[str, ...],
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    supported = (schema_version,) if isinstance(schema_version, str) else schema_version
    if payload.get("schema_version") not in supported:
        raise ValueError(f"unsupported artifact schema in {path}")
    if payload.get("status") != "complete":
        raise ValueError(f"artifact is not complete: {path}")
    return payload


def _verified_student_training_artifact(
    report_path: Path,
    dataset_path: Path,
    config: DenseStudentTrainingConfig,
) -> dict[str, object] | None:
    report = _load_complete_artifact(
        report_path,
        schema_version="conditional_student_training_report_v1",
    )
    if report is None:
        return None
    if _json_compatible(report.get("config")) != _json_compatible(asdict(config)):
        raise ValueError(f"Student training config changed since {report_path}")
    dataset_reference = report.get("dataset_manifest")
    if not isinstance(dataset_reference, dict):
        raise ValueError(f"Student report is missing dataset provenance: {report_path}")
    if Path(str(dataset_reference.get("path"))).resolve() != dataset_path.resolve():
        raise ValueError(
            f"Student report references a different dataset: {report_path}"
        )
    if str(dataset_reference.get("sha256")) != sha256_file(dataset_path):
        raise ValueError(f"Student dataset hash changed since {report_path}")
    checkpoint = Path(str(report.get("checkpoint")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Student checkpoint is missing: {checkpoint}")
    return report


def _verified_evaluation_artifact(
    report_path: Path,
    student_checkpoint: Path,
    teacher_bank_path: Path,
) -> dict[str, object] | None:
    report = _load_complete_artifact(
        report_path,
        schema_version="dense_student_closed_loop_evaluation_v1",
    )
    if report is None:
        return None
    student_reference = report.get("student_checkpoint")
    teacher_reference = report.get("teacher_bank")
    if not isinstance(student_reference, dict) or not isinstance(
        teacher_reference, dict
    ):
        raise ValueError(f"evaluation provenance is incomplete: {report_path}")
    if str(student_reference.get("sha256")) != sha256_file(student_checkpoint):
        raise ValueError(f"evaluation uses a different Student: {report_path}")
    if str(teacher_reference.get("sha256")) != sha256_file(teacher_bank_path):
        raise ValueError(f"evaluation uses a different Teacher Bank: {report_path}")
    return report


def _load_complete_bank(path: Path) -> dict[str, object]:
    bank = json.loads(path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "specialist_teacher_bank_v1":
        raise ValueError("unsupported specialist Teacher Bank schema")
    if bank.get("status") != "complete":
        raise ValueError("student-driven distillation requires a complete Teacher Bank")
    teachers = bank.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Teacher Bank has no teachers")
    if any(entry.get("status") != "complete" for entry in teachers):
        raise ValueError("Teacher Bank contains an incomplete Teacher")
    return bank


def _profile_splits(
    teacher_count: int,
    plant_id: str,
    training_mode: str,
    duration_s: float,
    train_plants: set[str],
    split_strategy: str,
) -> tuple[tuple[RollRateCommandProfile, np.uint8], ...]:
    training = (
        specialist_step_commands(duration_s)
        if training_mode == "step"
        else specialist_extended_commands(duration_s)
    )
    if teacher_count == 1:
        return tuple((profile, TRAIN_SPLIT) for profile in training) + tuple(
            (profile, VALIDATION_SPLIT)
            for profile in specialist_evaluation_commands(duration_s)
        )
    if split_strategy == "all_aircraft_command_holdout":
        return tuple((profile, TRAIN_SPLIT) for profile in training) + tuple(
            (profile, VALIDATION_SPLIT)
            for profile in specialist_evaluation_commands(duration_s)
        )
    split = TRAIN_SPLIT if plant_id in train_plants else VALIDATION_SPLIT
    return tuple((profile, split) for profile in training)


def _collect_student_driven_profile(
    teacher_policy: object,
    student_policy: DenseStudentPolicy,
    record: object,
    training_config: object,
    profile: RollRateCommandProfile,
    *,
    sample_stride: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    environment = build_specialist_env(record, training_config, (profile,))
    observation, _ = environment.reset(seed=seed)
    observations: list[np.ndarray] = []
    teacher_actions: list[np.ndarray] = []
    student_actions: list[np.ndarray] = []
    policy_steps: list[int] = []
    squared_action_error = 0.0
    action_elements = 0
    episode_return = 0.0
    step = 0
    last_info: dict[str, object] = {}
    while True:
        student_action = np.asarray(
            student_policy.predict(observation, deterministic=True), dtype=np.float32
        )
        if step % sample_stride == 0:
            teacher_action = np.asarray(
                teacher_policy.predict(observation, deterministic=True),
                dtype=np.float32,
            )
            observations.append(observation.copy())
            teacher_actions.append(teacher_action.copy())
            student_actions.append(student_action.copy())
            policy_steps.append(step)
            squared_action_error += float(
                np.square(student_action - teacher_action).sum()
            )
            action_elements += student_action.size
        observation, reward, terminated, truncated, last_info = environment.step(
            student_action
        )
        episode_return += reward
        step += 1
        if terminated or truncated:
            break
    observations_array = np.asarray(observations, dtype=np.float32)
    teacher_action_array = np.asarray(teacher_actions, dtype=np.float32)
    student_action_array = np.asarray(student_actions, dtype=np.float32)
    policy_step_array = np.asarray(policy_steps, dtype=np.int32)
    if len(student_action_array) > 1:
        step_delta = np.diff(policy_step_array).astype(np.float32)[:, None]
        student_rate = np.diff(student_action_array, axis=0) / step_delta
        teacher_rate = np.diff(teacher_action_array, axis=0) / step_delta
        action_delta_rmse = float(
            np.sqrt(np.mean(np.square(student_rate - teacher_rate)))
        )
    else:
        action_delta_rmse = 0.0
    return (
        observations_array,
        teacher_action_array,
        student_action_array,
        policy_step_array,
        {
            "visited_action_rmse": float(
                np.sqrt(squared_action_error / max(action_elements, 1))
            ),
            "student_episode_return": float(episode_return),
            "student_saturation_fraction": float(
                last_info.get("action_saturation_fraction", 0.0)
            ),
            "visited_action_delta_rmse_per_policy_step": action_delta_rmse,
            "student_normalized_action_total_variation": float(
                np.sum(np.abs(np.diff(student_action_array, axis=0)))
            ),
            "teacher_on_visited_states_normalized_action_total_variation": float(
                np.sum(np.abs(np.diff(teacher_action_array, axis=0)))
            ),
            "hard_tracking_state_fraction": float(
                np.mean(np.abs(observations_array[:, 3]) >= 0.2)
            ),
        },
    )


def _prior_shards(
    previous_manifest_path: Path,
    destination: Path,
) -> list[dict[str, object]]:
    previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    shards = previous.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("previous distillation manifest has no shards")
    references: list[dict[str, object]] = []
    for shard in shards:
        absolute_path = (previous_manifest_path.parent / str(shard["path"])).resolve()
        expected_hash = str(shard["sha256"])
        if sha256_file(absolute_path) != expected_hash:
            raise ValueError(f"prior distillation shard hash changed: {absolute_path}")
        references.append(
            {
                **shard,
                "path": os.path.relpath(absolute_path, destination),
                "collection_round": int(shard.get("collection_round", 0)),
                "driver": str(shard.get("driver", "teacher")),
            }
        )
    return references


def collect_student_driven_round(
    teacher_bank_path: str | Path,
    student_checkpoint_path: str | Path,
    previous_dataset_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    round_index: int,
    sample_stride: int,
    seed: int,
    device: str,
) -> dict[str, object]:
    """Roll out the Student, label visited states with the matching Teacher."""

    if round_index <= 0 or sample_stride <= 0:
        raise ValueError(
            "student-driven rounds start at one and require a positive stride"
        )
    bank_path = Path(teacher_bank_path)
    checkpoint_path = Path(student_checkpoint_path)
    previous_path = Path(previous_dataset_manifest_path)
    destination = Path(output_dir)
    shard_dir = destination / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    bank = _load_complete_bank(bank_path)
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if previous.get("schema_version") not in SUPPORTED_DISTILLATION_DATASET_SCHEMAS:
        raise ValueError("unsupported prior distillation dataset schema")
    model, _ = load_dense_student(checkpoint_path, device=device)
    teachers = bank["teachers"]
    train_plants = set(previous["train_plant_ids"])
    validation_plants = set(previous["validation_plant_ids"])
    command_ids = list(previous["command_ids"])
    command_lookup = {command_id: index for index, command_id in enumerate(command_ids)}
    prior_shards = _prior_shards(previous_path, destination)
    new_shards: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    new_train_rows = 0
    new_validation_rows = 0
    observation_dim: int | None = None

    for teacher_index, entry in enumerate(teachers):
        actor_path = bank_path.parent / str(entry["actor_checkpoint"])
        teacher_policy, record, training_config, teacher_payload = (
            load_specialist_actor(actor_path, device=device)
        )
        if int(teacher_payload["actor_observation_dim"]) != model.observation_dim:
            raise ValueError("Student and Teacher observation contracts differ")
        observation_dim = model.observation_dim
        student_policy = DenseStudentPolicy(model, record.parameters, device=device)
        profile_splits = _profile_splits(
            len(teachers),
            record.plant_id,
            training_config.command_mode,
            training_config.episode_duration_s,
            train_plants,
            str(previous["split_strategy"]),
        )
        observation_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        command_parts: list[np.ndarray] = []
        split_parts: list[np.ndarray] = []
        episode_parts: list[np.ndarray] = []
        policy_step_parts: list[np.ndarray] = []
        driver_action_parts: list[np.ndarray] = []
        for profile_index, (profile, split_code) in enumerate(profile_splits):
            if profile.command_id not in command_lookup:
                command_lookup[profile.command_id] = len(command_ids)
                command_ids.append(profile.command_id)
            (
                observations,
                actions,
                driver_actions,
                policy_steps,
                profile_diagnostics,
            ) = _collect_student_driven_profile(
                teacher_policy,
                student_policy,
                record,
                training_config,
                profile,
                sample_stride=sample_stride,
                seed=seed
                + round_index * 100_000
                + teacher_index * 1000
                + profile_index,
            )
            row_count = len(observations)
            observation_parts.append(observations)
            action_parts.append(actions)
            command_parts.append(
                np.full(row_count, command_lookup[profile.command_id], dtype=np.int32)
            )
            split_parts.append(np.full(row_count, split_code, dtype=np.uint8))
            episode_parts.append(np.full(row_count, profile_index, dtype=np.int64))
            policy_step_parts.append(policy_steps)
            driver_action_parts.append(driver_actions)
            diagnostics.append(
                {
                    "plant_id": record.plant_id,
                    "command_id": profile.command_id,
                    "split": "train" if split_code == TRAIN_SPLIT else "validation",
                    **profile_diagnostics,
                }
            )

        observations = np.concatenate(observation_parts)
        actions = np.concatenate(action_parts)
        split_codes = np.concatenate(split_parts)
        rows = len(observations)
        theta = normalize_aircraft_parameters(record.parameters)
        arrays = DistillationArrays(
            observations=observations,
            aircraft_parameters=np.repeat(theta[None, :], rows, axis=0),
            teacher_actions=actions,
            plant_indices=np.full(rows, teacher_index, dtype=np.int32),
            command_indices=np.concatenate(command_parts),
            split_codes=split_codes,
            episode_indices=np.concatenate(episode_parts),
            policy_step_indices=np.concatenate(policy_step_parts),
            driver_actions=np.concatenate(driver_action_parts),
        )
        shard_path = save_distillation_shard(
            shard_dir / f"{teacher_index:04d}-{record.plant_id}.npz", arrays
        )
        train_rows = int(np.sum(split_codes == TRAIN_SPLIT))
        validation_rows = int(np.sum(split_codes == VALIDATION_SPLIT))
        new_train_rows += train_rows
        new_validation_rows += validation_rows
        new_shards.append(
            {
                "path": str(shard_path.relative_to(destination)),
                "sha256": sha256_file(shard_path),
                "plant_index": teacher_index,
                "plant_id": record.plant_id,
                "teacher_actor": str(actor_path.resolve()),
                "teacher_actor_sha256": sha256_file(actor_path),
                "rows": rows,
                "train_rows": train_rows,
                "validation_rows": validation_rows,
                "episode_count": len(profile_splits),
                "temporal_pair_count": rows - len(profile_splits),
                "collection_round": round_index,
                "driver": "student",
                "labeler": "matching_specialist_teacher",
            }
        )
        aircraft_diagnostics = diagnostics[-len(profile_splits) :]
        print(
            json.dumps(
                {
                    "event": "student_driven_collection_aircraft",
                    "round": round_index,
                    "plant_id": record.plant_id,
                    "aircraft_index": teacher_index,
                    "aircraft_count": len(teachers),
                    "rows": rows,
                    "mean_visited_action_rmse": float(
                        np.mean(
                            [row["visited_action_rmse"] for row in aircraft_diagnostics]
                        )
                    ),
                    "mean_visited_action_delta_rmse_per_policy_step": float(
                        np.mean(
                            [
                                row["visited_action_delta_rmse_per_policy_step"]
                                for row in aircraft_diagnostics
                            ]
                        )
                    ),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    all_shards = prior_shards + new_shards
    manifest: dict[str, object] = {
        "schema_version": DISTILLATION_DATASET_SCHEMA_V2,
        "status": "complete",
        "source": git_source_revision(),
        "collection_method": "student_driven_dagger",
        "collection_round": round_index,
        "driver_checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "teacher_bank": {
            "path": str(bank_path.resolve()),
            "sha256": sha256_file(bank_path),
        },
        "previous_dataset": {
            "path": str(previous_path.resolve()),
            "sha256": sha256_file(previous_path),
        },
        "split_strategy": previous["split_strategy"],
        "aircraft_split_method": previous.get("aircraft_split_method"),
        "observation_dim": observation_dim,
        "actor_observation_contract": previous.get("actor_observation_contract"),
        "aircraft_parameter_dim": len(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_names": list(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_normalization": AIRCRAFT_PARAMETER_NORMALIZATION,
        "action_dim": 1,
        "action_definition": "normalized_direct_full_F_as",
        "temporal_contract": {
            "episode_index": "one deterministic command rollout within one shard",
            "policy_step_index": "environment policy step before action application",
            "driver_action": "Student action used to visit the recorded state",
            "predecessor_scope": "same shard and episode only",
        },
        "command_ids": command_ids,
        "train_plant_ids": sorted(train_plants),
        "validation_plant_ids": sorted(validation_plants),
        "row_count": int(sum(int(shard["rows"]) for shard in all_shards)),
        "train_rows": int(sum(int(shard["train_rows"]) for shard in all_shards)),
        "validation_rows": int(
            sum(int(shard["validation_rows"]) for shard in all_shards)
        ),
        "episode_count": int(
            sum(int(shard.get("episode_count", 0)) for shard in all_shards)
        ),
        "temporal_pair_count": int(
            sum(int(shard.get("temporal_pair_count", 0)) for shard in all_shards)
        ),
        "new_rows": new_train_rows + new_validation_rows,
        "new_train_rows": new_train_rows,
        "new_validation_rows": new_validation_rows,
        "sample_stride": sample_stride,
        "shards": all_shards,
        "rollout_diagnostics": diagnostics,
    }
    _write_json(destination / "dataset.json", manifest)
    return manifest


def _validation_summary(evaluation: dict[str, object]) -> dict[str, object]:
    by_split = evaluation["by_distillation_split"]
    if "validation_aircraft" in by_split:
        return by_split["validation_aircraft"]
    if "single_aircraft_command_holdout" in by_split:
        return by_split["single_aircraft_command_holdout"]
    if "all_aircraft_command_holdout" in by_split:
        return by_split["all_aircraft_command_holdout"]
    return evaluation


def _round_score(evaluation: dict[str, object]) -> tuple[float, float, float]:
    summary = _validation_summary(evaluation)
    return (
        float(summary["mean_student_tracking_rmse_deg_s"]),
        float(summary["maximum_student_peak_error_deg_s"]),
        float(summary["mean_student_requested_force_total_variation_n"]),
    )


def _quality_gate_fallback_score(
    row: dict[str, object], thresholds: dict[str, float]
) -> tuple[float, ...]:
    """Rank failed-gate rounds by safety violations before tracking accuracy."""

    summary = _validation_summary(row["closed_loop_evaluation"])
    checks = row["quality_checks"]
    comparisons = (
        (
            "student_teacher_rmse_gap",
            float(np.rad2deg(summary["median_student_minus_teacher_rmse_rad_s"])),
            thresholds["max_student_teacher_rmse_gap_deg_s"],
            "maximum",
            max(thresholds["max_student_teacher_rmse_gap_deg_s"], 1.0),
        ),
        (
            "student_improvement_rate",
            float(summary["student_improvement_rate"]),
            thresholds["minimum_student_improvement_rate"],
            "minimum",
            1.0,
        ),
        (
            "student_harm_rate",
            float(summary["student_harm_rate"]),
            thresholds["maximum_student_harm_rate"],
            "maximum",
            1.0,
        ),
        (
            "student_peak_error",
            float(summary["maximum_student_peak_error_deg_s"]),
            thresholds["maximum_student_peak_error_deg_s"],
            "maximum",
            thresholds["maximum_student_peak_error_deg_s"],
        ),
        (
            "student_requested_force_variation",
            float(summary["mean_student_requested_force_total_variation_n"]),
            thresholds["maximum_mean_student_requested_force_variation_n"],
            "maximum",
            thresholds["maximum_mean_student_requested_force_variation_n"],
        ),
        (
            "student_teacher_force_variation_ratio",
            float(summary["student_teacher_requested_force_variation_ratio"]),
            thresholds["maximum_student_teacher_force_variation_ratio"],
            "maximum",
            thresholds["maximum_student_teacher_force_variation_ratio"],
        ),
    )
    violations: list[float] = []
    for check_name, observed, threshold, direction, scale in comparisons:
        if bool(checks[check_name]):
            violations.append(0.0)
        elif direction == "maximum":
            violations.append(max(observed - threshold, 0.0) / scale)
        else:
            violations.append(max(threshold - observed, 0.0) / scale)
    return (
        float(sum(not bool(value) for value in checks.values())),
        float(sum(violations)),
        float(max(violations, default=0.0)),
        *_round_score(row["closed_loop_evaluation"]),
    )


def _select_student_round(
    rounds: list[dict[str, object]], thresholds: dict[str, float]
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    eligible_rounds = [row for row in rounds if all(row["quality_checks"].values())]
    if eligible_rounds:
        return (
            min(
                eligible_rounds,
                key=lambda row: _round_score(row["closed_loop_evaluation"]),
            ),
            eligible_rounds,
            "validation_mean_student_rmse_then_peak_then_force_variation_among_quality_eligible_rounds",
        )
    return (
        min(
            rounds,
            key=lambda row: _quality_gate_fallback_score(row, thresholds),
        ),
        [],
        "fewest_quality_gate_violations_then_normalized_excess_then_validation_tracking_fallback",
    )


def _student_quality_checks(
    evaluation: dict[str, object], config: StudentDrivenDistillationConfig
) -> dict[str, bool]:
    summary = _validation_summary(evaluation)
    return {
        "student_teacher_rmse_gap": float(
            np.rad2deg(summary["median_student_minus_teacher_rmse_rad_s"])
        )
        <= config.max_student_teacher_rmse_gap_deg_s,
        "student_improvement_rate": float(summary["student_improvement_rate"])
        >= config.minimum_student_improvement_rate,
        "student_harm_rate": float(summary["student_harm_rate"])
        <= config.maximum_student_harm_rate,
        "student_peak_error": float(summary["maximum_student_peak_error_deg_s"])
        <= config.maximum_student_peak_error_deg_s,
        "student_requested_force_variation": float(
            summary["mean_student_requested_force_total_variation_n"]
        )
        <= config.maximum_mean_student_requested_force_variation_n,
        "student_teacher_force_variation_ratio": float(
            summary["student_teacher_requested_force_variation_ratio"]
        )
        <= config.maximum_student_teacher_force_variation_ratio,
    }


def _save_progress_plot(rounds: list[dict[str, object]], output_path: Path) -> None:
    indices = np.asarray([row["round"] for row in rounds], dtype=int)
    action_rmse = np.asarray(
        [
            row["student_training"]["validation_metrics"]["action_rmse"]
            for row in rounds
        ],
        dtype=float,
    )
    gap_deg_s = np.asarray(
        [
            np.rad2deg(
                _validation_summary(row["closed_loop_evaluation"])[
                    "median_student_minus_teacher_rmse_rad_s"
                ]
            )
            for row in rounds
        ],
        dtype=float,
    )
    variation_ratio = np.asarray(
        [
            _validation_summary(row["closed_loop_evaluation"])[
                "student_teacher_requested_force_variation_ratio"
            ]
            for row in rounds
        ],
        dtype=float,
    )
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="constrained")
    axes[0].plot(indices, action_rmse, marker="o", color="#2878b5")
    axes[0].set_ylabel("Validation action RMSE")
    axes[0].grid(alpha=0.25)
    axes[1].plot(indices, gap_deg_s, marker="o", color="#c82423")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Distillation round")
    axes[1].set_ylabel("Student - Teacher RMSE (deg/s)")
    axes[1].grid(alpha=0.25)
    axes[2].plot(indices, variation_ratio, marker="o", color="#2f8f46")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("Distillation round")
    axes[2].set_ylabel("Student / Teacher requested-force TV")
    axes[2].grid(alpha=0.25)
    figure.suptitle("Student-driven distillation progress")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_round_csv(rounds: list[dict[str, object]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "round",
                "driver",
                "dataset_rows",
                "validation_action_rmse",
                "validation_student_minus_teacher_rmse_deg_s",
                "student_improvement_rate",
                "student_harm_rate",
                "student_peak_error_deg_s",
                "student_requested_force_variation_n",
                "student_teacher_force_variation_ratio",
            ),
        )
        writer.writeheader()
        for row in rounds:
            evaluation = _validation_summary(row["closed_loop_evaluation"])
            writer.writerow(
                {
                    "round": row["round"],
                    "driver": row["driver"],
                    "dataset_rows": row["dataset"]["row_count"],
                    "validation_action_rmse": row["student_training"][
                        "validation_metrics"
                    ]["action_rmse"],
                    "validation_student_minus_teacher_rmse_deg_s": np.rad2deg(
                        evaluation["median_student_minus_teacher_rmse_rad_s"]
                    ),
                    "student_improvement_rate": evaluation["student_improvement_rate"],
                    "student_harm_rate": evaluation["student_harm_rate"],
                    "student_peak_error_deg_s": evaluation[
                        "maximum_student_peak_error_deg_s"
                    ],
                    "student_requested_force_variation_n": evaluation[
                        "mean_student_requested_force_total_variation_n"
                    ],
                    "student_teacher_force_variation_ratio": evaluation[
                        "student_teacher_requested_force_variation_ratio"
                    ],
                }
            )


def run_student_driven_distillation(
    teacher_bank_path: str | Path,
    output_dir: str | Path,
    config: StudentDrivenDistillationConfig = StudentDrivenDistillationConfig(),
    *,
    resume: bool = False,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    """Run initial Teacher collection followed by pure Student-driven DAgger rounds."""

    started = time.perf_counter()
    bank_path = Path(teacher_bank_path)
    _load_complete_bank(bank_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    pipeline_path = destination / "pipeline_report.json"
    expected_config = _json_compatible(asdict(config))
    bootstrap_checkpoint: dict[str, object] | None = None
    if initial_checkpoint_path is not None:
        initial_path = Path(initial_checkpoint_path)
        if not initial_path.is_file():
            raise FileNotFoundError(
                f"initial Student checkpoint is missing: {initial_path}"
            )
        bootstrap_checkpoint = {
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
        }
    rounds: list[dict[str, object]] = []
    if resume and pipeline_path.is_file():
        prior = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if prior.get("schema_version") != "student_driven_distillation_pipeline_v1":
            raise ValueError("unsupported Student-driven pipeline schema")
        if prior.get("status") in {"complete", "quality_gate_failed"}:
            return prior
        if prior.get("status") != "running":
            raise ValueError("only a running Student-driven pipeline can be resumed")
        if Path(str(prior.get("teacher_bank"))).resolve() != bank_path.resolve():
            raise ValueError("Teacher Bank changed since the interrupted run")
        if _json_compatible(prior.get("config")) != expected_config:
            raise ValueError("pipeline config changed since the interrupted run")
        if prior.get("bootstrap_student_checkpoint") != bootstrap_checkpoint:
            raise ValueError("bootstrap Student changed since the interrupted run")
        prior_rounds = prior.get("rounds")
        if not isinstance(prior_rounds, list):
            raise ValueError("interrupted pipeline has invalid round metadata")
        rounds = prior_rounds
        if [row.get("round") for row in rounds] != list(range(len(rounds))):
            raise ValueError("completed Student-driven rounds are not contiguous")

    def write_running_report() -> None:
        _write_json(
            pipeline_path,
            {
                "schema_version": "student_driven_distillation_pipeline_v1",
                "status": "running",
                "teacher_bank": str(bank_path.resolve()),
                "bootstrap_student_checkpoint": bootstrap_checkpoint,
                "config": asdict(config),
                "rounds": rounds,
            },
        )

    if not resume or not pipeline_path.is_file():
        write_running_report()

    if rounds:
        first = rounds[0]
        dataset_path = Path(str(first["dataset_manifest"]))
        dataset = _load_complete_artifact(
            dataset_path,
            schema_version=SUPPORTED_DISTILLATION_DATASET_SCHEMAS,
        )
        if dataset is None:
            raise FileNotFoundError(
                f"completed round dataset is missing: {dataset_path}"
            )
        student_checkpoint = Path(str(first["student_checkpoint"]))
        evaluation_path = (
            student_checkpoint.parent.parent / "evaluation/evaluation.json"
        )
        if (
            _verified_evaluation_artifact(
                evaluation_path, student_checkpoint, bank_path
            )
            is None
        ):
            raise FileNotFoundError(
                f"completed round evaluation is missing: {evaluation_path}"
            )
    else:
        round_dir = destination / "round_000_teacher_driven"
        dataset_path = round_dir / "dataset/dataset.json"
        dataset = (
            _load_complete_artifact(
                dataset_path,
                schema_version=SUPPORTED_DISTILLATION_DATASET_SCHEMAS,
            )
            if resume
            else None
        )
        if dataset is None:
            dataset = collect_teacher_bank_data(
                bank_path,
                round_dir / "dataset",
                DistillationCollectionConfig(
                    sample_stride=config.initial_sample_stride,
                    validation_aircraft_fraction=config.validation_aircraft_fraction,
                    split_strategy=config.split_strategy,
                    validation_plant_ids=config.validation_plant_ids,
                    seed=config.seed,
                    device=config.device,
                ),
            )
            dataset["collection_method"] = "teacher_driven_initialization"
            dataset["collection_round"] = 0
            for shard in dataset["shards"]:
                shard["collection_round"] = 0
                shard["driver"] = "teacher"
            _write_json(dataset_path, dataset)
        student_training = (
            _verified_student_training_artifact(
                round_dir / "student/report.json",
                dataset_path,
                config.student_training,
            )
            if resume
            else None
        )
        if student_training is None:
            student_training = train_dense_student(
                dataset_path,
                round_dir / "student",
                config.student_training,
                initial_checkpoint_path=initial_checkpoint_path,
            )
        student_checkpoint = Path(str(student_training["checkpoint"]))
        evaluation = (
            _verified_evaluation_artifact(
                round_dir / "evaluation/evaluation.json",
                student_checkpoint,
                bank_path,
            )
            if resume
            else None
        )
        if evaluation is None:
            evaluation = evaluate_dense_student_bank(
                student_checkpoint,
                bank_path,
                round_dir / "evaluation",
                device=config.device,
            )
        rounds.append(
            {
                "round": 0,
                "driver": "teacher",
                "dataset": dataset,
                "dataset_manifest": str(dataset_path),
                "student_training": student_training,
                "student_checkpoint": str(student_checkpoint),
                "closed_loop_evaluation": evaluation,
            }
        )
        write_running_report()

    for round_index in range(1, config.dagger_rounds + 1):
        if round_index < len(rounds):
            completed = rounds[round_index]
            dataset_path = Path(str(completed["dataset_manifest"]))
            dataset = _load_complete_artifact(
                dataset_path,
                schema_version=SUPPORTED_DISTILLATION_DATASET_SCHEMAS,
            )
            if dataset is None:
                raise FileNotFoundError(
                    f"completed round dataset is missing: {dataset_path}"
                )
            student_checkpoint = Path(str(completed["student_checkpoint"]))
            evaluation_path = (
                student_checkpoint.parent.parent / "evaluation/evaluation.json"
            )
            if (
                _verified_evaluation_artifact(
                    evaluation_path, student_checkpoint, bank_path
                )
                is None
            ):
                raise FileNotFoundError(
                    f"completed round evaluation is missing: {evaluation_path}"
                )
            continue
        round_dir = destination / f"round_{round_index:03d}_student_driven"
        next_dataset_path = round_dir / "dataset/dataset.json"
        dataset = (
            _load_complete_artifact(
                next_dataset_path,
                schema_version=SUPPORTED_DISTILLATION_DATASET_SCHEMAS,
            )
            if resume
            else None
        )
        if dataset is None:
            dataset = collect_student_driven_round(
                bank_path,
                student_checkpoint,
                dataset_path,
                round_dir / "dataset",
                round_index=round_index,
                sample_stride=config.student_sample_stride,
                seed=config.seed,
                device=config.device,
            )
        dataset_path = round_dir / "dataset/dataset.json"
        student_training = (
            _verified_student_training_artifact(
                round_dir / "student/report.json",
                dataset_path,
                config.student_training,
            )
            if resume
            else None
        )
        if student_training is None:
            student_training = train_dense_student(
                dataset_path,
                round_dir / "student",
                config.student_training,
                initial_checkpoint_path=student_checkpoint,
            )
        student_checkpoint = Path(str(student_training["checkpoint"]))
        evaluation = (
            _verified_evaluation_artifact(
                round_dir / "evaluation/evaluation.json",
                student_checkpoint,
                bank_path,
            )
            if resume
            else None
        )
        if evaluation is None:
            evaluation = evaluate_dense_student_bank(
                student_checkpoint,
                bank_path,
                round_dir / "evaluation",
                device=config.device,
            )
        rounds.append(
            {
                "round": round_index,
                "driver": "student",
                "dataset": dataset,
                "dataset_manifest": str(dataset_path),
                "student_training": student_training,
                "student_checkpoint": str(student_checkpoint),
                "closed_loop_evaluation": evaluation,
            }
        )
        write_running_report()

    for row in rounds:
        row["quality_checks"] = _student_quality_checks(
            row["closed_loop_evaluation"], config
        )
    thresholds = {
        "max_student_teacher_rmse_gap_deg_s": config.max_student_teacher_rmse_gap_deg_s,
        "minimum_student_improvement_rate": config.minimum_student_improvement_rate,
        "maximum_student_harm_rate": config.maximum_student_harm_rate,
        "maximum_student_peak_error_deg_s": config.maximum_student_peak_error_deg_s,
        "maximum_mean_student_requested_force_variation_n": config.maximum_mean_student_requested_force_variation_n,
        "maximum_student_teacher_force_variation_ratio": config.maximum_student_teacher_force_variation_ratio,
    }
    best, eligible_rounds, selection_metric = _select_student_round(rounds, thresholds)
    best_evaluation = _validation_summary(best["closed_loop_evaluation"])
    gap_deg_s = float(
        np.rad2deg(best_evaluation["median_student_minus_teacher_rmse_rad_s"])
    )
    checks = best["quality_checks"]
    quality_gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "median_student_minus_teacher_rmse_deg_s": gap_deg_s,
            "student_improvement_rate": best_evaluation["student_improvement_rate"],
            "student_harm_rate": best_evaluation["student_harm_rate"],
            "maximum_student_peak_error_deg_s": best_evaluation[
                "maximum_student_peak_error_deg_s"
            ],
            "mean_student_requested_force_total_variation_n": best_evaluation[
                "mean_student_requested_force_total_variation_n"
            ],
            "student_teacher_requested_force_variation_ratio": best_evaluation[
                "student_teacher_requested_force_variation_ratio"
            ],
        },
        "thresholds": thresholds,
    }
    final_dir = destination / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = final_dir / Path(str(best["student_checkpoint"])).name
    shutil.copy2(best["student_checkpoint"], final_checkpoint)
    best_evaluation_dir = Path(best["student_checkpoint"]).parent.parent / "evaluation"
    shutil.copytree(best_evaluation_dir, final_dir / "evaluation", dirs_exist_ok=True)
    _save_progress_plot(rounds, destination / "distillation_progress.png")
    _write_round_csv(rounds, destination / "round_metrics.csv")
    report: dict[str, object] = {
        "schema_version": "student_driven_distillation_pipeline_v1",
        "status": "complete" if quality_gate["passed"] else "quality_gate_failed",
        "source": git_source_revision(),
        "teacher_bank": {
            "path": str(bank_path.resolve()),
            "sha256": sha256_file(bank_path),
        },
        "bootstrap_student_checkpoint": bootstrap_checkpoint,
        "config": asdict(config),
        "round_count": len(rounds),
        "student_driven_round_count": config.dagger_rounds,
        "best_round": best["round"],
        "quality_eligible_rounds": [row["round"] for row in eligible_rounds],
        "selection_metric": selection_metric,
        "quality_gate": quality_gate,
        "final_checkpoint": {
            "path": str(final_checkpoint),
            "sha256": sha256_file(final_checkpoint),
        },
        "elapsed_s": time.perf_counter() - started,
        "rounds": rounds,
        "artifacts": {
            "progress_plot": str(destination / "distillation_progress.png"),
            "round_metrics_csv": str(destination / "round_metrics.csv"),
            "final_evaluation": str(final_dir / "evaluation/evaluation.json"),
        },
    }
    _write_json(pipeline_path, report)
    return report


def reselect_student_driven_distillation(output_dir: str | Path) -> dict[str, object]:
    """Reapply the current selection rule to an already completed run."""

    destination = Path(output_dir)
    pipeline_path = destination / "pipeline_report.json"
    report = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "student_driven_distillation_pipeline_v1":
        raise ValueError("unsupported Student-driven distillation report schema")
    rounds = report.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Student-driven distillation report has no rounds")
    if any("quality_checks" not in row for row in rounds):
        raise ValueError("Student-driven rounds have not completed quality checks")

    prior_gate = report.get("quality_gate", {})
    thresholds = prior_gate.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("Student-driven report is missing quality thresholds")
    best, eligible_rounds, selection_metric = _select_student_round(rounds, thresholds)
    best_evaluation = _validation_summary(best["closed_loop_evaluation"])
    checks = best["quality_checks"]
    quality_gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "median_student_minus_teacher_rmse_deg_s": float(
                np.rad2deg(best_evaluation["median_student_minus_teacher_rmse_rad_s"])
            ),
            "student_improvement_rate": best_evaluation["student_improvement_rate"],
            "student_harm_rate": best_evaluation["student_harm_rate"],
            "maximum_student_peak_error_deg_s": best_evaluation[
                "maximum_student_peak_error_deg_s"
            ],
            "mean_student_requested_force_total_variation_n": best_evaluation[
                "mean_student_requested_force_total_variation_n"
            ],
            "student_teacher_requested_force_variation_ratio": best_evaluation[
                "student_teacher_requested_force_variation_ratio"
            ],
        },
        "thresholds": thresholds,
    }

    final_dir = destination / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = final_dir / Path(str(best["student_checkpoint"])).name
    shutil.copy2(best["student_checkpoint"], final_checkpoint)
    best_evaluation_dir = Path(best["student_checkpoint"]).parent.parent / "evaluation"
    shutil.copytree(best_evaluation_dir, final_dir / "evaluation", dirs_exist_ok=True)
    _save_progress_plot(rounds, destination / "distillation_progress.png")
    _write_round_csv(rounds, destination / "round_metrics.csv")

    report.update(
        {
            "status": "complete" if quality_gate["passed"] else "quality_gate_failed",
            "source": git_source_revision(),
            "best_round": best["round"],
            "quality_eligible_rounds": [row["round"] for row in eligible_rounds],
            "selection_metric": selection_metric,
            "quality_gate": quality_gate,
            "final_checkpoint": {
                "path": str(final_checkpoint),
                "sha256": sha256_file(final_checkpoint),
            },
        }
    )
    _write_json(pipeline_path, report)
    return report
