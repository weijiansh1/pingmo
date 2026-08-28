"""Collect deterministic specialist behavior into aircraft-level shards."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from src.aircraft.sampler import PlantRecord
from src.context.aircraft_parameters import (
    AIRCRAFT_PARAMETER_NAMES,
    AIRCRAFT_PARAMETER_NORMALIZATION,
    normalize_aircraft_parameters,
)
from src.distillation.dataset import (
    DistillationArrays,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    save_distillation_shard,
)
from src.envs.roll_rate_commands import (
    RollRateCommandProfile,
    specialist_evaluation_commands,
    specialist_extended_commands,
    specialist_step_commands,
)
from src.teacher.specialist.trainer import (
    PredictivePolicy,
    SpecialistTrainingConfig,
    build_specialist_env,
    load_specialist_actor,
)
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class DistillationCollectionConfig:
    sample_stride: int = 10
    validation_aircraft_fraction: float = 0.2
    seed: int = 20260828
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if not 0 < self.validation_aircraft_fraction < 1:
            raise ValueError("validation_aircraft_fraction must be in (0, 1)")


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aircraft_split(teacher_count: int, fraction: float, seed: int) -> set[int]:
    if teacher_count < 2:
        return set()
    validation_count = min(teacher_count - 1, max(1, int(round(teacher_count * fraction))))
    shuffled = np.random.default_rng(seed).permutation(teacher_count)
    return {int(index) for index in shuffled[:validation_count]}


def _collect_profile(
    policy: PredictivePolicy,
    record: PlantRecord,
    training_config: SpecialistTrainingConfig,
    profile: RollRateCommandProfile,
    *,
    sample_stride: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    env = build_specialist_env(record, training_config, (profile,))
    observation, _ = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    step = 0
    while True:
        action = np.asarray(policy.predict(observation, deterministic=True), dtype=np.float32)
        if step % sample_stride == 0:
            observations.append(observation.copy())
            actions.append(action.copy())
        observation, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    return np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def collect_teacher_bank_data(
    teacher_bank_path: str | Path,
    output_dir: str | Path,
    config: DistillationCollectionConfig = DistillationCollectionConfig(),
) -> dict[str, object]:
    """Collect `(observation, theta, Teacher action)` without feeding theta to Teachers."""

    bank_path = Path(teacher_bank_path)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "specialist_teacher_bank_v1" or bank.get("status") != "complete":
        raise ValueError("a complete specialist Teacher Bank manifest is required")
    teachers = bank.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Teacher Bank has no completed teachers")
    if any(entry.get("status") != "complete" for entry in teachers):
        raise ValueError("Teacher Bank contains an incomplete specialist")

    destination = Path(output_dir)
    shard_dir = destination / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    validation_aircraft = _aircraft_split(
        len(teachers), config.validation_aircraft_fraction, config.seed
    )
    command_ids: list[str] = []
    command_lookup: dict[str, int] = {}
    shard_entries: list[dict[str, object]] = []
    train_plants: list[str] = []
    validation_plants: list[str] = []
    expected_observation_dim: int | None = None
    total_rows = 0
    train_rows = 0
    validation_rows = 0

    for teacher_index, entry in enumerate(teachers):
        actor_path = bank_path.parent / str(entry["actor_checkpoint"])
        policy, record, training_config, actor_payload = load_specialist_actor(
            actor_path,
            device=config.device,
        )
        observation_dim = int(actor_payload["actor_observation_dim"])
        if expected_observation_dim is None:
            expected_observation_dim = observation_dim
        elif observation_dim != expected_observation_dim:
            raise ValueError("all specialist Teachers must share one observation contract")

        training_profiles = (
            specialist_step_commands(training_config.episode_duration_s)
            if training_config.command_mode == "step"
            else specialist_extended_commands(training_config.episode_duration_s)
        )
        evaluation_profiles = specialist_evaluation_commands(training_config.episode_duration_s)
        if len(teachers) == 1:
            profile_splits = [(profile, TRAIN_SPLIT) for profile in training_profiles]
            profile_splits += [(profile, VALIDATION_SPLIT) for profile in evaluation_profiles]
            train_plants.append(record.plant_id)
            validation_plants.append(record.plant_id)
        else:
            split_code = VALIDATION_SPLIT if teacher_index in validation_aircraft else TRAIN_SPLIT
            profile_splits = [(profile, split_code) for profile in training_profiles]
            target = validation_plants if split_code == VALIDATION_SPLIT else train_plants
            target.append(record.plant_id)

        observation_parts: list[np.ndarray] = []
        action_parts: list[np.ndarray] = []
        command_parts: list[np.ndarray] = []
        split_parts: list[np.ndarray] = []
        for profile_index, (profile, split_code) in enumerate(profile_splits):
            if profile.command_id not in command_lookup:
                command_lookup[profile.command_id] = len(command_ids)
                command_ids.append(profile.command_id)
            observations, actions = _collect_profile(
                policy,
                record,
                training_config,
                profile,
                sample_stride=config.sample_stride,
                seed=config.seed + teacher_index * 1000 + profile_index,
            )
            row_count = len(observations)
            observation_parts.append(observations)
            action_parts.append(actions)
            command_parts.append(
                np.full(row_count, command_lookup[profile.command_id], dtype=np.int32)
            )
            split_parts.append(np.full(row_count, split_code, dtype=np.uint8))

        observations = np.concatenate(observation_parts)
        actions = np.concatenate(action_parts)
        split_codes = np.concatenate(split_parts)
        theta = normalize_aircraft_parameters(record.parameters)
        rows = len(observations)
        arrays = DistillationArrays(
            observations=observations,
            aircraft_parameters=np.repeat(theta[None, :], rows, axis=0),
            teacher_actions=actions,
            plant_indices=np.full(rows, teacher_index, dtype=np.int32),
            command_indices=np.concatenate(command_parts),
            split_codes=split_codes,
        )
        shard_name = f"{teacher_index:04d}-{record.plant_id}.npz"
        shard_path = save_distillation_shard(shard_dir / shard_name, arrays)
        shard_train_rows = int(np.sum(split_codes == TRAIN_SPLIT))
        shard_validation_rows = int(np.sum(split_codes == VALIDATION_SPLIT))
        train_rows += shard_train_rows
        validation_rows += shard_validation_rows
        total_rows += rows
        shard_entries.append(
            {
                "path": str(shard_path.relative_to(destination)),
                "sha256": sha256_file(shard_path),
                "plant_index": teacher_index,
                "plant_id": record.plant_id,
                "teacher_actor": str(actor_path.resolve()),
                "teacher_actor_sha256": sha256_file(actor_path),
                "rows": rows,
                "train_rows": shard_train_rows,
                "validation_rows": shard_validation_rows,
            }
        )

    split_strategy = (
        "single_aircraft_command_holdout"
        if len(teachers) == 1
        else "aircraft_holdout"
    )
    manifest: dict[str, object] = {
        "schema_version": "specialist_distillation_dataset_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {
            "path": str(bank_path.resolve()),
            "sha256": sha256_file(bank_path),
        },
        "config": asdict(config),
        "split_strategy": split_strategy,
        "observation_dim": expected_observation_dim,
        "aircraft_parameter_dim": len(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_names": list(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_normalization": AIRCRAFT_PARAMETER_NORMALIZATION,
        "action_dim": 1,
        "action_definition": "normalized_direct_full_F_as",
        "command_ids": command_ids,
        "train_plant_ids": train_plants,
        "validation_plant_ids": validation_plants,
        "row_count": total_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "shards": shard_entries,
    }
    _write_json(destination / "dataset.json", manifest)
    return manifest
