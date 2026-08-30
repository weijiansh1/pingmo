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
    DISTILLATION_DATASET_SCHEMA_V2,
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
    sample_stride: int = 2
    validation_aircraft_fraction: float = 0.2
    split_strategy: str = "aircraft_holdout"
    validation_plant_ids: tuple[str, ...] = ()
    seed: int = 20260828
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if not 0 < self.validation_aircraft_fraction < 1:
            raise ValueError("validation_aircraft_fraction must be in (0, 1)")
        if self.split_strategy not in {
            "aircraft_holdout",
            "all_aircraft_command_holdout",
        }:
            raise ValueError("unsupported distillation split strategy")
        if len(set(self.validation_plant_ids)) != len(self.validation_plant_ids):
            raise ValueError("validation plant IDs must be unique")
        if self.validation_plant_ids and self.split_strategy != "aircraft_holdout":
            raise ValueError(
                "explicit validation plant IDs require the aircraft_holdout strategy"
            )


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _aircraft_split(
    teachers: list[dict[str, object]],
    fraction: float,
    seed: int,
    explicit_validation_plant_ids: tuple[str, ...] = (),
) -> set[int]:
    teacher_count = len(teachers)
    if teacher_count < 2:
        return set()
    if explicit_validation_plant_ids:
        index_by_plant_id = {
            str(entry["plant_id"]): index for index, entry in enumerate(teachers)
        }
        missing = sorted(set(explicit_validation_plant_ids) - set(index_by_plant_id))
        if missing:
            raise ValueError(f"validation plant IDs are absent from Teacher Bank: {missing}")
        validation = {
            index_by_plant_id[plant_id] for plant_id in explicit_validation_plant_ids
        }
        if len(validation) == teacher_count:
            raise ValueError("aircraft holdout must retain at least one training aircraft")
        groups: dict[str, set[int]] = {}
        for index, entry in enumerate(teachers):
            groups.setdefault(str(entry.get("quality_region", "unknown")), set()).add(
                index
            )
        emptied_groups = sorted(
            region for region, indices in groups.items() if indices <= validation
        )
        if emptied_groups:
            raise ValueError(
                "explicit aircraft holdout removes every training aircraft from "
                f"quality regions: {emptied_groups}"
            )
        return validation
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index, entry in enumerate(teachers):
        groups.setdefault(str(entry.get("quality_region", "unknown")), []).append(index)
    validation: set[int] = set()
    for indices in groups.values():
        if len(indices) < 2:
            continue
        count = min(len(indices) - 1, max(1, int(round(len(indices) * fraction))))
        selected = rng.choice(indices, size=count, replace=False)
        validation.update(int(index) for index in selected)
    if not validation:
        validation.add(int(rng.integers(teacher_count)))
    if len(validation) == teacher_count:
        validation.remove(next(iter(validation)))
    return validation


def _collect_profile(
    policy: PredictivePolicy,
    record: PlantRecord,
    training_config: SpecialistTrainingConfig,
    profile: RollRateCommandProfile,
    *,
    sample_stride: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    env = build_specialist_env(record, training_config, (profile,))
    observation, _ = env.reset(seed=seed)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    policy_steps: list[int] = []
    step = 0
    while True:
        action = np.asarray(policy.predict(observation, deterministic=True), dtype=np.float32)
        if step % sample_stride == 0:
            observations.append(observation.copy())
            actions.append(action.copy())
            policy_steps.append(step)
        observation, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            break
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(policy_steps, dtype=np.int32),
    )


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
    validation_aircraft = (
        _aircraft_split(
            teachers,
            config.validation_aircraft_fraction,
            config.seed,
            config.validation_plant_ids,
        )
        if config.split_strategy == "aircraft_holdout"
        else set()
    )
    command_ids: list[str] = []
    command_lookup: dict[str, int] = {}
    shard_entries: list[dict[str, object]] = []
    train_plants: list[str] = []
    validation_plants: list[str] = []
    expected_observation_dim: int | None = None
    expected_observation_contract: dict[str, object] | None = None
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
        observation_contract = actor_payload.get("actor_observation_contract")
        if not isinstance(observation_contract, dict):
            observation_contract = build_specialist_env(
                record, training_config
            ).actor_observation_contract()
        if expected_observation_dim is None:
            expected_observation_dim = observation_dim
            expected_observation_contract = observation_contract
        elif observation_dim != expected_observation_dim:
            raise ValueError("all specialist Teachers must share one observation contract")
        elif observation_contract != expected_observation_contract:
            raise ValueError("all specialist Teachers must share observation semantics")

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
        elif config.split_strategy == "all_aircraft_command_holdout":
            profile_splits = [
                (profile, TRAIN_SPLIT) for profile in training_profiles
            ] + [(profile, VALIDATION_SPLIT) for profile in evaluation_profiles]
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
        episode_parts: list[np.ndarray] = []
        policy_step_parts: list[np.ndarray] = []
        for profile_index, (profile, split_code) in enumerate(profile_splits):
            if profile.command_id not in command_lookup:
                command_lookup[profile.command_id] = len(command_ids)
                command_ids.append(profile.command_id)
            observations, actions, policy_steps = _collect_profile(
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
            episode_parts.append(
                np.full(row_count, profile_index, dtype=np.int64)
            )
            policy_step_parts.append(policy_steps)

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
            episode_indices=np.concatenate(episode_parts),
            policy_step_indices=np.concatenate(policy_step_parts),
            driver_actions=actions,
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
                "episode_count": len(profile_splits),
                "temporal_pair_count": rows - len(profile_splits),
                "collection_round": 0,
                "driver": "teacher",
                "labeler": "matching_specialist_teacher",
            }
        )
        print(
            json.dumps(
                {
                    "event": "teacher_driven_collection_aircraft",
                    "plant_id": record.plant_id,
                    "aircraft_index": teacher_index,
                    "aircraft_count": len(teachers),
                    "rows": rows,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    split_strategy = (
        "single_aircraft_command_holdout"
        if len(teachers) == 1
        else config.split_strategy
    )
    manifest: dict[str, object] = {
        "schema_version": DISTILLATION_DATASET_SCHEMA_V2,
        "status": "complete",
        "source": git_source_revision(),
        "collection_method": "teacher_driven_initialization",
        "collection_round": 0,
        "teacher_bank": {
            "path": str(bank_path.resolve()),
            "sha256": sha256_file(bank_path),
        },
        "config": asdict(config),
        "split_strategy": split_strategy,
        "aircraft_split_method": (
            "single_aircraft_command_holdout"
            if len(teachers) == 1
            else (
                "per_aircraft_unseen_command_profiles"
                if config.split_strategy == "all_aircraft_command_holdout"
                else (
                    "explicit_quality_preserving_aircraft_holdout"
                    if config.validation_plant_ids
                    else "quality_region_stratified_holdout"
                )
            )
        ),
        "observation_dim": expected_observation_dim,
        "actor_observation_contract": expected_observation_contract,
        "aircraft_parameter_dim": len(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_names": list(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_normalization": AIRCRAFT_PARAMETER_NORMALIZATION,
        "action_dim": 1,
        "action_definition": "normalized_direct_full_F_as",
        "temporal_contract": {
            "episode_index": "one deterministic command rollout within one shard",
            "policy_step_index": "environment policy step before action application",
            "driver_action": "teacher action; identical to label in round zero",
            "predecessor_scope": "same shard and episode only",
        },
        "command_ids": command_ids,
        "train_plant_ids": train_plants,
        "validation_plant_ids": validation_plants,
        "row_count": total_rows,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "episode_count": int(
            sum(int(shard["episode_count"]) for shard in shard_entries)
        ),
        "temporal_pair_count": int(
            sum(int(shard["temporal_pair_count"]) for shard in shard_entries)
        ),
        "shards": shard_entries,
    }
    _write_json(destination / "dataset.json", manifest)
    return manifest
