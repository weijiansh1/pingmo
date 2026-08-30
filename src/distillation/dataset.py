"""Validated, aircraft-split datasets for specialist-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.provenance import sha256_file


TRAIN_SPLIT = np.uint8(0)
VALIDATION_SPLIT = np.uint8(1)
SplitName = Literal["train", "validation"]
DISTILLATION_DATASET_SCHEMA_V1 = "specialist_distillation_dataset_v1"
DISTILLATION_DATASET_SCHEMA_V2 = "specialist_distillation_dataset_v2"
SUPPORTED_DISTILLATION_DATASET_SCHEMAS = (
    DISTILLATION_DATASET_SCHEMA_V1,
    DISTILLATION_DATASET_SCHEMA_V2,
)


@dataclass(frozen=True, slots=True)
class DistillationArrays:
    observations: np.ndarray
    aircraft_parameters: np.ndarray
    teacher_actions: np.ndarray
    plant_indices: np.ndarray
    command_indices: np.ndarray
    split_codes: np.ndarray
    episode_indices: np.ndarray | None = None
    policy_step_indices: np.ndarray | None = None
    driver_actions: np.ndarray | None = None

    def __post_init__(self) -> None:
        row_count = len(self.observations)
        if self.episode_indices is None:
            object.__setattr__(
                self,
                "episode_indices",
                np.asarray(self.command_indices, dtype=np.int64),
            )
        if self.policy_step_indices is None:
            counters: dict[int, int] = {}
            policy_steps = np.empty(row_count, dtype=np.int32)
            for index, episode in enumerate(self.episode_indices):
                episode_id = int(episode)
                policy_steps[index] = counters.get(episode_id, 0)
                counters[episode_id] = int(policy_steps[index]) + 1
            object.__setattr__(self, "policy_step_indices", policy_steps)
        if self.driver_actions is None:
            object.__setattr__(self, "driver_actions", self.teacher_actions)

        arrays = (
            self.aircraft_parameters,
            self.teacher_actions,
            self.plant_indices,
            self.command_indices,
            self.split_codes,
            self.episode_indices,
            self.policy_step_indices,
            self.driver_actions,
        )
        if row_count <= 0 or any(len(array) != row_count for array in arrays):
            raise ValueError("distillation arrays must be non-empty and row aligned")
        if self.observations.ndim != 2 or self.aircraft_parameters.ndim != 2:
            raise ValueError("observations and aircraft parameters must be matrices")
        if self.teacher_actions.ndim != 2 or self.teacher_actions.shape[1] != 1:
            raise ValueError("teacher actions must have shape [N, 1]")
        if self.driver_actions.ndim != 2 or self.driver_actions.shape != self.teacher_actions.shape:
            raise ValueError("driver actions must match the teacher-action matrix")
        if self.episode_indices.ndim != 1 or self.policy_step_indices.ndim != 1:
            raise ValueError("episode and policy-step indices must be vectors")
        if np.min(self.episode_indices) < 0 or np.min(self.policy_step_indices) < 0:
            raise ValueError("episode and policy-step indices cannot be negative")
        if not all(
            np.isfinite(array).all()
            for array in (
                self.observations,
                self.aircraft_parameters,
                self.teacher_actions,
                self.driver_actions,
            )
        ):
            raise ValueError("distillation features and labels must be finite")
        if max(
            np.max(np.abs(self.teacher_actions)),
            np.max(np.abs(self.driver_actions)),
        ) > 1.0 + 1e-6:
            raise ValueError("teacher and driver actions must be normalized to [-1, 1]")
        if not bool(np.all(np.isin(self.split_codes, (TRAIN_SPLIT, VALIDATION_SPLIT)))):
            raise ValueError("unsupported distillation split code")


class DistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory PyTorch view of one validated train or validation split."""

    def __init__(
        self,
        arrays: DistillationArrays,
        split: SplitName,
        *,
        hard_case_weight_boost: float = 0.0,
        hard_tracking_error_scale: float = 0.2,
        hard_teacher_mismatch_scale: float = 0.1,
        hard_action_rate_scale: float = 0.05,
        tracking_error_index: int = 3,
    ) -> None:
        if hard_case_weight_boost < 0 or min(
            hard_tracking_error_scale,
            hard_teacher_mismatch_scale,
            hard_action_rate_scale,
        ) <= 0:
            raise ValueError("invalid hard-case weighting configuration")
        if not 0 <= tracking_error_index < arrays.observations.shape[1]:
            raise ValueError("tracking-error observation index is out of range")
        split_code = TRAIN_SPLIT if split == "train" else VALIDATION_SPLIT
        indices = np.flatnonzero(arrays.split_codes == split_code)
        if not len(indices):
            raise ValueError(f"distillation dataset has no {split} rows")
        self.observations = torch.from_numpy(
            np.ascontiguousarray(arrays.observations[indices], dtype=np.float32)
        )
        self.aircraft_parameters = torch.from_numpy(
            np.ascontiguousarray(arrays.aircraft_parameters[indices], dtype=np.float32)
        )
        self.teacher_actions = torch.from_numpy(
            np.ascontiguousarray(arrays.teacher_actions[indices], dtype=np.float32)
        )
        self.driver_actions = torch.from_numpy(
            np.ascontiguousarray(arrays.driver_actions[indices], dtype=np.float32)
        )
        self.plant_indices = torch.from_numpy(
            np.ascontiguousarray(arrays.plant_indices[indices], dtype=np.int64)
        )
        episodes = np.asarray(arrays.episode_indices[indices], dtype=np.int64)
        policy_steps = np.asarray(arrays.policy_step_indices[indices], dtype=np.int64)
        previous = np.arange(len(indices), dtype=np.int64)
        temporal_mask = np.zeros(len(indices), dtype=np.float32)
        step_delta = np.ones(len(indices), dtype=np.float32)
        latest_by_episode: dict[int, int] = {}
        for local_index, (episode, policy_step) in enumerate(
            zip(episodes, policy_steps, strict=True)
        ):
            previous_index = latest_by_episode.get(int(episode))
            if previous_index is not None and policy_step > policy_steps[previous_index]:
                previous[local_index] = previous_index
                temporal_mask[local_index] = 1.0
                step_delta[local_index] = float(
                    policy_step - policy_steps[previous_index]
                )
            latest_by_episode[int(episode)] = local_index

        self.previous_observations = self.observations[torch.from_numpy(previous)]
        self.previous_teacher_actions = self.teacher_actions[torch.from_numpy(previous)]
        self.previous_driver_actions = self.driver_actions[torch.from_numpy(previous)]
        self.temporal_mask = torch.from_numpy(temporal_mask)
        self.policy_step_delta = torch.from_numpy(step_delta)

        tracking_hardness = np.clip(
            np.abs(arrays.observations[indices, tracking_error_index])
            / hard_tracking_error_scale,
            0.0,
            1.0,
        )
        teacher_mismatch_hardness = np.clip(
            np.max(
                np.abs(
                    arrays.driver_actions[indices] - arrays.teacher_actions[indices]
                ),
                axis=1,
            )
            / hard_teacher_mismatch_scale,
            0.0,
            1.0,
        )
        driver_delta = (
            self.driver_actions.numpy() - self.previous_driver_actions.numpy()
        ) / step_delta[:, None]
        action_rate_hardness = np.clip(
            np.max(np.abs(driver_delta), axis=1) / hard_action_rate_scale,
            0.0,
            1.0,
        ) * temporal_mask
        hardness = np.maximum.reduce(
            (tracking_hardness, teacher_mismatch_hardness, action_rate_hardness)
        ).astype(np.float32)
        self.hardness_scores = torch.from_numpy(hardness)
        self.sample_weights = 1.0 + hard_case_weight_boost * self.hardness_scores

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observations[index],
            "aircraft_parameters": self.aircraft_parameters[index],
            "teacher_action": self.teacher_actions[index],
            "driver_action": self.driver_actions[index],
            "previous_observation": self.previous_observations[index],
            "previous_teacher_action": self.previous_teacher_actions[index],
            "temporal_mask": self.temporal_mask[index],
            "policy_step_delta": self.policy_step_delta[index],
            "hardness_score": self.hardness_scores[index],
            "sample_weight": self.sample_weights[index],
            "plant_index": self.plant_indices[index],
        }


def save_distillation_shard(path: str | Path, arrays: DistillationArrays) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            observations=np.asarray(arrays.observations, dtype=np.float32),
            aircraft_parameters=np.asarray(arrays.aircraft_parameters, dtype=np.float32),
            teacher_actions=np.asarray(arrays.teacher_actions, dtype=np.float32),
            plant_indices=np.asarray(arrays.plant_indices, dtype=np.int32),
            command_indices=np.asarray(arrays.command_indices, dtype=np.int32),
            split_codes=np.asarray(arrays.split_codes, dtype=np.uint8),
            episode_indices=np.asarray(arrays.episode_indices, dtype=np.int64),
            policy_step_indices=np.asarray(arrays.policy_step_indices, dtype=np.int32),
            driver_actions=np.asarray(arrays.driver_actions, dtype=np.float32),
        )
    temporary.replace(destination)
    return destination


def load_distillation_shard(path: str | Path) -> DistillationArrays:
    with np.load(path, allow_pickle=False) as payload:
        optional = set(payload.files)
        return DistillationArrays(
            observations=payload["observations"],
            aircraft_parameters=payload["aircraft_parameters"],
            teacher_actions=payload["teacher_actions"],
            plant_indices=payload["plant_indices"],
            command_indices=payload["command_indices"],
            split_codes=payload["split_codes"],
            episode_indices=(
                payload["episode_indices"] if "episode_indices" in optional else None
            ),
            policy_step_indices=(
                payload["policy_step_indices"]
                if "policy_step_indices" in optional
                else None
            ),
            driver_actions=(
                payload["driver_actions"] if "driver_actions" in optional else None
            ),
        )


def load_distillation_arrays(manifest_path: str | Path) -> tuple[DistillationArrays, dict[str, object]]:
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") not in SUPPORTED_DISTILLATION_DATASET_SCHEMAS:
        raise ValueError("unsupported distillation dataset schema")
    if manifest.get("status") != "complete":
        raise ValueError("distillation dataset manifest is incomplete")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("distillation dataset manifest has no shards")
    loaded: list[DistillationArrays] = []
    for shard in shards:
        shard_path = source.parent / str(shard["path"])
        expected_hash = shard.get("sha256")
        if expected_hash is None or sha256_file(shard_path) != str(expected_hash):
            raise ValueError(f"distillation shard hash mismatch: {shard_path}")
        loaded.append(load_distillation_shard(shard_path))
    episode_parts: list[np.ndarray] = []
    episode_offset = 0
    for item in loaded:
        local_episodes = np.asarray(item.episode_indices, dtype=np.int64)
        _, dense_episodes = np.unique(local_episodes, return_inverse=True)
        episode_parts.append(dense_episodes.astype(np.int64) + episode_offset)
        episode_offset += int(dense_episodes.max()) + 1
    arrays = DistillationArrays(
        observations=np.concatenate([item.observations for item in loaded]),
        aircraft_parameters=np.concatenate([item.aircraft_parameters for item in loaded]),
        teacher_actions=np.concatenate([item.teacher_actions for item in loaded]),
        plant_indices=np.concatenate([item.plant_indices for item in loaded]),
        command_indices=np.concatenate([item.command_indices for item in loaded]),
        split_codes=np.concatenate([item.split_codes for item in loaded]),
        episode_indices=np.concatenate(episode_parts),
        policy_step_indices=np.concatenate(
            [item.policy_step_indices for item in loaded]
        ),
        driver_actions=np.concatenate([item.driver_actions for item in loaded]),
    )
    if len(arrays.observations) != int(manifest["row_count"]):
        raise ValueError("distillation shard rows do not match the manifest")
    if arrays.observations.shape[1] != int(manifest["observation_dim"]):
        raise ValueError("distillation observation dimension does not match the manifest")
    if arrays.aircraft_parameters.shape[1] != int(manifest["aircraft_parameter_dim"]):
        raise ValueError("distillation theta dimension does not match the manifest")
    if int(np.sum(arrays.split_codes == TRAIN_SPLIT)) != int(manifest["train_rows"]):
        raise ValueError("distillation train rows do not match the manifest")
    if int(np.sum(arrays.split_codes == VALIDATION_SPLIT)) != int(
        manifest["validation_rows"]
    ):
        raise ValueError("distillation validation rows do not match the manifest")
    return arrays, manifest
