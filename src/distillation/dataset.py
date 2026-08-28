"""Validated, aircraft-split datasets for specialist-policy distillation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


TRAIN_SPLIT = np.uint8(0)
VALIDATION_SPLIT = np.uint8(1)
SplitName = Literal["train", "validation"]


@dataclass(frozen=True, slots=True)
class DistillationArrays:
    observations: np.ndarray
    aircraft_parameters: np.ndarray
    teacher_actions: np.ndarray
    plant_indices: np.ndarray
    command_indices: np.ndarray
    split_codes: np.ndarray

    def __post_init__(self) -> None:
        row_count = len(self.observations)
        arrays = (
            self.aircraft_parameters,
            self.teacher_actions,
            self.plant_indices,
            self.command_indices,
            self.split_codes,
        )
        if row_count <= 0 or any(len(array) != row_count for array in arrays):
            raise ValueError("distillation arrays must be non-empty and row aligned")
        if self.observations.ndim != 2 or self.aircraft_parameters.ndim != 2:
            raise ValueError("observations and aircraft parameters must be matrices")
        if self.teacher_actions.ndim != 2 or self.teacher_actions.shape[1] != 1:
            raise ValueError("teacher actions must have shape [N, 1]")
        if not all(
            np.isfinite(array).all()
            for array in (self.observations, self.aircraft_parameters, self.teacher_actions)
        ):
            raise ValueError("distillation features and labels must be finite")
        if np.max(np.abs(self.teacher_actions)) > 1.0 + 1e-6:
            raise ValueError("teacher actions must be normalized to [-1, 1]")
        if not bool(np.all(np.isin(self.split_codes, (TRAIN_SPLIT, VALIDATION_SPLIT)))):
            raise ValueError("unsupported distillation split code")


class DistillationDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory PyTorch view of one validated train or validation split."""

    def __init__(self, arrays: DistillationArrays, split: SplitName) -> None:
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
        self.plant_indices = torch.from_numpy(
            np.ascontiguousarray(arrays.plant_indices[indices], dtype=np.int64)
        )

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observations[index],
            "aircraft_parameters": self.aircraft_parameters[index],
            "teacher_action": self.teacher_actions[index],
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
        )
    temporary.replace(destination)
    return destination


def load_distillation_shard(path: str | Path) -> DistillationArrays:
    with np.load(path, allow_pickle=False) as payload:
        return DistillationArrays(
            observations=payload["observations"],
            aircraft_parameters=payload["aircraft_parameters"],
            teacher_actions=payload["teacher_actions"],
            plant_indices=payload["plant_indices"],
            command_indices=payload["command_indices"],
            split_codes=payload["split_codes"],
        )


def load_distillation_arrays(manifest_path: str | Path) -> tuple[DistillationArrays, dict[str, object]]:
    source = Path(manifest_path)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "specialist_distillation_dataset_v1":
        raise ValueError("unsupported distillation dataset schema")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("distillation dataset manifest has no shards")
    loaded = [load_distillation_shard(source.parent / str(shard["path"])) for shard in shards]
    arrays = DistillationArrays(
        observations=np.concatenate([item.observations for item in loaded]),
        aircraft_parameters=np.concatenate([item.aircraft_parameters for item in loaded]),
        teacher_actions=np.concatenate([item.teacher_actions for item in loaded]),
        plant_indices=np.concatenate([item.plant_indices for item in loaded]),
        command_indices=np.concatenate([item.command_indices for item in loaded]),
        split_codes=np.concatenate([item.split_codes for item in loaded]),
    )
    if len(arrays.observations) != int(manifest["row_count"]):
        raise ValueError("distillation shard rows do not match the manifest")
    return arrays, manifest
