"""Smoke-test the V6 incremental-action Student without the gymnasium dependency.

Exercises the pieces that do not import the training environment:
``IncrementalDenseStudent`` forward + odd symmetry, the incremental dataset
fields (``previous_driver_action`` zeroed at episode starts, ``residual_delta_label``),
and ``incremental_student_losses``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.dataset import (
    DistillationArrays,
    DistillationDataset,
    TRAIN_SPLIT,
)
from src.distillation.losses import incremental_student_losses
from src.student.dense.network import IncrementalDenseStudent


def _synthetic_arrays() -> DistillationArrays:
    # Two episodes of three stride-1 steps each: six rows total.
    obs = np.random.RandomState(0).randn(6, 7).astype(np.float32)
    theta = np.random.RandomState(1).randn(6, 8).astype(np.float32) * 0.1
    teacher = np.clip(np.random.RandomState(2).randn(6, 1), -1, 1).astype(np.float32)
    # driver_actions differ from teacher only in the second episode (student-driven).
    driver = teacher.copy()
    driver[3:, 0] += 0.05
    driver = np.clip(driver, -1, 1).astype(np.float32)
    return DistillationArrays(
        observations=obs,
        aircraft_parameters=theta,
        teacher_actions=teacher,
        plant_indices=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        command_indices=np.array([0, 0, 0, 0, 0, 0], dtype=np.int32),
        split_codes=np.full(6, TRAIN_SPLIT, dtype=np.uint8),
        episode_indices=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        policy_step_indices=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        driver_actions=driver,
    )


def test_network_forward_and_odd_symmetry() -> None:
    model = IncrementalDenseStudent(
        7, 8, 1, width=32, residual_blocks=2, enforce_odd_policy=True
    )
    obs = torch.randn(4, 7)
    theta = torch.randn(4, 8)
    prev = torch.randn(4, 1)
    action, delta = model(obs, theta, prev)
    assert action.shape == (4, 1) and delta.shape == (4, 1)
    # action must equal clip(prev + delta) exactly.
    assert torch.allclose(action, (prev + delta).clamp(-1.0, 1.0), atol=1e-6)
    # Odd symmetry: mirroring observation AND previous action negates the action.
    mirrored_action, mirrored_delta = model(-obs, theta, -prev)
    assert torch.allclose(mirrored_action, -action, atol=1e-6)
    assert torch.allclose(mirrored_delta, -delta, atol=1e-6)


def _t(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def test_dataset_incremental_fields() -> None:
    arrays = _synthetic_arrays()
    dataset = DistillationDataset(arrays, "train", hard_case_weight_boost=0.0)
    sample = dataset[1]  # episode 0, step 1 -> valid predecessor
    assert torch.allclose(
        sample["previous_driver_action"], _t(arrays.driver_actions[0:1]), atol=1e-6
    )
    expected_label = _t(arrays.teacher_actions[1:2] - arrays.driver_actions[0:1])
    assert torch.allclose(sample["residual_delta_label"], expected_label, atol=1e-6)

    first = dataset[0]  # episode 0, step 0 -> boundary
    assert torch.allclose(first["previous_driver_action"], torch.zeros(1), atol=1e-6)
    assert torch.allclose(
        first["residual_delta_label"], _t(arrays.teacher_actions[0:1]), atol=1e-6
    )
    # The legacy own-delta predecessor (previous_teacher_action) still self-references
    # at the boundary, proving the incremental zeroing is a distinct field.
    assert torch.allclose(
        first["previous_teacher_action"], _t(arrays.teacher_actions[0:1]), atol=1e-6
    )


def test_incremental_losses() -> None:
    arrays = _synthetic_arrays()
    dataset = DistillationDataset(arrays, "train", hard_case_weight_boost=0.0)
    samples = [dataset[i] for i in range(len(dataset))]
    stacked = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}
    model = IncrementalDenseStudent(7, 8, 1, width=32, residual_blocks=2)
    action, delta = model(
        stacked["observation"],
        stacked["aircraft_parameters"],
        stacked["previous_driver_action"],
    )
    losses = incremental_student_losses(
        action,
        delta,
        stacked["teacher_action"],
        stacked["residual_delta_label"],
        stacked["sample_weight"],
        excess_margin=0.0,
    )
    for name, value in losses.items():
        assert torch.isfinite(value) and value >= 0, (name, value)
    # With delta exactly matching the label and margin 0, excess should be ~0.
    exact = incremental_student_losses(
        (stacked["previous_driver_action"] + stacked["residual_delta_label"]).clamp(
            -1.0, 1.0
        ),
        stacked["residual_delta_label"],
        stacked["teacher_action"],
        stacked["residual_delta_label"],
        stacked["sample_weight"],
        excess_margin=0.0,
    )
    assert float(exact["excess"]) < 1e-6


def main() -> None:
    test_network_forward_and_odd_symmetry()
    test_dataset_incremental_fields()
    test_incremental_losses()
    print("V6 incremental smoke test: PASS")


if __name__ == "__main__":
    main()
