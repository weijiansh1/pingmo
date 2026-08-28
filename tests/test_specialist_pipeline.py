from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.aircraft.sampler import generate_plant_library, persist_plant_library
from src.context.aircraft_parameters import AIRCRAFT_PARAMETER_NAMES, normalize_aircraft_parameters
from src.distillation.collect_data import DistillationCollectionConfig, collect_teacher_bank_data
from src.distillation.dataset import (
    DistillationArrays,
    DistillationDataset,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    load_distillation_arrays,
    load_distillation_shard,
    save_distillation_shard,
)
from src.distillation.distill import DenseStudentTrainingConfig, train_dense_student
from src.distillation.validate import evaluate_dense_student_bank
from src.envs.reference_model import SecondOrderReferenceConfig, SecondOrderRollRateReference
from src.envs.roll_rate_commands import RollRateCommandProfile, specialist_step_commands
from src.envs.specialist_tracking_env import SpecialistRollRateEnv
from src.student.dense.network import DenseConditionalStudent
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.manager import train_teacher_bank
from src.teacher.specialist.trainer import SpecialistTrainingConfig


def _short_profile() -> RollRateCommandProfile:
    return RollRateCommandProfile(
        "test-step",
        "step",
        amplitude_deg_s=10.0,
        onset_s=0.0,
        duration_s=0.02,
    )


def test_second_order_reference_converges_to_roll_rate_command() -> None:
    model = SecondOrderRollRateReference(
        SecondOrderReferenceConfig(natural_frequency_rad_s=2.0, damping_ratio=0.7),
        dt_s=0.001,
    )
    command = np.full(10_000, np.deg2rad(20.0))
    response = model.rollout(command)

    assert response.shape == (10_001,)
    assert response[0] == 0.0
    assert np.rad2deg(response[-1]) == pytest.approx(20.0, abs=0.001)


def test_specialist_command_bank_has_signed_training_steps() -> None:
    profiles = specialist_step_commands(1.0)
    amplitudes = {profile.amplitude_deg_s for profile in profiles}

    assert amplitudes == {10.0, 20.0, 30.0, -10.0, -20.0, -30.0}
    assert all(profile.kind == "step" for profile in profiles)


def test_specialist_actor_observation_excludes_theta_and_action_is_full_force() -> None:
    first, second = generate_plant_library(17, {"train_core": 2})
    first_env = SpecialistRollRateEnv(first, command_profiles=(_short_profile(),), history_steps=2)
    second_env = SpecialistRollRateEnv(second, command_profiles=(_short_profile(),), history_steps=2)
    first_observation, first_info = first_env.reset(seed=3)
    second_observation, _ = second_env.reset(seed=3)

    assert first_observation.shape == (16,)
    assert np.array_equal(first_observation, second_observation)
    assert first_info["actor_receives_theta"] is False
    assert first_info["critic_state"].shape == (21,)

    _, reward, _, _, info = first_env.step(np.asarray([1.0], dtype=np.float32))
    assert info["commanded_f_as_n"] == pytest.approx(0.088)
    assert info["f_as_n"] == pytest.approx(0.088)
    assert reward <= 0.0

    zero_env = SpecialistRollRateEnv(first, command_profiles=(_short_profile(),), history_steps=2)
    zero_env.reset(seed=3)
    _, _, _, _, zero_info = zero_env.step(np.asarray([0.0], dtype=np.float32))
    assert zero_info["f_as_n"] == 0.0


def test_specialist_rejects_nonfinite_action() -> None:
    record = generate_plant_library(19, {"train_core": 1})[0]
    env = SpecialistRollRateEnv(record, command_profiles=(_short_profile(),), history_steps=2)
    env.reset(seed=1)
    with pytest.raises(ValueError, match="finite normalized force"):
        env.step(np.asarray([np.nan], dtype=np.float32))


def test_parameter_normalization_has_one_canonical_student_order() -> None:
    record = generate_plant_library(21, {"train_core": 1})[0]
    theta = normalize_aircraft_parameters(record.parameters)

    assert len(AIRCRAFT_PARAMETER_NAMES) == 8
    assert theta.shape == (8,)
    assert np.isfinite(theta).all()


def test_distillation_shards_preserve_aircraft_splits(tmp_path: Path) -> None:
    arrays = DistillationArrays(
        observations=np.arange(24, dtype=np.float32).reshape(6, 4),
        aircraft_parameters=np.zeros((6, 8), dtype=np.float32),
        teacher_actions=np.linspace(-0.5, 0.5, 6, dtype=np.float32).reshape(-1, 1),
        plant_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        command_indices=np.arange(6),
        split_codes=np.asarray(
            [TRAIN_SPLIT, TRAIN_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT, VALIDATION_SPLIT, VALIDATION_SPLIT]
        ),
    )
    path = save_distillation_shard(tmp_path / "shard.npz", arrays)
    loaded = load_distillation_shard(path)
    train = DistillationDataset(loaded, "train")
    validation = DistillationDataset(loaded, "validation")

    assert len(train) == 3
    assert len(validation) == 3
    assert set(train.plant_indices.tolist()) == {0}
    assert set(validation.plant_indices.tolist()) == {1}


def test_dense_student_conditions_on_theta() -> None:
    torch.manual_seed(7)
    model = DenseConditionalStudent(16, width=16, residual_blocks=1)
    observation = torch.zeros(2, 16)
    theta = torch.stack((torch.zeros(8), torch.ones(8)))
    actions = model(observation, theta)

    assert actions.shape == (2, 1)
    assert torch.all(actions.abs() <= 1.0)
    assert not torch.allclose(actions[0], actions[1])


def test_one_aircraft_teacher_to_student_pipeline(tmp_path: Path) -> None:
    library_dir = persist_plant_library(tmp_path / "library", 23, {"train_core": 1})
    library_path = library_dir / "plants.jsonl"
    teacher_dir = tmp_path / "teachers"
    teacher_config = SpecialistTrainingConfig(
        total_steps=12,
        warmup_steps=4,
        batch_size=4,
        replay_capacity=32,
        episode_duration_s=0.1,
        history_steps=2,
        network_width=16,
        residual_blocks=1,
        progress_interval_steps=6,
        seed=29,
        device="cpu",
    )
    bank = train_teacher_bank(library_path, teacher_dir, teacher_config, count=1)
    assert bank["status"] == "complete"
    assert (teacher_dir / "teacher_bank.json").is_file()

    data_dir = tmp_path / "distillation"
    dataset_manifest = collect_teacher_bank_data(
        teacher_dir / "teacher_bank.json",
        data_dir,
        DistillationCollectionConfig(sample_stride=20, seed=31, device="cpu"),
    )
    assert dataset_manifest["split_strategy"] == "single_aircraft_command_holdout"
    assert dataset_manifest["train_rows"] > 0
    assert dataset_manifest["validation_rows"] > 0
    arrays, _ = load_distillation_arrays(data_dir / "dataset.json")
    assert arrays.observations.shape[1] == 16
    assert arrays.aircraft_parameters.shape[1] == 8

    student_dir = tmp_path / "student"
    student_report = train_dense_student(
        data_dir / "dataset.json",
        student_dir,
        DenseStudentTrainingConfig(
            epochs=2,
            batch_size=32,
            network_width=16,
            residual_blocks=1,
            patience_epochs=2,
            seed=37,
            device="cpu",
        ),
    )
    assert student_report["status"] == "complete"
    model, payload = load_dense_student(student_dir / "dense_student.pt")
    assert payload["aircraft_parameter_dim"] == 8
    teacher_record = json.loads(library_path.read_text(encoding="utf-8").splitlines()[0])
    assert teacher_record["plant_id"] in dataset_manifest["train_plant_ids"]

    evaluation = evaluate_dense_student_bank(
        student_dir / "dense_student.pt",
        teacher_dir / "teacher_bank.json",
        tmp_path / "student_evaluation",
        device="cpu",
    )
    assert evaluation["status"] == "complete"
    assert evaluation["aircraft_count"] == 1
    assert evaluation["pair_count"] == 6
    assert evaluation["distillation_split_strategy"] == "single_aircraft_command_holdout"
    assert evaluation["by_distillation_split"]["single_aircraft_command_holdout"]["pair_count"] == 6
    assert (tmp_path / "student_evaluation/evaluation.json").is_file()

    policy_theta = arrays.aircraft_parameters[0]
    policy = DenseStudentPolicy(model, policy_theta)
    action = policy.predict(np.zeros(16, dtype=np.float32))
    assert action.shape == (1,)
