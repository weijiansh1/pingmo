from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.aircraft.sampler import generate_plant_library, persist_plant_library
from src.context.aircraft_parameters import (
    AIRCRAFT_PARAMETER_NAMES,
    normalize_aircraft_parameters,
)
from src.controllers.pid import PIDGains, RollRatePIDPolicy
from src.distillation.collect_data import (
    DistillationCollectionConfig,
    _aircraft_split,
    collect_teacher_bank_data,
)
from src.distillation.dataset import (
    DISTILLATION_DATASET_SCHEMA_V2,
    DistillationArrays,
    DistillationDataset,
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    load_distillation_arrays,
    load_distillation_shard,
    save_distillation_shard,
)
from src.distillation.distill import DenseStudentTrainingConfig, train_dense_student
from src.distillation.losses import teacher_action_rate_mse
from src.distillation.student_driven import (
    StudentDrivenDistillationConfig,
    _profile_splits,
    _round_score,
    _select_student_round,
    run_student_driven_distillation,
)
from src.distillation.validate import evaluate_dense_student_bank
from src.envs.reference_model import (
    SecondOrderReferenceConfig,
    SecondOrderRollRateReference,
)
from src.envs.roll_rate_commands import (
    RandomCommandDistribution,
    RollRateCommandProfile,
    RollRateCommandSequence,
    SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION,
    sample_mixed_duration_training_episode,
    sample_random_long_dwell_step,
    sample_random_training_command,
    sample_random_training_sequence,
    specialist_evaluation_commands,
    specialist_independent_test_commands,
    specialist_step_commands,
)
from src.envs.specialist_tracking_env import SpecialistRollRateEnv
from src.experiments.exploratory_sac import load_persisted_records
from src.student.dense.network import DenseConditionalStudent
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.student.moe.network import ThetaRoutedLinearMoEStudent
from src.teacher.sac.actor import SquashedGaussianActor
from src.teacher.specialist.manager import train_teacher_bank
from src.teacher.specialist.pure_td3_trainer import (
    PureRewardTD3Config,
    train_pure_reward_td3,
)
from src.teacher.specialist.td3_manager import train_pid_guided_teacher_bank
from src.teacher.specialist.td3_trainer import (
    PIDGuidedTD3Config,
    train_pid_guided_td3,
)
from src.teacher.specialist.trainer import (
    SpecialistTrainingConfig,
    load_specialist_actor,
    tracking_metrics,
)
from src.teacher.td3.actor import DeterministicActor, PIDResidualActor
from src.teacher.transitions import continuing_task_transition_flags
from src.utils.provenance import git_source_revision, sha256_file


def _short_profile() -> RollRateCommandProfile:
    return RollRateCommandProfile(
        "test-step",
        "step",
        amplitude_deg_s=10.0,
        onset_s=0.0,
        duration_s=0.02,
    )


def test_explicit_aircraft_holdout_is_deterministic_and_quality_preserving() -> None:
    teachers = [
        {"plant_id": "l1-train", "quality_region": "level_1"},
        {"plant_id": "l1-validation", "quality_region": "level_1"},
        {"plant_id": "l2-train", "quality_region": "level_2"},
        {"plant_id": "l2-validation", "quality_region": "level_2"},
    ]

    selected = _aircraft_split(
        teachers,
        fraction=0.2,
        seed=1,
        explicit_validation_plant_ids=("l1-validation", "l2-validation"),
    )

    assert selected == {1, 3}


def test_explicit_aircraft_holdout_rejects_missing_or_emptied_quality_region() -> None:
    teachers = [
        {"plant_id": "l1-only", "quality_region": "level_1"},
        {"plant_id": "l2-a", "quality_region": "level_2"},
        {"plant_id": "l2-b", "quality_region": "level_2"},
    ]

    with pytest.raises(ValueError, match="absent from Teacher Bank"):
        _aircraft_split(
            teachers,
            fraction=0.2,
            seed=1,
            explicit_validation_plant_ids=("missing",),
        )
    with pytest.raises(ValueError, match="quality regions"):
        _aircraft_split(
            teachers,
            fraction=0.2,
            seed=1,
            explicit_validation_plant_ids=("l1-only",),
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


def test_second_order_reference_applies_fractional_transport_delay() -> None:
    command = np.ones(100, dtype=float)
    immediate = SecondOrderRollRateReference(dt_s=0.001).rollout(command)
    one_sample = SecondOrderRollRateReference(dt_s=0.001, delay_s=0.001).rollout(
        command
    )
    two_samples = SecondOrderRollRateReference(dt_s=0.001, delay_s=0.002).rollout(
        command
    )
    fractional = SecondOrderRollRateReference(dt_s=0.001, delay_s=0.0015).rollout(
        command
    )

    assert one_sample[1:] == pytest.approx(immediate[:-1])
    assert two_samples[2:] == pytest.approx(immediate[:-2])
    assert fractional == pytest.approx(0.5 * (one_sample + two_samples))


def test_specialist_command_bank_has_signed_training_steps() -> None:
    profiles = specialist_step_commands(1.0)
    amplitudes = {profile.amplitude_deg_s for profile in profiles}

    assert amplitudes == {10.0, 20.0, 30.0, -10.0, -20.0, -30.0}
    assert all(profile.kind == "step" for profile in profiles)


def test_independent_test_command_bank_is_frozen_and_disjoint() -> None:
    profiles = specialist_independent_test_commands(30.0)
    validation_profiles = specialist_evaluation_commands(30.0)
    validation_ids = {profile.command_id for profile in validation_profiles}

    assert SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION == (
        "specialist-independent-test-v1"
    )
    assert [profile.command_id for profile in profiles] == [
        "test-v1-step-pos-18deg-s",
        "test-v1-step-neg-18deg-s",
        "test-v1-step-pos-27deg-s",
        "test-v1-doublet-neg-18deg-s",
        "test-v1-sine-0.43hz",
        "test-v1-multisine",
    ]
    assert not validation_ids.intersection(profile.command_id for profile in profiles)
    assert all(
        not np.array_equal(test.samples(0.02), validation.samples(0.02))
        for test in profiles
        for validation in validation_profiles
    )
    assert all(profile.duration_s == 30.0 for profile in profiles)
    assert all(profile.samples(0.02).shape == (1_500,) for profile in profiles)


def test_random_training_commands_are_reproducible_and_continuous() -> None:
    first_rng = np.random.default_rng(20260829)
    second_rng = np.random.default_rng(20260829)
    first = [
        sample_random_training_command(first_rng, index, policy_dt_s=0.02)
        for index in range(500)
    ]
    second = [
        sample_random_training_command(second_rng, index, policy_dt_s=0.02)
        for index in range(500)
    ]

    assert first == second
    assert {profile.kind for profile in first} == {
        "step",
        "doublet",
        "sine",
        "multisine",
    }
    assert len({profile.command_id for profile in first}) == len(first)
    assert all(4.0 <= profile.duration_s <= 8.0 for profile in first)
    assert all(
        profile.duration_s / 0.02 == pytest.approx(round(profile.duration_s / 0.02))
        for profile in first
    )
    step_amplitudes = [
        abs(profile.amplitude_deg_s) for profile in first if profile.kind == "step"
    ]
    assert any(
        not np.isclose(amplitude, round(amplitude)) for amplitude in step_amplitudes
    )
    assert all(
        sum(abs(amplitude) for amplitude, _ in profile.multisine_components)
        <= 25.0 + 1e-12
        for profile in first
        if profile.kind == "multisine"
    )


def test_random_training_sequence_is_reproducible_and_keeps_one_long_episode() -> None:
    first = sample_random_training_sequence(
        np.random.default_rng(20260830),
        7,
        policy_dt_s=0.02,
        duration_s=30.0,
        segment_duration_range_s=(2.0, 5.0),
    )
    second = sample_random_training_sequence(
        np.random.default_rng(20260830),
        7,
        policy_dt_s=0.02,
        duration_s=30.0,
        segment_duration_range_s=(2.0, 5.0),
    )

    assert first == second
    assert isinstance(first, RollRateCommandSequence)
    assert first.kind == "sequence"
    assert first.duration_s == pytest.approx(30.0)
    assert len(first.segments) > 1
    assert all(2.0 <= segment.duration_s <= 5.0 for segment in first.segments)
    assert first.samples(0.001).shape == (30_000,)


def test_long_dwell_and_mixed_duration_commands_are_reproducible() -> None:
    first_rng = np.random.default_rng(20260831)
    second_rng = np.random.default_rng(20260831)
    first = [
        sample_mixed_duration_training_episode(
            first_rng,
            index,
            policy_dt_s=0.02,
            duration_s=30.0,
            long_dwell_step_probability=0.5,
        )
        for index in range(100)
    ]
    second = [
        sample_mixed_duration_training_episode(
            second_rng,
            index,
            policy_dt_s=0.02,
            duration_s=30.0,
            long_dwell_step_probability=0.5,
        )
        for index in range(100)
    ]

    assert first == second
    long_steps = [
        profile for profile in first if isinstance(profile, RollRateCommandProfile)
    ]
    short_sequences = [
        profile for profile in first if isinstance(profile, RollRateCommandSequence)
    ]
    assert long_steps
    assert short_sequences
    assert all(profile.kind == "step" for profile in long_steps)
    assert all(
        15.0 <= profile.duration_s - profile.onset_s <= 30.0 for profile in long_steps
    )
    assert all(
        2.0 <= segment.duration_s <= 5.0
        for profile in short_sequences
        for segment in profile.segments
    )
    assert all(profile.samples(0.001).shape == (30_000,) for profile in first)

    held = sample_random_long_dwell_step(
        np.random.default_rng(9),
        1,
        policy_dt_s=0.02,
        duration_s=30.0,
        dwell_duration_range_s=(15.0, 30.0),
    )
    assert 15.0 <= held.duration_s - held.onset_s <= 30.0


def test_sequence_command_requires_critic_without_future_profile_context() -> None:
    record = generate_plant_library(202, {"train_core": 1})[0]
    sequence = sample_random_training_sequence(
        np.random.default_rng(203),
        0,
        policy_dt_s=0.02,
        duration_s=0.12,
        segment_duration_range_s=(0.04, 0.06),
        config=RandomCommandDistribution(
            onset_range_s=(0.005, 0.010),
            doublet_segment_range_s=(0.005, 0.010),
        ),
    )

    with pytest.raises(ValueError, match="command_context=False"):
        SpecialistRollRateEnv(record, command_profiles=(sequence,))
    env = SpecialistRollRateEnv(
        record,
        command_profiles=(sequence,),
        critic_include_episode_progress=False,
        critic_include_command_context=False,
    )
    _, info = env.reset(seed=204)
    contract = env.critic_observation_contract()

    assert info["command_kind"] == "sequence"
    assert contract["includes_command_context"] is False
    assert contract["command_context_width"] == 0
    assert contract["future_reference_determined_by_profile_and_progress"] is False
    assert info["critic_state"].shape == (len(contract["names"]),)


def test_continuing_task_bootstraps_only_across_time_limit() -> None:
    assert continuing_task_transition_flags(False, False) == (False, False)
    assert continuing_task_transition_flags(False, True) == (True, False)
    assert continuing_task_transition_flags(True, False) == (True, True)
    assert continuing_task_transition_flags(True, True) == (True, True)


def test_specialist_actor_observation_excludes_theta_and_action_is_full_force() -> None:
    first, second = generate_plant_library(17, {"train_core": 2})
    first_env = SpecialistRollRateEnv(
        first, command_profiles=(_short_profile(),), history_steps=0
    )
    second_env = SpecialistRollRateEnv(
        second, command_profiles=(_short_profile(),), history_steps=0
    )
    first_observation, first_info = first_env.reset(seed=3)
    second_observation, _ = second_env.reset(seed=3)

    assert first_observation.shape == (7,)
    assert np.array_equal(first_observation, second_observation)
    assert first_info["actor_receives_theta"] is False
    assert first_info["reference_delay_s"] == first.parameters.tau_p
    critic_contract = first_env.critic_observation_contract()
    expected_delay_width = int(np.floor(first.parameters.tau_p / 0.001)) + 3
    assert first_info["critic_state"].shape == (28 + expected_delay_width,)
    assert first_info["critic_state"].shape == (len(critic_contract["names"]),)
    assert critic_contract["deployment_input"] is False
    assert critic_contract["transport_delay_fifo_width"] == expected_delay_width
    assert critic_contract["includes_actuator_state"] is True
    assert (
        critic_contract["future_reference_determined_by_profile_and_progress"] is True
    )

    _, reward, _, _, info = first_env.step(np.asarray([1.0], dtype=np.float32))
    assert info["requested_f_as_n"] == pytest.approx(22.0)
    assert info["commanded_f_as_n"] == pytest.approx(1.76)
    assert info["f_as_n"] == pytest.approx(1.76)
    assert reward <= 0.0
    metrics = tracking_metrics(first_env.trajectory(), first_env.force_limit_n)
    assert metrics["force_rate_limit_active_fraction"] == pytest.approx(0.5)
    assert metrics["mean_abs_force_rate_limit_gap_n"] == pytest.approx(10.12)
    assert metrics["maximum_abs_force_rate_limit_gap_n"] == pytest.approx(20.24)
    critic_state = np.asarray(info["critic_state"])
    critic_names = list(critic_contract["names"])
    delay_indices = [
        index
        for index, name in enumerate(critic_names)
        if name.startswith("plant_force_delay_fifo")
    ]
    assert np.any(critic_state[delay_indices] > 0.0)
    assert critic_state[
        critic_names.index("commanded_force_normalized")
    ] == pytest.approx(0.08)
    assert critic_state[
        critic_names.index("applied_force_normalized")
    ] == pytest.approx(0.08)
    assert critic_state[critic_names.index("command_kind_one_hot.step")] == 1.0
    assert critic_state[
        critic_names.index("command_amplitude_normalized")
    ] == pytest.approx(1.0 / 3.0)
    assert critic_state[critic_names.index("command_onset_fraction")] == 0.0

    zero_env = SpecialistRollRateEnv(
        first, command_profiles=(_short_profile(),), history_steps=0
    )
    zero_env.reset(seed=3)
    _, _, _, _, zero_info = zero_env.step(np.asarray([0.0], dtype=np.float32))
    assert zero_info["f_as_n"] == 0.0


def test_specialist_reset_can_select_one_command_deterministically() -> None:
    record = generate_plant_library(171, {"train_core": 1})[0]
    profiles = specialist_step_commands(1.0)
    env = SpecialistRollRateEnv(record, command_profiles=profiles, history_steps=0)

    _, info = env.reset(seed=3, options={"command_index": 4})

    assert info["command_id"] == profiles[4].command_id
    with pytest.raises(ValueError, match="command_index"):
        env.reset(options={"command_index": len(profiles)})


def test_actor_action_memory_exposes_delay_state_with_fixed_width() -> None:
    record = generate_plant_library(172, {"train_core": 1})[0]
    env = SpecialistRollRateEnv(
        record,
        command_profiles=(_short_profile(),),
        history_steps=0,
        requested_action_history_steps=6,
        include_actor_actuator_state=True,
        critic_include_episode_progress=False,
    )
    profile = RollRateCommandProfile(
        "dynamic-step",
        "step",
        amplitude_deg_s=13.7,
        onset_s=0.01,
        duration_s=0.08,
    )
    observation, info = env.reset(options={"command_profile": profile})

    assert observation.shape == (14,)
    assert info["episode_duration_s"] == pytest.approx(0.08)
    assert env.horizon_steps == 4
    actor_contract = env.actor_observation_contract()
    assert actor_contract["requested_action_history_steps"] == 6
    assert actor_contract["delay_coverage_s"] == pytest.approx(0.12)
    assert actor_contract["names"][6:] == [
        "previous_force_normalized",
        "commanded_force_normalized",
        "applied_force_normalized",
        "requested_force_lag_2_normalized",
        "requested_force_lag_3_normalized",
        "requested_force_lag_4_normalized",
        "requested_force_lag_5_normalized",
        "requested_force_lag_6_normalized",
    ]

    for action in (0.1, 0.2, 0.3):
        observation, _, _, truncated, info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        assert truncated is False
    assert observation[6] == pytest.approx(0.3)
    assert observation[7] == pytest.approx(0.24)
    assert observation[8] == pytest.approx(0.24)
    assert observation[9] == pytest.approx(0.2)
    assert observation[10] == pytest.approx(0.1)
    critic_contract = env.critic_observation_contract()
    assert critic_contract["includes_episode_progress"] is False
    assert "episode_progress" not in critic_contract["names"]
    assert len(info["critic_state"]) == len(critic_contract["names"])

    _, _, _, truncated, _ = env.step(np.asarray([0.3], dtype=np.float32))
    assert truncated is True


def test_reference_derivative_is_optional_causal_actor_state() -> None:
    record = generate_plant_library(173, {"train_core": 1})[0]
    profile = RollRateCommandProfile(
        "reference-derivative-step",
        "step",
        amplitude_deg_s=15.0,
        onset_s=0.0,
        duration_s=0.08,
    )
    default_env = SpecialistRollRateEnv(
        record,
        command_profiles=(profile,),
        history_steps=0,
        reference_delay_s=0.0,
    )
    derivative_env = SpecialistRollRateEnv(
        record,
        command_profiles=(profile,),
        history_steps=0,
        include_reference_derivative=True,
        reference_delay_s=0.0,
    )

    default_observation, _ = default_env.reset(seed=5)
    observation, info = derivative_env.reset(seed=5)
    assert default_observation.shape == (7,)
    assert observation.shape == (8,)
    assert observation[-1] == pytest.approx(0.0)
    assert info["actor_includes_reference_derivative"] is True

    observation, _, _, _, _ = derivative_env.step(np.asarray([0.0], dtype=np.float32))
    contract = derivative_env.actor_observation_contract()
    trace = derivative_env.trajectory()
    assert observation[-1] > 0.0
    assert contract["names"][7] == "p_reference_dot_normalized"
    assert contract["includes_reference_derivative"] is True
    assert trace["p_reference_dot_rad_s2"][-1] > 0.0


def test_specialist_smoothness_cost_uses_requested_teacher_action() -> None:
    record = generate_plant_library(18, {"train_core": 1})[0]
    profile = RollRateCommandProfile(
        "two-step-test", "step", amplitude_deg_s=10.0, onset_s=0.0, duration_s=0.04
    )
    steady = SpecialistRollRateEnv(record, command_profiles=(profile,), history_steps=0)
    jitter = SpecialistRollRateEnv(record, command_profiles=(profile,), history_steps=0)
    steady.reset(seed=5)
    jitter.reset(seed=5)

    steady.step(np.asarray([1.0], dtype=np.float32))
    jitter.step(np.asarray([1.0], dtype=np.float32))
    steady.step(np.asarray([1.0], dtype=np.float32))
    jitter.step(np.asarray([-1.0], dtype=np.float32))

    steady_trace = steady.trajectory()
    jitter_trace = jitter.trajectory()
    assert steady_trace["force_delta_cost"][-1] == pytest.approx(0.0)
    assert jitter_trace["force_delta_cost"][-1] == pytest.approx(0.08)
    assert jitter_trace["requested_f_as_n"][-1] == pytest.approx(-22.0)


def test_specialist_rejects_nonfinite_action() -> None:
    record = generate_plant_library(19, {"train_core": 1})[0]
    env = SpecialistRollRateEnv(
        record, command_profiles=(_short_profile(),), history_steps=0
    )
    env.reset(seed=1)
    with pytest.raises(ValueError, match="finite normalized force"):
        env.step(np.asarray([np.nan], dtype=np.float32))


def test_pid_policy_uses_explicit_tracking_integral() -> None:
    policy = RollRatePIDPolicy(
        PIDGains(10.0, 2.0, 0.5),
        policy_dt_s=0.02,
        command_scale_rad_s=np.deg2rad(30.0),
        integral_error_scale_rad=np.deg2rad(30.0),
        roll_acceleration_scale_rad_s2=5.0,
        force_limit_n=22.0,
    )
    observation = np.zeros(7, dtype=np.float32)
    observation[3] = 0.5
    observation[4] = 0.25
    first = policy.predict(observation)
    second = policy.predict(observation)
    policy.reset()

    assert first.shape == (1,)
    assert 0.0 < first[0] <= 1.0
    assert second == pytest.approx(first)
    assert policy.predict(observation) == pytest.approx(first)


def test_parameter_normalization_has_one_canonical_student_order() -> None:
    record = generate_plant_library(21, {"train_core": 1})[0]
    theta = normalize_aircraft_parameters(record.parameters)

    assert len(AIRCRAFT_PARAMETER_NAMES) == 8
    assert theta.shape == (8,)
    assert np.isfinite(theta).all()


def test_source_provenance_hashes_the_executable_working_tree() -> None:
    source = git_source_revision()

    assert isinstance(source["working_tree_sha256"], str)
    assert len(source["working_tree_sha256"]) == 64


def test_distillation_shards_preserve_aircraft_splits(tmp_path: Path) -> None:
    arrays = DistillationArrays(
        observations=np.arange(24, dtype=np.float32).reshape(6, 4),
        aircraft_parameters=np.zeros((6, 8), dtype=np.float32),
        teacher_actions=np.linspace(-0.5, 0.5, 6, dtype=np.float32).reshape(-1, 1),
        plant_indices=np.asarray([0, 0, 0, 1, 1, 1]),
        command_indices=np.arange(6),
        split_codes=np.asarray(
            [
                TRAIN_SPLIT,
                TRAIN_SPLIT,
                TRAIN_SPLIT,
                VALIDATION_SPLIT,
                VALIDATION_SPLIT,
                VALIDATION_SPLIT,
            ]
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


def test_temporal_distillation_pairs_stay_inside_one_episode() -> None:
    arrays = DistillationArrays(
        observations=np.asarray(
            [
                [0.0, 0.0, 0.0, 0.00],
                [1.0, 0.0, 0.0, 0.25],
                [2.0, 0.0, 0.0, 0.00],
                [3.0, 0.0, 0.0, 0.00],
            ],
            dtype=np.float32,
        ),
        aircraft_parameters=np.zeros((4, 8), dtype=np.float32),
        teacher_actions=np.asarray([[0.0], [0.1], [0.0], [0.0]], dtype=np.float32),
        plant_indices=np.zeros(4, dtype=np.int32),
        command_indices=np.asarray([0, 0, 1, 1], dtype=np.int32),
        split_codes=np.full(4, TRAIN_SPLIT, dtype=np.uint8),
        episode_indices=np.asarray([0, 0, 1, 1], dtype=np.int64),
        policy_step_indices=np.asarray([0, 2, 0, 2], dtype=np.int32),
        driver_actions=np.asarray([[0.0], [0.4], [0.7], [0.7]], dtype=np.float32),
    )
    dataset = DistillationDataset(
        arrays,
        "train",
        hard_case_weight_boost=7.0,
        hard_tracking_error_scale=0.2,
        hard_teacher_mismatch_scale=0.1,
        hard_action_rate_scale=0.05,
    )

    assert dataset[0]["temporal_mask"] == 0.0
    assert dataset[1]["temporal_mask"] == 1.0
    assert dataset[1]["policy_step_delta"] == 2.0
    assert dataset[1]["previous_observation"] == pytest.approx(
        torch.tensor([0.0, 0.0, 0.0, 0.0])
    )
    assert dataset[2]["temporal_mask"] == 0.0
    assert dataset[2]["previous_observation"] == pytest.approx(
        torch.tensor([2.0, 0.0, 0.0, 0.0])
    )
    assert dataset[1]["sample_weight"] == pytest.approx(8.0)


def test_teacher_action_rate_loss_uses_policy_step_interval() -> None:
    loss = teacher_action_rate_mse(
        torch.tensor([[0.4], [0.9]]),
        torch.tensor([[0.1], [0.9]]),
        torch.tensor([[0.3], [0.0]]),
        torch.tensor([[0.1], [0.0]]),
        torch.tensor([2.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.ones(2),
    )

    assert loss == pytest.approx(0.0025)


def test_v2_distillation_shard_records_temporal_contract(tmp_path: Path) -> None:
    arrays = DistillationArrays(
        observations=np.zeros((2, 4), dtype=np.float32),
        aircraft_parameters=np.zeros((2, 8), dtype=np.float32),
        teacher_actions=np.asarray([[0.0], [0.1]], dtype=np.float32),
        plant_indices=np.zeros(2, dtype=np.int32),
        command_indices=np.zeros(2, dtype=np.int32),
        split_codes=np.asarray([TRAIN_SPLIT, VALIDATION_SPLIT], dtype=np.uint8),
        episode_indices=np.zeros(2, dtype=np.int64),
        policy_step_indices=np.asarray([0, 2], dtype=np.int32),
        driver_actions=np.asarray([[0.0], [0.2]], dtype=np.float32),
    )
    shard = save_distillation_shard(tmp_path / "v2.npz", arrays)
    loaded = load_distillation_shard(shard)

    assert loaded.episode_indices.tolist() == [0, 0]
    assert loaded.policy_step_indices.tolist() == [0, 2]
    assert loaded.driver_actions[:, 0].tolist() == pytest.approx([0.0, 0.2])
    assert DISTILLATION_DATASET_SCHEMA_V2 == "specialist_distillation_dataset_v2"


def test_legacy_distillation_shard_gets_safe_temporal_defaults(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "legacy-v1.npz"
    with shard.open("wb") as stream:
        np.savez_compressed(
            stream,
            observations=np.zeros((3, 4), dtype=np.float32),
            aircraft_parameters=np.zeros((3, 8), dtype=np.float32),
            teacher_actions=np.asarray([[0.0], [0.1], [0.2]], dtype=np.float32),
            plant_indices=np.zeros(3, dtype=np.int32),
            command_indices=np.asarray([4, 4, 5], dtype=np.int32),
            split_codes=np.full(3, TRAIN_SPLIT, dtype=np.uint8),
        )

    loaded = load_distillation_shard(shard)

    assert loaded.episode_indices.tolist() == [4, 4, 5]
    assert loaded.policy_step_indices.tolist() == [0, 1, 0]
    assert loaded.driver_actions == pytest.approx(loaded.teacher_actions)


def test_all_aircraft_command_holdout_is_preserved_in_student_driven_rounds() -> None:
    profile_splits = _profile_splits(
        11,
        "plant-a",
        "extended",
        5.0,
        {"plant-a"},
        "all_aircraft_command_holdout",
    )

    train_ids = [
        profile.command_id for profile, split in profile_splits if split == TRAIN_SPLIT
    ]
    validation_ids = [
        profile.command_id
        for profile, split in profile_splits
        if split == VALIDATION_SPLIT
    ]
    assert len(train_ids) == 11
    assert len(validation_ids) == 6
    assert all(command_id.startswith("train-") for command_id in train_ids)
    assert all(command_id.startswith("eval-") for command_id in validation_ids)
    assert not set(train_ids) & set(validation_ids)


def test_student_round_selection_prioritizes_absolute_tracking_quality() -> None:
    smoother_but_less_accurate = {
        "mean_student_tracking_rmse_deg_s": 0.60,
        "maximum_student_peak_error_deg_s": 4.0,
        "mean_student_requested_force_total_variation_n": 100.0,
    }
    more_accurate = {
        "mean_student_tracking_rmse_deg_s": 0.57,
        "maximum_student_peak_error_deg_s": 3.0,
        "mean_student_requested_force_total_variation_n": 132.0,
    }

    assert _round_score(
        {"by_distillation_split": {"all_aircraft_command_holdout": more_accurate}}
    ) < _round_score(
        {
            "by_distillation_split": {
                "all_aircraft_command_holdout": smoother_but_less_accurate
            }
        }
    )


def test_failed_gate_round_selection_prioritizes_fewer_quality_violations() -> None:
    thresholds = {
        "max_student_teacher_rmse_gap_deg_s": 0.5,
        "minimum_student_improvement_rate": 1.0,
        "maximum_student_harm_rate": 0.0,
        "maximum_student_peak_error_deg_s": 5.0,
        "maximum_mean_student_requested_force_variation_n": 360.0,
        "maximum_student_teacher_force_variation_ratio": 1.25,
    }

    def round_row(
        round_index: int, rmse: float, peak: float, variation_ratio: float
    ) -> dict[str, object]:
        summary = {
            "mean_student_tracking_rmse_deg_s": rmse,
            "median_student_minus_teacher_rmse_rad_s": np.deg2rad(0.2),
            "student_improvement_rate": 1.0,
            "student_harm_rate": 0.0,
            "maximum_student_peak_error_deg_s": peak,
            "mean_student_requested_force_total_variation_n": 250.0,
            "student_teacher_requested_force_variation_ratio": variation_ratio,
        }
        return {
            "round": round_index,
            "closed_loop_evaluation": {
                "by_distillation_split": {"validation_aircraft": summary}
            },
            "quality_checks": {
                "student_teacher_rmse_gap": True,
                "student_improvement_rate": True,
                "student_harm_rate": True,
                "student_peak_error": peak <= 5.0,
                "student_requested_force_variation": True,
                "student_teacher_force_variation_ratio": variation_ratio <= 1.25,
            },
        }

    stable = round_row(0, rmse=0.95, peak=4.75, variation_ratio=2.13)
    slightly_more_accurate = round_row(1, rmse=0.92, peak=5.45, variation_ratio=2.54)
    best, eligible, selection_metric = _select_student_round(
        [stable, slightly_more_accurate], thresholds
    )

    assert best["round"] == 0
    assert eligible == []
    assert selection_metric.startswith("fewest_quality_gate_violations")


def test_dense_student_conditions_on_theta() -> None:
    torch.manual_seed(7)
    model = DenseConditionalStudent(16, width=16, residual_blocks=1)
    observation = torch.full((2, 16), 0.25)
    theta = torch.stack((torch.zeros(8), torch.ones(8)))
    actions = model(observation, theta)

    assert actions.shape == (2, 1)
    assert torch.all(actions.abs() <= 1.0)
    assert not torch.allclose(actions[0], actions[1])


def test_theta_routed_linear_moe_has_no_state_dependent_routing() -> None:
    model = ThetaRoutedLinearMoEStudent(
        7,
        8,
        1,
        torch.stack((torch.zeros(8), torch.ones(8))),
        router_temperature=0.1,
    )
    with torch.no_grad():
        model.expert_weights.normal_(std=0.1)
    theta = torch.full((3, 8), 0.2)
    first_observation = torch.randn(3, 7)
    second_observation = torch.randn(3, 7)

    actions, first_routes, _ = model.forward_with_routing(first_observation, theta)
    _, second_routes, _ = model.forward_with_routing(second_observation, theta)
    redundant_signal_change = first_observation.clone()
    redundant_signal_change[:, [0, 1, 2, 6]] = torch.randn(3, 4)
    mirrored_actions = model(-first_observation, theta)

    torch.testing.assert_close(first_routes, second_routes)
    torch.testing.assert_close(actions, -mirrored_actions)
    torch.testing.assert_close(actions, model(redundant_signal_change, theta))
    torch.testing.assert_close(model(torch.zeros(3, 7), theta), torch.zeros(3, 1))
    assert not torch.allclose(
        model.routing_weights(torch.zeros(1, 8)),
        model.routing_weights(torch.ones(1, 8)),
    )


def test_theta_routed_linear_moe_distillation_checkpoint(tmp_path: Path) -> None:
    rng = np.random.default_rng(13)
    rows_per_plant = 48
    observations = rng.normal(size=(3 * rows_per_plant, 7)).astype(np.float32)
    theta_values = np.asarray(
        [
            [-0.6] * 8,
            [0.6] * 8,
            [0.0] * 8,
        ],
        dtype=np.float32,
    )
    theta = np.repeat(theta_values, rows_per_plant, axis=0)
    coefficients = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.7, 0.2, -0.1, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.3, -0.2, 0.0],
            [0.0, 0.0, 0.0, 0.85, 0.25, -0.15, 0.0],
        ],
        dtype=np.float32,
    )
    plant_indices = np.repeat(np.arange(3), rows_per_plant).astype(np.int32)
    teacher_actions = np.asarray(
        [
            np.clip(observation @ coefficients[plant_index], -1.0, 1.0)
            for observation, plant_index in zip(
                observations, plant_indices, strict=True
            )
        ],
        dtype=np.float32,
    ).reshape(-1, 1)
    split_codes = np.where(plant_indices == 2, VALIDATION_SPLIT, TRAIN_SPLIT)
    arrays = DistillationArrays(
        observations=observations,
        aircraft_parameters=theta,
        teacher_actions=teacher_actions,
        plant_indices=plant_indices,
        command_indices=np.zeros(len(observations), dtype=np.int32),
        split_codes=split_codes.astype(np.uint8),
    )
    dataset_dir = tmp_path / "dataset"
    shard_path = save_distillation_shard(dataset_dir / "shard.npz", arrays)
    manifest = {
        "schema_version": "specialist_distillation_dataset_v1",
        "status": "complete",
        "split_strategy": "aircraft_holdout",
        "aircraft_split_method": "test",
        "observation_dim": 7,
        "actor_observation_contract": {
            "raw_history_steps": 0,
            "uses_raw_history_window": False,
        },
        "aircraft_parameter_dim": 8,
        "action_dim": 1,
        "train_plant_ids": ["train-a", "train-b"],
        "validation_plant_ids": ["validation"],
        "row_count": len(observations),
        "train_rows": 2 * rows_per_plant,
        "validation_rows": rows_per_plant,
        "shards": [
            {
                "path": shard_path.name,
                "sha256": sha256_file(shard_path),
            }
        ],
    }
    manifest_path = dataset_dir / "dataset.json"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    report = train_dense_student(
        manifest_path,
        tmp_path / "student",
        DenseStudentTrainingConfig(
            architecture="theta_routed_linear_moe",
            epochs=3,
            batch_size=32,
            patience_epochs=3,
            moe_expert_count=0,
            moe_prototype_movement_limit=0.0,
            seed=17,
            device="cpu",
        ),
    )
    model, payload = load_dense_student(tmp_path / "student/student.pt")

    assert isinstance(model, ThetaRoutedLinearMoEStudent)
    assert payload["schema_version"] == "theta_routed_linear_moe_student_v2"
    assert payload["control_feature_indices"] == [3, 4, 5]
    assert payload["expert_count"] == 2
    assert payload["temporal_contract"]["raw_history_steps"] == 0
    assert payload["temporal_contract"]["uses_raw_history_window"] is False
    assert report["train_routing"]["router_input"] == "normalized_aircraft_theta_only"
    assert (tmp_path / "student/routing_report.json").is_file()


def test_teacher_and_student_policies_enforce_sign_symmetry() -> None:
    torch.manual_seed(8)
    teacher = SquashedGaussianActor(
        7, 1, width=16, residual_blocks=1, enforce_odd_symmetry=True
    )
    student = DenseConditionalStudent(7, width=16, residual_blocks=1)
    observation = torch.randn(3, 7)
    theta = torch.randn(3, 8)

    teacher_action, _ = teacher.sample(observation, deterministic=True)
    mirrored_teacher_action, _ = teacher.sample(-observation, deterministic=True)
    student_action = student(observation, theta)
    mirrored_student_action = student(-observation, theta)

    torch.testing.assert_close(teacher_action, -mirrored_teacher_action)
    torch.testing.assert_close(student_action, -mirrored_student_action)
    zero_teacher_action, _ = teacher.sample(torch.zeros(3, 7), deterministic=True)
    torch.testing.assert_close(zero_teacher_action, torch.zeros(3, 1))
    torch.testing.assert_close(student(torch.zeros(3, 7), theta), torch.zeros(3, 1))


def test_pid_residual_actor_starts_at_exact_bounded_control_prior() -> None:
    coefficients = torch.zeros((1, 7))
    coefficients[0, 3:6] = torch.tensor([1.5, 0.75, -0.25])
    actor = PIDResidualActor(
        7,
        1,
        coefficients,
        width=16,
        residual_blocks=1,
        residual_action_limit=0.05,
        enforce_odd_symmetry=True,
    )
    observations = torch.randn(32, 7)
    expected = torch.nn.functional.linear(observations, coefficients).clamp(-1.0, 1.0)

    actions, _ = actor.sample(observations, deterministic=True)
    mirrored_actions, _ = actor.sample(-observations, deterministic=True)

    torch.testing.assert_close(actions, expected)
    torch.testing.assert_close(actions, -mirrored_actions)
    assert actor.architecture_name == "pid_initialized_bounded_residual_mlp_v1"


def test_deterministic_actor_enforces_exact_sign_symmetry() -> None:
    torch.manual_seed(9)
    actor = DeterministicActor(
        7, 1, width=16, residual_blocks=1, enforce_odd_symmetry=True
    )
    observations = torch.randn(32, 7)

    actions, log_probability = actor.sample(observations, deterministic=True)
    mirrored_actions, _ = actor.sample(-observations, deterministic=True)
    zero_actions, _ = actor.sample(torch.zeros(4, 7), deterministic=True)

    torch.testing.assert_close(actions, -mirrored_actions)
    torch.testing.assert_close(zero_actions, torch.zeros(4, 1))
    torch.testing.assert_close(log_probability, torch.zeros(32, 1))
    assert actor.architecture_name == "odd_deterministic_residual_mlp_v1"


def test_teacher_defaults_to_inference_only_odd_projection() -> None:
    config = SpecialistTrainingConfig()

    assert config.enforce_odd_policy is True
    assert config.odd_policy_projection_stage == "inference"


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
        history_steps=0,
        network_width=16,
        residual_blocks=1,
        progress_interval_steps=6,
        enforce_quality_gate=False,
        seed=29,
        device="cpu",
    )
    bank = train_teacher_bank(library_path, teacher_dir, teacher_config, count=1)
    assert bank["status"] == "complete"
    assert (teacher_dir / "teacher_bank.json").is_file()
    teacher_entry = bank["teachers"][0]
    teacher_report = json.loads(
        (teacher_dir / teacher_entry["report"]).read_text(encoding="utf-8")
    )
    assert teacher_report["odd_policy_contract"] == {
        "enabled": True,
        "projection_stage": "inference",
        "applied_during_sac_training": False,
        "applied_to_deterministic_teacher": True,
        "applied_to_distillation_labels": True,
    }
    assert teacher_report["training_contract"] == {
        "algorithm": "soft_actor_critic",
        "supervision": "environment_reward_only",
        "uses_pid_demonstrations": False,
        "uses_behavior_cloning": False,
        "uses_embedded_control_prior": False,
        "uses_pid_regularization": False,
        "actor_action": "normalized_direct_full_F_as",
    }
    critic_contract = teacher_report["critic_observation_contract"]
    assert critic_contract["deployment_input"] is False
    assert critic_contract["transport_delay_fifo_width"] > 3
    assert critic_contract["includes_actuator_state"] is True

    data_dir = tmp_path / "distillation"
    dataset_manifest = collect_teacher_bank_data(
        teacher_dir / "teacher_bank.json",
        data_dir,
        DistillationCollectionConfig(sample_stride=20, seed=31, device="cpu"),
    )
    assert dataset_manifest["schema_version"] == DISTILLATION_DATASET_SCHEMA_V2
    assert dataset_manifest["episode_count"] > 0
    assert dataset_manifest["temporal_pair_count"] == (
        dataset_manifest["row_count"] - dataset_manifest["episode_count"]
    )
    assert dataset_manifest["split_strategy"] == "single_aircraft_command_holdout"
    assert dataset_manifest["train_rows"] > 0
    assert dataset_manifest["validation_rows"] > 0
    arrays, _ = load_distillation_arrays(data_dir / "dataset.json")
    assert arrays.observations.shape[1] == 7
    assert arrays.aircraft_parameters.shape[1] == 8
    assert arrays.driver_actions.shape == arrays.teacher_actions.shape
    assert arrays.policy_step_indices.shape == arrays.plant_indices.shape
    observation_contract = dataset_manifest["actor_observation_contract"]
    assert observation_contract["raw_history_steps"] == 0
    assert observation_contract["uses_raw_history_window"] is False
    assert observation_contract["instantaneous_signal_names"] == [
        "p_command_normalized",
        "p_reference_normalized",
        "p_normalized",
        "tracking_error_normalized",
    ]
    assert observation_contract["controller_state_names"] == [
        "integrated_tracking_error_normalized",
        "p_dot_normalized",
        "previous_force_normalized",
    ]

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
    teacher_record = json.loads(
        library_path.read_text(encoding="utf-8").splitlines()[0]
    )
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
    assert (
        evaluation["distillation_split_strategy"] == "single_aircraft_command_holdout"
    )
    assert (
        evaluation["by_distillation_split"]["single_aircraft_command_holdout"][
            "pair_count"
        ]
        == 6
    )
    assert (tmp_path / "student_evaluation/evaluation.json").is_file()

    policy_theta = arrays.aircraft_parameters[0]
    policy = DenseStudentPolicy(model, policy_theta)
    action = policy.predict(np.zeros(7, dtype=np.float32))
    assert action.shape == (1,)

    student_driven = run_student_driven_distillation(
        teacher_dir / "teacher_bank.json",
        tmp_path / "student_driven",
        StudentDrivenDistillationConfig(
            dagger_rounds=1,
            initial_sample_stride=20,
            student_sample_stride=20,
            student_training=DenseStudentTrainingConfig(
                epochs=1,
                batch_size=32,
                network_width=16,
                residual_blocks=1,
                patience_epochs=1,
                seed=41,
                device="cpu",
            ),
            seed=41,
            device="cpu",
            max_student_teacher_rmse_gap_deg_s=1e6,
            minimum_student_improvement_rate=0.0,
            maximum_student_harm_rate=1.0,
        ),
    )
    assert student_driven["status"] == "complete"
    assert student_driven["round_count"] == 2
    assert student_driven["student_driven_round_count"] == 1
    assert student_driven["rounds"][1]["driver"] == "student"
    assert student_driven["rounds"][1]["dataset"]["new_rows"] > 0
    assert (tmp_path / "student_driven/final/dense_student.pt").is_file()
    assert (tmp_path / "student_driven/distillation_progress.png").is_file()
    assert (
        tmp_path
        / "student_driven/round_001_student_driven/evaluation"
        / teacher_record["plant_id"]
        / "teacher_student_comparison.png"
    ).is_file()


def test_pid_guided_td3_specialist_smoke(tmp_path: Path) -> None:
    record = generate_plant_library(41, {"train_core": 1})[0]
    environment_config = SpecialistTrainingConfig(
        episode_duration_s=0.04,
        history_steps=0,
        enforce_quality_gate=False,
        seed=43,
        device="cpu",
    )
    report = train_pid_guided_td3(
        record,
        PIDGains(1.0, 1.0, 0.0),
        tmp_path / "td3",
        environment_config,
        PIDGuidedTD3Config(
            total_steps=8,
            batch_size=4,
            replay_capacity=64,
            behavior_clone_epochs=2,
            behavior_clone_batch_size=8,
            update_interval_steps=2,
            updates_per_interval=1,
            network_width=16,
            residual_blocks=1,
            critic_warmup_updates=0,
            actor_trust_region_l2=1.0,
            progress_interval_steps=4,
            seed=43,
            device="cpu",
        ),
    )

    assert report["status"] == "complete"
    assert report["algorithm"] == "pid_guided_td3"
    assert report["demonstration_steps"] == 12
    assert report["online_updates"] == 4
    assert report["online_actor_updates"] == 2
    assert report["rl_actor_parameter_delta_l2"] > 0.0
    assert report["rl_actor_parameter_delta_l2"] <= 1.0 + 1e-6
    assert report["actor_observation_dim"] == 7
    assert report["actor_observation_contract"]["uses_raw_history_window"] is False
    assert report["critic_observation_contract"]["deployment_input"] is False
    assert report["embedded_linear_control_prior"] is True
    assert report["pid_controller_object_used_at_deployment"] is False
    assert report["initialization_quality_gate"]["passed"] is True
    assert report["quality_gate"]["checks"]["online_actor_updates"] is True
    assert (tmp_path / "td3/teacher_actor.pt").is_file()
    assert (tmp_path / "td3/response_comparison.png").is_file()
    policy, loaded_record, loaded_config, payload = load_specialist_actor(
        tmp_path / "td3/teacher_actor.pt"
    )
    assert loaded_record.plant_id == record.plant_id
    assert loaded_config.network_width == 16
    assert payload["algorithm"] == "pid_guided_td3"
    assert policy.predict(np.zeros(7, dtype=np.float32)).shape == (1,)


def test_pid_guided_td3_rejects_raw_history(tmp_path: Path) -> None:
    record = generate_plant_library(42, {"train_core": 1})[0]
    with pytest.raises(ValueError, match="no raw history"):
        train_pid_guided_td3(
            record,
            PIDGains(1.0, 1.0, 0.0),
            tmp_path / "td3-history",
            SpecialistTrainingConfig(
                episode_duration_s=0.04,
                history_steps=1,
                seed=43,
                device="cpu",
            ),
            PIDGuidedTD3Config(
                total_steps=2,
                batch_size=2,
                replay_capacity=8,
                behavior_clone_epochs=1,
                behavior_clone_batch_size=2,
                network_width=8,
                residual_blocks=1,
                seed=43,
                device="cpu",
            ),
        )


def test_pure_reward_td3_specialist_smoke(tmp_path: Path) -> None:
    record = generate_plant_library(44, {"train_core": 1})[0]
    environment_config = SpecialistTrainingConfig(
        episode_duration_s=0.12,
        history_steps=0,
        requested_action_history_steps=26,
        include_actor_actuator_state=True,
        include_reference_derivative=True,
        critic_include_episode_progress=False,
        critic_include_command_context=False,
        enforce_quality_gate=False,
        seed=45,
        device="cpu",
    )
    report = train_pure_reward_td3(
        record,
        tmp_path / "pure-td3",
        environment_config,
        PureRewardTD3Config(
            total_steps=12,
            warmup_steps=4,
            batch_size=4,
            replay_capacity=64,
            exploration_std_initial=0.1,
            exploration_std_final=0.01,
            exploration_decay_steps=8,
            network_width=16,
            residual_blocks=1,
            critic_warmup_updates=0,
            progress_interval_steps=4,
            evaluation_interval_steps=4,
            random_command_sequence=True,
            random_sequence_segment_duration_range_s=(0.04, 0.06),
            long_dwell_step_probability=0.5,
            long_dwell_duration_range_s=(0.04, 0.10),
            random_command_distribution=RandomCommandDistribution(
                duration_range_s=(0.04, 0.04),
                onset_range_s=(0.005, 0.010),
                doublet_segment_range_s=(0.005, 0.010),
            ),
            seed=45,
            device="cpu",
        ),
    )

    assert report["status"] == "complete"
    assert report["algorithm"] == "pure_reward_td3"
    assert report["steps"] == 12
    assert report["updates"] == 8
    assert report["actor_updates"] == 4
    assert report["rl_actor_parameter_delta_l2"] > 0.0
    assert report["actor_observation_dim"] == 35
    assert report["actor_uses_raw_history"] is False
    assert report["actor_uses_requested_action_history"] is True
    assert report["privileged_critic_training_only"] is True
    assert report["training_contract"] == {
        "algorithm": "twin_delayed_deep_deterministic_policy_gradient",
        "supervision": "environment_reward_only",
        "uses_pid_demonstrations": False,
        "uses_behavior_cloning": False,
        "uses_embedded_control_prior": False,
        "uses_pid_regularization": False,
        "uses_entropy_regularization": False,
        "actor_action": "normalized_direct_full_F_as",
    }
    assert report["odd_policy_contract"]["applied_during_td3_training"] is True
    assert report["continuing_task_contract"] == {
        "task_type": "continuing_flight_control",
        "episode_reset_condition": "terminated_or_truncated",
        "bellman_terminal_mask": "terminated_only",
        "bootstraps_across_time_limit_truncation": True,
        "critic_includes_artificial_episode_progress": False,
    }
    assert (
        report["training_command_scheduler"]
        == "mixed_long_dwell_step_and_random_sequence_v1"
    )
    assert report["command_sampling_contract"]["sampled_profile_count"] > 1
    assert report["command_sampling_contract"]["sampled_segment_count"] > 1
    assert [point["step"] for point in report["learning_curve"]] == [0, 4, 8, 12]
    assert report["best_validation"] in report["learning_curve"]
    assert report["quality_gate"]["checks"]["online_actor_updates"] is True
    assert (tmp_path / "pure-td3/teacher_actor.pt").is_file()
    assert (tmp_path / "pure-td3/training_checkpoint.pt").is_file()
    assert (tmp_path / "pure-td3/learning_curve.json").is_file()
    validation_checkpoints = sorted(
        (tmp_path / "pure-td3/validation_checkpoints").glob("*.pt")
    )
    assert len(validation_checkpoints) == 4
    assert Path(report["best_validation_actor_checkpoint"]).is_file()
    validation_policy, validation_record, _, validation_payload = load_specialist_actor(
        validation_checkpoints[1]
    )
    assert validation_record.plant_id == record.plant_id
    assert validation_payload["step"] == 4
    assert validation_policy.predict(np.zeros(35, dtype=np.float32)).shape == (1,)

    policy, loaded_record, loaded_config, payload = load_specialist_actor(
        tmp_path / "pure-td3/teacher_actor.pt"
    )
    assert loaded_record.plant_id == record.plant_id
    assert loaded_config.network_width == 16
    assert payload["algorithm"] == "pure_reward_td3"
    observation = np.linspace(-0.5, 0.5, 35, dtype=np.float32)
    assert policy.predict(observation) == pytest.approx(-policy.predict(-observation))
    checkpoint = torch.load(
        tmp_path / "pure-td3/training_checkpoint.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["schema_version"] == "pure_reward_td3_checkpoint_v1"


def test_pure_reward_td3_rejects_raw_history(tmp_path: Path) -> None:
    record = generate_plant_library(46, {"train_core": 1})[0]
    with pytest.raises(ValueError, match="no raw history"):
        train_pure_reward_td3(
            record,
            tmp_path / "pure-td3-history",
            SpecialistTrainingConfig(
                episode_duration_s=0.04,
                history_steps=1,
                seed=47,
                device="cpu",
            ),
            PureRewardTD3Config(
                total_steps=2,
                warmup_steps=1,
                batch_size=2,
                replay_capacity=8,
                network_width=8,
                residual_blocks=1,
                seed=47,
                device="cpu",
            ),
        )


def test_pid_guided_td3_bank_is_distillation_loadable(tmp_path: Path) -> None:
    library_dir = persist_plant_library(tmp_path / "library", 47, {"train_core": 1})
    library_path = library_dir / "plants.jsonl"
    plant_id = json.loads(library_path.read_text(encoding="utf-8").splitlines()[0])[
        "plant_id"
    ]
    record = load_persisted_records(library_path, [plant_id])[0]
    environment_config = SpecialistTrainingConfig(
        episode_duration_s=0.04,
        history_steps=0,
        enforce_quality_gate=False,
        quality_gate_minimum_improvement_rate=0.0,
        quality_gate_maximum_mean_rmse_deg_s=1e6,
        quality_gate_maximum_peak_error_deg_s=1e6,
        quality_gate_maximum_mean_requested_force_variation_n=1e6,
        quality_gate_maximum_saturation_fraction=1.0,
        seed=53,
        device="cpu",
    )
    pid_root = tmp_path / "pid"
    pid_report_path = pid_root / record.plant_id / "pid_report.json"
    pid_report_path.parent.mkdir(parents=True)
    pid_report_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "controller": "direct_reference_tracking_pid",
                "plant_id": record.plant_id,
                "plant_parameters": asdict(record.parameters),
                "gains": {
                    "proportional": 1.0,
                    "integral": 1.0,
                    "derivative": 0.0,
                },
                "episode_duration_s": environment_config.episode_duration_s,
                "plant_dt_s": environment_config.plant_dt_s,
                "policy_dt_s": environment_config.policy_dt_s,
                "reference_natural_frequency_rad_s": (
                    environment_config.reference_natural_frequency_rad_s
                ),
                "reference_damping_ratio": (environment_config.reference_damping_ratio),
                "reference_delay_mode": environment_config.reference_delay_mode,
                "evaluation": {
                    "tracking_improvement_rate": 1.0,
                    "mean_PID_tracking_rmse_deg_s": 0.0,
                    "maximum_PID_peak_error_deg_s": 0.0,
                    "mean_PID_requested_force_total_variation_n": 0.0,
                    "mean_PID_force_saturation_fraction": 0.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bank_dir = tmp_path / "bank"
    result = train_pid_guided_teacher_bank(
        library_path,
        pid_root,
        bank_dir,
        [record],
        environment_config,
        PIDGuidedTD3Config(
            total_steps=4,
            batch_size=2,
            replay_capacity=64,
            behavior_clone_epochs=1,
            behavior_clone_batch_size=4,
            update_interval_steps=1,
            updates_per_interval=1,
            network_width=8,
            residual_blocks=1,
            critic_warmup_updates=0,
            actor_trust_region_l2=1.0,
            progress_interval_steps=2,
            seed=53,
            device="cpu",
        ),
    )

    assert result["status"] == "complete"
    assert result["accepted_teacher_count"] == 1
    assert result["teachers"][0]["algorithm"] == "pid_guided_td3"
    dataset = collect_teacher_bank_data(
        bank_dir / "teacher_bank.json",
        tmp_path / "bank-dataset",
        DistillationCollectionConfig(sample_stride=2, seed=53, device="cpu"),
    )
    assert dataset["observation_dim"] == 7
    assert dataset["row_count"] > 0
