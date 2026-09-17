"""Smoke-test the P1 advantage-weighted distillation machinery without gymnasium.

Covers the pieces that do not import the training environment:
``SpecialistCriticPolicy`` / ``load_specialist_critic`` (critic reconstruction
and Q-value queries from a synthetic SAC checkpoint) and the advantage factor
folded into ``DistillationDataset.sample_weights``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.critic import (
    SpecialistCriticPolicy,
    _infer_critic_shape,
    load_specialist_critic,
)
from src.distillation.dataset import (
    DistillationArrays,
    DistillationDataset,
    TRAIN_SPLIT,
)
from src.teacher.sac.critic import TwinQCritic


def _synthetic_arrays(advantages: np.ndarray | None = None) -> DistillationArrays:
    obs = np.random.RandomState(0).randn(6, 7).astype(np.float32)
    theta = np.random.RandomState(1).randn(6, 8).astype(np.float32) * 0.1
    teacher = np.clip(np.random.RandomState(2).randn(6, 1), -1, 1).astype(np.float32)
    if advantages is None:
        advantages = np.zeros((6, 1), dtype=np.float32)
    return DistillationArrays(
        observations=obs,
        aircraft_parameters=theta,
        teacher_actions=teacher,
        plant_indices=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        command_indices=np.array([0, 0, 0, 0, 0, 0], dtype=np.int32),
        split_codes=np.full(6, TRAIN_SPLIT, dtype=np.uint8),
        episode_indices=np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
        policy_step_indices=np.array([0, 1, 2, 0, 1, 2], dtype=np.int32),
        driver_actions=teacher,
        advantages=advantages,
    )


def test_infer_critic_shape() -> None:
    critic = TwinQCritic(5, 1, width=16, residual_blocks=3)
    dim, width, blocks = _infer_critic_shape(critic.state_dict())
    assert (dim, width, blocks) == (5, 16, 3)


def test_load_specialist_critic() -> None:
    critic = TwinQCritic(5, 1, width=16, residual_blocks=2)
    critic.eval()
    payload = {
        "schema_version": "specialist_actor_v1",
        "critic": critic.state_dict(),
        "critic_observation_contract": {"names": ["a", "b", "c", "d", "e"]},
    }
    path = Path("test_v7_critic_checkpoint.pt")
    torch.save(payload, path)
    try:
        policy, contract = load_specialist_critic(path, device="cpu")
        assert contract == {"names": ["a", "b", "c", "d", "e"]}
        state = np.random.RandomState(3).randn(4, 5).astype(np.float32)
        action = np.random.RandomState(4).randn(4, 1).astype(np.float32)
        q1, q2 = policy.q_values(state, action)
        assert q1.shape == (4,) and q2.shape == (4,)
        # Reconstructed critic must reproduce the original network exactly.
        with torch.no_grad():
            ref1, ref2 = critic(
                torch.as_tensor(state), torch.as_tensor(action)
            )
        assert np.allclose(q1, ref1.cpu().numpy().reshape(-1), atol=1e-6)
        assert np.allclose(q2, ref2.cpu().numpy().reshape(-1), atol=1e-6)
    finally:
        path.unlink(missing_ok=True)


def test_advantage_sign_convention() -> None:
    critic = TwinQCritic(5, 1, width=16, residual_blocks=2)
    policy = SpecialistCriticPolicy(critic, "cpu")
    state = np.random.RandomState(5).randn(3, 5).astype(np.float32)
    teacher = np.random.RandomState(6).randn(3, 1).astype(np.float32)
    student = np.random.RandomState(7).randn(3, 1).astype(np.float32)
    advantage = policy.advantage(state, teacher, student)
    expected = policy.q_min(state, teacher) - policy.q_min(state, student)
    assert advantage.shape == (3,)
    assert np.allclose(advantage, expected, atol=1e-6)


def test_advantage_weight_folding() -> None:
    advantages = np.array(
        [[0.0], [0.5], [-0.5], [1.5], [-2.0], [0.2]], dtype=np.float32
    )
    arrays = _synthetic_arrays(advantages)
    dataset = DistillationDataset(
        arrays,
        "train",
        hard_case_weight_boost=0.0,
        advantage_weight=2.0,
        advantage_clip=1.0,
    )
    expected_factor = np.exp(2.0 * np.clip(advantages.reshape(-1), -1.0, 1.0))
    assert np.allclose(
        dataset.advantage_factors.numpy(), expected_factor.astype(np.float32), atol=1e-5
    )
    # With zero hard-case boost, sample weight == advantage factor.
    assert np.allclose(
        dataset.sample_weights.numpy(), expected_factor.astype(np.float32), atol=1e-5
    )
    sample = dataset[2]
    assert torch.allclose(
        sample["advantage"], torch.tensor(advantages[2, 0], dtype=torch.float32)
    )
    # Disabling advantage weighting must recover the base weight of one.
    off = DistillationDataset(
        arrays, "train", hard_case_weight_boost=0.0, advantage_weight=0.0
    )
    assert np.allclose(off.sample_weights.numpy(), np.ones(6, dtype=np.float32))


def main() -> None:
    test_infer_critic_shape()
    test_load_specialist_critic()
    test_advantage_sign_convention()
    test_advantage_weight_folding()
    print("V7 advantage-weighting smoke test: PASS")


if __name__ == "__main__":
    main()
