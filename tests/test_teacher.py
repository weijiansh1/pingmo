import torch
import numpy as np
import pytest

from src.teacher.moe.actor import MoEActor
from src.teacher.moe.teacher import MoETeacher
from src.teacher.sac.actor import SquashedGaussianActor
from src.teacher.sac.critic import TwinQCritic
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.sac.teacher import PrivilegedSAC


def _small_sac(actor_dim: int = 13, critic_dim: int = 25) -> PrivilegedSAC:
    return PrivilegedSAC(
        actor_observation_dim=actor_dim,
        critic_observation_dim=critic_dim,
        action_dim=1,
        actor_width=32,
        actor_residual_blocks=2,
        critic_width=32,
        critic_residual_blocks=2,
    )


def _small_moe(actor_dim: int = 13, critic_dim: int = 25) -> MoETeacher:
    return MoETeacher(
        actor_observation_dim=actor_dim,
        critic_observation_dim=critic_dim,
        action_dim=1,
        experts=4,
        actor_width=32,
        shared_residual_blocks=2,
        expert_residual_blocks=1,
        expert_bottleneck_width=16,
        critic_width=32,
        critic_residual_blocks=2,
    )


def test_privileged_sac_uses_critic_only_privileged_features() -> None:
    torch.manual_seed(3)
    learner = _small_sac()
    actor_observation = torch.zeros(1, 13)
    action = learner.act(actor_observation, deterministic=True)
    q_without_privilege = learner.q_values(torch.zeros(1, 25), action)[0]
    q_with_privilege = learner.q_values(torch.cat((torch.zeros(1, 13), torch.ones(1, 12)), dim=1), action)[0]

    assert action.shape == (1, 1)
    assert torch.allclose(action, learner.act(actor_observation, deterministic=True))
    assert not torch.allclose(q_without_privilege, q_with_privilege)


def test_two_stream_replay_preserves_actor_and_critic_states() -> None:
    replay = TwoStreamReplayBuffer(capacity=4, actor_observation_dim=13, critic_observation_dim=25, action_dim=1, seed=5)
    replay.add(np.full(13, 1.0, dtype=np.float32), np.full(25, 2.0, dtype=np.float32), np.array([0.25], dtype=np.float32), 3.0, np.full(13, 4.0, dtype=np.float32), np.full(25, 5.0, dtype=np.float32), False)
    batch = replay.sample(1, device="cpu")

    assert batch["actor_obs"].shape == (1, 13)
    assert batch["critic_obs"].shape == (1, 25)
    assert batch["next_actor_obs"].shape == (1, 13)
    assert batch["next_critic_obs"].shape == (1, 25)
    assert batch["critic_obs"][0, 0].item() == 2.0
    assert batch["next_critic_obs"][0, 0].item() == 5.0


def test_two_stream_replay_add_batch_wraps_without_losing_shape() -> None:
    replay = TwoStreamReplayBuffer(capacity=4, actor_observation_dim=2, critic_observation_dim=3, action_dim=1, seed=9)
    for offset in (0.0, 2.0, 4.0):
        actor = np.arange(offset, offset + 4, dtype=np.float32).reshape(2, 2)
        critic = np.arange(offset, offset + 6, dtype=np.float32).reshape(2, 3)
        replay.add_batch(
            actor,
            critic,
            np.zeros((2, 1), dtype=np.float32),
            np.array([offset, offset + 1], dtype=np.float32),
            actor + 1,
            critic + 1,
            np.zeros(2, dtype=np.float32),
        )

    assert len(replay) == 4
    assert set(replay.rewards[:, 0]) == {2.0, 3.0, 4.0, 5.0}


def test_privileged_sac_update_has_finite_losses() -> None:
    learner = _small_sac()
    batch = {
        "actor_obs": torch.randn(16, 13), "critic_obs": torch.randn(16, 25), "action": torch.tanh(torch.randn(16, 1)),
        "reward": torch.randn(16, 1), "next_actor_obs": torch.randn(16, 13), "next_critic_obs": torch.randn(16, 25), "done": torch.zeros(16, 1),
    }
    losses = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_moe_actor_mixes_features_with_theta_only_router() -> None:
    actor = MoEActor(
        observation_dim=13,
        action_dim=1,
        experts=4,
        model_width=32,
        shared_residual_blocks=2,
        expert_residual_blocks=1,
        expert_bottleneck_width=16,
    )
    action, log_prob, weights = actor.sample(torch.randn(8, 13))
    assert action.shape == (8, 1)
    assert log_prob.shape == (8, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(8))
    assert torch.all(action.abs() <= 1.0)


def test_moe_teacher_update_reports_balance_loss_and_router_usage() -> None:
    learner = _small_moe()
    batch = {"actor_obs": torch.randn(16, 13), "critic_obs": torch.randn(16, 25), "action": torch.tanh(torch.randn(16, 1)), "reward": torch.randn(16, 1), "next_actor_obs": torch.randn(16, 13), "next_critic_obs": torch.randn(16, 25), "done": torch.zeros(16, 1)}
    losses = learner.update(batch)
    assert losses["balance_loss"] >= 0.0
    assert 0.0 < losses["router_max_mean"] <= 1.0


def test_moe_teacher_critic_reads_privileged_state_only() -> None:
    torch.manual_seed(4)
    learner = _small_moe()
    actor_obs = torch.zeros(1, 13)
    action, _, _ = learner.actor.sample(actor_obs)
    q_without_privilege = learner.q_values(torch.zeros(1, 25), action)[0]
    q_with_privilege = learner.q_values(torch.cat((torch.zeros(1, 13), torch.ones(1, 12)), dim=1), action)[0]

    assert not torch.allclose(q_without_privilege, q_with_privilege)


def test_moe_router_reads_theta_but_not_response_state() -> None:
    actor = MoEActor(
        observation_dim=13,
        action_dim=1,
        experts=4,
        model_width=32,
        shared_residual_blocks=2,
        expert_residual_blocks=1,
        expert_bottleneck_width=16,
    )
    first = torch.randn(4, 13)
    second = first.clone()
    second[:, :-8] = torch.randn_like(second[:, :-8])
    _, _, first_weights = actor.sample(first, deterministic=True)
    _, _, second_weights = actor.sample(second, deterministic=True)

    assert torch.allclose(first_weights, second_weights)


def test_deep_residual_actor_propagates_finite_gradients() -> None:
    actor = SquashedGaussianActor(13, 1, width=32, residual_blocks=10)
    action, log_probability = actor.sample(torch.randn(8, 13))
    loss = action.square().mean() + log_probability.mean()
    loss.backward()

    gradients = [parameter.grad for parameter in actor.parameters() if parameter.requires_grad]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_production_residual_mlp_and_moe_have_matched_parameter_budgets() -> None:
    mlp_actor = SquashedGaussianActor(268, 1)
    moe_actor = MoEActor(268, 1, experts=4)
    critic = TwinQCritic(273, 1)

    assert mlp_actor.parameter_count == 22_773_634
    assert moe_actor.parameter_count == 23_046_790
    assert critic.parameter_count == 45_556_226
    assert mlp_actor.parameter_count + critic.parameter_count + 1 == 68_329_861
    assert moe_actor.parameter_count + critic.parameter_count + 1 == 68_603_017
    assert moe_actor.parameter_count / mlp_actor.parameter_count == pytest.approx(1.01199, rel=1e-4)
