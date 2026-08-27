import torch
import numpy as np

from src.teacher.moe.actor import MoEActor
from src.teacher.moe.teacher import MoETeacher
from src.teacher.sac.replay import TwoStreamReplayBuffer
from src.teacher.sac.teacher import PrivilegedSAC


def test_privileged_sac_uses_critic_only_privileged_features() -> None:
    torch.manual_seed(3)
    learner = PrivilegedSAC(actor_observation_dim=13, critic_observation_dim=25, action_dim=1)
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


def test_privileged_sac_update_has_finite_losses() -> None:
    learner = PrivilegedSAC(actor_observation_dim=13, critic_observation_dim=25, action_dim=1)
    batch = {
        "actor_obs": torch.randn(16, 13), "critic_obs": torch.randn(16, 25), "action": torch.tanh(torch.randn(16, 1)),
        "reward": torch.randn(16, 1), "next_actor_obs": torch.randn(16, 13), "next_critic_obs": torch.randn(16, 25), "done": torch.zeros(16, 1),
    }
    losses = learner.update(batch)
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_moe_actor_mixes_features_with_theta_only_router() -> None:
    actor = MoEActor(observation_dim=13, action_dim=1, experts=4)
    action, log_prob, weights = actor.sample(torch.randn(8, 13))
    assert action.shape == (8, 1)
    assert log_prob.shape == (8, 1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(8))
    assert torch.all(action.abs() <= 1.0)


def test_moe_teacher_update_reports_balance_loss_and_router_usage() -> None:
    learner = MoETeacher(actor_observation_dim=13, critic_observation_dim=25, action_dim=1, experts=4)
    batch = {"actor_obs": torch.randn(16, 13), "critic_obs": torch.randn(16, 25), "action": torch.tanh(torch.randn(16, 1)), "reward": torch.randn(16, 1), "next_actor_obs": torch.randn(16, 13), "next_critic_obs": torch.randn(16, 25), "done": torch.zeros(16, 1)}
    losses = learner.update(batch)
    assert losses["balance_loss"] >= 0.0
    assert 0.0 < losses["router_max_mean"] <= 1.0


def test_moe_teacher_critic_reads_privileged_state_only() -> None:
    torch.manual_seed(4)
    learner = MoETeacher(actor_observation_dim=13, critic_observation_dim=25, action_dim=1, experts=4)
    actor_obs = torch.zeros(1, 13)
    action, _, _ = learner.actor.sample(actor_obs)
    q_without_privilege = learner.q_values(torch.zeros(1, 25), action)[0]
    q_with_privilege = learner.q_values(torch.cat((torch.zeros(1, 13), torch.ones(1, 12)), dim=1), action)[0]

    assert not torch.allclose(q_without_privilege, q_with_privilege)
