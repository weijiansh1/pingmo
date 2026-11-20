"""CPU regression checks for PPO samples and time-limit value bootstrapping."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aircraft.parameters import PChannelParameters
from src.envs.batched_p_channel import (
    BatchedRollRateEnv, EPISODE_STEPS, OBS_DIM, build_command_set,
)
from src.teacher.ppo.networks import PPOTeacherActor
from src.teacher.ppo import trainer

torch.set_num_threads(1)


def parameters():
    return PChannelParameters(15.0, -0.02, 0.4, 0.5, 1.2, 1.05, 1.0, 0.04)


def actor():
    return PPOTeacherActor(width=8, blocks=1, conv_width=2, initial_log_std=1.0)


class PPORegressionTests(unittest.TestCase):
    def test_saturated_samples_have_unit_ratio_before_any_update(self):
        torch.manual_seed(42)
        policy = actor()
        obs = torch.randn(256, OBS_DIM)
        priv = torch.randn(256, 12)
        with torch.no_grad():
            bounded, old_logp, _, raw = policy.sample_for_ppo(obs, priv)
            new_logp, _, _ = policy.evaluate(obs, priv, raw)
        self.assertTrue((raw > 1).any().item())
        self.assertTrue((raw < -1).any().item())
        self.assertLessEqual(bounded.abs().max().item(), 1.0)
        torch.testing.assert_close(bounded, raw.clamp(-1, 1))
        torch.testing.assert_close((new_logp-old_logp).exp(), torch.ones_like(old_logp))
        # Reproduce the old failure under the same unchanged policy.
        with torch.no_grad():
            wrong_logp, _, _ = policy.evaluate(obs, priv, bounded)
        self.assertGreater(((wrong_logp-old_logp).exp()-1).abs().max().item(), 0.1)

    def test_probability_update_has_finite_gradients_for_stored_raw_actions(self):
        torch.manual_seed(7)
        policy = actor()
        obs = torch.randn(32, OBS_DIM)
        priv = torch.randn(32, 12)
        with torch.no_grad():
            _, old_logp, _, raw = policy.sample_for_ppo(obs, priv)
        new_logp, entropy, _ = policy.evaluate(obs, priv, raw)
        loss = -((new_logp-old_logp).exp()*torch.linspace(-1, 1, 32)[:, None]).mean() - .001*entropy.mean()
        loss.backward()
        grads = [p.grad for p in policy.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_existing_deterministic_api_and_checkpoint_keys_are_compatible(self):
        spec = importlib.util.spec_from_file_location(
            "archived_ppo_networks", ROOT/'before/src/teacher/ppo/networks.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        previous = old.PPOTeacherActor(width=8, blocks=1, conv_width=2)
        fixed = actor()
        fixed.load_state_dict(previous.state_dict(), strict=True)
        obs, priv = torch.randn(4, OBS_DIM), torch.randn(4, 12)
        with torch.no_grad():
            expected = previous.act(obs, priv, deterministic=True)
            actual = fixed.act(obs, priv, deterministic=True)
        self.assertEqual(len(actual), 3)
        self.assertEqual(list(previous.state_dict()), list(fixed.state_dict()))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)

    def make_env(self):
        env = BatchedRollRateEnv(parameters(), 2, commands=build_command_set()[:1], device='cpu', seed=4)
        env.t[:] = EPISODE_STEPS-1
        env.state[:] = .01
        env.requested[:] = 1.0
        return env

    def test_final_state_is_retained_for_time_limit_bootstrap(self):
        env = self.make_env()
        obs, _, output = env.step(torch.full((2, 1), .25), auto_reset=False)
        self.assertTrue((env.t == EPISODE_STEPS).all().item())
        self.assertGreater(env.state.abs().sum().item(), 0)
        torch.testing.assert_close(obs, env.observation())
        torch.testing.assert_close(obs[:, 2], output / trainer.COMMAND_SCALE_RAD_S)
        torch.testing.assert_close(env.privileged()[:, -2], torch.ones(2))
        self.assertGreater(env.requested.abs().sum().item(), 0)
        env.reset()
        self.assertTrue((env.t == 0).all().item())
        self.assertEqual(env.state.abs().sum().item(), 0)

    def test_existing_automatic_reset_and_dynamics_are_preserved(self):
        retained, automatic = self.make_env(), self.make_env()
        action = torch.full((2, 1), .25)
        _, reward_a, output_a = retained.step(action, auto_reset=False)
        obs, reward_b, output_b = automatic.step(action)
        torch.testing.assert_close(reward_a, reward_b)
        torch.testing.assert_close(output_a, output_b)
        self.assertTrue((automatic.t == 0).all().item())
        self.assertEqual(automatic.state.abs().sum().item(), 0)
        torch.testing.assert_close(obs, automatic.observation())

    def test_default_reset_only_resets_truncated_rows(self):
        env = self.make_env()
        env.t[1] -= 1
        env.step(torch.zeros(2, 1))
        self.assertEqual(env.t.tolist(), [0, EPISODE_STEPS-1])
        self.assertEqual(env.state[0].abs().sum().item(), 0)
        self.assertGreater(env.state[1].abs().sum().item(), 0)

    def test_gae_uses_the_provided_final_state_value(self):
        adv, ret = trainer._gae(torch.ones(2, 1), torch.zeros(2, 1),
                                torch.tensor([5.0]), .9, 1.0)
        expected = torch.tensor([[5.95], [5.5]])
        torch.testing.assert_close(adv, expected)
        torch.testing.assert_close(ret, expected)

    def test_trainer_passes_raw_samples_and_bootstraps_before_reset(self):
        # Two synthetic transitions exercise the real collection/update wiring.
        # Adam.step is disabled: no policy is trained or persisted as a result.
        seen_raw, bootstrap_values, auto_reset_flags = [], [], []

        class Env:
            def __init__(self, p, n_envs, **kwargs):
                self.n_envs = n_envs
                self.reset()
            def reset(self):
                self.t = 0
            def observation(self):
                return torch.full((self.n_envs, 1), float(self.t))
            def actor_privileged(self):
                return torch.zeros(self.n_envs, 12)
            def privileged(self):
                return self.observation()
            def step(self, action, *, auto_reset=True):
                auto_reset_flags.append(auto_reset)
                self.t += 1
                if auto_reset and self.t >= 2:
                    self.reset()
                return self.observation(), torch.ones(self.n_envs), torch.zeros(self.n_envs)

        class Actor(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.mean = nn.Parameter(torch.zeros(()))
            def sample_for_ppo(self, obs, priv):
                raw = torch.full((len(obs), 1), 2.0)
                dist = torch.distributions.Normal(self.mean, torch.ones_like(raw))
                return raw.clamp(-1, 1), dist.log_prob(raw), self.mean.expand_as(raw), raw
            def evaluate(self, obs, priv, raw):
                seen_raw.append(raw.detach().clone())
                dist = torch.distributions.Normal(self.mean, torch.ones_like(raw))
                return dist.log_prob(raw), dist.entropy(), self.mean.expand_as(raw)

        class Critic(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(()))
            def forward(self, obs, priv):
                return priv[:, :1] + self.bias

        gae = trainer._gae
        def capture_gae(rewards, values, bootstrap, *args):
            bootstrap_values.append(bootstrap.clone())
            return gae(rewards, values, bootstrap, *args)

        cfg = trainer.PPOTrainingConfig(total_rollouts=1, n_envs=2, epochs=1,
                                        minibatch=4, device='cpu', amp=False)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(trainer, 'EPISODE_STEPS', 2), \
             patch.object(trainer, 'BatchedRollRateEnv', Env), \
             patch.object(trainer, 'PPOTeacherActor', Actor), \
             patch.object(trainer, 'PPOTeacherCritic', Critic), \
             patch.object(trainer, '_gae', side_effect=capture_gae), \
             patch.object(trainer, 'evaluate', return_value={'rmse_deg_s': 0.0}), \
             patch.object(torch.optim.Adam, 'step', return_value=None):
            result = trainer.train(parameters(), cfg, Path(tmp), verbose=False)
            summary = json.loads((Path(tmp)/'summary.json').read_text())
            self.assertEqual(result['ppo_protocol'], trainer.PPO_TRAINING_PROTOCOL)
            self.assertEqual(summary['ppo_protocol'], trainer.PPO_TRAINING_PROTOCOL)
        self.assertEqual(auto_reset_flags, [False, False])
        self.assertEqual(len(seen_raw), 1)
        torch.testing.assert_close(seen_raw[0], torch.full((4, 1), 2.0))
        torch.testing.assert_close(bootstrap_values[0], torch.full((2,), 2.0))


if __name__ == '__main__':
    unittest.main()
