"""Fixed-shape replay storage for separate actor and privileged critic states."""

import numpy as np
import torch


class TwoStreamReplayBuffer:
    def __init__(self, capacity: int, actor_observation_dim: int, critic_observation_dim: int, action_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self.actor_obs = np.zeros((capacity, actor_observation_dim), dtype=np.float32)
        self.critic_obs = np.zeros((capacity, critic_observation_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_actor_obs = np.zeros((capacity, actor_observation_dim), dtype=np.float32)
        self.next_critic_obs = np.zeros((capacity, critic_observation_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._position = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, actor_obs: np.ndarray, critic_obs: np.ndarray, action: np.ndarray, reward: float, next_actor_obs: np.ndarray, next_critic_obs: np.ndarray, done: bool) -> None:
        index = self._position
        self.actor_obs[index] = actor_obs
        self.critic_obs[index] = critic_obs
        self.actions[index] = action
        self.rewards[index, 0] = reward
        self.next_actor_obs[index] = next_actor_obs
        self.next_critic_obs[index] = next_critic_obs
        self.dones[index, 0] = float(done)
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str | torch.device) -> dict[str, torch.Tensor]:
        if batch_size > self._size:
            raise ValueError("cannot sample more transitions than stored")
        indices = self._rng.integers(self._size, size=batch_size)
        return {
            "actor_obs": torch.as_tensor(self.actor_obs[indices], device=device),
            "critic_obs": torch.as_tensor(self.critic_obs[indices], device=device),
            "action": torch.as_tensor(self.actions[indices], device=device),
            "reward": torch.as_tensor(self.rewards[indices], device=device),
            "next_actor_obs": torch.as_tensor(self.next_actor_obs[indices], device=device),
            "next_critic_obs": torch.as_tensor(self.next_critic_obs[indices], device=device),
            "done": torch.as_tensor(self.dones[indices], device=device),
        }
