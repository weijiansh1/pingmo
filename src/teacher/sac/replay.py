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
        self.add_batch(
            np.asarray(actor_obs)[None, :],
            np.asarray(critic_obs)[None, :],
            np.asarray(action)[None, :],
            np.asarray([reward], dtype=np.float32),
            np.asarray(next_actor_obs)[None, :],
            np.asarray(next_critic_obs)[None, :],
            np.asarray([done], dtype=np.float32),
        )

    def add_batch(
        self,
        actor_obs: np.ndarray,
        critic_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_actor_obs: np.ndarray,
        next_critic_obs: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Append one synchronous vector-environment transition batch."""

        batch_size = int(np.asarray(actor_obs).shape[0])
        if batch_size <= 0 or batch_size > self.capacity:
            raise ValueError("batch size must be between one and replay capacity")
        arrays = (critic_obs, actions, rewards, next_actor_obs, next_critic_obs, dones)
        if any(int(np.asarray(array).shape[0]) != batch_size for array in arrays):
            raise ValueError("all replay batch arrays must have the same leading dimension")

        indices = (self._position + np.arange(batch_size)) % self.capacity
        self.actor_obs[indices] = actor_obs
        self.critic_obs[indices] = critic_obs
        self.actions[indices] = actions
        self.rewards[indices, 0] = np.asarray(rewards).reshape(batch_size)
        self.next_actor_obs[indices] = next_actor_obs
        self.next_critic_obs[indices] = next_critic_obs
        self.dones[indices, 0] = np.asarray(dones).reshape(batch_size)
        self._position = (self._position + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

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
