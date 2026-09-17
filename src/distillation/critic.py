"""Specialist Teacher critic access for advantage-weighted distillation (P1).

The specialist Teachers are SAC/TD3 learners whose deployable Actor never sees
privileged plant state, but whose twin critic does. This module re-loads that
critic and exposes Q-value queries so the distillation loop can weight each
collected transition by the Teacher's advantage ``Q(s, u^T) - Q(s, u^S)``.

It is deliberately free of the gymnasium/``specialist`` environment dependency so
the loading and weighting machinery can be smoke-tested on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.teacher.sac.critic import TwinQCritic


def _infer_critic_shape(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Recover (critic_observation_dim, width, residual_blocks) from a state dict.

    Both the SAC and TD3 Teachers use ``TwinQCritic``, whose first layer is
    ``q1.body.projection.weight`` of shape ``[width, critic_observation_dim + action_dim]``.
    This inference keeps the loader independent of the two Teachers' differing
    checkpoint layouts (SAC stores the critic in the actor file; TD3 in a sibling
    full checkpoint) and their differing config-key names.
    """

    projection = state_dict.get("q1.body.projection.weight")
    if projection is None:
        raise ValueError("critic state dict is missing the TwinQCritic projection layer")
    width = int(projection.shape[0])
    input_dim = int(projection.shape[1])
    block_count = sum(
        1
        for key in state_dict
        if key.startswith("q1.body.blocks.") and key.endswith(".norm.weight")
    )
    if block_count <= 0:
        raise ValueError("critic state dict has no residual blocks")
    return input_dim - 1, width, block_count


class SpecialistCriticPolicy:
    """Query a specialist Teacher's privileged twin critic for Q-values."""

    def __init__(self, critic: TwinQCritic, device: str | torch.device) -> None:
        self.critic = critic
        self.device = torch.device(device)

    @torch.no_grad()
    def q_values(
        self, critic_state: np.ndarray, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the twin Q-values ``(q1, q2)`` for batched or single inputs."""

        state = torch.as_tensor(critic_state, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        single = state.ndim == 1
        if single:
            state = state.unsqueeze(0)
            act = act.unsqueeze(0)
        if state.shape[0] != act.shape[0]:
            raise ValueError("critic state and action batch sizes must align")
        q1, q2 = self.critic(state, act)
        q1 = q1.cpu().numpy().reshape(-1)
        q2 = q2.cpu().numpy().reshape(-1)
        return (q1[0], q2[0]) if single else (q1, q2)

    def q_min(self, critic_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Conservative twin-critic minimum, following the Teacher's own convention."""

        q1, q2 = self.q_values(critic_state, action)
        return np.minimum(q1, q2)

    def advantage(
        self,
        critic_state: np.ndarray,
        teacher_action: np.ndarray,
        student_action: np.ndarray,
    ) -> np.ndarray:
        """Teacher advantage over the Student action: ``Q(s, u^T) - Q(s, u^S)``.

        Positive means the Student's visited action was worse than the Teacher's,
        so the sample should be up-weighted.
        """

        return self.q_min(critic_state, teacher_action) - self.q_min(
            critic_state, student_action
        )


def _load_critic_payload(path: Path) -> tuple[dict[str, torch.Tensor], object | None]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    critic_state = payload.get("critic")
    contract = payload.get("critic_observation_contract")
    if critic_state is not None:
        return critic_state, contract
    # TD3 Teachers store the critic in a sibling full training checkpoint, not in
    # the deployable actor file referenced by the Teacher Bank.
    sibling = path.parent / "training_checkpoint.pt"
    if not sibling.is_file():
        raise FileNotFoundError(
            f"critic is missing from {path} and no sibling {sibling} exists"
        )
    full = torch.load(sibling, map_location="cpu", weights_only=True)
    critic_state = full.get("critic")
    if critic_state is None:
        raise ValueError(f"TD3 sibling checkpoint {sibling} has no critic")
    return critic_state, full.get("critic_observation_contract", contract)


def load_specialist_critic(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SpecialistCriticPolicy, object | None]:
    """Load a specialist Teacher critic and return a queryable policy wrapper.

    Accepts either the SAC actor checkpoint (``critic`` key present) or the TD3
    actor checkpoint (critic resolved from the sibling ``training_checkpoint.pt``).
    Returns ``(policy, critic_observation_contract)``; the contract may be ``None``
    when the source checkpoint does not record it.
    """

    critic_state, contract = _load_critic_payload(Path(checkpoint_path))
    critic_observation_dim, width, residual_blocks = _infer_critic_shape(critic_state)
    critic = TwinQCritic(
        critic_observation_dim,
        1,
        width=width,
        residual_blocks=residual_blocks,
    )
    critic.load_state_dict(critic_state)
    critic.eval().to(device)
    return SpecialistCriticPolicy(critic, device), contract
