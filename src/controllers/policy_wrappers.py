"""Stateful deployment wrappers for normalized continuous-control policies."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class PredictivePolicy(Protocol):
    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray: ...


class ForceSlewLimitedPolicy:
    """Limit requested-force slew before the action reaches the environment."""

    def __init__(
        self,
        policy: PredictivePolicy,
        *,
        force_rate_limit_n_s: float,
        policy_dt_s: float,
        force_limit_n: float,
    ) -> None:
        if not np.isfinite(force_rate_limit_n_s) or force_rate_limit_n_s <= 0:
            raise ValueError("force_rate_limit_n_s must be finite and positive")
        if not np.isfinite(policy_dt_s) or policy_dt_s <= 0:
            raise ValueError("policy_dt_s must be finite and positive")
        if not np.isfinite(force_limit_n) or force_limit_n <= 0:
            raise ValueError("force_limit_n must be finite and positive")
        self.policy = policy
        self.force_rate_limit_n_s = float(force_rate_limit_n_s)
        self.policy_dt_s = float(policy_dt_s)
        self.force_limit_n = float(force_limit_n)
        self.maximum_normalized_increment = (
            self.force_rate_limit_n_s * self.policy_dt_s / self.force_limit_n
        )
        self._previous_action: np.ndarray | None = None

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self._previous_action = None

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        requested = np.asarray(
            self.policy.predict(observation, deterministic=deterministic),
            dtype=np.float32,
        )
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("slew-limited policy requires one action vector")
        if not np.isfinite(requested).all():
            raise ValueError("policy action must be finite")
        if np.max(np.abs(requested)) > 1.0 + 1e-6:
            raise ValueError("policy action must be normalized to [-1, 1]")
        requested = np.clip(requested, -1.0, 1.0)
        previous = (
            np.zeros_like(requested)
            if self._previous_action is None
            else self._previous_action
        )
        if previous.shape != requested.shape:
            raise ValueError("policy action shape changed within one rollout")
        increment = np.clip(
            requested - previous,
            -self.maximum_normalized_increment,
            self.maximum_normalized_increment,
        )
        limited = np.clip(previous + increment, -1.0, 1.0).astype(
            np.float32, copy=False
        )
        self._previous_action = limited.copy()
        return limited
