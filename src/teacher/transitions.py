"""Episode-boundary semantics for continuing flight-control training."""

from __future__ import annotations


def continuing_task_transition_flags(
    terminated: bool,
    truncated: bool,
) -> tuple[bool, bool]:
    """Return ``(reset_episode, bellman_terminal)`` for a continuing task.

    A time-limit truncation resets the simulator, but it must not suppress the
    bootstrap term in a Bellman target. Only a physical/task termination is a
    terminal transition.
    """

    return bool(terminated or truncated), bool(terminated)


def continuing_task_contract(
    *, critic_includes_episode_progress: bool
) -> dict[str, object]:
    return {
        "task_type": "continuing_flight_control",
        "episode_reset_condition": "terminated_or_truncated",
        "bellman_terminal_mask": "terminated_only",
        "bootstraps_across_time_limit_truncation": True,
        "critic_includes_artificial_episode_progress": (
            critic_includes_episode_progress
        ),
    }
