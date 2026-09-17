"""Loss functions for direct specialist-action distillation."""

from __future__ import annotations

import torch


def teacher_action_mse(student_action: torch.Tensor, teacher_action: torch.Tensor) -> torch.Tensor:
    if student_action.shape != teacher_action.shape:
        raise ValueError("student and Teacher actions must have identical shapes")
    return (student_action - teacher_action).square().mean()


def weighted_teacher_action_mse(
    student_action: torch.Tensor,
    teacher_action: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Teacher-action MSE with a stable, batch-size-independent weight scale."""

    if student_action.shape != teacher_action.shape:
        raise ValueError("student and Teacher actions must have identical shapes")
    if sample_weight.shape != student_action.shape[:-1]:
        raise ValueError("sample weights must have one value per action row")
    if bool(torch.any(sample_weight <= 0)):
        raise ValueError("sample weights must be positive")
    per_sample = (student_action - teacher_action).square().mean(dim=-1)
    return torch.sum(per_sample * sample_weight) / torch.sum(sample_weight)


def teacher_action_rate_mse(
    student_action: torch.Tensor,
    previous_student_action: torch.Tensor,
    teacher_action: torch.Tensor,
    previous_teacher_action: torch.Tensor,
    policy_step_delta: torch.Tensor,
    temporal_mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    """Match Teacher action increments per policy step on valid trajectory pairs."""

    action_shape = student_action.shape
    if any(
        value.shape != action_shape
        for value in (
            previous_student_action,
            teacher_action,
            previous_teacher_action,
        )
    ):
        raise ValueError("current and previous Student/Teacher actions must align")
    row_shape = action_shape[:-1]
    if any(
        value.shape != row_shape
        for value in (policy_step_delta, temporal_mask, sample_weight)
    ):
        raise ValueError("temporal metadata must have one value per action row")
    effective_weight = temporal_mask * sample_weight
    denominator = torch.sum(effective_weight)
    if float(denominator.detach()) <= 0:
        return student_action.sum() * 0.0
    step_delta = policy_step_delta.clamp_min(1.0).unsqueeze(-1)
    student_rate = (student_action - previous_student_action) / step_delta
    teacher_rate = (teacher_action - previous_teacher_action) / step_delta
    per_sample = (student_rate - teacher_rate).square().mean(dim=-1)
    return torch.sum(per_sample * effective_weight) / denominator


def incremental_student_losses(
    action_t: torch.Tensor,
    delta_t: torch.Tensor,
    teacher_action: torch.Tensor,
    residual_delta_label: torch.Tensor,
    sample_weight: torch.Tensor,
    excess_margin: float,
) -> dict[str, torch.Tensor]:
    """V6 losses for the incremental-action Student.

    ``residual_delta_label`` is the Teacher's target increment from the Student's
    actual previous action: ``u^T_t - u_{t-1}`` (own-delta during round-zero
    teacher-forcing, residual during student-driven DAgger rounds). ``action_t``
    and ``delta_t`` are the reconstructed action and the emitted delta, so
    ``action_t = clip(u_{t-1} + delta_t)``.

    Returns ``action_mse`` (L_u), ``delta_mse`` (L_du), and ``excess`` (L_excess),
    the last penalizing only Student deltas that overshoot the Teacher's
    necessary magnitude by more than ``excess_margin``.
    """

    if action_t.shape != teacher_action.shape or delta_t.shape != teacher_action.shape:
        raise ValueError("incremental Student outputs must match the Teacher action")
    if residual_delta_label.shape != teacher_action.shape:
        raise ValueError("residual delta labels must match the Teacher action")
    if sample_weight.shape != action_t.shape[:-1]:
        raise ValueError("sample weights must have one value per action row")
    if bool(torch.any(sample_weight <= 0)):
        raise ValueError("sample weights must be positive")
    if excess_margin < 0:
        raise ValueError("excess_margin cannot be negative")

    weight_sum = torch.sum(sample_weight)

    action_per_sample = (action_t - teacher_action).square().mean(dim=-1)
    action_mse = torch.sum(action_per_sample * sample_weight) / weight_sum

    delta_per_sample = (delta_t - residual_delta_label).square().mean(dim=-1)
    delta_mse = torch.sum(delta_per_sample * sample_weight) / weight_sum

    excess = torch.clamp(
        delta_t.abs() - residual_delta_label.abs() - excess_margin, min=0.0
    )
    excess_per_sample = excess.square().mean(dim=-1)
    excess_mse = torch.sum(excess_per_sample * sample_weight) / weight_sum

    return {
        "action_mse": action_mse,
        "delta_mse": delta_mse,
        "excess": excess_mse,
    }
