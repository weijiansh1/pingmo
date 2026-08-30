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
