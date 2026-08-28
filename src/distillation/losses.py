"""Loss functions for direct specialist-action distillation."""

from __future__ import annotations

import torch


def teacher_action_mse(student_action: torch.Tensor, teacher_action: torch.Tensor) -> torch.Tensor:
    if student_action.shape != teacher_action.shape:
        raise ValueError("student and Teacher actions must have identical shapes")
    return (student_action - teacher_action).square().mean()
