"""Stable deployment entry point for the current dense conditional Student."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.aircraft.parameters import PChannelParameters
from src.student.dense.policy import DenseStudentPolicy, load_dense_student


def load_student_controller(
    checkpoint_path: str | Path,
    aircraft_parameters: PChannelParameters | np.ndarray,
    *,
    device: str | torch.device = "cpu",
) -> tuple[DenseStudentPolicy, dict[str, object]]:
    model, payload = load_dense_student(checkpoint_path, device=device)
    return DenseStudentPolicy(model, aircraft_parameters, device=device), payload
