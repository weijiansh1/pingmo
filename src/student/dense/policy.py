"""Deployment wrapper and checkpoint loading for the dense Student."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.aircraft.parameters import PChannelParameters
from src.context.aircraft_parameters import normalize_aircraft_parameters
from src.student.dense.network import DenseConditionalStudent


class DenseStudentPolicy:
    def __init__(
        self,
        model: DenseConditionalStudent,
        aircraft_parameters: PChannelParameters | np.ndarray,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        theta = (
            normalize_aircraft_parameters(aircraft_parameters)
            if isinstance(aircraft_parameters, PChannelParameters)
            else np.asarray(aircraft_parameters, dtype=np.float32)
        )
        if theta.shape != (model.aircraft_parameter_dim,) or not np.isfinite(theta).all():
            raise ValueError("student policy requires one finite normalized theta vector")
        self.theta = torch.as_tensor(theta, dtype=torch.float32, device=self.device)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        values = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        single = values.ndim == 1
        if single:
            values = values.unsqueeze(0)
        theta = self.theta.unsqueeze(0).expand(values.shape[0], -1)
        with torch.no_grad():
            actions = self.model(values, theta)
        result = actions.cpu().numpy()
        return result[0] if single else result


def load_dense_student(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[DenseConditionalStudent, dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != "dense_conditional_student_v1":
        raise ValueError("unsupported dense Student checkpoint schema")
    model = DenseConditionalStudent(
        int(payload["observation_dim"]),
        int(payload["aircraft_parameter_dim"]),
        int(payload["action_dim"]),
        width=int(payload["network_width"]),
        residual_blocks=int(payload["residual_blocks"]),
        residual_scale=float(payload["residual_scale"]),
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload
