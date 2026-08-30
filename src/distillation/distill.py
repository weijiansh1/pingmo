"""Supervised distillation from fixed-aircraft Teachers to one Student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.distillation.dataset import DistillationDataset, load_distillation_arrays
from src.distillation.losses import (
    teacher_action_rate_mse,
    weighted_teacher_action_mse,
)
from src.distillation.validate import imitation_metrics
from src.student.dense.network import DenseConditionalStudent
from src.student.moe.network import ThetaRoutedLinearMoEStudent
from src.utils.provenance import git_source_revision, sha256_file


StudentModel = DenseConditionalStudent | ThetaRoutedLinearMoEStudent
SUPPORTED_STUDENT_ARCHITECTURES = ("dense", "theta_routed_linear_moe")


@dataclass(frozen=True, slots=True)
class DenseStudentTrainingConfig:
    architecture: str = "dense"
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    network_width: int = 512
    residual_blocks: int = 8
    residual_scale: float = 0.1
    gradient_norm_limit: float = 10.0
    patience_epochs: int = 15
    action_delta_weight: float = 1.0
    hard_case_weight_boost: float = 7.0
    hard_tracking_error_scale: float = 0.2
    hard_teacher_mismatch_scale: float = 0.1
    hard_action_rate_scale: float = 0.05
    enforce_odd_policy: bool = True
    moe_expert_count: int = 0
    moe_router_temperature: float = 0.2
    moe_prototype_movement_limit: float = 0.05
    moe_router_balance_weight: float = 1e-3
    moe_router_z_loss_weight: float = 1e-5
    moe_prototype_anchor_weight: float = 1e-3
    moe_linear_ridge: float = 1e-4
    seed: int = 20260828
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = (
            self.epochs,
            self.batch_size,
            self.learning_rate,
            self.network_width,
            self.residual_blocks,
            self.residual_scale,
            self.gradient_norm_limit,
            self.patience_epochs,
        )
        if min(positive) <= 0 or min(
            self.weight_decay,
            self.action_delta_weight,
            self.hard_case_weight_boost,
        ) < 0:
            raise ValueError("invalid dense Student training configuration")
        if min(
            self.hard_tracking_error_scale,
            self.hard_teacher_mismatch_scale,
            self.hard_action_rate_scale,
        ) <= 0:
            raise ValueError("hard-case scales must be positive")
        if self.architecture not in SUPPORTED_STUDENT_ARCHITECTURES:
            raise ValueError(f"unsupported Student architecture: {self.architecture}")
        if self.moe_expert_count < 0:
            raise ValueError("MoE expert count cannot be negative")
        if self.moe_router_temperature <= 0 or self.moe_prototype_movement_limit < 0:
            raise ValueError("invalid MoE routing configuration")
        if min(
            self.moe_router_balance_weight,
            self.moe_router_z_loss_weight,
            self.moe_prototype_anchor_weight,
            self.moe_linear_ridge,
        ) < 0:
            raise ValueError("MoE regularization values cannot be negative")
        if self.architecture == "theta_routed_linear_moe" and not self.enforce_odd_policy:
            raise ValueError("the linear MoE Student always enforces an odd policy")


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _training_theta_by_plant(arrays: object) -> tuple[np.ndarray, np.ndarray]:
    train_mask = arrays.split_codes == 0
    plant_indices = np.unique(arrays.plant_indices[train_mask])
    if not len(plant_indices):
        raise ValueError("MoE initialization requires training-aircraft rows")
    theta = np.stack(
        [
            arrays.aircraft_parameters[
                np.flatnonzero(train_mask & (arrays.plant_indices == plant_index))[0]
            ]
            for plant_index in plant_indices
        ]
    )
    return plant_indices, theta.astype(np.float32, copy=False)


def _select_theta_prototypes(theta: np.ndarray, expert_count: int) -> np.ndarray:
    if expert_count <= 0:
        expert_count = len(theta)
    if expert_count > len(theta):
        raise ValueError(
            "MoE expert count cannot exceed the number of training aircraft"
        )
    if expert_count == len(theta):
        return theta.copy()

    centroid = theta.mean(axis=0)
    selected = [int(np.argmin(np.square(theta - centroid).sum(axis=1)))]
    while len(selected) < expert_count:
        distances = np.square(theta[:, None, :] - theta[selected][None, :, :]).mean(
            axis=2
        )
        nearest_distance = distances.min(axis=1)
        nearest_distance[selected] = -1.0
        selected.append(int(np.argmax(nearest_distance)))
    return theta[selected].copy()


def _initialize_linear_experts(
    model: ThetaRoutedLinearMoEStudent,
    arrays: object,
    ridge: float,
) -> dict[str, object]:
    """Fit a stable per-prototype linear control law before gradient training."""

    train_mask = arrays.split_codes == 0
    plant_indices, plant_theta = _training_theta_by_plant(arrays)
    prototypes = model.anchor_prototypes.detach().cpu().numpy().astype(np.float64)
    plant_coefficients = np.zeros(
        (len(plant_indices), model.action_dim, model.control_feature_dim),
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    identity = np.eye(model.control_feature_dim, dtype=np.float64)
    for row_index, plant_index in enumerate(plant_indices):
        plant_mask = train_mask & (arrays.plant_indices == plant_index)
        plant_observations = arrays.observations[plant_mask].astype(
            np.float64, copy=False
        )
        plant_targets = arrays.teacher_actions[plant_mask].astype(
            np.float64, copy=False
        )
        control_observations = plant_observations[
            :, model.control_feature_indices
        ]
        unsaturated = np.max(np.abs(plant_targets), axis=1) < 0.98
        if int(np.sum(unsaturated)) >= model.control_feature_dim:
            fit_observations = control_observations[unsaturated]
            fit_targets = plant_targets[unsaturated]
        else:
            fit_observations = control_observations
            fit_targets = plant_targets
        if len(fit_observations) < model.control_feature_dim:
            raise ValueError(
                f"MoE training plant {int(plant_index)} has too few initialization rows"
            )
        gram = fit_observations.T @ fit_observations + ridge * identity
        coefficient = np.linalg.solve(
            gram, fit_observations.T @ fit_targets
        ).T
        plant_coefficients[row_index] = coefficient
        prediction = np.clip(control_observations @ coefficient.T, -1.0, 1.0)
        rows.append(
            {
                "plant_index": int(plant_index),
                "rows": int(len(plant_observations)),
                "unsaturated_fit_rows": int(len(fit_observations)),
                "linearized_teacher_action_rmse": float(
                    np.sqrt(np.mean(np.square(prediction - plant_targets)))
                ),
            }
        )

    route_distance = np.square(
        plant_theta.astype(np.float64)[:, None, :] - prototypes[None, :, :]
    ).mean(axis=2)
    route_logits = -route_distance / model.router_temperature
    route_logits -= route_logits.max(axis=1, keepdims=True)
    route_matrix = np.exp(route_logits)
    route_matrix /= route_matrix.sum(axis=1, keepdims=True)
    route_gram = route_matrix.T @ route_matrix + ridge * np.eye(
        model.expert_count, dtype=np.float64
    )
    flattened_coefficients = plant_coefficients.reshape(len(plant_indices), -1)
    expert_coefficients = np.linalg.solve(
        route_gram, route_matrix.T @ flattened_coefficients
    )
    weights = expert_coefficients.reshape(
        model.expert_count, model.action_dim, model.control_feature_dim
    ).astype(np.float32)
    with torch.no_grad():
        model.expert_weights.copy_(torch.from_numpy(weights))
    return {
        "method": "per_aircraft_linearization_then_soft_route_solve_v1",
        "router_temperature": model.router_temperature,
        "route_matrix_condition_number": float(np.linalg.cond(route_matrix)),
        "route_matrix": route_matrix.tolist(),
        "training_plants": rows,
    }


def _build_student_model(
    arrays: object,
    dataset_manifest: dict[str, object],
    config: DenseStudentTrainingConfig,
) -> tuple[StudentModel, dict[str, object] | None]:
    observation_dim = int(dataset_manifest["observation_dim"])
    theta_dim = int(dataset_manifest["aircraft_parameter_dim"])
    action_dim = int(dataset_manifest["action_dim"])
    if config.architecture == "dense":
        return (
            DenseConditionalStudent(
                observation_dim,
                theta_dim,
                action_dim,
                width=config.network_width,
                residual_blocks=config.residual_blocks,
                residual_scale=config.residual_scale,
                enforce_odd_policy=config.enforce_odd_policy,
            ),
            None,
        )

    _, training_theta = _training_theta_by_plant(arrays)
    anchors = _select_theta_prototypes(training_theta, config.moe_expert_count)
    model = ThetaRoutedLinearMoEStudent(
        observation_dim,
        theta_dim,
        action_dim,
        torch.from_numpy(anchors),
        router_temperature=config.moe_router_temperature,
        prototype_movement_limit=config.moe_prototype_movement_limit,
    )
    initialization = _initialize_linear_experts(model, arrays, config.moe_linear_ridge)
    initialization["training_aircraft_count"] = len(training_theta)
    initialization["expert_count"] = model.expert_count
    return model, initialization


def _routing_diagnostics(
    model: StudentModel,
    dataset: DistillationDataset,
    device: torch.device,
) -> dict[str, object] | None:
    if not isinstance(model, ThetaRoutedLinearMoEStudent):
        return None
    unique_theta = torch.unique(dataset.aircraft_parameters, dim=0).to(device)
    with torch.no_grad():
        weights = model.routing_weights(unique_theta)
        regularization = model.router_regularization(unique_theta)
    mean_usage = weights.mean(dim=0)
    hard_counts = torch.bincount(
        weights.argmax(dim=-1), minlength=model.expert_count
    )
    return {
        "router_input": "normalized_aircraft_theta_only",
        "expert_input": "current_error_integral_error_p_dot_only",
        "uses_raw_history_window": False,
        "aircraft_count": len(unique_theta),
        "expert_count": model.expert_count,
        "mean_expert_usage": mean_usage.cpu().tolist(),
        "hard_route_aircraft_counts": hard_counts.cpu().tolist(),
        "router_entropy": float(regularization["router_entropy"].cpu()),
        "router_normalized_entropy": float(
            regularization["router_normalized_entropy"].cpu()
        ),
        "router_max_mean_usage": float(
            regularization["router_max_mean_usage"].cpu()
        ),
        "router_min_mean_usage": float(
            regularization["router_min_mean_usage"].cpu()
        ),
        "prototype_max_displacement": float(
            (model.prototypes - model.anchor_prototypes)
            .abs()
            .max()
            .detach()
            .cpu()
        ),
    }


def _save_routing_plot(
    train_routing: dict[str, object],
    validation_routing: dict[str, object],
    output_path: Path,
) -> None:
    train_usage = np.asarray(train_routing["mean_expert_usage"], dtype=float)
    validation_usage = np.asarray(
        validation_routing["mean_expert_usage"], dtype=float
    )
    indices = np.arange(len(train_usage))
    figure, axis = plt.subplots(figsize=(9, 4.8), layout="constrained")
    axis.bar(indices - 0.2, train_usage, width=0.4, label="Train aircraft")
    axis.bar(indices + 0.2, validation_usage, width=0.4, label="Held-out aircraft")
    axis.axhline(
        1.0 / len(train_usage),
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Uniform mean usage",
    )
    axis.set_xticks(indices, [f"E{index + 1}" for index in indices])
    axis.set_xlabel("Linear control expert")
    axis.set_ylabel("Mean soft-routing weight")
    axis.set_ylim(0.0, max(float(train_usage.max()), float(validation_usage.max())) * 1.15)
    axis.set_title("Theta-only Student router utilization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _save_training_plot(
    history: list[dict[str, float | int]], output_path: Path
) -> None:
    epochs = np.asarray([row["epoch"] for row in history], dtype=int)
    figure, axes = plt.subplots(
        2, 1, figsize=(8.5, 7.0), sharex=True, layout="constrained"
    )
    axes[0].plot(
        epochs,
        [row["train_action_mse"] for row in history],
        label="Train weighted action MSE",
    )
    axes[0].plot(
        epochs,
        [row["action_mse"] for row in history],
        label="Validation action MSE",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Action MSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(
        epochs,
        [row["train_action_delta_mse_per_policy_step"] for row in history],
        label="Train weighted delta MSE",
    )
    axes[1].plot(
        epochs,
        [row["action_delta_mse_per_policy_step"] for row in history],
        label="Validation delta MSE",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Action-delta MSE / policy step")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    figure.suptitle("Stability-aware Student distillation")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _checkpoint_payload(
    model: StudentModel,
    config: DenseStudentTrainingConfig,
    dataset_path: Path,
    dataset_manifest: dict[str, object],
    initial_checkpoint: dict[str, object] | None,
    initialization: dict[str, object] | None,
    best_epoch: int,
    validation: dict[str, float | int],
) -> dict[str, object]:
    common: dict[str, object] = {
        "model": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "student_architecture": config.architecture,
        "observation_dim": model.observation_dim,
        "aircraft_parameter_dim": model.aircraft_parameter_dim,
        "action_dim": model.action_dim,
        "enforce_odd_policy": model.enforce_odd_policy,
        "parameter_count": model.parameter_count,
        "training_config": asdict(config),
        "temporal_contract": {
            "uses_raw_history_window": False,
            "raw_history_steps": 0,
            "uses_predecessor_pairs_during_training": config.action_delta_weight > 0,
            "action_delta_definition": "normalized_action_increment_per_policy_step",
            "action_delta_weight": config.action_delta_weight,
            "hard_case_weight_boost": config.hard_case_weight_boost,
            "router_input": (
                "normalized_aircraft_theta_only"
                if isinstance(model, ThetaRoutedLinearMoEStudent)
                else "not_applicable"
            ),
            "expert_input": (
                "current_error_integral_error_p_dot_only"
                if isinstance(model, ThetaRoutedLinearMoEStudent)
                else "current_actor_observation_only"
            ),
        },
        "dataset_manifest": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "split_strategy": dataset_manifest["split_strategy"],
            "aircraft_split_method": dataset_manifest.get("aircraft_split_method"),
            "train_plant_ids": dataset_manifest["train_plant_ids"],
            "validation_plant_ids": dataset_manifest["validation_plant_ids"],
            "actor_observation_contract": dataset_manifest.get(
                "actor_observation_contract"
            ),
        },
        "initial_checkpoint": initial_checkpoint,
        "initialization": initialization,
        "best_epoch": best_epoch,
        "best_validation": validation,
        "source": git_source_revision(),
    }
    if isinstance(model, DenseConditionalStudent):
        return {
            "schema_version": "dense_conditional_student_v1",
            **common,
            "network_width": config.network_width,
            "residual_blocks": config.residual_blocks,
            "residual_scale": config.residual_scale,
        }
    return {
        "schema_version": "theta_routed_linear_moe_student_v2",
        **common,
        "expert_count": model.expert_count,
        "router_temperature": model.router_temperature,
        "prototype_movement_limit": model.prototype_movement_limit,
        "control_feature_indices": list(model.control_feature_indices),
        "control_feature_names": [
            "tracking_error_normalized",
            "integrated_tracking_error_normalized",
            "p_dot_normalized",
        ],
        "router_anchor_prototypes": model.anchor_prototypes.detach().cpu(),
    }


def train_dense_student(
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    config: DenseStudentTrainingConfig = DenseStudentTrainingConfig(),
    *,
    initial_checkpoint_path: str | Path | None = None,
) -> dict[str, object]:
    dataset_path = Path(dataset_manifest_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    arrays, dataset_manifest = load_distillation_arrays(dataset_path)
    observation_contract = dataset_manifest.get("actor_observation_contract")
    if isinstance(observation_contract, dict) and (
        int(observation_contract.get("raw_history_steps", 0)) != 0
        or bool(observation_contract.get("uses_raw_history_window", False))
    ):
        raise ValueError("Student distillation requires the no-raw-history contract")
    dataset_weighting = {
        "hard_tracking_error_scale": config.hard_tracking_error_scale,
        "hard_teacher_mismatch_scale": config.hard_teacher_mismatch_scale,
        "hard_action_rate_scale": config.hard_action_rate_scale,
    }
    train_dataset = DistillationDataset(
        arrays,
        "train",
        hard_case_weight_boost=config.hard_case_weight_boost,
        **dataset_weighting,
    )
    validation_dataset = DistillationDataset(
        arrays,
        "validation",
        hard_case_weight_boost=0.0,
        **dataset_weighting,
    )
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    model, initialization = _build_student_model(arrays, dataset_manifest, config)
    model = model.to(device)
    initial_checkpoint: dict[str, object] | None = None
    if initial_checkpoint_path is not None:
        initial_path = Path(initial_checkpoint_path)
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=True)
        expected = {
            "student_architecture": config.architecture,
            "observation_dim": model.observation_dim,
            "aircraft_parameter_dim": model.aircraft_parameter_dim,
            "action_dim": model.action_dim,
            "enforce_odd_policy": config.enforce_odd_policy,
        }
        if isinstance(model, DenseConditionalStudent):
            expected.update(
                {
                    "network_width": config.network_width,
                    "residual_blocks": config.residual_blocks,
                }
            )
        else:
            expected.update(
                {
                    "expert_count": model.expert_count,
                    "router_temperature": model.router_temperature,
                    "prototype_movement_limit": model.prototype_movement_limit,
                }
            )
        observed = dict(initial_payload)
        if (
            "student_architecture" not in observed
            and observed.get("schema_version") == "dense_conditional_student_v1"
        ):
            observed["student_architecture"] = "dense"
        mismatches = {
            name: (observed.get(name), value)
            for name, value in expected.items()
            if observed.get(name) != value
        }
        if mismatches:
            raise ValueError(f"initial Student checkpoint is incompatible: {mismatches}")
        model.load_state_dict(initial_payload["model"])
        initialization = {
            "method": "continued_from_previous_student_checkpoint",
            "original_initialization": initialization,
        }
        initial_checkpoint = {
            "path": str(initial_path.resolve()),
            "sha256": sha256_file(initial_path),
        }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_path = destination / (
        "student.pt"
        if config.architecture == "theta_routed_linear_moe"
        else "dense_student.pt"
    )
    best_validation_objective = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        imitation_loss_sum = 0.0
        action_delta_loss_sum = 0.0
        balance_loss_sum = 0.0
        z_loss_sum = 0.0
        anchor_loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            observation = batch["observation"].to(device)
            theta = batch["aircraft_parameters"].to(device)
            target = batch["teacher_action"].to(device)
            prediction = model(observation, theta)
            sample_weight = batch["sample_weight"].to(device)
            imitation_loss = weighted_teacher_action_mse(
                prediction,
                target,
                sample_weight,
            )
            previous_prediction = model(
                batch["previous_observation"].to(device), theta
            )
            action_delta_loss = teacher_action_rate_mse(
                prediction,
                previous_prediction,
                target,
                batch["previous_teacher_action"].to(device),
                batch["policy_step_delta"].to(device),
                batch["temporal_mask"].to(device),
                sample_weight,
            )
            loss = imitation_loss + config.action_delta_weight * action_delta_loss
            if isinstance(model, ThetaRoutedLinearMoEStudent):
                router = model.router_regularization(theta)
                loss = (
                    loss
                    + config.moe_router_balance_weight
                    * router["router_balance_loss"]
                    + config.moe_router_z_loss_weight * router["router_z_loss"]
                    + config.moe_prototype_anchor_weight
                    * router["prototype_anchor_loss"]
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_norm_limit)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(observation)
            imitation_loss_sum += float(imitation_loss.detach()) * len(observation)
            action_delta_loss_sum += float(action_delta_loss.detach()) * len(
                observation
            )
            if isinstance(model, ThetaRoutedLinearMoEStudent):
                balance_loss_sum += float(router["router_balance_loss"].detach()) * len(
                    observation
                )
                z_loss_sum += float(router["router_z_loss"].detach()) * len(observation)
                anchor_loss_sum += float(
                    router["prototype_anchor_loss"].detach()
                ) * len(observation)
            sample_count += len(observation)

        validation = imitation_metrics(model, validation_loader, device)
        validation_objective = float(validation["action_mse"]) + (
            config.action_delta_weight
            * float(validation["action_delta_mse_per_policy_step"])
        )
        epoch_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_total_loss": loss_sum / sample_count,
            "train_action_mse": imitation_loss_sum / sample_count,
            "train_action_delta_mse_per_policy_step": (
                action_delta_loss_sum / sample_count
            ),
            "validation_objective": validation_objective,
            **validation,
        }
        if isinstance(model, ThetaRoutedLinearMoEStudent):
            epoch_row.update(
                {
                    "train_router_balance_loss": balance_loss_sum / sample_count,
                    "train_router_z_loss": z_loss_sum / sample_count,
                    "train_prototype_anchor_loss": anchor_loss_sum / sample_count,
                }
            )
        history.append(epoch_row)
        print(
            json.dumps(
                {
                    "event": "student_distillation_epoch",
                    "epoch": epoch,
                    "train_action_mse": epoch_row["train_action_mse"],
                    "train_action_delta_mse_per_policy_step": epoch_row[
                        "train_action_delta_mse_per_policy_step"
                    ],
                    "validation_action_mse": validation["action_mse"],
                    "validation_action_delta_mse_per_policy_step": validation[
                        "action_delta_mse_per_policy_step"
                    ],
                    "validation_objective": validation_objective,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        if validation_objective < best_validation_objective:
            best_validation_objective = validation_objective
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = _checkpoint_payload(
                model,
                config,
                dataset_path,
                dataset_manifest,
                initial_checkpoint,
                initialization,
                best_epoch,
                {**validation, "selection_objective": validation_objective},
            )
            _save_checkpoint(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience_epochs:
                break

    model_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(model_payload["model"])
    train_metrics = imitation_metrics(model, train_loader, device)
    validation_metrics = imitation_metrics(model, validation_loader, device)
    train_routing = _routing_diagnostics(model, train_dataset, device)
    validation_routing = _routing_diagnostics(model, validation_dataset, device)
    report: dict[str, object] = {
        "schema_version": "conditional_student_training_report_v1",
        "status": "complete",
        "source": git_source_revision(),
        "config": asdict(config),
        "dataset_manifest": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "split_strategy": dataset_manifest["split_strategy"],
            "train_rows": len(train_dataset),
            "validation_rows": len(validation_dataset),
        },
        "initial_checkpoint": initial_checkpoint,
        "student_architecture": config.architecture,
        "temporal_contract": {
            "uses_raw_history_window": False,
            "raw_history_steps": 0,
            "uses_predecessor_pairs_during_training": config.action_delta_weight > 0,
            "action_delta_definition": "normalized_action_increment_per_policy_step",
        },
        "hard_case_statistics": {
            "train_mean_hardness": float(train_dataset.hardness_scores.mean()),
            "train_full_weight_fraction": float(
                (train_dataset.hardness_scores >= 1.0 - 1e-6).float().mean()
            ),
            "train_mean_sample_weight": float(train_dataset.sample_weights.mean()),
            "train_max_sample_weight": float(train_dataset.sample_weights.max()),
            "validation_mean_hardness": float(
                validation_dataset.hardness_scores.mean()
            ),
        },
        "checkpoint_selection_metric": (
            "validation_action_mse_plus_weighted_action_delta_mse"
        ),
        "initialization": initialization,
        "parameter_count": model.parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_s": time.perf_counter() - started,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "train_routing": train_routing,
        "validation_routing": validation_routing,
        "checkpoint": str(checkpoint_path),
        "artifacts": {
            "training_curves": str(destination / "training_curves.png"),
        },
        "history": history,
    }
    _save_training_plot(history, destination / "training_curves.png")
    _write_json(destination / "report.json", report)
    if train_routing is not None:
        _write_json(
            destination / "routing_report.json",
            {
                "schema_version": "theta_routed_student_routing_report_v1",
                "status": "complete",
                "train": train_routing,
                "validation": validation_routing,
            },
        )
        if validation_routing is None:
            raise RuntimeError("MoE validation routing diagnostics are missing")
        _save_routing_plot(
            train_routing,
            validation_routing,
            destination / "router_utilization.png",
        )
    return report
