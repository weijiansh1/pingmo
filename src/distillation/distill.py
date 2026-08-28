"""Supervised distillation from fixed-aircraft Teachers to one dense Student."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.distillation.dataset import DistillationDataset, load_distillation_arrays
from src.distillation.losses import teacher_action_mse
from src.distillation.validate import imitation_metrics
from src.student.dense.network import DenseConditionalStudent
from src.utils.provenance import git_source_revision, sha256_file


@dataclass(frozen=True, slots=True)
class DenseStudentTrainingConfig:
    epochs: int = 100
    batch_size: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    network_width: int = 512
    residual_blocks: int = 8
    residual_scale: float = 0.1
    gradient_norm_limit: float = 10.0
    patience_epochs: int = 15
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
        if min(positive) <= 0 or self.weight_decay < 0:
            raise ValueError("invalid dense Student training configuration")


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_checkpoint(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_dense_student(
    dataset_manifest_path: str | Path,
    output_dir: str | Path,
    config: DenseStudentTrainingConfig = DenseStudentTrainingConfig(),
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
    train_dataset = DistillationDataset(arrays, "train")
    validation_dataset = DistillationDataset(arrays, "validation")
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
    model = DenseConditionalStudent(
        int(dataset_manifest["observation_dim"]),
        int(dataset_manifest["aircraft_parameter_dim"]),
        int(dataset_manifest["action_dim"]),
        width=config.network_width,
        residual_blocks=config.residual_blocks,
        residual_scale=config.residual_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    checkpoint_path = destination / "dense_student.pt"
    best_validation_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            observation = batch["observation"].to(device)
            theta = batch["aircraft_parameters"].to(device)
            target = batch["teacher_action"].to(device)
            prediction = model(observation, theta)
            loss = teacher_action_mse(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_norm_limit)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(observation)
            sample_count += len(observation)

        validation = imitation_metrics(model, validation_loader, device)
        epoch_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_action_mse": loss_sum / sample_count,
            **validation,
        }
        history.append(epoch_row)
        if validation["action_mse"] < best_validation_mse:
            best_validation_mse = validation["action_mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "schema_version": "dense_conditional_student_v1",
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "observation_dim": model.observation_dim,
                "aircraft_parameter_dim": model.aircraft_parameter_dim,
                "action_dim": model.action_dim,
                "network_width": config.network_width,
                "residual_blocks": config.residual_blocks,
                "residual_scale": config.residual_scale,
                "parameter_count": model.parameter_count,
                "training_config": asdict(config),
                "dataset_manifest": {
                    "path": str(dataset_path.resolve()),
                    "sha256": sha256_file(dataset_path),
                },
                "best_epoch": best_epoch,
                "best_validation": validation,
                "source": git_source_revision(),
            }
            _save_checkpoint(checkpoint, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience_epochs:
                break

    model_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(model_payload["model"])
    train_metrics = imitation_metrics(model, train_loader, device)
    validation_metrics = imitation_metrics(model, validation_loader, device)
    report: dict[str, object] = {
        "schema_version": "dense_student_training_report_v1",
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
        "parameter_count": model.parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "elapsed_s": time.perf_counter() - started,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    _write_json(destination / "report.json", report)
    return report
