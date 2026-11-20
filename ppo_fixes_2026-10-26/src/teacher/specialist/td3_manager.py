"""Gated Teacher-Bank orchestration for PID-guided TD3 specialists."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from src.aircraft.sampler import PlantRecord
from src.controllers.pid import PIDGains
from src.teacher.specialist.td3_trainer import (
    PIDGuidedTD3Config,
    train_pid_guided_td3,
)
from src.teacher.specialist.trainer import (
    SpecialistTrainingConfig,
    specialist_quality_gate,
)
from src.utils.provenance import git_source_revision, sha256_file


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_pid_oracle(
    report_path: Path,
    record: PlantRecord,
    environment_config: SpecialistTrainingConfig,
) -> tuple[PIDGains, dict[str, object], dict[str, object]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"PID oracle report is missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"PID oracle is incomplete: {report_path}")
    if payload.get("controller") != "direct_reference_tracking_pid":
        raise ValueError(f"unsupported PID oracle contract: {report_path}")
    if payload.get("plant_id") != record.plant_id:
        raise ValueError("PID oracle plant ID does not match the selected aircraft")

    reported_parameters = payload.get("plant_parameters")
    expected_parameters = asdict(record.parameters)
    if not isinstance(reported_parameters, dict) or set(reported_parameters) != set(
        expected_parameters
    ):
        raise ValueError("PID oracle aircraft parameters are incomplete")
    if any(
        not np.isclose(
            float(reported_parameters[name]), value, rtol=0.0, atol=1e-12
        )
        for name, value in expected_parameters.items()
    ):
        raise ValueError("PID oracle aircraft parameters do not match the library")

    expected_contract = {
        "episode_duration_s": environment_config.episode_duration_s,
        "plant_dt_s": environment_config.plant_dt_s,
        "policy_dt_s": environment_config.policy_dt_s,
        "reference_natural_frequency_rad_s": (
            environment_config.reference_natural_frequency_rad_s
        ),
        "reference_damping_ratio": environment_config.reference_damping_ratio,
        "reference_delay_mode": environment_config.reference_delay_mode,
    }
    mismatches = {
        name: (payload.get(name), expected)
        for name, expected in expected_contract.items()
        if payload.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"PID oracle environment contract mismatch: {mismatches}")

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("PID oracle has no closed-loop evaluation")
    gate = specialist_quality_gate(
        evaluation,
        replace(environment_config, enforce_quality_gate=True),
        controller_label="PID",
    )
    if not gate["passed"]:
        raise ValueError(
            f"PID oracle fails the Teacher quality thresholds: {gate['observed']}"
        )
    return PIDGains(**payload["gains"]), payload, gate


def _run_fingerprint(
    environment_config: SpecialistTrainingConfig,
    td3_config: PIDGuidedTD3Config,
    source: dict[str, object],
    pid_report_sha256: str,
) -> str:
    canonical = json.dumps(
        {
            "environment_config": asdict(environment_config),
            "td3_config": asdict(td3_config),
            "source": source,
            "pid_report_sha256": pid_report_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def train_pid_guided_teacher_bank(
    library_path: str | Path,
    pid_report_root: str | Path,
    output_dir: str | Path,
    records: list[PlantRecord],
    environment_config: SpecialistTrainingConfig,
    td3_config: PIDGuidedTD3Config,
    *,
    skip_completed: bool = True,
) -> dict[str, object]:
    """Train one independent, gated TD3 Teacher for each explicit aircraft."""

    if not records:
        raise ValueError("PID-guided Teacher Bank requires at least one aircraft")
    if len({record.plant_id for record in records}) != len(records):
        raise ValueError("PID-guided Teacher Bank cannot repeat an aircraft")
    if environment_config.history_steps != 0:
        raise ValueError("PID-guided Teacher Bank does not permit raw Actor history")
    library = Path(library_path)
    pid_root = Path(pid_report_root)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "teacher_bank.json"
    source = git_source_revision()
    manifest: dict[str, object] = {
        "schema_version": "specialist_teacher_bank_v1",
        "status": "running",
        "algorithm": "pid_guided_td3",
        "source": source,
        "library": {
            "path": str(library.resolve()),
            "sha256": sha256_file(library),
        },
        "pid_report_root": str(pid_root.resolve()),
        "base_environment_config": asdict(environment_config),
        "base_td3_config": asdict(td3_config),
        "teachers": [],
    }
    _write_manifest(manifest_path, manifest)

    entries: list[dict[str, object]] = []
    for index, record in enumerate(records):
        teacher_environment = replace(
            environment_config,
            seed=environment_config.seed + index,
            device=td3_config.device,
        )
        teacher_td3 = replace(td3_config, seed=td3_config.seed + index)
        pid_report_path = pid_root / record.plant_id / "pid_report.json"
        try:
            gains, _, pid_gate = _validate_pid_oracle(
                pid_report_path, record, teacher_environment
            )
            pid_hash = sha256_file(pid_report_path)
            fingerprint = _run_fingerprint(
                teacher_environment, teacher_td3, source, pid_hash
            )
            run_id = (
                f"{record.plant_id}-td3-seed-{teacher_td3.seed}"
                f"-cfg-{fingerprint}"
            )
            run_dir = destination / run_id
            report_path = run_dir / "report.json"
            actor_path = run_dir / "teacher_actor.pt"
            skipped = False
            if skip_completed and report_path.is_file() and actor_path.is_file():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                skipped = bool(report.get("accepted_for_distillation", False))
            if not skipped:
                report = train_pid_guided_td3(
                    record,
                    gains,
                    run_dir,
                    teacher_environment,
                    teacher_td3,
                    library_path=library,
                )
        except Exception as error:
            manifest["status"] = "training_failed"
            manifest["failed_teacher_index"] = index
            manifest["failed_plant_id"] = record.plant_id
            manifest["error"] = f"{type(error).__name__}: {error}"
            manifest["teachers"] = entries
            _write_manifest(manifest_path, manifest)
            raise

        accepted = bool(report.get("accepted_for_distillation", False))
        entry = {
            "run_id": run_id,
            "algorithm": "pid_guided_td3",
            "plant_id": record.plant_id,
            "quality_region": record.quality_region,
            "seed": teacher_td3.seed,
            "config_fingerprint": fingerprint,
            "status": "complete" if accepted else "quality_gate_failed",
            "quality_gate": report.get("quality_gate"),
            "skipped_completed": skipped,
            "run_dir": str(run_dir.relative_to(destination)),
            "actor_checkpoint": str(actor_path.relative_to(destination)),
            "report": str(report_path.relative_to(destination)),
            "pid_oracle": {
                "path": str(pid_report_path.resolve()),
                "sha256": pid_hash,
                "quality_gate": pid_gate,
            },
        }
        entries.append(entry)
        manifest["teachers"] = entries
        _write_manifest(manifest_path, manifest)

    manifest["status"] = (
        "complete"
        if all(entry["status"] == "complete" for entry in entries)
        else "quality_gate_failed"
    )
    manifest["teacher_count"] = len(entries)
    manifest["accepted_teacher_count"] = sum(
        entry["status"] == "complete" for entry in entries
    )
    manifest["quality_region_counts"] = {
        region: sum(entry["quality_region"] == region for entry in entries)
        for region in sorted({str(entry["quality_region"]) for entry in entries})
    }
    _write_manifest(manifest_path, manifest)
    return manifest
