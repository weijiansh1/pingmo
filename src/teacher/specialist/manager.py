"""Selection and recoverable orchestration for a bank of specialist Teachers."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from src.aircraft.sampler import PlantRecord
from src.experiments.exploratory_sac import load_persisted_records
from src.teacher.specialist.trainer import SpecialistTrainingConfig, train_specialist
from src.utils.provenance import git_source_revision, sha256_file


def select_specialist_records(
    library_path: str | Path,
    count: int,
    *,
    seed: int,
    splits: tuple[str, ...] = ("train_core", "train_boundary"),
) -> list[PlantRecord]:
    """Select a level-balanced deterministic subset without repeating aircraft."""

    if count <= 0:
        raise ValueError("specialist count must be positive")
    source = Path(library_path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    eligible = [row for row in rows if row.get("split") in splits]
    if count > len(eligible):
        raise ValueError(f"requested {count} specialists from only {len(eligible)} eligible plants")
    groups: dict[str, list[dict[str, object]]] = {}
    for row in eligible:
        groups.setdefault(str(row["quality_region"]), []).append(row)
    rng = np.random.default_rng(seed)
    levels = sorted(groups)
    shuffled = {level: list(rng.permutation(groups[level])) for level in levels}
    positions = {level: 0 for level in levels}
    selected_ids: list[str] = []
    while len(selected_ids) < count:
        available = [level for level in levels if positions[level] < len(shuffled[level])]
        if not available:
            raise RuntimeError("specialist selection exhausted eligible aircraft")
        for level in available:
            if len(selected_ids) >= count:
                break
            selected_ids.append(str(shuffled[level][positions[level]]["plant_id"]))
            positions[level] += 1
    return load_persisted_records(source, selected_ids)


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _config_fingerprint(config: SpecialistTrainingConfig) -> str:
    canonical = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def train_teacher_bank(
    library_path: str | Path,
    output_dir: str | Path,
    config: SpecialistTrainingConfig,
    *,
    records: list[PlantRecord] | None = None,
    count: int = 1,
    skip_completed: bool = True,
) -> dict[str, object]:
    """Train one independent SAC run per aircraft and maintain a bank manifest."""

    library = Path(library_path)
    selected = records or select_specialist_records(library, count, seed=config.seed)
    if len({record.plant_id for record in selected}) != len(selected):
        raise ValueError("teacher bank cannot contain duplicate plant IDs")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "teacher_bank.json"
    manifest: dict[str, object] = {
        "schema_version": "specialist_teacher_bank_v1",
        "status": "running",
        "source": git_source_revision(),
        "library": {"path": str(library.resolve()), "sha256": sha256_file(library)},
        "base_config": asdict(config),
        "teachers": [],
    }
    _write_manifest(manifest_path, manifest)
    entries: list[dict[str, object]] = []
    for index, record in enumerate(selected):
        teacher_config = replace(config, seed=config.seed + index)
        config_fingerprint = _config_fingerprint(teacher_config)
        run_id = f"{record.plant_id}-seed-{teacher_config.seed}-cfg-{config_fingerprint}"
        run_dir = destination / run_id
        report_path = run_dir / "report.json"
        actor_path = run_dir / "teacher_actor.pt"
        skipped = False
        if skip_completed and report_path.is_file() and actor_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("status") == "complete":
                skipped = True
            else:
                report = train_specialist(record, run_dir, teacher_config, library_path=library)
        else:
            report = train_specialist(record, run_dir, teacher_config, library_path=library)
        entry = {
            "run_id": run_id,
            "plant_id": record.plant_id,
            "quality_region": record.quality_region,
            "seed": teacher_config.seed,
            "config_fingerprint": config_fingerprint,
            "status": report["status"],
            "skipped_completed": skipped,
            "run_dir": str(run_dir.relative_to(destination)),
            "actor_checkpoint": str(actor_path.relative_to(destination)),
            "report": str(report_path.relative_to(destination)),
        }
        entries.append(entry)
        manifest["teachers"] = entries
        _write_manifest(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["teacher_count"] = len(entries)
    _write_manifest(manifest_path, manifest)
    return manifest
