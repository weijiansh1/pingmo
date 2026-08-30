"""Merge validated specialist Teacher Banks into one provenance-preserving bank."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.teacher.specialist.trainer import load_specialist_actor
from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-report", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_from_manifest(manifest: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def main() -> None:
    args = _parse_args()
    bank_paths = [path.resolve() for path in args.bank]
    if len(set(bank_paths)) != len(bank_paths):
        raise ValueError("source Teacher Banks must be unique")
    destination = args.output.resolve()
    actor_dir = destination / "actors"
    actor_dir.mkdir(parents=True, exist_ok=True)

    merged_teachers: list[dict[str, object]] = []
    merged_rejected: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    seen_plant_ids: set[str] = set()
    expected_library_hash: str | None = None
    expected_contract: dict[str, object] | None = None
    for bank_path in bank_paths:
        bank = _read_json(bank_path)
        if bank.get("schema_version") != "specialist_teacher_bank_v1":
            raise ValueError(f"unsupported Teacher Bank schema: {bank_path}")
        teachers = bank.get("teachers")
        rejected = bank.get("rejected_teachers", [])
        if not isinstance(teachers, list) or not isinstance(rejected, list):
            raise ValueError(f"invalid Teacher Bank entries: {bank_path}")
        library = bank.get("library")
        if not isinstance(library, dict) or not library.get("sha256"):
            raise ValueError(f"Teacher Bank has no library provenance: {bank_path}")
        library_hash = str(library["sha256"])
        if expected_library_hash is None:
            expected_library_hash = library_hash
        elif library_hash != expected_library_hash:
            raise ValueError("source Teacher Banks use different aircraft libraries")
        source_rows.append(
            {
                "path": str(bank_path),
                "sha256": sha256_file(bank_path),
                "status": bank.get("status"),
                "accepted_teacher_count": len(teachers),
                "rejected_teacher_count": len(rejected),
            }
        )

        for entry in teachers:
            if entry.get("status") != "complete":
                raise ValueError(f"source bank contains incomplete Teacher: {bank_path}")
            plant_id = str(entry["plant_id"])
            if plant_id in seen_plant_ids:
                raise ValueError(f"duplicate Teacher aircraft across banks: {plant_id}")
            actor_source = _resolve_from_manifest(bank_path, entry["actor_checkpoint"])
            if not actor_source.is_file():
                raise FileNotFoundError(actor_source)
            _, record, _, actor_payload = load_specialist_actor(
                actor_source, device="cpu"
            )
            if record.plant_id != plant_id:
                raise ValueError(f"Teacher actor identity mismatch: {plant_id}")
            contract = actor_payload.get("actor_observation_contract")
            if not isinstance(contract, dict):
                raise ValueError(f"Teacher actor has no observation contract: {plant_id}")
            if expected_contract is None:
                expected_contract = contract
            elif contract != expected_contract:
                raise ValueError("source Teacher actors have different observations")
            expected_actor_hash = str(entry.get("actor_checkpoint_sha256", ""))
            observed_actor_hash = sha256_file(actor_source)
            if expected_actor_hash and expected_actor_hash != observed_actor_hash:
                raise ValueError(f"Teacher actor hash mismatch: {plant_id}")
            actor_destination = actor_dir / f"{plant_id}.pt"
            shutil.copy2(actor_source, actor_destination)
            source_report = entry.get("source_report")
            merged_entry = {
                **entry,
                "actor_checkpoint": os.path.relpath(actor_destination, destination),
                "actor_checkpoint_sha256": sha256_file(actor_destination),
                "source_bank": str(bank_path),
                "source_bank_sha256": sha256_file(bank_path),
                "source_actor_checkpoint": str(actor_source),
            }
            if source_report:
                merged_entry["source_report"] = str(
                    _resolve_from_manifest(bank_path, source_report)
                )
            merged_teachers.append(merged_entry)
            seen_plant_ids.add(plant_id)

        for entry in rejected:
            merged_rejected.append(
                {
                    **entry,
                    "source_bank": str(bank_path),
                    "source_bank_sha256": sha256_file(bank_path),
                }
            )

    if len(merged_teachers) < 2 or expected_contract is None:
        raise ValueError("merged Teacher Bank requires at least two valid Teachers")
    selection: dict[str, object] | None = None
    if args.selection_report:
        selection_path = args.selection_report.resolve()
        selection_payload = _read_json(selection_path)
        selected_ids = {
            str(entry["plant_id"])
            for entry in selection_payload.get("selected_candidates", [])
        }
        realized_accepted = sorted(selected_ids & seen_plant_ids)
        realized_rejected = sorted(
            selected_ids
            & {str(entry["plant_id"]) for entry in merged_rejected}
        )
        unaccounted = sorted(selected_ids - set(realized_accepted) - set(realized_rejected))
        if unaccounted:
            raise ValueError(
                f"selected Teacher candidates are absent from source banks: {unaccounted}"
            )
        selection = {
            "path": str(selection_path),
            "sha256": sha256_file(selection_path),
            "selected_candidate_count": len(selected_ids),
            "accepted_candidate_ids": realized_accepted,
            "rejected_candidate_ids": realized_rejected,
            "acceptance_rate": len(realized_accepted) / max(len(selected_ids), 1),
        }

    region_counts = Counter(
        str(entry["quality_region"]) for entry in merged_teachers
    )
    split_counts = Counter(str(entry["split"]) for entry in merged_teachers)
    first_bank = _read_json(bank_paths[0])
    report = {
        "schema_version": "specialist_teacher_bank_v1",
        "status": "complete",
        "algorithm": "merged_pure_reward_td3_best_validation",
        "source": git_source_revision(),
        "library": first_bank["library"],
        "source_banks": source_rows,
        "coverage_selection": selection,
        "attempted_teacher_count": len(merged_teachers) + len(merged_rejected),
        "teacher_count": len(merged_teachers),
        "accepted_teacher_count": len(merged_teachers),
        "rejected_teacher_count": len(merged_rejected),
        "quality_region_counts": dict(region_counts),
        "split_counts": dict(split_counts),
        "actor_observation_contract": expected_contract,
        "selection_contract": {
            "method": "union_of_source_bank_quality_eligible_teachers",
            "duplicate_aircraft_allowed": False,
            "actor_hashes_verified": True,
            "observation_contracts_identical": True,
        },
        "teachers": merged_teachers,
        "rejected_teachers": merged_rejected,
    }
    _write_json(destination / "teacher_bank.json", report)
    print(
        json.dumps(
            {
                "status": "complete",
                "accepted_teacher_count": len(merged_teachers),
                "rejected_teacher_count": len(merged_rejected),
                "teacher_bank": str(destination / "teacher_bank.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
