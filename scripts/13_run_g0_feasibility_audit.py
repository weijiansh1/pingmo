"""Run the read-only G0 constrained-reference feasibility audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.commands import DEFAULT_COMMAND_DURATION_S, CommandProfile, default_command_suite
from src.experiments.feasibility_audit import (
    FeasibilityPolicy,
    iter_audit_library,
    summarize_audit_rows,
    write_audit,
)


ACTION_DT_S = 0.02
PILOT_FORCE_N = 22.0
CORRECTION_RATIO = 0.30
NORMALIZED_RATE_LIMIT_S_INV = 4.0
ACTUATOR_TIME_CONSTANT_S = 0.08


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=ROOT / "data" / "aircraft" / "generated" / "p_channel_library_iv_a_manual_v1" / "plants.jsonl",
        help="immutable 3,000-aircraft parameter library",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results" / "g0_feasibility_audit",
        help="parent directory for immutable run directories",
    )
    parser.add_argument("--run-id", help="explicit run identifier; defaults to hashes of source, library, and commands")
    parser.add_argument(
        "--relative-rmse-limit",
        type=float,
        default=0.10,
        help="exploratory normalized tracking-error limit used only for diagnostic labels",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(24, os.cpu_count() or 1)),
        help="parallel CPU processes; G0 is a deterministic simulation, not GPU training",
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_record(profile: CommandProfile) -> dict[str, object]:
    samples = profile.samples(ACTION_DT_S, DEFAULT_COMMAND_DURATION_S, PILOT_FORCE_N)
    return {
        **asdict(profile),
        "action_dt_s": ACTION_DT_S,
        "sample_count": len(samples),
        "force_history_sha256": hashlib.sha256(np.asarray(samples, dtype="<f8").tobytes()).hexdigest(),
    }


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_revision() -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("G0 must run from a clean Git revision")
    return revision


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(path: Path, expected_keys: set[tuple[str, str]]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid checkpoint JSON on line {line_number}") from exc
        key = (str(row["plant_id"]), str(row["command_id"]))
        if key not in expected_keys:
            raise RuntimeError(f"checkpoint contains unexpected pair {key}")
        if key in seen:
            raise RuntimeError(f"checkpoint contains duplicate pair {key}")
        seen.add(key)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    library = args.library.resolve()
    profiles = default_command_suite()
    policy = FeasibilityPolicy(args.relative_rmse_limit)
    source_revision = _git_revision()
    library_sha256 = _sha256_file(library)
    command_records = [_command_record(profile) for profile in profiles]
    command_suite_sha256 = _json_sha256(command_records)
    plant_rows = [json.loads(line) for line in library.read_text(encoding="utf-8").splitlines()]
    plant_ids = [str(row["plant_id"]) for row in plant_rows]
    expected_keys = {(plant_id, profile.command_id) for plant_id in plant_ids for profile in profiles}
    if len(expected_keys) != len(plant_ids) * len(profiles):
        raise RuntimeError("plant and command identifiers must be unique")

    run_id = args.run_id or f"g0-{source_revision[:12]}-{library_sha256[:8]}-{command_suite_sha256[:8]}"
    destination = args.output_root.resolve() / run_id
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    checkpoint_path = destination / "g0_feasibility_pairs.checkpoint.jsonl"
    completed_marker = destination / "COMPLETED"
    configuration = {
        "schema_version": "2.0",
        "run_id": run_id,
        "source_revision": source_revision,
        "library_path": str(library),
        "library_sha256": library_sha256,
        "command_suite_sha256": command_suite_sha256,
        "commands": command_records,
        "physics": {
            "duration_s": DEFAULT_COMMAND_DURATION_S,
            "action_dt_s": ACTION_DT_S,
            "plant_dt_s": 0.005,
            "pilot_force_n": PILOT_FORCE_N,
            "augmentation_limit_n": PILOT_FORCE_N * CORRECTION_RATIO,
            "correction_ratio": CORRECTION_RATIO,
            "normalized_rate_limit_s_inv": NORMALIZED_RATE_LIMIT_S_INV,
            "actuator_time_constant_s": ACTUATOR_TIME_CONSTANT_S,
        },
        "feasibility_policy": {
            **asdict(policy),
            "status": "exploratory_non_gjb",
            "normalizer": "reference_response_rms",
        },
        "expected_pair_count": len(expected_keys),
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("configuration") != configuration:
            raise RuntimeError(f"run directory {destination} belongs to a different configuration")
        if manifest.get("status") == "completed" and completed_marker.exists():
            print(f"G0 run already completed: {destination}", flush=True)
            return
    else:
        manifest = {
            "configuration": configuration,
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_pair_count": 0,
        }
        _atomic_write_json(manifest_path, manifest)

    rows = _load_checkpoint(checkpoint_path, expected_keys)
    completed_keys = {(str(row["plant_id"]), str(row["command_id"])) for row in rows}
    manifest["status"] = "running"
    manifest["completed_pair_count"] = len(rows)
    manifest["resumed_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(manifest_path, manifest)
    print(
        f"run_id={run_id} completed={len(rows)}/{len(expected_keys)} workers={args.workers}",
        flush=True,
    )

    start = time.monotonic()
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for row in iter_audit_library(
            library,
            profiles,
            duration_s=DEFAULT_COMMAND_DURATION_S,
            workers=args.workers,
            feasibility_policy=policy,
            completed_keys=completed_keys,
        ):
            checkpoint.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            rows.append(row)
            count = len(rows)
            if count % 32 == 0:
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
            if count % 250 == 0 or count == len(expected_keys):
                checkpoint.flush()
                os.fsync(checkpoint.fileno())
                elapsed = max(time.monotonic() - start, 1e-9)
                newly_completed = count - len(completed_keys)
                rate = newly_completed / elapsed
                remaining = len(expected_keys) - count
                manifest["completed_pair_count"] = count
                manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
                _atomic_write_json(manifest_path, manifest)
                print(
                    f"progress={count}/{len(expected_keys)} rate={rate:.2f}_pairs_per_s eta_s={remaining / max(rate, 1e-9):.0f}",
                    flush=True,
                )

    if len(rows) != len(expected_keys):
        raise RuntimeError(f"audit ended with {len(rows)} of {len(expected_keys)} pairs")
    plant_order = {plant_id: index for index, plant_id in enumerate(plant_ids)}
    command_order = {profile.command_id: index for index, profile in enumerate(profiles)}
    rows.sort(key=lambda row: (plant_order[str(row["plant_id"])], command_order[str(row["command_id"])]))
    write_audit(destination, rows)
    summary = summarize_audit_rows(rows)
    manifest["status"] = "completed"
    manifest["completed_pair_count"] = len(rows)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["summary_sha256"] = _sha256_file(destination / "g0_feasibility_summary.json")
    manifest["pairs_sha256"] = _sha256_file(destination / "g0_feasibility_pairs.jsonl")
    _atomic_write_json(manifest_path, manifest)
    _atomic_write_text(completed_marker, f"{manifest['completed_at_utc']}\n")
    checkpoint_path.unlink()
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print(f"wrote {len(rows)} aircraft-command records to {destination}", flush=True)


if __name__ == "__main__":
    main()
