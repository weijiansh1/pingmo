"""Read-only all-split constrained-reference feasibility audit."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.aircraft.parameters import PChannelParameters
from src.envs.commands import CommandProfile
from src.experiments.reference_oracle_check import simulate_reference_oracle


_PARAMETER_NAMES = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")


def audit_pair(plant_id: str, split: str, parameters: PChannelParameters, command_profile: CommandProfile, *, duration_s: float = 10.0) -> dict[str, object]:
    """Measure one aircraft-command pair without learning from its split."""
    trace = simulate_reference_oracle(parameters, command_profile=command_profile, duration_s=duration_s)
    metrics = trace.metrics
    raw = float(metrics["raw_tracking_rmse"])
    constrained = float(metrics["constrained_tracking_rmse"])
    return {
        "plant_id": plant_id,
        "split": split,
        "command_id": command_profile.command_id,
        "command_kind": command_profile.kind,
        "raw_tracking_rmse": raw,
        "oracle_tracking_rmse": float(metrics["oracle_tracking_rmse"]),
        "constrained_tracking_rmse": constrained,
        "oracle_gap_rmse": float(metrics["oracle_gap_rmse"]),
        "constrained_saturation_fraction": float(metrics["constrained_saturation_fraction"]),
        "constrained_command_total_variation_n": float(metrics["constrained_command_total_variation_n"]),
        "constrained_applied_total_variation_n": float(metrics["constrained_applied_total_variation_n"]),
        "constrained_improves_raw": bool(constrained < raw),
    }


def _audit_row(task: tuple[str, str, PChannelParameters, CommandProfile, float]) -> dict[str, object]:
    plant_id, split, parameters, profile, duration_s = task
    return audit_pair(plant_id, split, parameters, profile, duration_s=duration_s)


def audit_library(library_path: str | Path, command_profiles: Iterable[CommandProfile], *, duration_s: float = 10.0, workers: int = 1) -> list[dict[str, object]]:
    """Audit every persisted aircraft while preserving its immutable split label."""
    rows = [json.loads(line) for line in Path(library_path).read_text(encoding="utf-8").splitlines()]
    profiles = tuple(command_profiles)
    if not profiles:
        raise ValueError("at least one command profile is required")
    tasks = [
        (
            str(row["plant_id"]),
            str(row["split"]),
            PChannelParameters(**{name: row["parameters"][name] for name in _PARAMETER_NAMES}),
            profile,
            duration_s,
        )
        for row in rows
        for profile in profiles
    ]
    if workers <= 1:
        return [_audit_row(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_audit_row, tasks, chunksize=16))


def summarize_audit_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_command: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)
        by_command[str(row["command_id"])].append(row)

    def summarize(values: list[dict[str, object]]) -> dict[str, float | int]:
        return {
            "pair_count": len(values),
            "improvement_rate": float(np.mean([bool(value["constrained_improves_raw"]) for value in values])),
            "median_raw_tracking_rmse": float(np.median([float(value["raw_tracking_rmse"]) for value in values])),
            "median_constrained_tracking_rmse": float(np.median([float(value["constrained_tracking_rmse"]) for value in values])),
            "mean_saturation_fraction": float(np.mean([float(value["constrained_saturation_fraction"]) for value in values])),
        }

    return {
        "pair_count": sum(len(values) for values in grouped.values()),
        "by_split": {split: summarize(values) for split, values in sorted(grouped.items())},
        "by_command": {command_id: summarize(values) for command_id, values in sorted(by_command.items())},
    }


def write_audit(destination: str | Path, rows: list[dict[str, object]]) -> Path:
    """Write immutable pair records and a split-separated summary."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    (output / "g0_feasibility_pairs.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    (output / "g0_feasibility_summary.json").write_text(json.dumps(summarize_audit_rows(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
