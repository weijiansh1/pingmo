"""Read-only all-split constrained-reference feasibility audit."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from src.aircraft.parameters import PChannelParameters
from src.envs.commands import CommandProfile
from src.experiments.reference_oracle_check import simulate_reference_oracle


_PARAMETER_NAMES = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")


@dataclass(frozen=True, slots=True)
class FeasibilityPolicy:
    """Exploratory relative-error gate; this is not a GJB threshold."""

    relative_tracking_rmse_limit: float = 0.10

    def __post_init__(self) -> None:
        if not 0 < self.relative_tracking_rmse_limit < 1:
            raise ValueError("relative_tracking_rmse_limit must be in (0, 1)")


def classify_feasibility(
    raw_relative_rmse: float,
    oracle_relative_rmse: float,
    constrained_relative_rmse: float,
    policy: FeasibilityPolicy,
) -> str:
    """Separate no-control, constrained-feasible, and authority-limited pairs."""
    limit = policy.relative_tracking_rmse_limit
    if raw_relative_rmse <= limit:
        return "control_not_needed"
    if constrained_relative_rmse <= limit:
        return "constrained_feasible"
    if oracle_relative_rmse <= limit:
        return "authority_limited"
    return "oracle_unreachable"


def audit_pair(
    plant_id: str,
    split: str,
    parameters: PChannelParameters,
    command_profile: CommandProfile,
    *,
    duration_s: float = 10.0,
    feasibility_policy: FeasibilityPolicy = FeasibilityPolicy(),
) -> dict[str, object]:
    """Measure one aircraft-command pair without learning from its split."""
    trace = simulate_reference_oracle(parameters, command_profile=command_profile, duration_s=duration_s)
    metrics = trace.metrics
    raw = float(metrics["raw_tracking_rmse"])
    constrained = float(metrics["constrained_tracking_rmse"])
    feasibility_label = classify_feasibility(
        float(metrics["raw_relative_tracking_rmse"]),
        float(metrics["oracle_relative_tracking_rmse"]),
        float(metrics["constrained_relative_tracking_rmse"]),
        feasibility_policy,
    )
    return {
        "plant_id": plant_id,
        "split": split,
        "command_id": command_profile.command_id,
        "command_kind": command_profile.kind,
        "reference_response_rms": float(metrics["reference_response_rms"]),
        "reference_tracking_rmse": float(metrics["reference_tracking_rmse"]),
        "raw_tracking_rmse": raw,
        "oracle_tracking_rmse": float(metrics["oracle_tracking_rmse"]),
        "constrained_tracking_rmse": constrained,
        "raw_relative_tracking_rmse": float(metrics["raw_relative_tracking_rmse"]),
        "oracle_relative_tracking_rmse": float(metrics["oracle_relative_tracking_rmse"]),
        "constrained_relative_tracking_rmse": float(metrics["constrained_relative_tracking_rmse"]),
        "oracle_gap_rmse": float(metrics["oracle_gap_rmse"]),
        "oracle_augmentation_rms_n": float(metrics["oracle_augmentation_rms_n"]),
        "oracle_peak_augmentation_n": float(metrics["oracle_peak_augmentation_n"]),
        "oracle_total_variation_n": float(metrics["oracle_total_variation_n"]),
        "oracle_authority_exceedance_fraction": float(metrics["oracle_authority_exceedance_fraction"]),
        "oracle_slew_exceedance_fraction": float(metrics["oracle_slew_exceedance_fraction"]),
        "constrained_saturation_fraction": float(metrics["constrained_saturation_fraction"]),
        "constrained_slew_bound_fraction": float(metrics["constrained_slew_bound_fraction"]),
        "constrained_command_rms_n": float(metrics["constrained_command_rms_n"]),
        "constrained_applied_rms_n": float(metrics["constrained_applied_rms_n"]),
        "constrained_max_increment_n": float(metrics["constrained_max_increment_n"]),
        "constrained_increment_limit_n": float(metrics["constrained_increment_limit_n"]),
        "constrained_command_total_variation_n": float(metrics["constrained_command_total_variation_n"]),
        "constrained_applied_total_variation_n": float(metrics["constrained_applied_total_variation_n"]),
        "constrained_improves_raw": bool(constrained < raw),
        "feasibility_label": feasibility_label,
        "oracle_feasible": feasibility_label in {"control_not_needed", "constrained_feasible"},
        "controller_improvable": feasibility_label == "constrained_feasible",
        "feasibility_policy": asdict(feasibility_policy),
    }


def _audit_row(task: tuple[str, str, PChannelParameters, CommandProfile, float, FeasibilityPolicy]) -> dict[str, object]:
    plant_id, split, parameters, profile, duration_s, feasibility_policy = task
    return audit_pair(
        plant_id,
        split,
        parameters,
        profile,
        duration_s=duration_s,
        feasibility_policy=feasibility_policy,
    )


def iter_audit_library(
    library_path: str | Path,
    command_profiles: Iterable[CommandProfile],
    *,
    duration_s: float = 10.0,
    workers: int = 1,
    feasibility_policy: FeasibilityPolicy = FeasibilityPolicy(),
    completed_keys: Iterable[tuple[str, str]] = (),
) -> Iterator[dict[str, object]]:
    """Yield deterministic pair records, skipping keys already checkpointed."""
    rows = [json.loads(line) for line in Path(library_path).read_text(encoding="utf-8").splitlines()]
    profiles = tuple(command_profiles)
    if not profiles:
        raise ValueError("at least one command profile is required")
    plant_ids = [str(row["plant_id"]) for row in rows]
    command_ids = [profile.command_id for profile in profiles]
    if len(set(plant_ids)) != len(plant_ids):
        raise ValueError("plant library contains duplicate plant_id values")
    if len(set(command_ids)) != len(command_ids):
        raise ValueError("command suite contains duplicate command_id values")
    completed = set(completed_keys)
    tasks = [
        (
            str(row["plant_id"]),
            str(row["split"]),
            PChannelParameters(**{name: row["parameters"][name] for name in _PARAMETER_NAMES}),
            profile,
            duration_s,
            feasibility_policy,
        )
        for row in rows
        for profile in profiles
        if (str(row["plant_id"]), profile.command_id) not in completed
    ]
    if workers <= 1:
        for task in tasks:
            yield _audit_row(task)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(_audit_row, tasks, chunksize=16)


def audit_library(
    library_path: str | Path,
    command_profiles: Iterable[CommandProfile],
    *,
    duration_s: float = 10.0,
    workers: int = 1,
    feasibility_policy: FeasibilityPolicy = FeasibilityPolicy(),
) -> list[dict[str, object]]:
    """Audit every persisted aircraft while preserving its immutable split label."""
    return list(
        iter_audit_library(
            library_path,
            command_profiles,
            duration_s=duration_s,
            workers=workers,
            feasibility_policy=feasibility_policy,
        )
    )


def summarize_audit_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_command: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split"])].append(row)
        by_command[str(row["command_id"])].append(row)

    def summarize(values: list[dict[str, object]]) -> dict[str, object]:
        labels = [str(value["feasibility_label"]) for value in values]
        return {
            "pair_count": len(values),
            "improvement_rate": float(np.mean([bool(value["constrained_improves_raw"]) for value in values])),
            "oracle_feasible_rate": float(np.mean([bool(value["oracle_feasible"]) for value in values])),
            "controller_improvable_rate": float(np.mean([bool(value["controller_improvable"]) for value in values])),
            "median_raw_tracking_rmse": float(np.median([float(value["raw_tracking_rmse"]) for value in values])),
            "median_constrained_tracking_rmse": float(np.median([float(value["constrained_tracking_rmse"]) for value in values])),
            "median_raw_relative_tracking_rmse": float(np.median([float(value["raw_relative_tracking_rmse"]) for value in values])),
            "median_constrained_relative_tracking_rmse": float(np.median([float(value["constrained_relative_tracking_rmse"]) for value in values])),
            "mean_saturation_fraction": float(np.mean([float(value["constrained_saturation_fraction"]) for value in values])),
            "feasibility_counts": {label: labels.count(label) for label in sorted(set(labels))},
        }

    return {
        "pair_count": sum(len(values) for values in grouped.values()),
        "by_split": {split: summarize(values) for split, values in sorted(grouped.items())},
        "by_command": {command_id: summarize(values) for command_id, values in sorted(by_command.items())},
    }


def write_audit(destination: str | Path, rows: list[dict[str, object]]) -> Path:
    """Atomically write immutable pair records and a split-separated summary."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    pairs = output / "g0_feasibility_pairs.jsonl"
    summary = output / "g0_feasibility_summary.json"
    pairs_tmp = output / f".{pairs.name}.tmp"
    summary_tmp = output / f".{summary.name}.tmp"
    pairs_tmp.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    summary_tmp.write_text(json.dumps(summarize_audit_rows(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pairs_tmp.replace(pairs)
    summary_tmp.replace(summary)
    return output
