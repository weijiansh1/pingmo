"""Select explicit, representative whole-aircraft validation holdouts."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context.aircraft_parameters import (
    AIRCRAFT_PARAMETER_NAMES,
    AIRCRAFT_PARAMETER_NORMALIZATION,
    normalize_aircraft_parameters,
)
from src.teacher.specialist.trainer import load_specialist_actor
from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-quality-region", type=int, default=1)
    parser.add_argument("--preferred-split", default="train_boundary")
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


def _resolve_actor(bank_path: Path, entry: dict[str, object]) -> Path:
    path = Path(str(entry["actor_checkpoint"]))
    return path.resolve() if path.is_absolute() else (bank_path.parent / path).resolve()


def _nearest_distance(
    plant_id: str,
    vectors: dict[str, np.ndarray],
    reference_ids: list[str],
) -> tuple[float, str]:
    distances = np.asarray(
        [np.linalg.norm(vectors[plant_id] - vectors[other]) for other in reference_ids],
        dtype=float,
    )
    index = int(np.argmin(distances))
    return float(distances[index]), reference_ids[index]


def _select_region_holdout(
    entries: list[dict[str, object]],
    vectors: dict[str, np.ndarray],
    count: int,
    preferred_split: str,
) -> list[str]:
    all_ids = [str(entry["plant_id"]) for entry in entries]
    if len(all_ids) <= count:
        raise ValueError("each quality region must retain at least one training aircraft")
    preferred_ids = [
        str(entry["plant_id"])
        for entry in entries
        if entry.get("split") == preferred_split
    ]
    candidate_ids = preferred_ids if len(preferred_ids) >= count else all_ids
    neighbor_distances = {
        plant_id: _nearest_distance(
            plant_id,
            vectors,
            [other for other in all_ids if other != plant_id],
        )[0]
        for plant_id in all_ids
    }
    median_distance = float(np.median(list(neighbor_distances.values())))
    return sorted(
        candidate_ids,
        key=lambda plant_id: (
            abs(neighbor_distances[plant_id] - median_distance),
            plant_id,
        ),
    )[:count]


def main() -> None:
    args = _parse_args()
    if args.per_quality_region <= 0:
        raise ValueError("per-quality-region must be positive")
    bank_path = args.teacher_bank.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bank = _read_json(bank_path)
    teachers = bank.get("teachers")
    if bank.get("status") != "complete" or not isinstance(teachers, list):
        raise ValueError("a complete Teacher Bank is required")

    vectors: dict[str, np.ndarray] = {}
    entries_by_region: dict[str, list[dict[str, object]]] = {}
    for entry in teachers:
        plant_id = str(entry["plant_id"])
        actor_path = _resolve_actor(bank_path, entry)
        _, record, _, payload = load_specialist_actor(actor_path, device="cpu")
        if record.plant_id != plant_id:
            raise ValueError(f"Teacher actor identity mismatch: {plant_id}")
        if payload.get("actor_observation_contract") != bank.get(
            "actor_observation_contract"
        ):
            raise ValueError(f"Teacher observation contract mismatch: {plant_id}")
        vectors[plant_id] = normalize_aircraft_parameters(record.parameters)
        entries_by_region.setdefault(str(entry["quality_region"]), []).append(entry)

    validation_ids: list[str] = []
    selection_rows: list[dict[str, object]] = []
    for region in sorted(entries_by_region):
        selected = _select_region_holdout(
            entries_by_region[region],
            vectors,
            args.per_quality_region,
            args.preferred_split,
        )
        validation_ids.extend(selected)
        for plant_id in selected:
            entry = next(
                row
                for row in entries_by_region[region]
                if row["plant_id"] == plant_id
            )
            selection_rows.append(
                {
                    "plant_id": plant_id,
                    "split": entry["split"],
                    "quality_region": region,
                    "selection_method": (
                        "nearest_neighbor_distance_closest_to_quality_region_median"
                    ),
                    "preferred_split": args.preferred_split,
                }
            )

    all_ids = [str(entry["plant_id"]) for entry in teachers]
    training_ids = [plant_id for plant_id in all_ids if plant_id not in validation_ids]
    training_regions = {
        str(entry["quality_region"])
        for entry in teachers
        if entry["plant_id"] in training_ids
    }
    if training_regions != set(entries_by_region):
        raise AssertionError("holdout selection emptied a quality region")
    train_matrix = np.stack([vectors[plant_id] for plant_id in training_ids])
    for row in selection_rows:
        plant_id = str(row["plant_id"])
        distance, nearest_id = _nearest_distance(
            plant_id, vectors, training_ids
        )
        vector = vectors[plant_id]
        row.update(
            {
                "nearest_training_plant_id": nearest_id,
                "nearest_training_distance": distance,
                "inside_training_axis_aligned_envelope": bool(
                    np.all(vector >= np.min(train_matrix, axis=0))
                    and np.all(vector <= np.max(train_matrix, axis=0))
                ),
                "normalized_theta": vector.tolist(),
            }
        )

    positions = np.arange(len(selection_rows), dtype=float)
    distances = [float(row["nearest_training_distance"]) for row in selection_rows]
    figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    axis.bar(
        positions,
        distances,
        color=[
            "#2f8f46" if row["quality_region"] == "level_1" else "#d18b19"
            if row["quality_region"] == "level_2"
            else "#c82423"
            for row in selection_rows
        ],
    )
    axis.set_xticks(
        positions,
        [str(row["plant_id"]).split("-", 1)[-1] for row in selection_rows],
    )
    axis.set_ylabel("Nearest training-aircraft distance")
    axis.set_title("Explicit whole-aircraft distillation holdout")
    axis.grid(axis="y", alpha=0.2)
    plot_path = destination / "holdout.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)

    report = {
        "schema_version": "distillation_aircraft_holdout_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "method": (
            "one_representative_preferred_split_aircraft_per_quality_region"
        ),
        "aircraft_parameter_names": list(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_normalization": AIRCRAFT_PARAMETER_NORMALIZATION,
        "preferred_split": args.preferred_split,
        "per_quality_region": args.per_quality_region,
        "training_plant_ids": training_ids,
        "validation_plant_ids": validation_ids,
        "training_distribution": {
            "by_quality_region": dict(
                Counter(
                    str(entry["quality_region"])
                    for entry in teachers
                    if entry["plant_id"] in training_ids
                )
            ),
            "by_split": dict(
                Counter(
                    str(entry["split"])
                    for entry in teachers
                    if entry["plant_id"] in training_ids
                )
            ),
        },
        "validation_distribution": {
            "by_quality_region": dict(
                Counter(str(row["quality_region"]) for row in selection_rows)
            ),
            "by_split": dict(Counter(str(row["split"]) for row in selection_rows)),
        },
        "selection": selection_rows,
        "artifacts": {"plot": str(plot_path)},
    }
    _write_json(destination / "holdout.json", report)
    print(
        json.dumps(
            {
                "status": "complete",
                "training_aircraft_count": len(training_ids),
                "validation_plant_ids": validation_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
