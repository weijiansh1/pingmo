"""Select new Teacher aircraft by stratified maximin coverage in normalized theta."""

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

from src.aircraft.parameters import PChannelParameters
from src.context.aircraft_parameters import (
    AIRCRAFT_PARAMETER_NAMES,
    AIRCRAFT_PARAMETER_NORMALIZATION,
    normalize_aircraft_parameters,
)
from src.utils.provenance import git_source_revision, sha256_file


DEFAULT_LIBRARY = (
    ROOT / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
)
DEFAULT_EXISTING_BANK = (
    ROOT / "results/pure_reward_teacher_bank_pilot_v1/teacher_bank.json"
)
DEFAULT_UNSEEN_REPORT = (
    ROOT
    / "results/pure_reward_teacher_bank_pilot_v1/unseen_aircraft_v1/report.json"
)
DEFAULT_SPLITS = ("train_core", "train_boundary")
DEFAULT_QUALITY_REGIONS = ("level_1", "level_2", "level_3")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    parser.add_argument("--existing-bank", type=Path, default=DEFAULT_EXISTING_BANK)
    parser.add_argument("--unseen-report", type=Path, default=DEFAULT_UNSEEN_REPORT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--split", action="append")
    parser.add_argument("--quality-region", action="append")
    parser.add_argument("--exclude-plant-id", action="append", default=[])
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


def _load_library(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid or empty aircraft library: {path}")
    plant_ids = [str(row["plant_id"]) for row in rows]
    if len(set(plant_ids)) != len(plant_ids):
        raise ValueError("aircraft library contains duplicate plant IDs")
    return rows


def _theta(row: dict[str, object]) -> np.ndarray:
    values = row.get("parameters")
    if not isinstance(values, dict):
        raise ValueError(f"aircraft has no parameter object: {row.get('plant_id')}")
    parameters = PChannelParameters(
        **{name: float(values[name]) for name in AIRCRAFT_PARAMETER_NAMES}
    )
    return normalize_aircraft_parameters(parameters)


def _distance_to_coverage(
    vector: np.ndarray, coverage: list[tuple[str, np.ndarray]]
) -> tuple[float, str]:
    distances = np.asarray(
        [np.linalg.norm(vector - reference) for _, reference in coverage],
        dtype=float,
    )
    index = int(np.argmin(distances))
    return float(distances[index]), coverage[index][0]


def _distance_summary(
    rows: list[dict[str, object]], coverage: list[tuple[str, np.ndarray]]
) -> dict[str, float | int]:
    distances = np.asarray(
        [_distance_to_coverage(_theta(row), coverage)[0] for row in rows], dtype=float
    )
    return {
        "aircraft_count": len(rows),
        "mean_nearest_distance": float(np.mean(distances)),
        "median_nearest_distance": float(np.median(distances)),
        "p90_nearest_distance": float(np.quantile(distances, 0.9)),
        "maximum_nearest_distance": float(np.max(distances)),
    }


def _quotas(
    splits: tuple[str, ...], quality_regions: tuple[str, ...], count: int
) -> dict[tuple[str, str], int]:
    cells = tuple(
        (split, region) for region in quality_regions for split in splits
    )
    if count < len(cells):
        raise ValueError(
            f"candidate count {count} cannot cover all {len(cells)} split-quality cells"
        )
    base, remainder = divmod(count, len(cells))
    return {
        cell: base + int(index < remainder) for index, cell in enumerate(cells)
    }


def _select(
    pools: dict[tuple[str, str], list[dict[str, object]]],
    quotas: dict[tuple[str, str], int],
    seed_coverage: list[tuple[str, np.ndarray]],
) -> list[dict[str, object]]:
    coverage = list(seed_coverage)
    selected: list[dict[str, object]] = []
    maximum_quota = max(quotas.values())
    for selection_round in range(maximum_quota):
        for cell, quota in quotas.items():
            if selection_round >= quota:
                continue
            candidates = pools[cell]
            scored = [
                (*_distance_to_coverage(_theta(row), coverage), row)
                for row in candidates
            ]
            maximum_distance = max(distance for distance, _, _ in scored)
            _, nearest_id, chosen = min(
                (
                    (distance, nearest_id, row)
                    for distance, nearest_id, row in scored
                    if np.isclose(distance, maximum_distance, rtol=0.0, atol=1e-12)
                ),
                key=lambda item: str(item[2]["plant_id"]),
            )
            candidates.remove(chosen)
            vector = _theta(chosen)
            selected.append(
                {
                    "selection_order": len(selected),
                    "selection_round": selection_round,
                    "plant_id": str(chosen["plant_id"]),
                    "split": str(chosen["split"]),
                    "quality_region": str(chosen["quality_region"]),
                    "nearest_covered_plant_id_before_selection": nearest_id,
                    "nearest_covered_distance_before_selection": maximum_distance,
                    "normalized_theta": vector.tolist(),
                    "parameters": {
                        name: float(chosen["parameters"][name])
                        for name in AIRCRAFT_PARAMETER_NAMES
                    },
                }
            )
            coverage.append((str(chosen["plant_id"]), vector))
    return selected


def _save_plot(
    all_rows: list[dict[str, object]],
    seed_ids: list[str],
    selected_ids: list[str],
    target_ids: list[str],
    path: Path,
) -> Path:
    row_by_id = {str(row["plant_id"]): row for row in all_rows}
    display_ids = [
        str(row["plant_id"])
        for row in all_rows
        if row["split"] in DEFAULT_SPLITS
        and row["quality_region"] in DEFAULT_QUALITY_REGIONS
    ]
    matrix = np.stack([_theta(row_by_id[plant_id]) for plant_id in display_ids])
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[:2].T
    coordinates = dict(zip(display_ids, projected, strict=True))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    colors = {"level_1": "#2f8f46", "level_2": "#d18b19", "level_3": "#c82423"}
    for region in DEFAULT_QUALITY_REGIONS:
        ids = [
            plant_id
            for plant_id in display_ids
            if row_by_id[plant_id]["quality_region"] == region
        ]
        points = np.stack([coordinates[plant_id] for plant_id in ids])
        axes[0].scatter(
            points[:, 0],
            points[:, 1],
            s=8,
            alpha=0.16,
            color=colors[region],
            label=region,
        )
    for ids, marker, color, label, size in (
        (seed_ids, "X", "#111111", "existing Teachers", 90),
        (selected_ids, "*", "#2b6cb0", "new candidates", 120),
        (target_ids, "o", "none", "frozen zero-shot test", 80),
    ):
        points = np.stack([coordinates[plant_id] for plant_id in ids])
        axes[0].scatter(
            points[:, 0],
            points[:, 1],
            s=size,
            marker=marker,
            facecolors=color,
            edgecolors="#111111" if color == "none" else color,
            linewidths=1.2,
            label=label,
        )
    axes[0].set_title("Normalized-theta PCA audit view")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)

    seed_coverage = [(plant_id, _theta(row_by_id[plant_id])) for plant_id in seed_ids]
    expanded_coverage = seed_coverage + [
        (plant_id, _theta(row_by_id[plant_id])) for plant_id in selected_ids
    ]
    before = [
        _distance_to_coverage(_theta(row_by_id[plant_id]), seed_coverage)[0]
        for plant_id in target_ids
    ]
    after = [
        _distance_to_coverage(_theta(row_by_id[plant_id]), expanded_coverage)[0]
        for plant_id in target_ids
    ]
    positions = np.arange(len(target_ids), dtype=float)
    width = 0.38
    axes[1].bar(positions - width / 2, before, width, label="4 current Teachers")
    axes[1].bar(positions + width / 2, after, width, label="+ selected candidates")
    axes[1].set_xticks(
        positions,
        [plant_id.split("-", 1)[-1] for plant_id in target_ids],
        rotation=30,
        ha="right",
    )
    axes[1].set_ylabel("Nearest normalized-theta distance")
    axes[1].set_title("Coverage of the frozen zero-shot test set")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = _parse_args()
    if args.candidate_count <= 0:
        raise ValueError("candidate count must be positive")
    library_path = args.library.resolve()
    bank_path = args.existing_bank.resolve()
    unseen_path = args.unseen_report.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    splits = tuple(args.split or DEFAULT_SPLITS)
    quality_regions = tuple(args.quality_region or DEFAULT_QUALITY_REGIONS)
    if len(set(splits)) != len(splits) or len(set(quality_regions)) != len(
        quality_regions
    ):
        raise ValueError("split and quality-region filters must be unique")

    rows = _load_library(library_path)
    row_by_id = {str(row["plant_id"]): row for row in rows}
    bank = _read_json(bank_path)
    unseen = _read_json(unseen_path)
    teachers = bank.get("teachers")
    rejected = bank.get("rejected_teachers", [])
    if bank.get("status") != "complete" or not isinstance(teachers, list):
        raise ValueError("existing Teacher Bank is incomplete")
    if not isinstance(rejected, list):
        raise ValueError("existing Teacher Bank rejected-teacher list is invalid")
    seed_ids = [str(entry["plant_id"]) for entry in teachers]
    attempted_ids = seed_ids + [str(entry["plant_id"]) for entry in rejected]
    target_ids = [str(value) for value in unseen["scope"]["target_plant_ids"]]
    missing_ids = sorted(
        (set(seed_ids) | set(target_ids) | set(args.exclude_plant_id)) - set(row_by_id)
    )
    if missing_ids:
        raise ValueError(f"selection contract references missing aircraft: {missing_ids}")

    excluded_ids = set(attempted_ids) | set(target_ids) | set(args.exclude_plant_id)
    eligible = [
        row
        for row in rows
        if row["split"] in splits
        and row["quality_region"] in quality_regions
        and row["plant_id"] not in excluded_ids
    ]
    quotas = _quotas(splits, quality_regions, args.candidate_count)
    pools = {
        cell: [
            row
            for row in eligible
            if (str(row["split"]), str(row["quality_region"])) == cell
        ]
        for cell in quotas
    }
    insufficient = {
        f"{split}/{region}": {"required": quotas[(split, region)], "available": len(pool)}
        for (split, region), pool in pools.items()
        if len(pool) < quotas[(split, region)]
    }
    if insufficient:
        raise ValueError(f"insufficient aircraft for coverage quotas: {insufficient}")

    seed_coverage = [(plant_id, _theta(row_by_id[plant_id])) for plant_id in seed_ids]
    selected = _select(pools, quotas, seed_coverage)
    selected_ids = [str(row["plant_id"]) for row in selected]
    if set(selected_ids) & excluded_ids:
        raise AssertionError("coverage selection leaked an excluded aircraft")
    expanded_coverage = seed_coverage + [
        (plant_id, _theta(row_by_id[plant_id])) for plant_id in selected_ids
    ]
    target_rows = [row_by_id[plant_id] for plant_id in target_ids]
    plot_path = _save_plot(
        rows,
        seed_ids,
        selected_ids,
        target_ids,
        destination / "coverage.png",
    )
    report = {
        "schema_version": "teacher_coverage_selection_v1",
        "status": "complete",
        "source": git_source_revision(),
        "method": "quality-and-source-split-stratified_farthest_first_maximin",
        "distance_metric": "euclidean_distance_in_log_linear_normalized_theta",
        "aircraft_parameter_names": list(AIRCRAFT_PARAMETER_NAMES),
        "aircraft_parameter_normalization": AIRCRAFT_PARAMETER_NORMALIZATION,
        "library": {"path": str(library_path), "sha256": sha256_file(library_path)},
        "existing_teacher_bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
        },
        "frozen_unseen_report": {
            "path": str(unseen_path),
            "sha256": sha256_file(unseen_path),
        },
        "contract": {
            "candidate_count": args.candidate_count,
            "splits": list(splits),
            "quality_regions": list(quality_regions),
            "seed_teacher_ids": seed_ids,
            "previously_attempted_teacher_ids": attempted_ids,
            "frozen_zero_shot_target_ids": target_ids,
            "additional_excluded_plant_ids": sorted(args.exclude_plant_id),
            "selected_target_overlap": sorted(set(selected_ids) & set(target_ids)),
            "selected_previous_attempt_overlap": sorted(
                set(selected_ids) & set(attempted_ids)
            ),
        },
        "cell_quotas": [
            {"split": split, "quality_region": region, "count": count}
            for (split, region), count in quotas.items()
        ],
        "selected_distribution": {
            "by_split": dict(Counter(row["split"] for row in selected)),
            "by_quality_region": dict(
                Counter(row["quality_region"] for row in selected)
            ),
        },
        "selected_candidates": selected,
        "coverage": {
            "eligible_population_before": _distance_summary(eligible, seed_coverage),
            "eligible_population_after": _distance_summary(eligible, expanded_coverage),
            "frozen_zero_shot_targets_before": _distance_summary(
                target_rows, seed_coverage
            ),
            "frozen_zero_shot_targets_after": _distance_summary(
                target_rows, expanded_coverage
            ),
        },
        "artifacts": {"coverage_plot": str(plot_path)},
    }
    if report["contract"]["selected_target_overlap"] or report["contract"][
        "selected_previous_attempt_overlap"
    ]:
        raise AssertionError("selection self-check failed")
    _write_json(destination / "selection.json", report)
    print(json.dumps({"status": "complete", "selected_ids": selected_ids}, indent=2))


if __name__ == "__main__":
    main()
