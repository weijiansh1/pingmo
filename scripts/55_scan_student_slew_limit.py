"""Select a frozen Student force-slew limit using training aircraft only."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import csv
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

from src.context.aircraft_parameters import normalize_aircraft_parameters
from src.controllers.policy_wrappers import ForceSlewLimitedPolicy
from src.distillation.dataset import TRAIN_SPLIT, load_distillation_arrays
from src.envs.roll_rate_commands import specialist_evaluation_commands
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import (
    build_specialist_env,
    load_specialist_actor,
    rollout_policy,
    tracking_metrics,
)
from src.utils.provenance import git_source_revision, sha256_file


TEACHER_RATE_QUANTILES = (0.90, 0.95, 0.975, 0.99)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-rate", action="append", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-mean-rmse-increase-deg-s", type=float, default=0.10)
    parser.add_argument("--max-peak-increase-deg-s", type=float, default=0.25)
    parser.add_argument("--max-peak-error-deg-s", type=float, default=5.0)
    parser.add_argument("--max-teacher-tv-ratio", type=float, default=1.25)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _teacher_action_rate_distribution(
    dataset_path: Path, *, force_limit_n: float, policy_dt_s: float
) -> tuple[dict[str, float | int], list[float], dict[str, object]]:
    arrays, manifest = load_distillation_arrays(dataset_path)
    action = np.asarray(arrays.teacher_actions[:, 0], dtype=np.float64)
    episode = np.asarray(arrays.episode_indices, dtype=np.int64)
    step = np.asarray(arrays.policy_step_indices, dtype=np.int64)
    split = np.asarray(arrays.split_codes)
    order = np.lexsort((step, episode))
    action, episode, step, split = (
        values[order] for values in (action, episode, step, split)
    )
    same_episode = episode[1:] == episode[:-1]
    step_delta = (step[1:] - step[:-1])[same_episode]
    if not len(step_delta) or np.any(step_delta <= 0):
        raise ValueError("dataset has no valid forward temporal pairs")
    action_delta = np.abs(action[1:] - action[:-1])[same_episode]
    rates = action_delta * force_limit_n / (step_delta * policy_dt_s)
    rates = rates[split[1:][same_episode] == TRAIN_SPLIT]
    if not len(rates):
        raise ValueError("dataset has no training temporal pairs")
    quantile_values = np.quantile(rates, TEACHER_RATE_QUANTILES)
    distribution: dict[str, float | int] = {
        "pair_count": int(len(rates)),
        "mean_n_s": float(np.mean(rates)),
        "rms_n_s": float(np.sqrt(np.mean(np.square(rates)))),
        "maximum_n_s": float(np.max(rates)),
    }
    for quantile, value in zip(
        TEACHER_RATE_QUANTILES, quantile_values, strict=True
    ):
        distribution[f"p{quantile * 100:g}_n_s"] = float(value)
    candidates = sorted(
        {
            max(1.0, float(round(value)))
            for value in quantile_values
        }
        | {88.0}
    )
    return distribution, candidates, manifest


def _select_stratified_medoids(
    loaded: dict[str, tuple[object, object, object, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    strata: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}
    for plant_id, (_, record, _, _) in loaded.items():
        key = (str(record.split), str(record.quality_region))
        strata.setdefault(key, []).append(
            (plant_id, normalize_aircraft_parameters(record.parameters))
        )
    selected: list[str] = []
    details: list[dict[str, object]] = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda row: row[0])
        theta = np.stack([row[1] for row in rows])
        distances = np.linalg.norm(theta[:, None, :] - theta[None, :, :], axis=2)
        medoid_index = int(np.argmin(np.sum(distances, axis=1)))
        plant_id = rows[medoid_index][0]
        selected.append(plant_id)
        details.append(
            {
                "split": key[0],
                "quality_region": key[1],
                "candidate_count": len(rows),
                "medoid_plant_id": plant_id,
                "sum_normalized_theta_distance": float(
                    np.sum(distances[medoid_index])
                ),
            }
        )
    return selected, details


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    return {
        "pair_count": len(rows),
        "mean_tracking_rmse_deg_s": float(
            np.mean([row["metrics"]["tracking_rmse_deg_s"] for row in rows])
        ),
        "maximum_peak_error_deg_s": float(
            np.max([row["metrics"]["tracking_peak_error_deg_s"] for row in rows])
        ),
        "mean_requested_force_total_variation_n": float(
            np.mean(
                [
                    row["metrics"]["requested_force_total_variation_n"]
                    for row in rows
                ]
            )
        ),
    }


def _save_tradeoff_plot(candidates: list[dict[str, object]], path: Path) -> Path:
    finite = [row for row in candidates if row["force_rate_limit_n_s"] is not None]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    rate = np.asarray([row["force_rate_limit_n_s"] for row in finite], dtype=float)
    rmse = np.asarray(
        [row["mean_tracking_rmse_deg_s"] for row in finite], dtype=float
    )
    variation = np.asarray(
        [row["mean_requested_force_total_variation_n"] for row in finite],
        dtype=float,
    )
    colors = ["#16803c" if row["eligible"] else "#c82d2d" for row in finite]
    axes[0].scatter(rate, rmse, c=colors, s=65)
    axes[0].plot(rate, rmse, color="#64748b", linewidth=1.0)
    axes[0].set_xlabel("Requested-force slew limit (N/s)")
    axes[0].set_ylabel("Mean tracking RMSE (deg/s)")
    axes[1].scatter(rate, variation, c=colors, s=65)
    axes[1].plot(rate, variation, color="#64748b", linewidth=1.0)
    axes[1].set_xlabel("Requested-force slew limit (N/s)")
    axes[1].set_ylabel("Mean requested-force TV (N)")
    for axis in axes:
        axis.grid(alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def _save_response_plot(
    traces: dict[str, dict[str, np.ndarray]], path: Path, title: str
) -> Path:
    figure, axes = plt.subplots(
        2, 1, figsize=(11, 7.5), sharex=True, layout="constrained"
    )
    reference = traces["teacher"]
    axes[0].plot(
        reference["time_s"],
        np.rad2deg(reference["p_command_rad_s"]),
        color="black",
        linestyle=":",
        label="p_c",
    )
    axes[0].plot(
        reference["time_s"],
        np.rad2deg(reference["p_reference_rad_s"]),
        color="black",
        linestyle="--",
        label="p_ref",
    )
    colors = {"teacher": "#2878b5", "unlimited": "#c82d2d", "selected": "#16803c"}
    labels = {"teacher": "Teacher", "unlimited": "v4 Student", "selected": "v4 + slew limit"}
    for name in ("teacher", "unlimited", "selected"):
        trace = traces[name]
        axes[0].plot(
            trace["time_s"],
            np.rad2deg(trace["p_rad_s"]),
            color=colors[name],
            linewidth=1.5,
            label=labels[name],
        )
        axes[1].plot(
            trace["time_s"],
            trace["requested_f_as_n"],
            color=colors[name],
            linewidth=1.2,
            label=labels[name],
        )
    axes[0].set_ylabel("Roll rate p (deg/s)")
    axes[0].set_title(title)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Requested F_as (N)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = _parse_args()
    bank_path = args.teacher_bank.resolve()
    dataset_path = args.dataset.resolve()
    student_path = args.student_checkpoint.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("status") != "complete":
        raise ValueError("a complete Teacher Bank is required")
    entries = {str(row["plant_id"]): row for row in bank["teachers"]}
    student_model, student_payload = load_dense_student(student_path, device=args.device)
    checkpoint_hash_before = sha256_file(student_path)

    first_entry = next(iter(entries.values()))
    first_actor = bank_path.parent / str(first_entry["actor_checkpoint"])
    _, _, first_config, teacher_payload = load_specialist_actor(
        first_actor, device=args.device
    )
    if int(teacher_payload["actor_observation_dim"]) != student_model.observation_dim:
        raise ValueError("Teacher and Student observation contracts differ")
    distribution, derived_candidates, dataset_manifest = (
        _teacher_action_rate_distribution(
            dataset_path,
            force_limit_n=first_config.force_limit_n,
            policy_dt_s=first_config.policy_dt_s,
        )
    )
    train_ids = list(map(str, dataset_manifest["train_plant_ids"]))
    validation_ids = set(map(str, dataset_manifest["validation_plant_ids"]))
    if validation_ids.intersection(train_ids):
        raise ValueError("training and validation aircraft overlap")
    missing = sorted(set(train_ids) - set(entries))
    if missing:
        raise ValueError(f"training aircraft are absent from Teacher Bank: {missing}")
    candidates = args.candidate_rate or derived_candidates
    candidates = sorted(set(map(float, candidates)))
    if not candidates or any(not np.isfinite(rate) or rate <= 0 for rate in candidates):
        raise ValueError("candidate rates must be finite and positive")

    loaded = {
        plant_id: load_specialist_actor(
            bank_path.parent / str(entries[plant_id]["actor_checkpoint"]),
            device=args.device,
        )
        for plant_id in train_ids
    }
    selected_ids, medoid_details = _select_stratified_medoids(loaded)
    if len(selected_ids) != 6:
        raise ValueError(
            "expected six train split/quality strata; "
            f"selected {len(selected_ids)}"
        )

    rows: list[dict[str, object]] = []
    trace_cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    for plant_index, plant_id in enumerate(selected_ids):
        teacher, record, config, _ = loaded[plant_id]
        if (
            config.policy_dt_s != first_config.policy_dt_s
            or config.force_limit_n != first_config.force_limit_n
        ):
            raise ValueError("Teacher Bank has inconsistent action contracts")
        profiles = specialist_evaluation_commands(config.episode_duration_s)
        for command_index, profile in enumerate(profiles):
            seed = config.seed + plant_index * 1000 + command_index
            teacher_trace = rollout_policy(
                teacher,
                build_specialist_env(record, config, (profile,)),
                seed=seed,
            )
            unlimited = DenseStudentPolicy(
                student_model, record.parameters, device=args.device
            )
            unlimited_trace = rollout_policy(
                unlimited,
                build_specialist_env(record, config, (profile,)),
                seed=seed,
            )
            for label, trace in (
                ("teacher", teacher_trace),
                ("unlimited", unlimited_trace),
            ):
                rows.append(
                    {
                        "plant_id": plant_id,
                        "split": record.split,
                        "quality_region": record.quality_region,
                        "command_id": profile.command_id,
                        "controller": label,
                        "force_rate_limit_n_s": None,
                        "metrics": tracking_metrics(trace, config.force_limit_n),
                    }
                )
                trace_cache[(plant_id, profile.command_id, label)] = trace
            for rate in candidates:
                limited = ForceSlewLimitedPolicy(
                    DenseStudentPolicy(
                        student_model, record.parameters, device=args.device
                    ),
                    force_rate_limit_n_s=rate,
                    policy_dt_s=config.policy_dt_s,
                    force_limit_n=config.force_limit_n,
                )
                trace = rollout_policy(
                    limited,
                    build_specialist_env(record, config, (profile,)),
                    seed=seed,
                )
                label = f"rate-{rate:g}"
                rows.append(
                    {
                        "plant_id": plant_id,
                        "split": record.split,
                        "quality_region": record.quality_region,
                        "command_id": profile.command_id,
                        "controller": label,
                        "force_rate_limit_n_s": rate,
                        "metrics": tracking_metrics(trace, config.force_limit_n),
                    }
                )
                trace_cache[(plant_id, profile.command_id, label)] = trace
        print(json.dumps({"completed_tuning_plant": plant_id}), flush=True)

    teacher_summary = _aggregate(
        [row for row in rows if row["controller"] == "teacher"]
    )
    baseline_summary = _aggregate(
        [row for row in rows if row["controller"] == "unlimited"]
    )
    candidate_summaries: list[dict[str, object]] = []
    for rate in candidates:
        summary = _aggregate(
            [row for row in rows if row["controller"] == f"rate-{rate:g}"]
        )
        allowed_peak = max(
            args.max_peak_error_deg_s,
            float(baseline_summary["maximum_peak_error_deg_s"])
            + args.max_peak_increase_deg_s,
        )
        checks = {
            "mean_rmse": float(summary["mean_tracking_rmse_deg_s"])
            <= float(baseline_summary["mean_tracking_rmse_deg_s"])
            + args.max_mean_rmse_increase_deg_s,
            "maximum_peak_error": float(summary["maximum_peak_error_deg_s"])
            <= allowed_peak,
            "requested_force_variation_not_increased": float(
                summary["mean_requested_force_total_variation_n"]
            )
            <= float(baseline_summary["mean_requested_force_total_variation_n"]),
            "teacher_variation_ratio": float(
                summary["mean_requested_force_total_variation_n"]
            )
            / max(
                float(teacher_summary["mean_requested_force_total_variation_n"]),
                1e-8,
            )
            <= args.max_teacher_tv_ratio,
        }
        candidate_summaries.append(
            {
                "force_rate_limit_n_s": rate,
                **summary,
                "student_teacher_requested_force_variation_ratio": float(
                    summary["mean_requested_force_total_variation_n"]
                )
                / max(
                    float(
                        teacher_summary["mean_requested_force_total_variation_n"]
                    ),
                    1e-8,
                ),
                "checks": checks,
                "eligible": all(checks.values()),
            }
        )
    eligible = [row for row in candidate_summaries if row["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda row: (
                row["mean_requested_force_total_variation_n"],
                row["mean_tracking_rmse_deg_s"],
            ),
        )
        if eligible
        else None
    )

    baseline_plot_row = {
        "force_rate_limit_n_s": None,
        **baseline_summary,
        "eligible": False,
    }
    tradeoff_plot = _save_tradeoff_plot(
        [baseline_plot_row, *candidate_summaries], destination / "tradeoff.png"
    )
    response_plot: Path | None = None
    if selected is not None:
        unlimited_rows = [row for row in rows if row["controller"] == "unlimited"]
        worst = max(
            unlimited_rows,
            key=lambda row: row["metrics"]["tracking_rmse_deg_s"],
        )
        plant_id = str(worst["plant_id"])
        command_id = str(worst["command_id"])
        selected_label = f"rate-{selected['force_rate_limit_n_s']:g}"
        response_plot = _save_response_plot(
            {
                name: trace_cache[(plant_id, command_id, label)]
                for name, label in (
                    ("teacher", "teacher"),
                    ("unlimited", "unlimited"),
                    ("selected", selected_label),
                )
            },
            destination / "worst_unlimited_response.png",
            f"{plant_id}: {command_id}",
        )

    csv_path = destination / "candidate_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            lineterminator="\n",
            fieldnames=(
                "force_rate_limit_n_s",
                "mean_tracking_rmse_deg_s",
                "maximum_peak_error_deg_s",
                "mean_requested_force_total_variation_n",
                "student_teacher_requested_force_variation_ratio",
                "eligible",
            ),
        )
        writer.writeheader()
        for row in candidate_summaries:
            writer.writerow({name: row[name] for name in writer.fieldnames})

    report = {
        "schema_version": "student_slew_limit_scan_v1",
        "status": "complete" if selected is not None else "selection_failed",
        "source": git_source_revision(),
        "selection_is_training_only": True,
        "teacher_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "student_checkpoint": {
            "path": str(student_path),
            "sha256": checkpoint_hash_before,
            "parameter_count": student_payload["parameter_count"],
            "unchanged_after_scan": sha256_file(student_path)
            == checkpoint_hash_before,
        },
        "action_contract": {
            "force_limit_n": first_config.force_limit_n,
            "policy_dt_s": first_config.policy_dt_s,
            "environment_commanded_force_rate_limit_n_s": (
                first_config.force_rate_limit_n_s
            ),
        },
        "teacher_training_action_rate_distribution": distribution,
        "candidate_rates_n_s": candidates,
        "tuning_aircraft": selected_ids,
        "excluded_validation_aircraft": sorted(validation_ids),
        "medoid_selection": medoid_details,
        "command_ids": [
            profile.command_id
            for profile in specialist_evaluation_commands(
                first_config.episode_duration_s
            )
        ],
        "selection_thresholds": {
            "max_mean_rmse_increase_deg_s": args.max_mean_rmse_increase_deg_s,
            "max_peak_increase_deg_s": args.max_peak_increase_deg_s,
            "max_peak_error_deg_s": args.max_peak_error_deg_s,
            "max_teacher_tv_ratio": args.max_teacher_tv_ratio,
        },
        "teacher_summary": teacher_summary,
        "unlimited_student_summary": baseline_summary,
        "candidates": candidate_summaries,
        "selected": selected,
        "plots": {
            "tradeoff": str(tradeoff_plot),
            "worst_unlimited_response": (
                str(response_plot) if response_plot is not None else None
            ),
        },
        "rows": rows,
    }
    _write_json(destination / "scan_report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "tuning_aircraft": selected_ids,
                "selected_force_rate_limit_n_s": (
                    selected["force_rate_limit_n_s"]
                    if selected is not None
                    else None
                ),
                "teacher": teacher_summary,
                "unlimited": baseline_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
