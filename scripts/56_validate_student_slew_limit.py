"""Validate a training-selected Student slew limit on held-out aircraft."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
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

from src.controllers.policy_wrappers import ForceSlewLimitedPolicy
from src.envs.roll_rate_commands import specialist_evaluation_commands
from src.student.dense.policy import DenseStudentPolicy, load_dense_student
from src.teacher.specialist.trainer import (
    CommandForceBaseline,
    build_specialist_env,
    load_specialist_actor,
    rollout_policy,
    tracking_metrics,
)
from src.utils.plotting import save_controller_comparison_grid
from src.utils.provenance import git_source_revision, sha256_file


CONTROLLERS = ("raw", "teacher", "unlimited_student", "limited_student")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--slew-limit-scan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-student-teacher-rmse-gap-deg-s", type=float, default=0.5)
    parser.add_argument("--minimum-student-improvement-rate", type=float, default=1.0)
    parser.add_argument("--maximum-student-harm-rate", type=float, default=0.0)
    parser.add_argument("--maximum-student-peak-error-deg-s", type=float, default=5.0)
    parser.add_argument(
        "--maximum-mean-student-requested-force-variation-n",
        type=float,
        default=360.0,
    )
    parser.add_argument(
        "--maximum-student-teacher-force-variation-ratio",
        type=float,
        default=1.25,
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _summary(rows: list[dict[str, object]], controller: str) -> dict[str, float | int]:
    metrics = [row[controller] for row in rows]
    return {
        "pair_count": len(metrics),
        "mean_tracking_rmse_deg_s": float(
            np.mean([metric["tracking_rmse_deg_s"] for metric in metrics])
        ),
        "maximum_peak_error_deg_s": float(
            np.max([metric["tracking_peak_error_deg_s"] for metric in metrics])
        ),
        "mean_episode_cost": float(
            np.mean([metric["episode_cost"] for metric in metrics])
        ),
        "mean_requested_force_total_variation_n": float(
            np.mean(
                [metric["requested_force_total_variation_n"] for metric in metrics]
            )
        ),
    }


def _quality_observed(
    rows: list[dict[str, object]], controller: str
) -> dict[str, float | int]:
    raw_rmse = np.asarray([row["raw"]["tracking_rmse_deg_s"] for row in rows])
    raw_cost = np.asarray([row["raw"]["episode_cost"] for row in rows])
    teacher_rmse = np.asarray(
        [row["teacher"]["tracking_rmse_deg_s"] for row in rows]
    )
    teacher_tv = np.asarray(
        [row["teacher"]["requested_force_total_variation_n"] for row in rows]
    )
    student_rmse = np.asarray(
        [row[controller]["tracking_rmse_deg_s"] for row in rows]
    )
    student_cost = np.asarray([row[controller]["episode_cost"] for row in rows])
    student_peak = np.asarray(
        [row[controller]["tracking_peak_error_deg_s"] for row in rows]
    )
    student_tv = np.asarray(
        [row[controller]["requested_force_total_variation_n"] for row in rows]
    )
    mean_teacher_tv = float(np.mean(teacher_tv))
    mean_student_tv = float(np.mean(student_tv))
    return {
        "pair_count": len(rows),
        "median_student_minus_teacher_rmse_deg_s": float(
            np.median(student_rmse - teacher_rmse)
        ),
        "student_improvement_rate": float(np.mean(student_rmse < raw_rmse)),
        "student_harm_rate": float(np.mean(student_cost > raw_cost)),
        "maximum_student_peak_error_deg_s": float(np.max(student_peak)),
        "mean_teacher_requested_force_total_variation_n": mean_teacher_tv,
        "mean_student_requested_force_total_variation_n": mean_student_tv,
        "student_teacher_requested_force_variation_ratio": mean_student_tv
        / max(mean_teacher_tv, 1e-8),
    }


def _quality_gate(
    observed: dict[str, float | int], args: argparse.Namespace
) -> dict[str, object]:
    checks = {
        "student_teacher_rmse_gap": float(
            observed["median_student_minus_teacher_rmse_deg_s"]
        )
        <= args.max_student_teacher_rmse_gap_deg_s,
        "student_improvement_rate": float(observed["student_improvement_rate"])
        >= args.minimum_student_improvement_rate,
        "student_harm_rate": float(observed["student_harm_rate"])
        <= args.maximum_student_harm_rate,
        "student_peak_error": float(observed["maximum_student_peak_error_deg_s"])
        <= args.maximum_student_peak_error_deg_s,
        "student_requested_force_variation": float(
            observed["mean_student_requested_force_total_variation_n"]
        )
        <= args.maximum_mean_student_requested_force_variation_n,
        "student_teacher_force_variation_ratio": float(
            observed["student_teacher_requested_force_variation_ratio"]
        )
        <= args.maximum_student_teacher_force_variation_ratio,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": observed,
        "thresholds": {
            "max_student_teacher_rmse_gap_deg_s": (
                args.max_student_teacher_rmse_gap_deg_s
            ),
            "minimum_student_improvement_rate": args.minimum_student_improvement_rate,
            "maximum_student_harm_rate": args.maximum_student_harm_rate,
            "maximum_student_peak_error_deg_s": (
                args.maximum_student_peak_error_deg_s
            ),
            "maximum_mean_student_requested_force_variation_n": (
                args.maximum_mean_student_requested_force_variation_n
            ),
            "maximum_student_teacher_force_variation_ratio": (
                args.maximum_student_teacher_force_variation_ratio
            ),
        },
    }


def _save_summary_plot(rows: list[dict[str, object]], path: Path) -> Path:
    labels = [
        f"{row['plant_id']}\n{row['command_id'].replace('eval-', '')}" for row in rows
    ]
    positions = np.arange(len(rows))
    figure, axes = plt.subplots(2, 1, figsize=(18, 10), layout="constrained")
    colors = {
        "teacher": "#2878b5",
        "unlimited_student": "#c82d2d",
        "limited_student": "#16803c",
    }
    for controller in ("teacher", "unlimited_student", "limited_student"):
        axes[0].plot(
            positions,
            [row[controller]["tracking_rmse_deg_s"] for row in rows],
            marker="o",
            markersize=3,
            linewidth=1.0,
            color=colors[controller],
            label=controller.replace("_", " "),
        )
        axes[1].plot(
            positions,
            [
                row[controller]["requested_force_total_variation_n"]
                for row in rows
            ],
            marker="o",
            markersize=3,
            linewidth=1.0,
            color=colors[controller],
            label=controller.replace("_", " "),
        )
    axes[0].set_ylabel("Tracking RMSE (deg/s)")
    axes[1].set_ylabel("Requested-force TV (N)")
    axes[1].set_xticks(positions, labels, rotation=70, ha="right", fontsize=7)
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
    scan_path = args.slew_limit_scan.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    if bank.get("status") != "complete" or scan.get("status") != "complete":
        raise ValueError("complete Teacher Bank and slew-limit scan are required")
    selected = scan.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("slew-limit scan has no selected candidate")
    rate = float(selected["force_rate_limit_n_s"])
    student_hash = sha256_file(student_path)
    if (
        scan.get("selection_is_training_only") is not True
        or scan.get("student_checkpoint", {}).get("sha256") != student_hash
        or scan.get("dataset", {}).get("sha256") != sha256_file(dataset_path)
    ):
        raise ValueError("scan provenance does not match this frozen evaluation")
    validation_ids = list(map(str, dataset["validation_plant_ids"]))
    if set(validation_ids).intersection(map(str, dataset["train_plant_ids"])):
        raise ValueError("training and validation aircraft overlap")
    entries = {str(row["plant_id"]): row for row in bank["teachers"]}
    missing = sorted(set(validation_ids) - set(entries))
    if missing:
        raise ValueError(f"validation aircraft are absent from Teacher Bank: {missing}")
    model, payload = load_dense_student(student_path, device=args.device)
    rows: list[dict[str, object]] = []
    worst_case: tuple[
        float,
        str,
        str,
        dict[str, dict[str, np.ndarray]],
    ] | None = None
    for plant_index, plant_id in enumerate(validation_ids):
        actor_path = bank_path.parent / str(entries[plant_id]["actor_checkpoint"])
        teacher, record, config, teacher_payload = load_specialist_actor(
            actor_path, device=args.device
        )
        if int(teacher_payload["actor_observation_dim"]) != model.observation_dim:
            raise ValueError("Teacher and Student observation contracts differ")
        for command_index, profile in enumerate(
            specialist_evaluation_commands(config.episode_duration_s)
        ):
            seed = config.seed + plant_index * 1000 + command_index
            policies = {
                "raw": CommandForceBaseline(),
                "teacher": teacher,
                "unlimited_student": DenseStudentPolicy(
                    model, record.parameters, device=args.device
                ),
                "limited_student": ForceSlewLimitedPolicy(
                    DenseStudentPolicy(model, record.parameters, device=args.device),
                    force_rate_limit_n_s=rate,
                    policy_dt_s=config.policy_dt_s,
                    force_limit_n=config.force_limit_n,
                ),
            }
            metrics = {}
            traces = {}
            for controller, policy in policies.items():
                trace = rollout_policy(
                    policy,
                    build_specialist_env(record, config, (profile,)),
                    seed=seed,
                )
                traces[controller] = trace
                metrics[controller] = tracking_metrics(trace, config.force_limit_n)
            limited_peak = float(
                metrics["limited_student"]["tracking_peak_error_deg_s"]
            )
            if worst_case is None or limited_peak > worst_case[0]:
                worst_case = (
                    limited_peak,
                    plant_id,
                    profile.command_id,
                    traces,
                )
            rows.append(
                {
                    "plant_id": plant_id,
                    "split": record.split,
                    "quality_region": record.quality_region,
                    "command_id": profile.command_id,
                    **metrics,
                }
            )
        print(json.dumps({"completed_holdout_plant": plant_id}), flush=True)

    unlimited_observed = _quality_observed(rows, "unlimited_student")
    limited_observed = _quality_observed(rows, "limited_student")
    unlimited_gate = _quality_gate(unlimited_observed, args)
    limited_gate = _quality_gate(limited_observed, args)
    summary_plot = _save_summary_plot(rows, destination / "holdout_summary.png")
    if worst_case is None:
        raise ValueError("holdout evaluation produced no worst case")
    _, worst_plant_id, worst_command_id, worst_traces = worst_case
    worst_case_plot = save_controller_comparison_grid(
        [
            (
                worst_command_id,
                worst_traces["raw"],
                {
                    "Teacher": worst_traces["teacher"],
                    "v4 Student": worst_traces["unlimited_student"],
                    f"v4 + {rate:g} N/s": worst_traces["limited_student"],
                },
            )
        ],
        destination / "worst_limited_response.png",
        title=f"{worst_plant_id}: frozen slew-limit holdout failure",
    )
    self_check = {
        "passed": bool(
            sha256_file(student_path) == student_hash
            and not set(validation_ids).intersection(scan["tuning_aircraft"])
            and len(rows) == 6 * len(validation_ids)
        ),
        "student_checkpoint_unchanged": sha256_file(student_path) == student_hash,
        "tuning_holdout_overlap": sorted(
            set(validation_ids).intersection(scan["tuning_aircraft"])
        ),
        "pair_count": len(rows),
    }
    status = (
        "complete"
        if self_check["passed"] and limited_gate["passed"]
        else "quality_gate_failed"
    )
    report = {
        "schema_version": "student_slew_limit_holdout_validation_v1",
        "status": status,
        "source": git_source_revision(),
        "selection_is_frozen": True,
        "selected_force_rate_limit_n_s": rate,
        "teacher_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "dataset": {"path": str(dataset_path), "sha256": sha256_file(dataset_path)},
        "student_checkpoint": {
            "path": str(student_path),
            "sha256": student_hash,
            "parameter_count": payload["parameter_count"],
        },
        "slew_limit_scan": {"path": str(scan_path), "sha256": sha256_file(scan_path)},
        "validation_plant_ids": validation_ids,
        "self_check": self_check,
        "controller_summaries": {
            controller: _summary(rows, controller) for controller in CONTROLLERS
        },
        "unlimited_student_quality_gate": unlimited_gate,
        "limited_student_quality_gate": limited_gate,
        "rows": rows,
        "artifacts": {
            "summary_plot": str(summary_plot),
            "worst_limited_response_plot": str(worst_case_plot),
            "worst_limited_plant_id": worst_plant_id,
            "worst_limited_command_id": worst_command_id,
        },
    }
    _write_json(destination / "holdout_report.json", report)
    print(
        json.dumps(
            {
                "status": status,
                "selected_force_rate_limit_n_s": rate,
                "unlimited_gate": unlimited_gate,
                "limited_gate": limited_gate,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
