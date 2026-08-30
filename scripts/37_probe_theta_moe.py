"""Probe theta-router temperatures with held-out-aircraft closed-loop evaluation."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.collect_data import (
    DistillationCollectionConfig,
    collect_teacher_bank_data,
)
from src.distillation.distill import DenseStudentTrainingConfig, train_dense_student
from src.distillation.validate import evaluate_dense_student_bank
from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-dataset", type=Path)
    parser.add_argument(
        "--temperature",
        type=float,
        action="append",
        dest="temperatures",
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--prototype-movement-limit", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _validation_summary(evaluation: dict[str, object]) -> dict[str, object]:
    by_split = evaluation["by_distillation_split"]
    if "validation_aircraft" not in by_split:
        raise ValueError("MoE temperature probe requires aircraft-level holdout")
    return by_split["validation_aircraft"]


def _passes_guardrails(summary: dict[str, object]) -> bool:
    return (
        float(summary["student_improvement_rate"]) >= 0.8
        and float(summary["student_harm_rate"]) <= 0.2
        and float(summary["maximum_student_peak_error_deg_s"]) <= 10.0
        and float(summary["mean_student_requested_force_total_variation_n"])
        <= 120.0
        and float(summary["student_teacher_requested_force_variation_ratio"])
        <= 1.25
    )


def _save_plot(rows: list[dict[str, object]], path: Path) -> None:
    temperatures = np.asarray([row["temperature"] for row in rows], dtype=float)
    rmse = np.asarray(
        [row["validation_mean_student_tracking_rmse_deg_s"] for row in rows],
        dtype=float,
    )
    gap = np.asarray(
        [row["validation_median_student_minus_teacher_rmse_deg_s"] for row in rows],
        dtype=float,
    )
    variation = np.asarray(
        [row["validation_student_teacher_force_variation_ratio"] for row in rows],
        dtype=float,
    )
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True, layout="constrained")
    axes[0].plot(temperatures, rmse, marker="o", color="#c82423")
    axes[0].set_ylabel("Student RMSE (deg/s)")
    axes[1].plot(temperatures, gap, marker="o", color="#2878b5")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Student - Teacher RMSE (deg/s)")
    axes[2].plot(temperatures, variation, marker="o", color="#2f8f46")
    axes[2].axhline(1.25, color="black", linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Student / Teacher force TV")
    axes[2].set_xlabel("Theta-router temperature")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle("Theta-only linear MoE temperature probe")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if any(temperature <= 0 for temperature in args.temperatures):
        raise ValueError("router temperatures must be positive")
    bank_path = args.teacher_bank.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    dataset_dir = destination / "dataset"
    if args.reuse_dataset is None:
        dataset = collect_teacher_bank_data(
            bank_path,
            dataset_dir,
            DistillationCollectionConfig(
                sample_stride=args.sample_stride,
                validation_aircraft_fraction=0.2,
                seed=args.seed,
                device=args.device,
            ),
        )
    else:
        source_manifest = args.reuse_dataset.resolve()
        dataset = json.loads(source_manifest.read_text(encoding="utf-8"))
        if (
            dataset.get("schema_version")
            != "specialist_distillation_dataset_v1"
            or dataset.get("status") != "complete"
        ):
            raise ValueError("reused distillation dataset is incomplete")
        shutil.copytree(source_manifest.parent, dataset_dir)
    dataset_path = dataset_dir / "dataset.json"

    rows: list[dict[str, object]] = []
    for index, temperature in enumerate(args.temperatures):
        run_dir = destination / f"temperature_{temperature:.4f}"
        config = DenseStudentTrainingConfig(
            architecture="theta_routed_linear_moe",
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=3e-4,
            weight_decay=1e-5,
            patience_epochs=15,
            moe_expert_count=0,
            moe_router_temperature=temperature,
            moe_prototype_movement_limit=args.prototype_movement_limit,
            moe_router_balance_weight=1e-3,
            moe_router_z_loss_weight=1e-5,
            moe_prototype_anchor_weight=1e-3,
            seed=args.seed + index,
            device=args.device,
        )
        training = train_dense_student(dataset_path, run_dir / "student", config)
        checkpoint = Path(str(training["checkpoint"]))
        evaluation = evaluate_dense_student_bank(
            checkpoint,
            bank_path,
            run_dir / "evaluation",
            device=args.device,
        )
        validation = _validation_summary(evaluation)
        rows.append(
            {
                "temperature": temperature,
                "guardrails_passed": _passes_guardrails(validation),
                "validation_mean_student_tracking_rmse_deg_s": validation[
                    "mean_student_tracking_rmse_deg_s"
                ],
                "validation_median_student_minus_teacher_rmse_deg_s": float(
                    np.rad2deg(
                        validation["median_student_minus_teacher_rmse_rad_s"]
                    )
                ),
                "validation_student_improvement_rate": validation[
                    "student_improvement_rate"
                ],
                "validation_student_harm_rate": validation["student_harm_rate"],
                "validation_maximum_student_peak_error_deg_s": validation[
                    "maximum_student_peak_error_deg_s"
                ],
                "validation_mean_student_requested_force_total_variation_n": validation[
                    "mean_student_requested_force_total_variation_n"
                ],
                "validation_student_teacher_force_variation_ratio": validation[
                    "student_teacher_requested_force_variation_ratio"
                ],
                "training": training,
                "evaluation": str(run_dir / "evaluation/evaluation.json"),
                "checkpoint": str(checkpoint),
            }
        )

    eligible = [row for row in rows if row["guardrails_passed"]]
    selection_pool = eligible or rows
    selected = min(
        selection_pool,
        key=lambda row: float(
            row["validation_median_student_minus_teacher_rmse_deg_s"]
        ),
    )
    csv_path = destination / "temperature_metrics.csv"
    csv_fields = [
        key
        for key in rows[0]
        if key not in {"training", "evaluation", "checkpoint"}
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in csv_fields} for row in rows
        )
    plot_path = destination / "temperature_probe.png"
    _save_plot(rows, plot_path)
    report = {
        "schema_version": "theta_routed_linear_moe_temperature_probe_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
        },
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "train_plant_ids": dataset["train_plant_ids"],
            "validation_plant_ids": dataset["validation_plant_ids"],
        },
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "sample_stride": args.sample_stride,
            "prototype_movement_limit": args.prototype_movement_limit,
            "seed": args.seed,
            "device": args.device,
        },
        "selection_rule": (
            "minimum_validation_median_student_minus_teacher_rmse_"
            "among_force_and_tracking_guardrail_eligible_temperatures"
        ),
        "selected_temperature": selected["temperature"],
        "selected_guardrails_passed": selected["guardrails_passed"],
        "rows": rows,
        "artifacts": {"metrics_csv": str(csv_path), "plot": str(plot_path)},
    }
    report_path = destination / "temperature_probe.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "selected_temperature": selected["temperature"],
                "selected_guardrails_passed": selected["guardrails_passed"],
                "rows": [
                    {key: row[key] for key in csv_fields} for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
