"""Probe a minimum p-dot damping coefficient on held-out aircraft."""

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
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context.aircraft_parameters import normalize_aircraft_parameters
from src.student.dense.policy import load_dense_student
from src.student.moe.network import ThetaRoutedLinearMoEStudent
from src.teacher.specialist.trainer import evaluate_specialist, load_specialist_actor
from src.utils.provenance import git_source_revision, sha256_file


class LinearControlPolicy:
    def __init__(self, coefficients: np.ndarray) -> None:
        values = np.asarray(coefficients, dtype=np.float32)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("linear control policy requires three finite coefficients")
        self.coefficients = values

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        del deterministic
        control_state = np.asarray(observation, dtype=np.float32)[[3, 4, 5]]
        action = np.clip(control_state @ self.coefficients, -1.0, 1.0)
        return np.asarray([action], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-bank", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-id", action="append")
    parser.add_argument(
        "--damping-floor", type=float, action="append", dest="damping_floors", required=True
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _effective_coefficients(
    model: ThetaRoutedLinearMoEStudent, parameters: object
) -> np.ndarray:
    theta = torch.as_tensor(
        normalize_aircraft_parameters(parameters), dtype=torch.float32
    ).unsqueeze(0)
    with torch.no_grad():
        route = model.routing_weights(theta)
        coefficients = torch.einsum(
            "be,eao->bao", route, model.expert_weights.cpu()
        )[0, 0]
    return coefficients.numpy()


def _save_plot(rows: list[dict[str, object]], output_path: Path) -> None:
    plants = sorted({str(row["plant_id"]) for row in rows})
    figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, layout="constrained")
    for plant_id in plants:
        selected = [row for row in rows if row["plant_id"] == plant_id]
        floor = [row["damping_floor"] for row in selected]
        axes[0].plot(
            floor,
            [row["tracking_rmse_deg_s"] for row in selected],
            marker="o",
            label=plant_id,
        )
        axes[1].plot(
            floor,
            [row["peak_error_deg_s"] for row in selected],
            marker="o",
            label=plant_id,
        )
        axes[2].plot(
            floor,
            [row["requested_force_tv_n"] for row in selected],
            marker="o",
            label=plant_id,
        )
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].axhline(10.0, color="black", linestyle="--", linewidth=1.0)
    axes[2].axhline(120.0, color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Tracking RMSE (deg/s)")
    axes[1].set_ylabel("Peak error (deg/s)")
    axes[2].set_ylabel("Requested-force TV (N)")
    axes[2].set_xlabel("Minimum normalized p-dot damping magnitude")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle("Held-out aircraft damping-floor probe")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    if any(value < 0 for value in args.damping_floors):
        raise ValueError("damping floors cannot be negative")
    bank_path = args.teacher_bank.resolve()
    checkpoint = args.student_checkpoint.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    model, payload = load_dense_student(checkpoint, device="cpu")
    if not isinstance(model, ThetaRoutedLinearMoEStudent):
        raise ValueError("damping-floor probe requires a linear MoE Student")
    dataset = payload.get("dataset_manifest")
    if not isinstance(dataset, dict):
        raise ValueError("Student checkpoint has no dataset split")
    plant_ids = args.plant_id or list(dataset["validation_plant_ids"])
    entries = {str(entry["plant_id"]): entry for entry in bank["teachers"]}

    rows: list[dict[str, object]] = []
    for plant_id in plant_ids:
        entry = entries.get(plant_id)
        if entry is None:
            raise ValueError(f"plant is absent from Teacher Bank: {plant_id}")
        _, record, config, _ = load_specialist_actor(
            bank_path.parent / str(entry["actor_checkpoint"]), device=args.device
        )
        base = _effective_coefficients(model, record.parameters)
        for damping_floor in args.damping_floors:
            coefficients = base.copy()
            coefficients[2] = min(float(coefficients[2]), -damping_floor)
            run_dir = destination / f"floor_{damping_floor:.4f}" / plant_id
            evaluation = evaluate_specialist(
                LinearControlPolicy(coefficients),
                record,
                config,
                output_dir=run_dir,
                controller_label="student",
                controller_display_label=f"Student damping >= {damping_floor:.3f}",
            )
            rows.append(
                {
                    "plant_id": plant_id,
                    "quality_region": record.quality_region,
                    "damping_floor": damping_floor,
                    "base_error_coefficient": float(base[0]),
                    "base_integral_coefficient": float(base[1]),
                    "base_p_dot_coefficient": float(base[2]),
                    "applied_p_dot_coefficient": float(coefficients[2]),
                    "tracking_rmse_deg_s": evaluation[
                        "mean_student_tracking_rmse_deg_s"
                    ],
                    "peak_error_deg_s": evaluation[
                        "maximum_student_peak_error_deg_s"
                    ],
                    "requested_force_tv_n": evaluation[
                        "mean_student_requested_force_total_variation_n"
                    ],
                    "tracking_improvement_rate": evaluation[
                        "tracking_improvement_rate"
                    ],
                    "harm_rate": evaluation["harm_rate"],
                    "evaluation": str(run_dir / "evaluation.json"),
                }
            )

    floor_summaries: list[dict[str, object]] = []
    for damping_floor in args.damping_floors:
        selected = [row for row in rows if row["damping_floor"] == damping_floor]
        guardrails = all(
            float(row["tracking_rmse_deg_s"]) <= 1.0
            and float(row["peak_error_deg_s"]) <= 10.0
            and float(row["requested_force_tv_n"]) <= 120.0
            and float(row["tracking_improvement_rate"]) >= 0.8
            and float(row["harm_rate"]) <= 0.2
            for row in selected
        )
        floor_summaries.append(
            {
                "damping_floor": damping_floor,
                "guardrails_passed": guardrails,
                "mean_tracking_rmse_deg_s": float(
                    np.mean([row["tracking_rmse_deg_s"] for row in selected])
                ),
                "maximum_peak_error_deg_s": float(
                    np.max([row["peak_error_deg_s"] for row in selected])
                ),
                "mean_requested_force_tv_n": float(
                    np.mean([row["requested_force_tv_n"] for row in selected])
                ),
            }
        )
    eligible = [row for row in floor_summaries if row["guardrails_passed"]]
    pool = eligible or floor_summaries
    selected = min(pool, key=lambda row: float(row["mean_tracking_rmse_deg_s"]))

    csv_path = destination / "damping_floor_metrics.csv"
    csv_fields = [key for key in rows[0] if key != "evaluation"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)
    plot_path = destination / "damping_floor_probe.png"
    _save_plot(rows, plot_path)
    report = {
        "schema_version": "student_damping_floor_probe_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {"path": str(bank_path), "sha256": sha256_file(bank_path)},
        "student_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "plant_ids": plant_ids,
        "selected_damping_floor": selected["damping_floor"],
        "selected_guardrails_passed": selected["guardrails_passed"],
        "floor_summaries": floor_summaries,
        "rows": rows,
        "artifacts": {"metrics_csv": str(csv_path), "plot": str(plot_path)},
    }
    report_path = destination / "damping_floor_probe.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {
                "selected_damping_floor": selected["damping_floor"],
                "selected_guardrails_passed": selected["guardrails_passed"],
                "floor_summaries": floor_summaries,
            },
            indent=2,
        )
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
