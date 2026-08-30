"""Summarize a complete RL Teacher Bank and its PID comparisons."""

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

from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-bank",
        type=Path,
        default=ROOT
        / "results/teacher_student_pipeline/01_teachers/teacher_bank.json",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "aircraft_count": len(rows),
        "mean_raw_tracking_rmse_deg_s": float(
            np.mean([row["raw_rmse_deg_s"] for row in rows])
        ),
        "mean_pid_tracking_rmse_deg_s": float(
            np.mean([row["pid_rmse_deg_s"] for row in rows])
        ),
        "mean_rl_teacher_tracking_rmse_deg_s": float(
            np.mean([row["rl_teacher_rmse_deg_s"] for row in rows])
        ),
        "maximum_rl_teacher_peak_error_deg_s": float(
            np.max([row["rl_teacher_peak_error_deg_s"] for row in rows])
        ),
        "mean_pid_requested_force_total_variation_n": float(
            np.mean([row["pid_force_tv_n"] for row in rows])
        ),
        "mean_rl_teacher_requested_force_total_variation_n": float(
            np.mean([row["rl_teacher_force_tv_n"] for row in rows])
        ),
        "rl_teacher_better_rmse_than_pid_count": sum(
            row["rl_teacher_rmse_deg_s"] <= row["pid_rmse_deg_s"]
            for row in rows
        ),
    }


def _save_plot(rows: list[dict[str, object]], path: Path) -> None:
    labels = [str(row["plant_id"]) for row in rows]
    positions = np.arange(len(rows), dtype=float)
    width = 0.26
    figure, axes = plt.subplots(3, 1, figsize=(13.0, 12.0), constrained_layout=True)

    axes[0].bar(
        positions - width,
        [row["raw_rmse_deg_s"] for row in rows],
        width,
        label="Raw",
        color="#64748b",
    )
    axes[0].bar(
        positions,
        [row["pid_rmse_deg_s"] for row in rows],
        width,
        label="PID",
        color="#16a34a",
    )
    axes[0].bar(
        positions + width,
        [row["rl_teacher_rmse_deg_s"] for row in rows],
        width,
        label="RL Teacher",
        color="#dc2626",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking RMSE (deg/s, log scale)")
    axes[0].set_title("Raw vs PID vs RL Teacher")
    axes[0].legend(ncol=3)

    axes[1].bar(
        positions - width / 2,
        [row["pid_rmse_deg_s"] for row in rows],
        width,
        label="PID",
        color="#16a34a",
    )
    axes[1].bar(
        positions + width / 2,
        [row["rl_teacher_rmse_deg_s"] for row in rows],
        width,
        label="RL Teacher",
        color="#dc2626",
    )
    axes[1].set_ylabel("Tracking RMSE (deg/s)")
    axes[1].set_title("Closed-loop tracking detail")
    axes[1].legend(ncol=2)

    axes[2].bar(
        positions - width / 2,
        [row["pid_force_tv_n"] for row in rows],
        width,
        label="PID",
        color="#16a34a",
    )
    axes[2].bar(
        positions + width / 2,
        [row["rl_teacher_force_tv_n"] for row in rows],
        width,
        label="RL Teacher",
        color="#dc2626",
    )
    axes[2].axhline(50.0, color="#111827", linestyle="--", label="Teacher gate")
    axes[2].set_ylabel("Mean requested-force TV (N)")
    axes[2].set_title("Control smoothness")
    axes[2].legend(ncol=3)

    for axis in axes:
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    bank_path = args.teacher_bank.resolve()
    output = (args.output or bank_path.parent / "summary").resolve()
    output.mkdir(parents=True, exist_ok=True)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "specialist_teacher_bank_v1":
        raise ValueError("unsupported Teacher Bank schema")
    if bank.get("status") != "complete":
        raise ValueError("Teacher Bank must be complete before summarization")
    teachers = bank.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("Teacher Bank has no teachers")

    rows: list[dict[str, object]] = []
    for entry in teachers:
        if entry.get("status") != "complete":
            raise ValueError("Teacher Bank contains a rejected Teacher")
        run_dir = bank_path.parent / str(entry["run_dir"])
        report_path = bank_path.parent / str(entry["report"])
        actor_path = bank_path.parent / str(entry["actor_checkpoint"])
        comparison_path = run_dir / "comparison_vs_pid/comparison.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        if not report.get("accepted_for_distillation"):
            raise ValueError(f"Teacher report is not accepted: {report_path}")
        if comparison.get("plant", {}).get("plant_id") != entry.get("plant_id"):
            raise ValueError("Teacher comparison and Bank plant IDs differ")
        summaries = comparison["summary"]
        row = {
            "plant_id": entry["plant_id"],
            "quality_region": entry["quality_region"],
            "seed": entry["seed"],
            "actor_parameters": report["parameter_counts"]["actor"],
            "online_actor_updates": report["online_actor_updates"],
            "actor_checkpoint_sha256": sha256_file(actor_path),
            "raw_rmse_deg_s": summaries["raw"]["mean_tracking_rmse_deg_s"],
            "pid_rmse_deg_s": summaries["PID"]["mean_tracking_rmse_deg_s"],
            "rl_teacher_rmse_deg_s": summaries["RL Teacher"][
                "mean_tracking_rmse_deg_s"
            ],
            "rl_teacher_peak_error_deg_s": summaries["RL Teacher"][
                "maximum_peak_error_deg_s"
            ],
            "pid_force_tv_n": summaries["PID"][
                "mean_requested_force_total_variation_n"
            ],
            "rl_teacher_force_tv_n": summaries["RL Teacher"][
                "mean_requested_force_total_variation_n"
            ],
            "rl_minus_pid_rmse_deg_s": (
                summaries["RL Teacher"]["mean_tracking_rmse_deg_s"]
                - summaries["PID"]["mean_tracking_rmse_deg_s"]
            ),
            "comparison": str(comparison_path),
            "comparison_plot": str(
                run_dir / "comparison_vs_pid/controller_comparison.png"
            ),
        }
        rows.append(row)

    csv_path = output / "teacher_bank_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                key
                for key in rows[0]
                if key not in {"comparison", "comparison_plot"}
            ],
        )
        writer.writeheader()
        writer.writerows(
            {
                key: value
                for key, value in row.items()
                if key not in {"comparison", "comparison_plot"}
            }
            for row in rows
        )
    plot_path = output / "teacher_bank_summary.png"
    _save_plot(rows, plot_path)
    payload = {
        "schema_version": "teacher_bank_summary_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
            "source": bank.get("source"),
        },
        "aggregate": _aggregate(rows),
        "rows": rows,
        "artifacts": {
            "metrics_csv": str(csv_path),
            "summary_plot": str(plot_path),
        },
    }
    report_path = output / "teacher_bank_summary.json"
    _write_json(report_path, payload)
    print(json.dumps(payload["aggregate"], indent=2))
    print(f"report={report_path}")
    print(f"plot={plot_path}")


if __name__ == "__main__":
    main()
