"""Compare two frozen-Student reports on the identical unseen-aircraft suite."""

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

from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version")
        != "unseen_aircraft_student_evaluation_v1"
        or payload.get("status") != "complete"
    ):
        raise ValueError(f"incomplete unseen-aircraft report: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _aircraft_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    return {str(row["plant_id"]): row for row in report["aircraft"]}


def _command_by_key(
    report: dict[str, object],
) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(aircraft["plant_id"]), str(command["command_id"])): command
        for aircraft in report["aircraft"]
        for command in aircraft["commands"]
    }


def _assert_fair_baselines(
    baseline: dict[str, object], candidate: dict[str, object]
) -> None:
    baseline_commands = _command_by_key(baseline)
    candidate_commands = _command_by_key(candidate)
    if baseline_commands.keys() != candidate_commands.keys():
        raise ValueError("unseen-aircraft reports use different aircraft-command pairs")
    metric_names = (
        "tracking_rmse_deg_s",
        "tracking_peak_error_deg_s",
        "requested_force_total_variation_n",
        "force_saturation_fraction",
    )
    for key in baseline_commands:
        for controller in ("raw", "PID"):
            for metric in metric_names:
                left = float(baseline_commands[key][controller][metric])
                right = float(candidate_commands[key][controller][metric])
                if not np.isclose(left, right, rtol=1e-10, atol=1e-10):
                    raise ValueError(
                        f"{controller} baseline changed for {key}: {metric}"
                    )


def _save_plot(rows: list[dict[str, object]], path: Path) -> Path:
    labels = [str(row["plant_id"]) for row in rows]
    positions = np.arange(len(rows), dtype=float)
    width = 0.25
    colors = {
        "PID": "#16803c",
        "baseline_student": "#7a5195",
        "candidate_student": "#c82d2d",
    }
    figure, axes = plt.subplots(3, 1, figsize=(15, 12), layout="constrained")
    for offset, name in enumerate(("PID", "baseline_student", "candidate_student")):
        axes[0].bar(
            positions + (offset - 1) * width,
            [row[name]["mean_tracking_rmse_deg_s"] for row in rows],
            width,
            color=colors[name],
            label=name,
        )
        axes[1].bar(
            positions + (offset - 1) * width,
            [row[name]["mean_requested_force_total_variation_n"] for row in rows],
            width,
            color=colors[name],
            label=name,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking RMSE (deg/s, log scale)")
    axes[0].set_title("Unseen-aircraft tracking")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Requested-force TV (N, log scale)")
    axes[1].set_title("Control smoothness")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    baseline_pair = np.asarray(
        [value for row in rows for value in row["baseline_pair_rmse_deg_s"]],
        dtype=float,
    )
    candidate_pair = np.asarray(
        [value for row in rows for value in row["candidate_pair_rmse_deg_s"]],
        dtype=float,
    )
    axes[2].scatter(baseline_pair, candidate_pair, color="#2878b5", alpha=0.75)
    bound = max(float(np.max(baseline_pair)), float(np.max(candidate_pair)), 1e-3)
    axes[2].plot([1e-3, bound], [1e-3, bound], color="black", linestyle="--")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Baseline Student RMSE (deg/s)")
    axes[2].set_ylabel("Coverage Student RMSE (deg/s)")
    axes[2].set_title("Each point is one aircraft-command pair")
    axes[2].grid(alpha=0.25)
    for axis in axes[:2]:
        axis.set_xticks(positions, labels, rotation=22, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main() -> None:
    args = _parse_args()
    baseline_path = args.baseline_report.resolve()
    candidate_path = args.candidate_report.resolve()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    baseline = _read_report(baseline_path)
    candidate = _read_report(candidate_path)
    if baseline["scope"]["target_plant_ids"] != candidate["scope"][
        "target_plant_ids"
    ]:
        raise ValueError("unseen-aircraft target order changed")
    if baseline["scope"]["command_suite"] != candidate["scope"]["command_suite"]:
        raise ValueError("unseen-aircraft command suite changed")
    _assert_fair_baselines(baseline, candidate)

    baseline_aircraft = _aircraft_by_id(baseline)
    candidate_aircraft = _aircraft_by_id(candidate)
    rows: list[dict[str, object]] = []
    for plant_id in baseline["scope"]["target_plant_ids"]:
        before = baseline_aircraft[plant_id]
        after = candidate_aircraft[plant_id]
        baseline_commands = {
            str(row["command_id"]): row for row in before["commands"]
        }
        candidate_commands = {
            str(row["command_id"]): row for row in after["commands"]
        }
        rows.append(
            {
                "plant_id": plant_id,
                "split": after["split"],
                "quality_region": after["quality_region"],
                "PID": after["summary"]["PID"],
                "baseline_student": before["summary"]["Student"],
                "candidate_student": after["summary"]["Student"],
                "baseline_pair_rmse_deg_s": [
                    baseline_commands[command_id]["Student"][
                        "tracking_rmse_deg_s"
                    ]
                    for command_id in baseline_commands
                ],
                "candidate_pair_rmse_deg_s": [
                    candidate_commands[command_id]["Student"][
                        "tracking_rmse_deg_s"
                    ]
                    for command_id in baseline_commands
                ],
            }
        )

    baseline_pair = np.asarray(
        [value for row in rows for value in row["baseline_pair_rmse_deg_s"]]
    )
    candidate_pair = np.asarray(
        [value for row in rows for value in row["candidate_pair_rmse_deg_s"]]
    )
    pid_pair = np.asarray(
        [
            command["PID"]["tracking_rmse_deg_s"]
            for aircraft in candidate["aircraft"]
            for command in aircraft["commands"]
        ]
    )
    baseline_aircraft_rmse = np.asarray(
        [row["baseline_student"]["mean_tracking_rmse_deg_s"] for row in rows]
    )
    candidate_aircraft_rmse = np.asarray(
        [row["candidate_student"]["mean_tracking_rmse_deg_s"] for row in rows]
    )
    plot_path = _save_plot(rows, destination / "comparison.png")
    self_check = {
        "passed": True,
        "same_target_aircraft": True,
        "same_command_suite": True,
        "raw_and_pid_metrics_identical": True,
        "baseline_student_checkpoint_sha256": baseline["student_checkpoint"][
            "sha256"
        ],
        "candidate_student_checkpoint_sha256": candidate["student_checkpoint"][
            "sha256"
        ],
    }
    report = {
        "schema_version": "unseen_aircraft_student_comparison_v1",
        "status": "complete",
        "source": git_source_revision(),
        "baseline_report": {
            "path": str(baseline_path),
            "sha256": sha256_file(baseline_path),
        },
        "candidate_report": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "self_check": self_check,
        "pair_count": len(baseline_pair),
        "aircraft_count": len(rows),
        "overall": {
            "PID_mean_rmse_deg_s": float(np.mean(pid_pair)),
            "baseline_student_mean_rmse_deg_s": float(np.mean(baseline_pair)),
            "candidate_student_mean_rmse_deg_s": float(np.mean(candidate_pair)),
            "candidate_improves_baseline_pair_rate": float(
                np.mean(candidate_pair < baseline_pair)
            ),
            "candidate_beats_or_equals_pid_pair_rate": float(
                np.mean(candidate_pair <= pid_pair)
            ),
            "candidate_improves_baseline_aircraft_rate": float(
                np.mean(candidate_aircraft_rmse < baseline_aircraft_rmse)
            ),
            "mean_candidate_minus_baseline_rmse_deg_s": float(
                np.mean(candidate_pair - baseline_pair)
            ),
        },
        "aircraft": rows,
        "artifacts": {"comparison_plot": str(plot_path)},
    }
    _write_json(destination / "comparison.json", report)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
