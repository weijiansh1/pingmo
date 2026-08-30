"""Plot fixed-validation learning curves for long-horizon reward-only TD3 runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("mean_episode_cost", "Mean episode cost", "mean_episode_cost"),
    ("mean_tracking_rmse_deg_s", "Tracking RMSE (deg/s)", "mean_tracking_rmse_deg_s"),
    ("maximum_peak_error_deg_s", "Peak error (deg/s)", "maximum_peak_error_deg_s"),
    (
        "mean_requested_force_total_variation_n",
        "Requested-force TV (N)",
        "mean_requested_force_total_variation_n",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _comparison_pid(path: Path, plant_id: str) -> dict[str, float]:
    payload = _load(path)
    if payload["plant"]["plant_id"] != plant_id:
        raise ValueError(f"comparison plant does not match {plant_id}: {path}")
    return payload["summary"]["PID"]


def main() -> None:
    args = _parse_args()
    if args.comparison and len(args.comparison) != len(args.report):
        raise ValueError("provide either zero comparisons or one comparison per report")
    reports = [_load(path.resolve()) for path in args.report]
    comparisons = (
        [path.resolve() for path in args.comparison]
        if args.comparison
        else [None] * len(reports)
    )

    figure, axes = plt.subplots(
        len(reports),
        len(METRICS),
        figsize=(17, max(4.2, 3.8 * len(reports))),
        squeeze=False,
        layout="constrained",
    )
    summaries: list[dict[str, object]] = []
    for row_index, (report, comparison_path) in enumerate(
        zip(reports, comparisons, strict=True)
    ):
        plant_id = str(report["plant_id"])
        points = report.get("learning_curve")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f"report has no multi-point learning curve: {plant_id}")
        steps = np.asarray([point["step"] for point in points], dtype=float)
        pid = (
            _comparison_pid(comparison_path, plant_id)
            if comparison_path is not None
            else None
        )
        metric_summary: dict[str, object] = {}
        for column_index, (curve_key, title, pid_key) in enumerate(METRICS):
            axis = axes[row_index, column_index]
            values = np.asarray([point[curve_key] for point in points], dtype=float)
            axis.plot(
                steps,
                values,
                color="#c82423",
                marker="o",
                markersize=4,
                linewidth=1.8,
                label="Reward-only TD3",
            )
            pid_value = None if pid is None else float(pid[pid_key])
            if pid_value is not None:
                axis.axhline(
                    pid_value,
                    color="#222222",
                    linestyle="--",
                    linewidth=1.4,
                    label="Tuned PID",
                )
            positive = values[values > 0.0]
            if len(positive) and float(np.max(positive) / np.min(positive)) >= 5.0:
                axis.set_yscale("log")
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
            if row_index == len(reports) - 1:
                axis.set_xlabel("Environment steps")
            if column_index == 0:
                axis.set_ylabel(plant_id)
            if row_index == 0 and column_index == 0:
                axis.legend(loc="best")
            best_index = int(np.argmin(values))
            metric_summary[curve_key] = {
                "initial": float(values[0]),
                "final": float(values[-1]),
                "best": float(values[best_index]),
                "best_step": int(steps[best_index]),
                "final_over_pid": (
                    None if pid_value is None else float(values[-1] / pid_value)
                ),
            }
        summaries.append({"plant_id": plant_id, "metrics": metric_summary})

    figure.suptitle("Long-horizon TD3 fixed-validation convergence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "pure_reward_td3_convergence_summary_v1",
                "runs": summaries,
                "plot": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"plot={args.output}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
