"""Plot a compact quality overview for the frozen specialist Teacher Bank."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    ROOT
    / "results/pure_reward_teacher_bank_coverage_v3/merged_bank/teacher_bank.json"
)
DEFAULT_EVALUATION_ROOT = (
    ROOT
    / "results/pure_reward_teacher_bank_coverage_v3"
    / "student_driven_dense_balanced_holdout/final/evaluation"
)
DEFAULT_OUTPUT = ROOT / "docs" / "figures" / "teacher_overview.png"
DEFAULT_CJK_FONT = Path.home() / ".fonts/NotoSansSC.ttf"

REGION_ORDER = {"level_1": 0, "level_2": 1, "level_3": 2}
REGION_LABELS = {
    "level_1": "Level 1",
    "level_2": "Level 2",
    "level_3": "Level 3",
}
REGION_COLORS = {
    "level_1": "#25885a",
    "level_2": "#d18b19",
    "level_3": "#c83e3e",
}
SPLIT_MARKERS = {"train_core": "o", "train_boundary": "s"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument(
        "--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT
    )
    parser.add_argument("--episode-duration-s", type=float, default=30.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _read_bank(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"Teacher Bank is not complete: {path}")
    teachers = payload.get("teachers")
    rejected = payload.get("rejected_teachers")
    if not isinstance(teachers, list) or not isinstance(rejected, list):
        raise ValueError(f"Teacher Bank entries are invalid: {path}")
    return payload


def _observed(entry: dict[str, object], name: str) -> float:
    quality_gate = entry.get("quality_gate")
    if not isinstance(quality_gate, dict):
        raise ValueError(f"missing quality gate for {entry.get('plant_id')}")
    observed = quality_gate.get("observed")
    if not isinstance(observed, dict) or name not in observed:
        raise ValueError(f"missing {name} for {entry.get('plant_id')}")
    return float(observed[name])


def _threshold(entry: dict[str, object], name: str) -> float:
    quality_gate = entry.get("quality_gate")
    if not isinstance(quality_gate, dict):
        raise ValueError(f"missing quality gate for {entry.get('plant_id')}")
    thresholds = quality_gate.get("thresholds")
    if not isinstance(thresholds, dict) or name not in thresholds:
        raise ValueError(f"missing {name} for {entry.get('plant_id')}")
    return float(thresholds[name])


def _worst_command_tv_rates(
    evaluation_root: Path,
    teachers: list[dict[str, object]],
    episode_duration_s: float,
) -> dict[str, float]:
    if episode_duration_s <= 0:
        raise ValueError("episode duration must be positive")
    rates: dict[str, float] = {}
    for entry in teachers:
        plant_id = str(entry["plant_id"])
        path = evaluation_root / plant_id / "teacher/evaluation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"missing evaluation rows: {path}")
        command_tv = []
        for row in rows:
            teacher = row.get("teacher") if isinstance(row, dict) else None
            if not isinstance(teacher, dict):
                raise ValueError(f"invalid Teacher evaluation row: {path}")
            command_tv.append(
                float(teacher["requested_force_total_variation_n"])
                / episode_duration_s
            )
        rates[plant_id] = max(command_tv)
    return rates


def _plot_summary(
    axis: plt.Axes,
    accepted_count: int,
    rejected_count: int,
) -> None:
    values = [accepted_count, rejected_count]
    colors = ["#25885a", "#c94b4b"]
    axis.pie(
        values,
        labels=[f"Bank 录用  {accepted_count}", f"淘汰  {rejected_count}"],
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct="%1.1f%%",
        pctdistance=0.72,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    axis.text(
        0,
        0.06,
        str(accepted_count + rejected_count),
        ha="center",
        va="center",
        fontsize=25,
        fontweight="bold",
        color="#202124",
    )
    axis.text(0, -0.19, "共训练", ha="center", va="center", fontsize=11)
    axis.set_title("Teacher Bank 筛选结果", fontsize=15, fontweight="bold", pad=12)


def _plot_rejection_reasons(
    axis: plt.Axes, rejected: list[dict[str, object]]
) -> None:
    check_names = (
        ("mean_tracking_rmse", "平均 RMSE"),
        ("peak_tracking_error", "最大峰值"),
        ("requested_force_tv_rate", "动作变化率"),
    )
    counts: Counter[str] = Counter()
    for entry in rejected:
        quality_gate = entry.get("quality_gate")
        checks = quality_gate.get("checks", {}) if isinstance(quality_gate, dict) else {}
        if not isinstance(checks, dict):
            continue
        for key, _ in check_names:
            if checks.get(key) is False:
                counts[key] += 1

    labels = [label for _, label in check_names]
    values = [counts[key] for key, _ in check_names]
    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=["#56738f", "#c94b4b", "#d18b19"])
    axis.set_yticks(positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) * 1.22)
    axis.set_xlabel("失败 Teacher 数")
    axis.set_title("淘汰原因（可重复计数）", fontsize=15, fontweight="bold", pad=12)
    axis.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_width() + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=12,
            fontweight="bold",
        )


def _plot_composition(axis: plt.Axes, teachers: list[dict[str, object]]) -> None:
    regions = tuple(REGION_ORDER)
    core = [
        sum(
            entry.get("quality_region") == region
            and entry.get("split") == "train_core"
            for entry in teachers
        )
        for region in regions
    ]
    boundary = [
        sum(
            entry.get("quality_region") == region
            and entry.get("split") == "train_boundary"
            for entry in teachers
        )
        for region in regions
    ]
    positions = np.arange(len(regions))
    axis.bar(positions, core, color="#4477aa", label=f"Core  {sum(core)}")
    axis.bar(
        positions,
        boundary,
        bottom=core,
        color="#cc6677",
        label=f"Boundary  {sum(boundary)}",
    )
    axis.set_xticks(positions, [REGION_LABELS[region] for region in regions])
    axis.set_ylabel("Bank 录用 Teacher 数")
    axis.set_title("录用 Teacher 构成", fontsize=15, fontweight="bold", pad=12)
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncols=2, loc="upper left")
    for index, total in enumerate(np.asarray(core) + np.asarray(boundary)):
        axis.text(index, total + 0.25, str(int(total)), ha="center", fontweight="bold")


def _plot_metric(
    axis: plt.Axes,
    teachers: list[dict[str, object]],
    observed_name: str,
    threshold_name: str,
    title: str,
    ylabel: str,
    show_labels: bool,
    values_by_plant: dict[str, float] | None = None,
) -> None:
    values = np.asarray(
        [
            _observed(entry, observed_name)
            if values_by_plant is None
            else values_by_plant[str(entry["plant_id"])]
            for entry in teachers
        ],
        dtype=float,
    )
    thresholds = np.asarray(
        [_threshold(entry, threshold_name) for entry in teachers], dtype=float
    )
    if not np.allclose(thresholds, thresholds[0]):
        raise ValueError(f"inconsistent thresholds for {observed_name}")
    threshold = float(thresholds[0])
    positions = np.arange(len(teachers), dtype=float)

    axis.axhspan(0, threshold, color="#25885a", alpha=0.055)
    axis.axhline(
        threshold,
        color="#b52b35",
        linewidth=1.8,
        linestyle="--",
        label=f"门限 {threshold:g}",
    )
    for region in REGION_ORDER:
        for split, marker in SPLIT_MARKERS.items():
            indices = [
                index
                for index, entry in enumerate(teachers)
                if entry.get("quality_region") == region
                and entry.get("split") == split
            ]
            if not indices:
                continue
            axis.scatter(
                positions[indices],
                values[indices],
                s=54,
                marker=marker,
                color=REGION_COLORS[region],
                edgecolors="white",
                linewidths=0.7,
                zorder=3,
            )

    axis.set_xlim(-0.7, len(teachers) - 0.3)
    axis.set_ylim(bottom=0)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=14, fontweight="bold", loc="left")
    axis.grid(axis="y", alpha=0.22)
    axis.text(
        0.995,
        0.88,
        f"均值 {values.mean():.3f}    最坏 {values.max():.3f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "#d5d9de", "alpha": 0.9},
    )
    if show_labels:
        labels = [
            ("C" if entry.get("split") == "train_core" else "B")
            + "-"
            + str(entry["plant_id"]).rsplit("-", 1)[-1]
            for entry in teachers
        ]
        axis.set_xticks(positions, labels, rotation=90, fontsize=8)
        axis.set_xlabel("Bank 录用 Teacher（C = Core，B = Boundary；按 Level 和编号排列）")
    else:
        axis.tick_params(axis="x", labelbottom=False)


def _save_plot(
    bank: dict[str, object],
    evaluation_root: Path,
    episode_duration_s: float,
    output: Path,
) -> None:
    teachers_raw = bank["teachers"]
    rejected_raw = bank["rejected_teachers"]
    assert isinstance(teachers_raw, list) and isinstance(rejected_raw, list)
    teachers = [entry for entry in teachers_raw if isinstance(entry, dict)]
    rejected = [entry for entry in rejected_raw if isinstance(entry, dict)]
    teachers.sort(
        key=lambda entry: (
            REGION_ORDER[str(entry["quality_region"])],
            0 if entry["split"] == "train_core" else 1,
            str(entry["plant_id"]),
        )
    )
    if not teachers:
        raise ValueError("Teacher Bank has no accepted teachers")
    worst_tv_rates = _worst_command_tv_rates(
        evaluation_root, teachers, episode_duration_s
    )
    tv_threshold = _threshold(
        teachers[0], "maximum_mean_requested_force_total_variation_rate_n_s"
    )
    worst_command_failures = sum(
        value > tv_threshold for value in worst_tv_rates.values()
    )

    font_family = "DejaVu Sans"
    if DEFAULT_CJK_FONT.is_file():
        font_manager.fontManager.addfont(str(DEFAULT_CJK_FONT))
        font_family = font_manager.FontProperties(
            fname=str(DEFAULT_CJK_FONT)
        ).get_name()
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfd",
        }
    )
    figure = plt.figure(figsize=(19, 16.5), layout="constrained")
    grid = figure.add_gridspec(
        6, 3, height_ratios=[1.15, 0.08, 1, 1, 1, 1.15]
    )
    summary_axis = figure.add_subplot(grid[0, 0])
    rejection_axis = figure.add_subplot(grid[0, 1])
    composition_axis = figure.add_subplot(grid[0, 2])
    rmse_axis = figure.add_subplot(grid[2, :])
    peak_axis = figure.add_subplot(grid[3, :], sharex=rmse_axis)
    mean_tv_axis = figure.add_subplot(grid[4, :], sharex=rmse_axis)
    worst_tv_axis = figure.add_subplot(grid[5, :], sharex=rmse_axis)

    _plot_summary(summary_axis, len(teachers), len(rejected))
    _plot_rejection_reasons(rejection_axis, rejected)
    _plot_composition(composition_axis, teachers)
    _plot_metric(
        rmse_axis,
        teachers,
        "mean_tracking_rmse_deg_s",
        "maximum_mean_tracking_rmse_deg_s",
        "32 个 Bank 录用 Teacher：平均跟踪误差",
        "RMSE (deg/s)",
        False,
    )
    _plot_metric(
        peak_axis,
        teachers,
        "maximum_peak_error_deg_s",
        "maximum_peak_error_deg_s",
        "32 个 Bank 录用 Teacher：最大峰值误差",
        "峰值误差 (deg/s)",
        False,
    )
    _plot_metric(
        mean_tv_axis,
        teachers,
        "mean_requested_force_total_variation_rate_n_s",
        "maximum_mean_requested_force_total_variation_rate_n_s",
        "Bank 原门禁：六条指令平均控制动作变化率",
        "TV rate (N/s)",
        False,
    )
    _plot_metric(
        worst_tv_axis,
        teachers,
        "mean_requested_force_total_variation_rate_n_s",
        "maximum_mean_requested_force_total_variation_rate_n_s",
        f"逐指令复核：最坏单条指令动作变化率（{worst_command_failures}/32 超过原平均门限）",
        "Worst-command TV rate (N/s)",
        True,
        values_by_plant=worst_tv_rates,
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=REGION_COLORS[region],
            markeredgecolor="white",
            markersize=9,
            label=REGION_LABELS[region],
        )
        for region in REGION_ORDER
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color="#555555",
                markersize=8,
                label="Core",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                color="#555555",
                markersize=8,
                label="Boundary",
            ),
            Line2D(
                [0],
                [0],
                color="#b52b35",
                linestyle="--",
                linewidth=1.8,
                label="质量门限",
            ),
        ]
    )
    rmse_axis.legend(
        handles=legend_handles,
        ncols=6,
        frameon=False,
        loc="upper left",
        fontsize=10,
    )

    figure.suptitle(
        "P 通道专用 Teacher Bank 质量总览\n"
        "官方平均门禁与最坏单指令 TV 复核",
        fontsize=21,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=210, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    bank_path = args.bank.resolve()
    evaluation_root = args.evaluation_root.resolve()
    output_path = args.output.resolve()
    _save_plot(
        _read_bank(bank_path),
        evaluation_root,
        args.episode_duration_s,
        output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
