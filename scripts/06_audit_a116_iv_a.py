"""Audit the IV-A plant bank against digitized GJB Figure A116."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aircraft.parameters import PChannelParameters
from src.quality.a116 import A116BoundarySet, audit_a116_parameters


def _parameters(row: dict[str, object]) -> PChannelParameters:
    raw = row["parameters"]
    if not isinstance(raw, dict):
        raise ValueError("plant parameters must be a mapping")
    return PChannelParameters(**{name: raw[name] for name in PChannelParameters.__dataclass_fields__})


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="A116 IV-A raw plant audit")
    parser.add_argument("--library", type=Path, default=root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=root / "results/手册G0_A116_IV-A审计.json")
    args = parser.parse_args()
    if args.duration <= 0 or not np.isclose(args.duration / .005, round(args.duration / .005)):
        raise ValueError("duration must be a positive multiple of 0.005 s")

    boundaries = A116BoundarySet.from_csv(root / "data/gjb_a116_boundary.csv")
    rows = [json.loads(line) for line in (args.library / "plants.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit is not None:
        rows = rows[:args.limit]
    steps = int(round(args.duration / .005))
    time = np.arange(steps, dtype=float) * .005
    audited = []
    for index, row in enumerate(rows):
        parameters = _parameters(row)
        audit = audit_a116_parameters(boundaries, parameters, "A_C", time)
        audited.append({"plant_id": row["plant_id"], "split": row["split"], "quality_region": row["quality_region"], **audit})
        if (index + 1) % 100 == 0:
            print(f"audited {index + 1}/{len(rows)}")

    level_counts = Counter(str(row["a116_level"]) for row in audited)
    status_counts = Counter(str(row["a116_status"]) for row in audited)
    reason_counts = Counter(str(row["a116_reason"]) for row in audited if row["a116_reason"] is not None)
    report = {
        "profile": "IV-A",
        "phase_group": "A_C",
        "source": {
            "document": "GJB 2874-97",
            "figure": "A116/A120",
            "a116_pdf_page": 240,
            "a116_print_page": 236,
            "a120_pdf_page": 244,
            "a120_print_page": 240,
            "digitization": "manual_trace_from_scanned_figure_grid_knots",
        },
        "test_condition": {
            "yaw_control": "free",
            "input": "per-aircraft held step",
            "target": "60 deg bank angle at 1.7 Dutch-roll time constants",
            "amplitude_limit": "none; A116 test amplitude is calibrated per aircraft",
            "p_osc_method": "first roll-rate peak P1, first valley P2, second peak P3 after spiral removal",
        },
        "duration_s": args.duration,
        "plant_count": len(audited),
        "level_counts": dict(sorted(level_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "plants": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, (boundary_axis, status_axis) = plt.subplots(2, 1, figsize=(10, 8), layout="constrained")
    phase = np.linspace(-360.0, 0.0, 361)
    for level, style in ((1, "--"), (2, "-")):
        boundary = [boundaries.interpolate("A_C", level, value) for value in phase]
        boundary_axis.plot(phase, boundary, color="black", linestyle=style, linewidth=1.5, label=f"A/C Level {level} 边界")
    valid = [row for row in audited if row["a116_status"] == "assessable"]
    visible_limit = 1.0
    visible = [row for row in valid if row["p_osc_over_p_av"] <= visible_limit]
    clipped_count = len(valid) - len(visible)
    if visible:
        colors = {"1": "#2ca02c", "2": "#1f77b4", "above_level_2": "#d62728", "not_available": "#7f7f7f"}
        for level in sorted({str(row["a116_level"]) for row in visible}):
            rows_at_level = [row for row in visible if str(row["a116_level"]) == level]
            boundary_axis.scatter(
                [row["psi_p_deg"] for row in rows_at_level],
                [row["p_osc_over_p_av"] for row in rows_at_level],
                s=10, alpha=.60, color=colors[level], label=f"严格提取：{level}",
            )
    boundary_axis.set_title("G0：A116 边界与严格 P1-P2-P3 原始机审计")
    boundary_axis.set_ylabel("P_osc / P_av 限值")
    boundary_axis.set_ylim(-0.02, 1.05)
    if clipped_count:
        boundary_axis.text(
            0.99, 0.97, f"另有 {clipped_count} 架严格样本的比值 > 1，图中截断；完整值见 JSON",
            transform=boundary_axis.transAxes, ha="right", va="top", fontsize=9,
            bbox={"facecolor": "white", "alpha": .8, "edgecolor": "none"},
        )
    boundary_axis.grid(alpha=.25)
    boundary_axis.legend(ncol=2)
    labels = list(reason_counts) or ["全部可评估"]
    values = [reason_counts[label] for label in labels] if reason_counts else [0]
    status_axis.bar(labels, values, color="#7f7f7f")
    status_axis.set_title("不可评估原因统计（未满足条件者不赋 A116 等级）")
    status_axis.set_xlabel("原因")
    status_axis.set_ylabel("飞机数量")
    status_axis.tick_params(axis="x", rotation=15)
    status_axis.grid(axis="y", alpha=.25)
    plot_name = "手册G0_A116_IV-A原始飞机审计_烟测.png" if args.limit is not None else "手册G0_A116_IV-A原始飞机审计.png"
    plot = root / "img" / plot_name
    plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot, dpi=180)
    plt.close(figure)
    print(json.dumps({**{key: value for key, value in report.items() if key != "plants"}, "plot": str(plot)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
