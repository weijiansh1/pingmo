"""Plot the separate roll-periphery integration run; never train or refit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "roll_autopilot" / "20261019"
ARMS = ("frozen", "full_b30")
LABELS = {"frozen": "冻结 W0", "full_b30": "既有 DHC-6 微调权重"}
COLORS = {"frozen": "#64748b", "full_b30": "#0f766e"}
COMMANDS = ("bank_step_20", "bank_reversal_15", "bank_sine_10_008Hz")
TITLES = ("20° 坡度角阶跃", "±15° 反转并回中", "10° / 0.08 Hz 正弦")


def main() -> None:
    matplotlib.rcParams["axes.unicode_minus"] = False
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if candidate in {f.name for f in font_manager.fontManager.ttflist}:
            matplotlib.rcParams["font.family"] = candidate
            break
    rows = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    states = sorted({r["condition"] for r in rows})
    data = {(r["arm"], r["condition"], r["command_id"]): r for r in rows}
    # A fixed representative condition, not the best-performing result.
    representative = "V080_H2500"
    fig, axes = plt.subplots(3, 3, figsize=(15.4, 9.6), sharex=True)
    for column, command in enumerate(COMMANDS):
        for arm in ARMS:
            with np.load(RUN / "trajectories" / f"{arm}__{representative}__{command}.npz") as trace:
                t = trace["time_s"]
                color = COLORS[arm]
                axes[0, column].plot(t, np.degrees(trace["bank_rad"]), color=color,
                                     lw=1.7, label=LABELS[arm])
                axes[1, column].plot(t, np.degrees(trace["p_rad_s"]), color=color, lw=1.5)
                axes[1, column].plot(t, np.degrees(trace["p_sp_rad_s"]), color=color, lw=1,
                                     ls="--", alpha=0.75)
                axes[2, column].plot(t, np.degrees(trace["aileron_applied_rad"]), color=color, lw=1.2)
                if arm == ARMS[0]:
                    axes[0, column].step(t, np.degrees(trace["command_rad"]), where="pre",
                                         color="#b91c1c", ls="--", lw=1.2, label="坡度角指令")
        axes[0, column].set_title(TITLES[column], fontsize=12)
        axes[2, column].set_xlabel("时间 (s)")
        axes[2, column].axhline(10, color="#b91c1c", ls=":", lw=0.9)
        axes[2, column].axhline(-10, color="#b91c1c", ls=":", lw=0.9)
        for row in range(3):
            axes[row, column].grid(alpha=0.24)
            axes[row, column].set_xlim(0, 20)
    axes[0, 0].set_ylabel("坡度角 φ (deg)")
    axes[1, 0].set_ylabel("机体滚转率 p (deg/s)\n实线响应 / 虚线外环给定")
    axes[2, 0].set_ylabel("实际副翼偏角 (deg)")
    axes[0, 0].legend(fontsize=8.3, frameon=False, loc="best")
    fig.suptitle(f"常规滚转外围 + 品模中心控制器：固定留出状态 {representative}", fontsize=15, y=0.98)
    fig.text(0.5, 0.012, "同一坡度角外环 τ=2 s；副翼 ±10° / 40°/s；两组既有权重均冻结。"
             "对象为独立辨识 ARX，φ 由 p 积分（pitch=q=r=0），仅为降阶集成试验。",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, 0.035, 1, 0.95])
    trajectory_path = HERE / "roll_periphery_trajectories.png"
    fig.savefig(trajectory_path, dpi=160, facecolor="white")
    plt.close(fig)

    matrix = np.full((len(states), 6), np.nan)
    for i, state in enumerate(states):
        for j, arm in enumerate(ARMS):
            for k, command in enumerate(COMMANDS):
                row = data[arm, state, command]
                if row["finite"]:
                    matrix[i, j * 3 + k] = row["bank_rmse_deg"]
    fig, ax = plt.subplots(figsize=(11.2, 7.8))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=np.nanmax(matrix), cmap="YlOrRd")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f"{value:.2f}" if np.isfinite(value) else "失败", ha="center", va="center",
                fontsize=10, color="white" if value > np.nanmax(matrix) * 0.67 else "#1e293b")
    ax.set_xticks(range(6), [f"{LABELS[arm]}\n{label}" for arm in ARMS
                            for label in ("阶跃", "反转", "正弦")], fontsize=9)
    ax.set_yticks(range(len(states)), [state + (" ★" if data[ARMS[0], state, COMMANDS[0]]["group"] == "heldout" else "")
                                     for state in states], fontsize=10)
    ax.axvline(2.5, color="white", lw=4)
    fig.colorbar(im, ax=ax, label="坡度角跟踪 RMSE (deg)")
    ax.set_title("全部 12 状态 × 3 指令 × 2 组既有权重（72 格，★跨状态留出）", fontsize=13, pad=16)
    fig.text(0.5, 0.018, "新坡度角协议；不与历史角速度 RMSE (deg/s) 混用。所有工况保留，未根据结果调整外环参数。",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    heatmap_path = HERE / "roll_periphery_heatmap.png"
    fig.savefig(heatmap_path, dpi=160, facecolor="white")
    plt.close(fig)
    audit = {"run": str(RUN), "n_cases": len(rows), "n_finite": sum(r["finite"] for r in rows),
             "representative_condition": representative, "selection": "fixed V080_H2500, not selected by score",
             "rate_parity": manifest["rate_parity"],
             "data_sha256": {name: hashlib.sha256((RUN / name).read_bytes()).hexdigest()
                              for name in ("manifest.json", "metrics.json")},
             "figure_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (trajectory_path, heatmap_path)}}
    (RUN / "figures_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
