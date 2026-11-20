"""Teacher results for §2.3: training curves, closed-loop rollouts, bank overview.

All inputs are existing, frozen artifacts:
  * results/pure_reward_teacher_bank_coverage_v3/merged_bank/teacher_bank.json
  * .../new_candidates_48/runs/<plant>/learning_curve.json
  * .../merged_bank/actors/<plant>.pt   (the exact 32 bank actors)
  * .../new_candidates_48/runs/<plant>/report.json (plant + config)

No training and no changes to the runs: the rollouts are deterministic
evaluation rollouts of the frozen bank actors through the existing env.

Writes PNG/SVG into dhc6_blog_assets/ and data/audit into
dhc6_blog_assets/evidence/teacher_figures/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
WS = HERE.parent
REPO = WS / "pingmo"
sys.path.insert(0, str(REPO))

from src.aircraft.sampler import PlantRecord  # noqa: E402
from src.context.aircraft_parameters import PChannelParameters  # noqa: E402
from src.envs.roll_rate_commands import specialist_evaluation_commands  # noqa: E402
from src.teacher.specialist.trainer import (  # noqa: E402
    SpecialistActorPolicy,
    SpecialistTrainingConfig,
    build_specialist_env,
    rollout_policy,
    tracking_metrics,
)
from src.teacher.td3.actor import DeterministicActor  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False
for _c in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if _c in {f.name for f in font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _c
        break

RES = REPO / "results/pure_reward_teacher_bank_coverage_v3"
BANK = RES / "merged_bank/teacher_bank.json"
RUNS = RES / "new_candidates_48/runs"
ACTORS = RES / "merged_bank/actors"
EVID = HERE / "evidence/teacher_figures"
EVID.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, name: str) -> None:
    fig.savefig(HERE / name, dpi=140, bbox_inches="tight", facecolor="white")
    fig.savefig(HERE / name.replace(".png", ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


bank = json.loads(BANK.read_text(encoding="utf-8"))
accepted = bank["teachers"]
rejected = bank["rejected_teachers"]
accepted_local = [t for t in accepted if (RUNS / t["plant_id"]).is_dir() and (ACTORS / f"{t['plant_id']}.pt").exists()]
rejected_local = [t for t in rejected if (RUNS / t["plant_id"]).is_dir()]
LEVELS = ("level_1", "level_2", "level_3")
LEVEL_COLOR = {"level_1": "#1d4ed8", "level_2": "#0f766e", "level_3": "#b45309"}
LEVEL_LABEL = {"level_1": "L1", "level_2": "L2", "level_3": "L3"}

# ------------------------------------------------------------------ A: training curves
curves = []
for t in accepted_local:
    path = RUNS / t["plant_id"] / "learning_curve.json"
    lc = json.loads(path.read_text(encoding="utf-8"))
    curves.append({"plant_id": t["plant_id"], "level": t["quality_region"],
                   "selected": t["selected_validation_step"],
                   "total": max(p["step"] for p in lc["points"]),
                   "points": lc["points"], "path": path})

steps = sorted({int(p["step"]) for c in curves for p in c["points"]})
rmse_at = defaultdict(list)
tv_at = defaultdict(list)
cost_at = defaultdict(list)
for step in steps:
    for c in curves:
        for p in c["points"]:
            if int(p["step"]) == step:
                rmse_at[step].append(float(p["mean_tracking_rmse_deg_s"]))
                tv_at[step].append(float(p["mean_requested_force_total_variation_n"]))
                cost_at[step].append(float(p["mean_episode_cost"]))
                break

fig, axes = plt.subplots(1, 3, figsize=(16.4, 5.4))
ax = axes[0]
for c in curves:
    xs = [p["step"] / 1000 for p in c["points"]]
    ys = [p["mean_tracking_rmse_deg_s"] for p in c["points"]]
    ax.plot(xs, ys, color="#94a3b8", lw=0.7, alpha=0.35)
med = [np.median(rmse_at[s]) for s in steps]
q1 = [np.percentile(rmse_at[s], 25) for s in steps]
q3 = [np.percentile(rmse_at[s], 75) for s in steps]
xs = [s / 1000 for s in steps]
ax.plot(xs, med, color="#1d4ed8", lw=2.2, marker="o", ms=4, label="中位数")
ax.fill_between(xs, q1, q3, color="#1d4ed8", alpha=0.16, label="25–75%")
ax.set_xlabel("训练步数 (k)", fontsize=15)
ax.set_ylabel("固定验证 平均跟踪 RMSE (deg/s)", fontsize=15)
ax.set_title(f"跟踪误差随训练下降（n={len(curves)} 个通过教师）", fontsize=18, loc="left")
ax.grid(alpha=0.25)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12, loc="upper right")
ax.text(0.02, 0.03, "细灰线 = 单个教师；粗线 = 中位数", transform=ax.transAxes, fontsize=8.4, color="#64748b")

ax = axes[1]
med = [np.median(tv_at[s]) for s in steps]
q1 = [np.percentile(tv_at[s], 25) for s in steps]
q3 = [np.percentile(tv_at[s], 75) for s in steps]
for c in curves:
    ax.plot([p["step"] / 1000 for p in c["points"]],
            [p["mean_requested_force_total_variation_n"] for p in c["points"]],
            color="#94a3b8", lw=0.7, alpha=0.35)
ax.plot(xs, med, color="#b45309", lw=2.2, marker="o", ms=4, label="中位数")
ax.fill_between(xs, q1, q3, color="#b45309", alpha=0.16, label="25–75%")
ax.set_xlabel("训练步数 (k)")
ax.set_ylabel("固定验证 请求力总变差 TV (N)", fontsize=15)
ax.set_title("控制活动（总变差，N，对数轴）", fontsize=18, loc="left")
ax.set_yscale("log")
tv_all = [v for s_ in steps for v in tv_at[s_]]
tv_pos = [v for v in tv_all if v > 0]
tv_min, tv_max = min(tv_pos), max(tv_pos)
ax.set_ylim(tv_min * 0.8, tv_max * 1.3)
ax.grid(alpha=0.25)
ax.tick_params(labelsize=12)
ax.legend(fontsize=12, loc="upper right")

ax = axes[2]
sel = Counter(int(c["selected"]) for c in curves)
labels = sorted(sel)
bars = ax.bar([f"{s // 1000}k" for s in labels], [sel[s] for s in labels],
              color=["#1d4ed8" if s < 80000 else "#93c5fd" for s in labels])
for bar, s in zip(bars, labels):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12, str(sel[s]),
            ha="center", fontsize=9.5)
ax.set_xlabel("选中 checkpoint 的步数", fontsize=15)
ax.set_ylabel("教师数", fontsize=15)
ax.set_title("选中点不一定在末步", fontsize=18, loc="left")
ax.set_ylim(0, max(sel.values()) + 1.4)
ax.grid(axis="y", alpha=0.25)
ax.text(0.02, 0.93, f"末步选中 {sel.get(80000, 0)}/{len(curves)}；其余选在更早的 checkpoint",
        transform=ax.transAxes, fontsize=8.6, color="#334155", va="top")
fig.suptitle("A · 教师训练过程：训练中的固定验证（30 s episode，确定性；不是探索回报或盲测）",
             fontsize=20.0, fontweight="bold", x=0.01, ha="left", y=1.03)
fig.tight_layout()
save(fig, "teacher_training_curves.png")

# training data
with (EVID / "teacher_training_points.csv").open("w", encoding="utf-8", newline="") as fh:
    fh.write("plant_id,level,step,mean_tracking_rmse_deg_s,mean_requested_force_total_variation_n,"
             "mean_episode_cost,selected_validation_step,is_selected\n")
    for c in curves:
        for p in c["points"]:
            fh.write(f"{c['plant_id']},{c['level']},{p['step']},{p['mean_tracking_rmse_deg_s']},"
                     f"{p['mean_requested_force_total_variation_n']},{p['mean_episode_cost']},"
                     f"{c['selected']},{int(int(p['step']) == int(c['selected']))}\n")

# ------------------------------------------------------------------ B: representatives
by_level = defaultdict(list)
for t in accepted_local:
    by_level[t["quality_region"]].append(t)
reps = []
for level in LEVELS:
    pool = sorted(by_level[level], key=lambda t: (t["quality_gate"]["observed"]["mean_tracking_rmse_deg_s"], t["plant_id"]))
    vals = [t["quality_gate"]["observed"]["mean_tracking_rmse_deg_s"] for t in pool]
    median = float(np.median(vals))
    best = min(pool, key=lambda t: (abs(t["quality_gate"]["observed"]["mean_tracking_rmse_deg_s"] - median),
                                    t["plant_id"]))
    reps.append({"level": level, "plant_id": best["plant_id"],
                 "selected_step": int(best["selected_validation_step"]),
                 "validation_rmse_deg_s": float(best["quality_gate"]["observed"]["mean_tracking_rmse_deg_s"]),
                 "median_pool_rmse_deg_s": median, "pool_size": len(pool),
                 "actor_sha256": best["actor_checkpoint_sha256"]})

commands = [("eval-step-pos-15deg-s", "阶跃 +15 deg/s", 6.0),
            ("eval-multisine", "多正弦（0.30/0.85/1.40 Hz）", 10.0)]
traces = {}
rep_metrics = {}
for rep in reps:
    run = RUNS / rep["plant_id"]
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    cfg = report["config"]
    config = SpecialistTrainingConfig(**cfg)
    p = report["plant"]
    record = PlantRecord(plant_id=p["plant_id"], split=p["split"], quality_region=p["quality_region"],
                         aircraft_class=p["aircraft_class"], flight_phase=p["flight_phase"],
                         parameters=PChannelParameters(**p["parameters"]))
    payload = torch.load(ACTORS / f"{rep['plant_id']}.pt", map_location="cpu", weights_only=False)
    actor = DeterministicActor(int(report["actor_observation_dim"]), 1,
                               width=int(cfg["network_width"]), residual_blocks=int(cfg["residual_blocks"]),
                               enforce_odd_symmetry=bool(cfg["enforce_odd_policy"]))
    actor.load_state_dict(payload["actor"])
    actor.eval()
    policy = SpecialistActorPolicy(actor, "cpu")
    profiles = {x.command_id: x for x in specialist_evaluation_commands(config.episode_duration_s)}
    rep_metrics[rep["plant_id"]] = {}
    for command_id, _, _ in commands:
        trace = rollout_policy(policy, build_specialist_env(record, config, (profiles[command_id],)),
                               seed=config.seed)
        metrics = tracking_metrics(trace, config.force_limit_n)
        rep_metrics[rep["plant_id"]][command_id] = {
            "tracking_rmse_deg_s": metrics["tracking_rmse_deg_s"],
            "tracking_peak_error_deg_s": metrics["tracking_peak_error_deg_s"],
            "force_rms_n": metrics["force_rms_n"],
            "requested_force_total_variation_n": metrics["requested_force_total_variation_n"],
            "force_saturation_fraction": metrics["force_saturation_fraction"],
        }
        traces[(rep["plant_id"], command_id)] = trace
        with (EVID / f"trajectory_{rep['plant_id']}_{command_id}.csv").open("w", encoding="utf-8", newline="") as fh:
            fh.write("time_s,p_command_deg_s,p_reference_deg_s,p_deg_s,requested_force_n,applied_force_n\n")
            for i in range(len(trace["time_s"])):
                fh.write(f"{trace['time_s'][i]:.4f},{np.rad2deg(trace['p_command_rad_s'][i]):.6f},"
                         f"{np.rad2deg(trace['p_reference_rad_s'][i]):.6f},{np.rad2deg(trace['p_rad_s'][i]):.6f},"
                         f"{trace['requested_f_as_n'][i]:.6f},{trace['f_as_n'][i]:.6f}\n")

fig, axes = plt.subplots(3, 2, figsize=(15.6, 10.8), sharex="col")
proxy = [Line2D([], [], color="#94a3b8", ls="--", lw=1.2, label="p_c 指令"),
         Line2D([], [], color="#111827", ls=":", lw=1.4, label="p_ref 参考"),
         Line2D([], [], color="#111827", lw=2.0, label="p 仿真响应（品模教师）"),
         Line2D([], [], color="#f59e0b", ls="--", lw=1.0, label="F_req 请求力"),
         Line2D([], [], color="#dc2626", lw=1.1, label="F_applied 实际施加力")]
for row, rep in enumerate(reps):
    for col, (command_id, command_label, window) in enumerate(commands):
        ax = axes[row, col]
        tr = traces[(rep["plant_id"], command_id)]
        m = tr["time_s"] <= window
        t = tr["time_s"][m]
        ax.plot(t, np.rad2deg(tr["p_command_rad_s"][m]), color="#94a3b8", ls="--", lw=1.2)
        ax.plot(t, np.rad2deg(tr["p_reference_rad_s"][m]), color="#111827", ls=":", lw=1.4)
        ax.plot(t, np.rad2deg(tr["p_rad_s"][m]), color=LEVEL_COLOR[rep["level"]], lw=2.0)
        ax.set_ylabel("p (deg/s)", fontsize=15.0)
        ax.grid(alpha=0.22)
        ax2 = ax.twinx()
        ax2.plot(t, tr["requested_f_as_n"][m], color="#f59e0b", ls="--", lw=1.0, alpha=0.9)
        ax2.plot(t, tr["f_as_n"][m], color="#dc2626", lw=1.1)
        ax2.set_ylabel("F (N)", fontsize=15.0, color="#b91c1c")
        ax2.tick_params(axis="y", labelsize=12.5, colors="#b91c1c")
        met = rep_metrics[rep["plant_id"]][command_id]
        ax.set_title(f"{LEVEL_LABEL[rep['level']]} · {rep['plant_id']} · {command_label}\n"
                     f"RMSE {met['tracking_rmse_deg_s']:.2f} deg/s · 峰值 {met['tracking_peak_error_deg_s']:.2f} deg/s · "
                     f"F_rms {met['force_rms_n']:.2f} N · TV {met['requested_force_total_variation_n']:.1f} N",
                     fontsize=15.0, loc="left")
        if row == 2:
            ax.set_xlabel("时间 (s)")
fig.legend(handles=proxy, loc="upper center", ncol=5, fontsize=14.0, frameon=False,
           bbox_to_anchor=(0.5, 1.015))
fig.suptitle("B · 训练完成后的闭环控制效果（冻结 bank actor 的确定性评测；1 kHz 对象 / 50 Hz 策略，源域，不是 DHC-6）",
             fontsize=18.6, fontweight="bold", x=0.01, ha="left", y=1.055)
fig.text(0.01, 0.005, "RMSE / 峰值 / F_rms / TV 均按完整 30 s 轨迹计算；跟踪误差相对 p_ref；左列显示 0–6 s、右列显示 0–10 s 的局部时窗。",
         fontsize=11.5, color="#475569")
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
save(fig, "teacher_closed_loop.png")

# ------------------------------------------------------------------ C: all 68
rows = []
for group, entries in (("accepted", accepted), ("rejected", rejected)):
    for t in entries:
        o = t["quality_gate"]["observed"]
        th = t["quality_gate"]["thresholds"]
        failed = [k for k, ok in t["quality_gate"]["checks"].items() if not ok]
        rows.append({
            "group": group, "plant_id": t["plant_id"], "level": t["quality_region"],
            "selected_step": int(t["selected_validation_step"]),
            "rmse_deg_s": float(o["mean_tracking_rmse_deg_s"]),
            "peak_deg_s": float(o["maximum_peak_error_deg_s"]),
            "tv_rate_n_s": float(o["mean_requested_force_total_variation_rate_n_s"]),
            "tv_total_n": float(o["mean_requested_force_total_variation_n"]),
            "saturation_fraction": float(o["mean_force_saturation_fraction"]),
            "rmse_ratio": float(o["mean_tracking_rmse_deg_s"] / th["maximum_mean_tracking_rmse_deg_s"]),
            "peak_ratio": float(o["maximum_peak_error_deg_s"] / th["maximum_peak_error_deg_s"]),
            "tv_rate_ratio": float(o["mean_requested_force_total_variation_rate_n_s"] /
                                   th["maximum_mean_requested_force_total_variation_rate_n_s"]),
            "failed_checks": "|".join(failed),
        })

with (EVID / "teacher_bank_overview.csv").open("w", encoding="utf-8", newline="") as fh:
    cols = list(rows[0].keys())
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join(str(r[c]) for c in cols) + "\n")

fig, axes = plt.subplots(1, 4, figsize=(16.6, 5.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
rng = np.random.default_rng(7)
panels = [("rmse_ratio", "平均跟踪 RMSE / 门限", "RMSE 比"),
          ("peak_ratio", "峰值误差 / 门限", "峰值比"),
          ("tv_rate_ratio", "请求力 TV 速率 / 门限", "TV 速率比")]
for ax, (key, ylabel, title) in zip(axes[:3], panels):
    for xi, group in enumerate(("accepted", "rejected")):
        vals = [r[key] for r in rows if r["group"] == group]
        jit = rng.uniform(-0.16, 0.16, size=len(vals))
        color = "#1d4ed8" if group == "accepted" else "#b91c1c"
        ax.scatter(np.full(len(vals), xi) + jit, vals, s=22, color=color, alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.hlines(np.median(vals), xi - 0.24, xi + 0.24, color=color, lw=2.4)
        over = int(sum(v > 1 for v in vals))
        ax.text(xi, 0.02, f"{over}/{len(vals)} >1", transform=ax.transAxes, ha="center", fontsize=9.2, color=color)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 40)
    ax.axhline(1.0, color="#dc2626", ls="--", lw=1.1)
    ax.set_xticks([0, 1], ["通过 (32)", "未通过 (36)"])
    ax.set_ylabel(ylabel, fontsize=15.0)
    ax.set_title(title, fontsize=17.0, loc="left")
    ax.grid(alpha=0.22, axis="y")
    ax.tick_params(labelsize=12.5)

ax = axes[3]
fail = Counter()
for r in rows:
    for k in filter(None, r["failed_checks"].split("|")):
        fail[k] += 1
labels = {"peak_tracking_error": "峰值误差", "mean_tracking_rmse": "平均 RMSE",
          "requested_force_tv_rate": "请求力 TV 速率"}
names = [labels.get(k, k) for k in fail]
ax.barh(names, [fail[k] for k in fail], color="#b91c1c", alpha=0.85)
for i, k in enumerate(fail):
    ax.text(fail[k] + 0.4, i, str(fail[k]), va="center", fontsize=9)
ax.set_xlabel("未通过教师数（可多选）", fontsize=15.0)
ax.set_title("哪些检查没通过", fontsize=17.0, loc="left")
ax.grid(alpha=0.2, axis="x")
ax.tick_params(labelsize=9)
fig.suptitle("C · 全部 68 个尝试：32 通过 / 36 未通过（各自门槛归一化；固定验证 selected checkpoint 口径）",
             fontsize=18.6, fontweight="bold", x=0.01, ha="left", y=1.02)
fig.tight_layout()
save(fig, "teacher_bank_overview.png")

# ------------------------------------------------------------------ audit + reps json
audit = {
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "script": str(Path(__file__).resolve().relative_to(WS)),
    "bank": {"path": str(BANK.relative_to(WS)), "sha256": sha(BANK)},
    "counts": {"accepted": len(accepted), "rejected": len(rejected),
               "accepted_local": len(accepted_local), "rejected_local": len(rejected_local),
               "training_curves": len(curves)},
    "levels_accepted": dict(Counter(t["quality_region"] for t in accepted)),
    "levels_accepted_local": dict(Counter(t["quality_region"] for t in accepted_local)),
    "thresholds": accepted[0]["quality_gate"]["thresholds"],
    "thresholds_uniform": all(
        t["quality_gate"]["thresholds"] == accepted[0]["quality_gate"]["thresholds"]
        for t in accepted + rejected),
    "selection_rule": "local accepted teachers; per level, minimum |fixed-validation mean RMSE - level median|, "
                      "tie-break by plant_id; actor = merged_bank actor (sha256 recorded)",
    "representatives": reps,
    "commands": [{"id": c, "label": l, "window_s": w} for c, l, w in commands],
    "tv_axis": {"data_min": tv_min, "data_max": tv_max, "ylim": [tv_min * 0.8, tv_max * 1.3],
                "zeros": sum(1 for v in tv_all if v == 0)},
    "traces": {f"{p}_{c}": {"rows": int(len(tr["time_s"])), "dt_s": 0.02}
               for (p, c), tr in traces.items()},
    "figures": {},
    "inputs": {},
}
for name in ("teacher_training_curves.png", "teacher_closed_loop.png", "teacher_bank_overview.png"):
    audit["figures"][name] = sha(HERE / name)
for rep in reps:
    run = RUNS / rep["plant_id"]
    audit["inputs"][rep["plant_id"]] = {
        "report_sha256": sha(run / "report.json"),
        "learning_curve_sha256": sha(run / "learning_curve.json"),
        "actor_sha256": sha(ACTORS / f"{rep['plant_id']}.pt"),
        "selected_step": rep["selected_step"],
        "command_metrics": rep_metrics[rep["plant_id"]],
    }
(EVID / "teacher_representatives.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"representatives": reps, "figures": audit["figures"]}, ensure_ascii=False, indent=1))
