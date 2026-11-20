"""Sampling-coverage scatter for the frozen IV-A object library (step 1).

Reads the FINAL library (not the 16384 candidates, not the 32 teachers):
``pingmo/data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl``
with its ``manifest.json``.  Groups come from the final ``quality_region`` label
(level_1/level_2/level_3/ood), cross-checked against the manifest quota; the raw
``gjb_level`` and the ``proposal_bucket`` are recorded but NOT used, because the
final quota reassigns them.

    python dhc6_blog_assets/make_sampling_figure.py

Writes sampling_distribution.png / .svg and sampling_distribution_audit.json.
No rows are dropped and no point is moved.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
WS = HERE.parent
LIB = WS / "pingmo" / "data" / "aircraft" / "generated" / "p_channel_library_iv_a_manual_v1"
PLANTS = LIB / "plants.jsonl"
MANIFEST = LIB / "manifest.json"

matplotlib.rcParams["axes.unicode_minus"] = False
for _c in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if _c in {f.name for f in font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _c
        break


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
rows = [json.loads(line) for line in PLANTS.read_text(encoding="utf-8").splitlines() if line.strip()]

GROUP_FIELD = "quality_region"
GROUP_LABEL = {"level_1": "L1", "level_2": "L2", "level_3": "L3", "ood": "OOD"}
COLOR = {"L1": "#1d4ed8", "L2": "#0f766e", "L3": "#b45309", "OOD": "#7c3aed"}
MARKER = {"L1": "o", "L2": "o", "L3": "o", "OOD": "X"}
SIZE = {"L1": 7.0, "L2": 7.0, "L3": 7.0, "OOD": 15.0}

params = [row["parameters"] for row in rows]
groups = [GROUP_LABEL[row[GROUP_FIELD]] for row in rows]
counts = {key: groups.count(key) for key in ("L1", "L2", "L3", "OOD")}
manifest_counts = dict(manifest["level_counts"])

# consistency checks that must hold before anything is plotted
plant_ids = [row["plant_id"] for row in rows]
assert len(rows) == 3000, len(rows)
assert len(set(plant_ids)) == len(rows), "plant_id is not unique"
assert counts == {"L1": 900, "L2": 900, "L3": 900, "OOD": 300}, counts
assert all(row["sampling_bucket"] == GROUP_LABEL[row["quality_region"]] for row in rows)
assert counts == {k: int(manifest_counts[k]) for k in counts}, (counts, manifest_counts)

TRACKED = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")
non_finite = {key: sum(1 for p in params if not math.isfinite(p[key])) for key in TRACKED}
assert sum(non_finite.values()) == 0, non_finite

data = {key: np.asarray([p[key] for p in params], dtype=float) for key in TRACKED}

PANELS = [
    ("t_r", "l_fa", "log", "log", "(a) T_R vs L_Fa", "T_R (s)", "L_Fa（模型增益，量纲未核实）"),
    ("zeta_d", "omega_d", "log", "log", "(b) ζ_d vs ω_d", "ζ_d", "ω_d (rad/s)"),
    ("r_omega", "r_zeta", "log", "log", "(c) R_ω vs R_ζ", "R_ω", "R_ζ"),
    ("lambda_s", "tau_p", "linear", "log", "(d) λ_s vs τ_p", "λ_s (1/s)", "τ_p (s)"),
]

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
axes = axes.ravel()
axes_info = []
for ax, (xk, yk, xscale, yscale, title, xlabel, ylabel) in zip(axes, PANELS):
    for key in ("L1", "L2", "L3", "OOD"):
        mask = np.asarray([g == key for g in groups])
        ax.scatter(data[xk][mask], data[yk][mask], s=SIZE[key], c=COLOR[key],
                   marker=MARKER[key], alpha=0.42, linewidths=0.0, rasterized=True,
                   label=f"{key}（{counts[key]}）")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_title(title, fontsize=11.4)
    ax.set_xlabel(xlabel + (" — 对数轴" if xscale == "log" else " — 线性轴"), fontsize=9.4)
    ax.set_ylabel(ylabel + (" — 对数轴" if yscale == "log" else " — 线性轴"), fontsize=9.4)
    ax.grid(alpha=0.25, which="both")
    axes_info.append({
        "panel": title, "x": xk, "x_scale": xscale,
        "x_range": [float(data[xk].min()), float(data[xk].max())],
        "y": yk, "y_scale": yscale,
        "y_range": [float(data[yk].min()), float(data[yk].max())],
    })

handles = [Line2D([0], [0], marker=MARKER[k], color="none", markerfacecolor=COLOR[k],
                  markeredgecolor=COLOR[k], markersize=7 if k != "OOD" else 8,
                  alpha=0.9, label=f"{k}（{counts[k]}）") for k in ("L1", "L2", "L3", "OOD")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.5,
           bbox_to_anchor=(0.5, 0.004))
fig.suptitle("冻结对象库的采样覆盖：8 维 θ 的二维投影（同一 3000 个对象）", fontsize=13.2)
fig.text(0.5, 0.055,
         "四个子图画的是同一批 3000 个对象，只是投影到不同的两维；这不等于 8 维联合空间覆盖，"
         "也不代表真实机队分布。L1/L2/L3 各 900、OOD 300 为库内配额。",
         ha="center", fontsize=8.8, color="#475569")
fig.tight_layout(rect=(0, 0.09, 1, 0.965))
fig.savefig(HERE / "sampling_distribution.png", dpi=150, bbox_inches="tight",
            pad_inches=0.25, facecolor="white")
fig.savefig(HERE / "sampling_distribution.svg", bbox_inches="tight",
            pad_inches=0.25, facecolor="white")
plt.close(fig)

audit = {
    "figure": "sampling_distribution.png / .svg",
    "data_source": {
        "plants": str(PLANTS.relative_to(WS)),
        "plants_sha256": sha256(PLANTS),
        "manifest": str(MANIFEST.relative_to(WS)),
        "manifest_sha256": sha256(MANIFEST),
    },
    "library": {
        "schema_version": manifest["schema_version"],
        "profile": manifest["profile"],
        "seed": manifest["seed"],
        "candidate_count": manifest["candidate_count"],
        "total_plants": manifest["total_plants"],
    },
    "n_lines": len(rows),
    "n_unique_plant_id": len(set(plant_ids)),
    "n_plotted": len(rows),
    "n_dropped": 0,
    "group_field_used": GROUP_FIELD,
    "group_counts": counts,
    "manifest_level_counts": manifest_counts,
    "group_counts_match_manifest": True,
    "not_used": {
        "gjb_level": {str(k): v for k, v in
                      __import__("collections").Counter(row["gjb_level"] for row in rows).items()},
        "proposal_bucket": dict(__import__("collections").Counter(row["proposal_bucket"] for row in rows)),
        "reason": "final quota reassigns the raw level/proposal; the figure uses the final quality_region",
    },
    "non_finite": non_finite,
    "non_finite_total": sum(non_finite.values()),
    "axes": axes_info,
    "note": ("2-D projections of the 8-D theta; the four panels share the same 3000 objects. "
             "Not a full joint-space coverage claim and not a real-fleet distribution."),
}
(HERE / "sampling_distribution_audit.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"groups": counts, "n": len(rows), "axes": axes_info}, ensure_ascii=False, indent=2))
print("wrote sampling_distribution.png / .svg / audit")
