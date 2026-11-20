"""Data for the paper figures, standard library only.

Reads the six-channel object family (v4 library) and the original roll library
(roll parameters kept verbatim in the v1 joint library), and writes pgfplots
tables under ../data:

* hist_<name>.dat       step histograms, original roll library vs this family
* gradient.dat          per-level 5th/50th/95th percentiles of graded quantities
* resp_<k>.dat          small-amplitude responses of three representative objects
* summary.json          numbers quoted in the text

The responses integrate the linear stability-axis models of the library
(longitudinal [u, alpha, q, theta], lateral [beta, p, r, phi]) with RK4 at
1 ms; inputs are those of verify_v4.py (lateral stick 0.02/L_Fa N, pedal for
0.1 deg steady sideslip, pitch stick for 0.5 g), each applied after its own
transport delay.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()            # six_channel_model_2026-10-29
OUT = Path(sys.argv[2]).resolve()             # report/paper/data
OUT.mkdir(parents=True, exist_ok=True)
G = 9.80665


def rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pct(values, q):
    """numpy.percentile with linear interpolation."""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


family = rows(ROOT / "library/six_channel_5000_v4/plants.jsonl")
original = rows(ROOT / "library/joint_six_channel_5000_v1/plants.jsonl")


def level(r):
    return "OOD" if r["ood_reasons"] else f"L{r['joint_level']}"


# ---------------------------------------------------------------- histograms
def orig_p(r):
    return r["roll"]["parameters"]


QUANTITIES = [
    # name, original getter, family getter, log bins
    ("tr", lambda r: orig_p(r)["t_r"], lambda r: r["roll"]["t_r"], True),
    ("lams", lambda r: orig_p(r)["lambda_s"], lambda r: r["roll"]["lambda_s"], False),
    ("wd", lambda r: orig_p(r)["omega_d"], lambda r: r["roll"]["omega_d"], True),
    ("zd", lambda r: orig_p(r)["zeta_d"], lambda r: r["roll"]["zeta_d"], False),
    ("rw", lambda r: orig_p(r)["r_omega"], lambda r: r["roll"]["r_omega"], False),
    ("zphi", lambda r: orig_p(r)["r_zeta"] * orig_p(r)["zeta_d"], lambda r: r["roll"]["zeta_phi"], False),
    ("taup", lambda r: orig_p(r)["tau_p"], lambda r: r["roll"]["tau_p"], False),
    ("eta", None, lambda r: r["eta"], False),
]
summary = {"histograms": {}}
for name, old, new, log in QUANTITIES:
    a = [old(r) for r in original] if old else []
    b = [new(r) for r in family]
    both = a + b
    lo, hi = pct(both, 0.5), pct(both, 99.5)
    n = 36
    if log:
        edges = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / n) for i in range(n + 1)]
    else:
        edges = [lo + (hi - lo) * i / n for i in range(n + 1)]

    def counts(values):
        c = [0] * n
        for x in values:
            if lo <= x <= hi:
                c[min(int((math.log(x / lo) if log else x - lo) / ((math.log(hi / lo) if log else hi - lo) / n)), n - 1)] += 1
        return c

    ca, cb = counts(a) if a else [0] * n, counts(b)
    with open(OUT / f"hist_{name}.dat", "w") as f:
        f.write("x orig fam\n")
        for i in range(n):
            f.write(f"{edges[i]:.6g} {ca[i]} {cb[i]}\n")
        f.write(f"{edges[n]:.6g} {ca[-1]} {cb[-1]}\n")
    summary["histograms"][name] = {
        "original_p5_p50_p95": [pct(a, 5), pct(a, 50), pct(a, 95)] if a else None,
        "family_p5_p50_p95": [pct(b, 5), pct(b, 50), pct(b, 95)],
        "ymax": max(ca + cb), "range": [lo, hi]}

# ---------------------------------------------------------------- level gradient
GRADED = [
    ("dbk", lambda r: r["criteria"]["sideslip_excursion"].get("delta_beta_over_k_deg")),
    ("wd", lambda r: r["roll"]["omega_d"]),
    ("tr", lambda r: r["roll"]["t_r"]),
    ("taup", lambda r: r["roll"]["tau_p"]),
    ("cap", lambda r: r["longitudinal_loes"]["omega_sp"] ** 2 / r["longitudinal_loes"]["n_alpha"]),
    ("taut", lambda r: r["longitudinal_loes"]["tau_theta"]),
    ("zp", lambda r: r["criteria"]["long_period"].get("zeta_p")),
]
LEVELS = ["L1", "L2", "L3", "OOD"]
summary["gradient"] = {}
with open(OUT / "gradient.dat", "w") as f:
    f.write("lv " + " ".join(f"{k}5 {k}50 {k}95" for k, _ in GRADED) + "\n")
    for i, lvl in enumerate(LEVELS, start=1):
        parts = []
        for key, get in GRADED:
            vals = [x for x in (get(r) for r in family if level(r) == lvl) if x is not None]
            trio = [pct(vals, 5), pct(vals, 50), pct(vals, 95)]
            summary["gradient"].setdefault(key, {})[lvl] = trio + [len(vals)]
            parts.append(" ".join(f"{x:.6g}" for x in trio))
        f.write(f"{i} " + " ".join(parts) + "\n")

# ---------------------------------------------------------------- library facts quoted in the text
counts = {}
for r in family:
    counts[level(r)] = counts.get(level(r), 0) + 1
at_level = {}
for lvl, n in (("L2", 2), ("L3", 3)):
    hist = {}
    for r in family:
        if level(r) == lvl:
            k = sum(1 for v in r["component_levels"].values() if v == n)
            hist[k] = hist.get(k, 0) + 1
    at_level[lvl] = dict(sorted(hist.items()))
l_r_env = (-0.6935, 7.274)
summary["library"] = {
    "levels": counts,
    "l1_all_components_level1": all(all(v == 1 for v in r["component_levels"].values())
                                    for r in family if level(r) == "L1"),
    "components_at_joint_level": at_level,
    "alpha_trim_deg_p5_p50_p95": [math.degrees(pct([r["longitudinal_loes"]["alpha_trim"] for r in family], q))
                                  for q in (5, 50, 95)],
    "ixz_over_iy_max": max(r["inertia"]["ixz"] / r["inertia"]["iy"] for r in family),
    "n_r_median": pct([r["lateral"]["n_r"] for r in family], 50),
    "l_r_median": pct([r["lateral"]["l_r"] for r in family], 50),
    "l_r_outside_real_fraction": sum(not (l_r_env[0] <= r["lateral"]["l_r"] <= l_r_env[1]) for r in family) / len(family),
    "max_n_r": max(r["lateral"]["n_r"] for r in family),
    "max_m_q": max(r["longitudinal"]["m_q"] for r in family),
    "v0_p5_p50_p95": [pct([r["longitudinal_loes"]["v0"] for r in family], q) for q in (5, 50, 95)],
    "splits": {},
}
for r in family:
    key = f"{r['split']}/{level(r)}"
    summary["library"]["splits"][key] = summary["library"]["splits"].get(key, 0) + 1


# ---------------------------------------------------------------- responses
def rk4(a, b_of_t, x, dt, steps, out_every, out):
    n = len(x)

    def f(t, s):
        bu = b_of_t(t)
        return [sum(a[i][j] * s[j] for j in range(n)) + bu[i] for i in range(n)]

    t = 0.0
    for k in range(steps + 1):
        if k % out_every == 0:
            out(t, x)
        if k == steps:
            break
        k1 = f(t, x)
        k2 = f(t + dt / 2, [x[i] + dt / 2 * k1[i] for i in range(n)])
        k3 = f(t + dt / 2, [x[i] + dt / 2 * k2[i] for i in range(n)])
        k4 = f(t + dt, [x[i] + dt * k3[i] for i in range(n)])
        x = [x[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) for i in range(n)]
        t = (k + 1) * dt


def lateral_matrices(lat):
    a = [[lat["y_b"], 0.0, -1.0, lat["g_over_v"]],
         [lat["l_b"], lat["l_p"], lat["l_r"], 0.0],
         [lat["n_b"], lat["n_p"], lat["n_r"], 0.0],
         [0.0, 1.0, 0.0, 0.0]]
    b = [[0.0, lat["y_rp"]], [lat["l_a"], lat["l_rp"]], [lat["n_a"], lat["n_rp"]], [0.0, 0.0]]
    return a, b


def longitudinal_matrix(lon):
    return [[lon["x_u"], lon["x_a"], 0.0, -lon["g"]],
            [lon["z_u"], lon["z_a"], 1.0, 0.0],
            [lon["m_u"], lon["m_a"], lon["m_q"], 0.0],
            [0.0, 0.0, 1.0, 0.0]]


DT, STEPS, EVERY = 0.001, 6000, 20
chosen = [next(r for r in family if r["split"] == "train_core" and r["sampling_bucket"] == f"L{k}") for k in (1, 2, 3)]
summary["responses"] = []
for index, r in enumerate(chosen, start=1):
    lat, lon, loes = r["lateral"], r["longitudinal"], r["longitudinal_loes"]
    a_lat, b_lat = lateral_matrices(lat)
    fas = 0.02 / r["roll"]["l_fa"]
    frp = 0.1 / r["sideslip_per_force_deg_per_n"]
    fs = 0.5 * loes["fs_per_g"]
    tau_p, tau_rp, tau_t = r["roll"]["tau_p"], r["tau_rp"], loes["tau_theta"]
    series = {}

    def run_lat(column, force, tau, tag):
        rec = []
        rk4(a_lat, lambda t: [b_lat[i][column] * force if t >= tau else 0.0 for i in range(4)],
            [0.0] * 4, DT, STEPS, EVERY, lambda t, x: rec.append((t, math.degrees(x[0]), math.degrees(x[1]),
                                                                  math.degrees(x[2]))))
        series[tag] = rec

    run_lat(0, fas, tau_p, "stick")
    run_lat(1, frp, tau_rp, "pedal")
    a_lon = longitudinal_matrix(lon)
    rec = []
    rk4(a_lon, lambda t: [0.0, 0.0, lon["m_s"] * fs if t >= tau_t else 0.0, 0.0], [0.0] * 4, DT, STEPS, EVERY,
        lambda t, x: rec.append((t, -(loes["v0"] / G) * (lon["z_u"] * x[0] + lon["z_a"] * x[1]))))
    with open(OUT / f"resp_{index}.dat", "w") as f:
        f.write("t pstick bstick rstick bpedal ppedal nz\n")
        for (t, b1, p1, r1), (_, b2, p2, _r2), (_, nz) in zip(series["stick"], series["pedal"], rec):
            f.write(f"{t:.3f} {p1:.6g} {b1:.6g} {r1:.6g} {b2:.6g} {p2:.6g} {nz:.6g}\n")
    s, pd, n = series["stick"], series["pedal"], rec
    summary["responses"].append({
        "object": index, "level": level(r), "speed_range": r["envelope"]["speed_range"],
        "binding": [k for k, v in r["component_levels"].items() if v == r["joint_level"] and v > 1],
        "stick_p_at_6s": s[-1][2], "stick_beta_at_6s": s[-1][1], "stick_r_at_6s": s[-1][3],
        "pedal_beta_min": min(x[1] for x in pd), "pedal_beta_max": max(x[1] for x in pd),
        "pedal_beta_at_6s": pd[-1][1], "pedal_p_max": max(x[2] for x in pd), "pedal_p_at_6s": pd[-1][2],
        "nz_max": max(x[1] for x in n), "nz_at_6s": n[-1][1],
        "inputs_N": {"lateral_stick": fas, "pedal": frp, "pitch_stick": fs}})

(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("library", "responses")}, ensure_ascii=False, indent=1))

# ---------------------------------------------------------------- 5-95 percentile segments for the gradient figure
for key, _ in GRADED:
    with open(OUT / f"grad_{key}.dat", "w") as f:
        f.write("x y\n")
        for i, lvl in enumerate(LEVELS, start=1):
            lo, _mid, hi, _n = summary["gradient"][key][lvl]
            f.write(f"{i} {lo:.6g}\n{i} {hi:.6g}\n{i} nan\n")

# ---------------------------------------------------------------- parameter-space scatter, one file per level
def nanf(x):
    return "nan" if x is None else f"{x:.6g}"


SCATTER = [
    ("tr", lambda r: r["roll"]["t_r"]),
    ("wd", lambda r: r["roll"]["omega_d"]),
    ("zd", lambda r: r["roll"]["zeta_d"]),
    ("nalpha", lambda r: r["longitudinal_loes"]["n_alpha"]),
    ("wsp", lambda r: r["longitudinal_loes"]["omega_sp"]),
    ("zsp", lambda r: r["longitudinal_loes"]["zeta_sp"]),
    ("taut", lambda r: r["longitudinal_loes"]["tau_theta"]),
    ("psib", lambda r: r["criteria"].get("psi_beta_deg")),
    ("dbk", lambda r: r["criteria"]["sideslip_excursion"].get("delta_beta_over_k_deg")),
    ("wp", lambda r: (r["longitudinal"].get("phugoid") or {}).get("omega_p")),
    ("zp", lambda r: (r["longitudinal"].get("phugoid") or {}).get("zeta_p")),
]
scatter_ranges = {}
for lvl in LEVELS:
    with open(OUT / f"scatter_{lvl}.dat", "w") as f:
        f.write(" ".join(k for k, _ in SCATTER) + "\n")
        for r in family:
            if level(r) != lvl:
                continue
            vals = [get(r) for _, get in SCATTER]
            f.write(" ".join(nanf(v) for v in vals) + "\n")
            for (k, _), v in zip(SCATTER, vals):
                if v is not None:
                    lo, hi = scatter_ranges.get(k, (v, v))
                    scatter_ranges[k] = (min(lo, v), max(hi, v))
summary["scatter_ranges"] = scatter_ranges

# ---------------------------------------------------------------- marginals stacked by level
MARGINALS = [
    ("tr", lambda r: r["roll"]["t_r"], True, (0.15, 15.0)),
    ("lams", lambda r: r["roll"]["lambda_s"], False, (-0.16, 0.30)),
    ("wd", lambda r: r["roll"]["omega_d"], True, (0.2, 7.0)),
    ("zd", lambda r: r["roll"]["zeta_d"], False, (0.0, 0.72)),
    ("taup", lambda r: r["roll"]["tau_p"], False, (0.0, 0.5)),
    ("cap", lambda r: r["longitudinal_loes"]["omega_sp"] ** 2 / r["longitudinal_loes"]["n_alpha"], True, (0.1, 60.0)),
    ("zsp", lambda r: r["longitudinal_loes"]["zeta_sp"], False, (0.1, 1.25)),
    ("zp", lambda r: (r["longitudinal"].get("phugoid") or {}).get("zeta_p"), False, (-0.1, 0.5)),
]
summary["marginals"] = {}
for name, get, log, (lo, hi) in MARGINALS:
    n = 30
    if log:
        edges = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / n) for i in range(n + 1)]
        centers = [math.sqrt(edges[i] * edges[i + 1]) for i in range(n)]
    else:
        edges = [lo + (hi - lo) * i / n for i in range(n + 1)]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n)]
    counts = {lvl: [0] * n for lvl in LEVELS}
    outside = 0
    for r in family:
        x = get(r)
        if x is None:
            continue
        if not lo <= x < hi:
            outside += 1
            continue
        pos = (math.log(x / lo) / math.log(hi / lo)) if log else (x - lo) / (hi - lo)
        counts[level(r)][min(int(pos * n), n - 1)] += 1
    with open(OUT / f"margin_{name}.dat", "w") as f:
        f.write("x " + " ".join(LEVELS) + "\n")
        for i in range(n):
            f.write(f"{centers[i]:.6g} " + " ".join(str(counts[l][i]) for l in LEVELS) + "\n")
    summary["marginals"][name] = {"outside_range": outside,
                                  "ymax": max(sum(counts[l][i] for l in LEVELS) for i in range(n))}

(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
