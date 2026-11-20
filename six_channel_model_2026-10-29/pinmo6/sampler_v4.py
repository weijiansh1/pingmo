"""v4 library: v3 with the corrections of the 2026-10-31 reasonableness review.

Same construction as v3 (GJB-level modes, CR-2144 coupling shape, exact
derivative inversion, joint worst-level grading, equal level quotas), with:

* the roll equivalent time delay graded (GJB 5.4.1.1.5, table A21:
  0.10 / 0.20 / 0.25 s) and sampled per level;
* requirements that hold at every level mapped to "no level" (OOD) when
  broken: steady-sideslip senses (GJB 5.4.6, 5.5.1.1.2), 1/T_theta1 > 0
  (table A4), and aperiodic long-period divergence (table A5 level 3 needs
  |zeta_p| < 1; MIL-STD-1797A 4.2.1.1);
* N'_r < 0 and M_q < 0 required of every candidate (no real aircraft in
  CR-2144 has N'_r > 0; positive pitch damping is not an aircraft);
* L'_r/N'_r drawn by inverse CDF of the real conditions, the trim alpha from
  n/alpha plus a clean-fighter zero-lift offset, and I_xz from the principal
  axis inclination of the fighter subset (literature/envelopes_v4.json).
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from pinmo6 import G
from pinmo6.criteria import (dutch_roll_eigenvector, dutch_roll_level, psi_beta_deg, roll_oscillation,
                             sideslip_excursion, steady_sideslip)
from pinmo6.derivatives import LongitudinalDerivatives, lateral_from_modes, roll_zeros, solve_longitudinal
from pinmo6.library_v2 import load_envelopes
from pinmo6.longitudinal import (COMPONENTS as LON_COMPONENTS, LongitudinalParameters, _cap, _fs_per_g,
                                 _log_between, _tau, classify_longitudinal)
from pinmo6.plausibility import lateral_ratios, violation
from pinmo6.sampler_v3 import (OMEGA_SP_MAX, SPLIT_COUNTS, ZETA_SP_MAX, _log_scale, _quantile, _round_robin,
                               _scale, _zeta_sp, binding_group, load_v3, object_parts, speed_range)
from src.aircraft.gain_calibration import calibrate_l_fa_for_sensitivity, sensitivity_1s_deg_per_n
from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import (IV_A_PLANT_DT_S, _iv_a_bucket_schedule, _level_one_dutch, _level_three_dutch,
                                  _level_two_dutch, classify_iv_a_quality)

__all__ = ["ENVELOPES_V4", "UNIT_DIM", "SPLIT_COUNTS", "ZETA_SP_MAX", "OMEGA_SP_MAX", "ROLL_DELAY_LIMITS",
           "roll_delay_level", "long_period_level_v4", "evaluate_candidate", "binding_group", "build_v4_library",
           "object_parts", "load_v4"]

ENVELOPES_V4 = Path(__file__).resolve().parents[1] / "literature/envelopes_v4.json"
UNIT_DIM = 34
ROLL_DELAY_LIMITS = (0.10, 0.20, 0.25)          # GJB table A21, levels 1/2/3


def roll_delay_level(tau_p: float) -> int | None:
    return next((level for level, limit in enumerate(ROLL_DELAY_LIMITS, start=1) if tau_p <= limit), None)


def long_period_level_v4(d: LongitudinalDerivatives) -> dict[str, object]:
    """GJB table A5: zeta_p >= 0.04 / >= 0 / (T_2 >= 55 s and |zeta_p| < 1); a divergent real root meets no level."""
    phugoid = d.phugoid
    roots = d.long_period_roots
    if phugoid is None:
        if float(np.max(roots.real)) <= 0.0:
            return {"level": 1, "zeta_p": None, "form": "two_stable_real_roots"}
        growth = float(np.max(roots.real))
        return {"level": None, "zeta_p": None, "time_to_double_s": math.log(2.0) / growth, "form": "aperiodic_divergence"}
    zeta, omega = phugoid["zeta_p"], phugoid["omega_p"]
    if zeta >= 0.04:
        return {"level": 1, "zeta_p": zeta, "form": "oscillatory"}
    if zeta >= 0.0:
        return {"level": 2, "zeta_p": zeta, "form": "oscillatory"}
    double = -math.log(2.0) / (zeta * omega)
    return {"level": 3 if double >= 55.0 else None, "zeta_p": zeta, "time_to_double_s": double, "form": "divergent"}


def _proposal(q: float) -> tuple[str, str | None]:
    """Proposed joint bucket and the anchored component (rest stay at level 1)."""
    lateral_l2 = ("t_r", "spiral", "dutch_roll", "sensitivity_1s", "roll_delay")
    lateral_l3 = ("t_r", "spiral", "dutch_roll", "roll_delay")
    if q < 0.55:
        return "L1", None
    if q < 0.75:
        anchors = lateral_l2 + LON_COMPONENTS
        return "L2", anchors[min(int((q - 0.55) / 0.20 * len(anchors)), len(anchors) - 1)]
    if q < 0.90:
        anchors = lateral_l3 + LON_COMPONENTS
        return "L3", anchors[min(int((q - 0.75) / 0.15 * len(anchors)), len(anchors) - 1)]
    anchors = ("t_r", "spiral", "omega_d", "sensitivity_1s", "tau_p")
    return "OOD", anchors[min(int((q - 0.90) / 0.10 * len(anchors)), len(anchors) - 1)]


def _roll_modes(bucket: str, anchor: str | None, u) -> dict[str, float]:
    """As v3, but tau_p is level 1 (<= 0.10 s) unless it is the anchor."""
    u_s, u_tr, u_lam, u_zeta, u_omega, u_tau = u
    sensitivity = _log_scale(u_s, 0.85, 3.38)
    t_r = _log_scale(u_tr, 0.18, 1.0)
    lam = _scale(u_lam, -0.15, math.log(2) / 12)
    zeta, omega = _level_one_dutch(u_zeta, u_omega)
    tau = _scale(u_tau, 0.001, ROLL_DELAY_LIMITS[0])
    level = {"L2": 2, "L3": 3}.get(bucket)
    if level == 2:
        if anchor == "t_r":
            t_r = _log_scale(u_tr, np.nextafter(1.0, math.inf), 1.4)
        elif anchor == "spiral":
            lam = _scale(u_lam, np.nextafter(math.log(2) / 12, math.inf), math.log(2) / 8)
        elif anchor == "dutch_roll":
            zeta, omega = _level_two_dutch(u_zeta, u_omega)
        elif anchor == "sensitivity_1s":
            sensitivity = _log_scale(u_s, np.nextafter(3.38, math.inf), 5.62)
        elif anchor == "roll_delay":
            tau = _scale(u_tau, np.nextafter(ROLL_DELAY_LIMITS[0], math.inf), ROLL_DELAY_LIMITS[1])
    elif level == 3:
        if anchor == "t_r":
            t_r = _log_scale(u_tr, np.nextafter(1.4, math.inf), 10.0)
        elif anchor == "spiral":
            lam = _scale(u_lam, np.nextafter(math.log(2) / 8, math.inf), math.log(2) / 4)
        elif anchor == "dutch_roll":
            zeta, omega = _level_three_dutch(u_zeta, u_omega)
        elif anchor == "roll_delay":
            tau = _scale(u_tau, np.nextafter(ROLL_DELAY_LIMITS[1], math.inf), ROLL_DELAY_LIMITS[2])
    elif bucket == "OOD":
        if anchor == "t_r":
            t_r = _log_scale(u_tr, np.nextafter(10.0, math.inf), 15.0)
        elif anchor == "spiral":
            lam = _scale(u_lam, np.nextafter(math.log(2) / 4, math.inf), 0.30)
        elif anchor == "omega_d":
            omega = _log_scale(u_omega, 0.20, np.nextafter(0.4, 0.0))
        elif anchor == "sensitivity_1s":
            sensitivity = _log_scale(u_s, np.nextafter(5.62, math.inf), 8.43)
        else:
            tau = _scale(u_tau, np.nextafter(ROLL_DELAY_LIMITS[2], math.inf), 0.50)
    if abs(lam) < 1e-6:
        lam = -1e-6
    return {"lambda_s": lam, "t_r": t_r, "zeta_d": zeta, "omega_d": omega, "tau_p": tau, "sensitivity": sensitivity}


def _longitudinal(levels: dict[str, int], v0: float, u, u_offset: float, offset_deg) -> LongitudinalParameters:
    """As v3, with the trim alpha 1/(n/alpha) (level flight, alpha from zero lift) plus a zero-lift offset."""
    u_t2, u_cap, b_cap, u_zeta, u_tau, u_fs, b_fs = u
    inv_t2 = _log_between(u_t2, 0.4, 2.5)
    n_alpha = v0 * inv_t2 / G
    cap = _cap(levels["frequency"], n_alpha, u_cap, b_cap)
    cap = min(cap, OMEGA_SP_MAX**2 / n_alpha)
    return LongitudinalParameters(
        omega_sp=math.sqrt(cap * n_alpha), zeta_sp=_zeta_sp(levels["damping"], u_zeta), n_alpha=n_alpha,
        v0=v0, fs_per_g=_fs_per_g(levels["stick_force"], n_alpha, u_fs, b_fs),
        tau_theta=_tau(levels["delay"], u_tau),
        alpha_trim=1.0 / n_alpha + math.radians(_scale(u_offset, *offset_deg)))


def _inertia(u_iy: float, u_iz: float, u_eps: float, envelopes: dict) -> dict[str, float]:
    """Body-axis ratios to I_x; I_xz from the principal-axis inclination eps."""
    iy = _scale(u_iy, *envelopes["iy_over_ix"])
    iz = _scale(u_iz, *envelopes["iz_fraction_of_ix_plus_iy"]) * (1.0 + iy)
    eps = math.radians(_scale(u_eps, *envelopes["principal_axis_inclination_deg"]))
    return {"iy": iy, "iz": iz, "ixz": 0.5 * (iz - 1.0) * math.tan(2.0 * eps)}


def evaluate_candidate(args) -> dict | None:
    """Build and grade one candidate; None when it cannot be a realistic aircraft."""
    index, unit, envelopes = args
    u = [float(x) for x in unit]
    bucket, anchor = _proposal(u[0])
    modes = _roll_modes(bucket, anchor, u[1:7])
    y_b = _scale(u[7], *envelopes["y_b_sampling"])
    c_np = _scale(u[8], *envelopes["n_p_over_l_p_sampling"])
    c_lr = _quantile(u[9], envelopes["l_r_over_n_r_quantiles"])
    eta = _scale(u[10], *envelopes["eta_sampling"])
    v_min = _scale(u[15], *envelopes["v_min_m_s"])
    v_max = _scale(u[16], *envelopes["v_max_m_s"])
    v0 = _log_scale(u[17], max(60.0, 1.15 * v_min), min(250.0, v_max))
    roll_modes = (modes["lambda_s"], modes["t_r"], modes["zeta_d"], modes["omega_d"])
    candidates = []
    for d in lateral_from_modes(roll_modes, v0, y_b, c_np, c_lr):
        if d.n_r >= 0.0:
            continue                                   # no real aircraft has positive yaw damping derivative
        c1z, c0z = roll_zeros(d, eta)
        if c0z <= 0 or c1z <= 0 or c1z * c1z >= 4 * c0z:
            continue                                   # outside the single-s P-channel family
        omega_phi = math.sqrt(c0z)
        probe = PChannelParameters(1.0, modes["lambda_s"], modes["t_r"], modes["zeta_d"], modes["omega_d"],
                                   omega_phi / modes["omega_d"], c1z / (2 * omega_phi) / modes["zeta_d"],
                                   modes["tau_p"])
        score, _ = violation(lateral_ratios(replace(d, l_a=1.0, n_a=eta), probe), envelopes["lateral_ratios"])
        if score == 0.0:
            candidates.append((d, probe))
    if not candidates:
        return None

    lon_levels = dict.fromkeys(LON_COMPONENTS, 1)
    if anchor in LON_COMPONENTS:
        lon_levels[anchor] = {"L2": 2, "L3": 3}[bucket]
    loes = _longitudinal(lon_levels, v0, u[18:25], u[33], envelopes["alpha_trim_offset_deg"])
    longitudinal = solve_longitudinal(
        loes, x_u=-2.0 * G / (v0 * _scale(u[25], *envelopes["lift_to_drag"])),
        x_a_aero=-G * _scale(u[26], *envelopes["drag_slope_over_lift"]), m_u=_scale(u[27], *envelopes["m_u"]))
    if longitudinal.m_q >= 0.0:
        return None                                    # positive pitch damping derivative

    d, probe = candidates[min(int(u[32] * len(candidates)), len(candidates) - 1)]
    roll = calibrate_l_fa_for_sensitivity(probe, modes["sensitivity"], dt=IV_A_PLANT_DT_S)
    measured = sensitivity_1s_deg_per_n(roll, dt=IV_A_PLANT_DT_S)
    n_rp = math.radians(_log_scale(u[11], *envelopes["sideslip_per_force_deg_per_n"])) * roll.omega_d**2
    dihedral = -d.l_b / d.n_b
    rudder_roll = (-_quantile(u[12], envelopes["rudder_roll_over_dihedral_quantiles"]) * dihedral
                   if dihedral > 0 else _scale(u[12], *envelopes["l_rp_over_n_rp"]))
    rudder_side = -_quantile(u[13], envelopes["rudder_side_force_ratio_quantiles"]) * abs(d.y_b) / d.n_b
    lateral = replace(d, l_a=roll.l_fa, n_a=eta * roll.l_fa, n_rp=n_rp, l_rp=rudder_roll * n_rp,
                      y_rp=rudder_side * n_rp)
    tau_rp = _scale(u[14], 0.001, 0.20)

    roll_grade = classify_iv_a_quality(roll, measured)
    shape = dutch_roll_eigenvector(lateral, roll)
    psi = psi_beta_deg(lateral, roll)
    dutch = dutch_roll_level(roll, shape["phi_over_beta"])
    oscillation = roll_oscillation(roll, psi, shape["p_lead_beta_deg"], shape["phi_over_beta"])
    oscillation_level = (max(oscillation["level"], oscillation["first_minimum_level"])
                         if oscillation["applicable"] else 1)
    band = speed_range(v0, v_min, v_max)
    excursion = sideslip_excursion(lateral, roll, psi, speed_range=band)
    steady = steady_sideslip(lateral)
    lon_grade = classify_longitudinal(loes)
    long_period = long_period_level_v4(longitudinal)

    components = {**{f"roll_{k}": v for k, v in roll_grade["gjb_component_levels"].items()},
                  "roll_dutch_roll": dutch["level"], "roll_oscillation": oscillation_level,
                  "roll_delay": roll_delay_level(roll.tau_p),
                  "sideslip_excursion": excursion.get("level"),
                  "steady_sideslip_sense": 1 if steady["dihedral_sense_ok"] and steady["pedal_sense_ok"] else None,
                  **{f"pitch_{k}": v for k, v in lon_grade["component_levels"].items()},
                  "pitch_inv_t_theta1": 1 if longitudinal.inv_t_theta1 > 0.0 else None,
                  "long_period": long_period["level"]}
    # the roll classifier already names its own components beyond level 3, and a delay beyond 0.25 s
    reported = {f"roll_{k}" for k, v in roll_grade["gjb_component_levels"].items() if v is None}
    if "transport_delay_above_research_envelope" in roll_grade["ood_reasons"]:
        reported.add("roll_delay")
    ood = list(roll_grade["ood_reasons"]) + [f"{k}_beyond_level_3" for k, v in components.items()
                                             if v is None and k not in reported]
    level = None if ood else max(components.values())
    return {
        "candidate": index, "proposal_bucket": bucket, "proposal_anchor": anchor,
        "sampling_bucket": "OOD" if ood else f"L{level}", "joint_level": level, "ood_reasons": ood,
        "component_levels": components,
        "roll": {**asdict(roll), "omega_phi": roll.omega_phi, "zeta_phi": roll.zeta_phi,
                 "sensitivity_1s_deg_per_n": measured},
        "lateral": asdict(lateral), "eta": eta, "tau_rp": tau_rp, "dihedral_ratio": dihedral,
        "sideslip_per_force_deg_per_n": math.degrees(n_rp / roll.omega_d**2),
        "envelope": {"v_min": v_min, "v_max": v_max, "speed_range": band},
        "longitudinal_loes": asdict(loes),
        "longitudinal": {**asdict(longitudinal), "phugoid": longitudinal.phugoid},
        "inertia": _inertia(u[28], u[29], u[30], envelopes),
        "span_m": _scale(u[31], *envelopes["span_m"]),
        "criteria": {"dutch_roll_shape": shape, "psi_beta_deg": psi, "dutch_roll": dutch,
                     "roll_oscillation": oscillation, "sideslip_excursion": excursion,
                     "steady_sideslip": steady, "long_period": long_period},
    }


def generate_candidates(count: int, seed: int, envelopes: dict, workers: int) -> list[dict]:
    unit = qmc.Sobol(d=UNIT_DIM, scramble=True, seed=seed).random_base2(math.ceil(math.log2(count)))[:count]
    jobs = [(index, row, envelopes) for index, row in enumerate(unit)]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(workers) as pool:
        results = pool.map(evaluate_candidate, jobs, chunksize=64)
    return [row for row in results if row is not None]


def build_v4_library(output_dir: Path, seed: int = 20261101, candidate_count: int = 65536,
                     workers: int = 32, envelopes_path: Path = ENVELOPES_V4) -> Path:
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    envelopes = load_envelopes(envelopes_path)
    candidates = generate_candidates(candidate_count, seed, envelopes, workers)
    pools = {bucket: [row for row in candidates if row["sampling_bucket"] == bucket]
             for bucket in ("L1", "L2", "L3", "OOD")}
    queues = {bucket: _round_robin(rows) for bucket, rows in pools.items()}
    selected = []
    for split, buckets in _iv_a_bucket_schedule(SPLIT_COUNTS).items():
        for bucket in buckets:
            row = next(queues[bucket], None)
            if row is None:
                raise ValueError(f"not enough {bucket} candidates: {len(pools[bucket])}")
            selected.append({**row, "plant_id": f"{split}-{len(selected):04d}", "split": split})
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plants.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, default=float) for row in selected) + "\n",
        encoding="utf-8")
    manifest = {
        "schema_version": "six-channel-v4", "seed": seed, "candidate_count": candidate_count,
        "valid_candidates": len(candidates),
        "pool_sizes": {bucket: len(rows) for bucket, rows in pools.items()},
        "selection": "within each level, round-robin over the binding component (designed anchor if it binds)",
        "proposal_counts": {bucket: sum(r["proposal_bucket"] == bucket for r in candidates) for bucket in pools},
        "split_counts": SPLIT_COUNTS,
        "level_basis": "worst of roll T_R/spiral/S1s (A18/A19/A31), roll delay (A21), Dutch roll with |phi/beta| "
                       "correction, roll oscillation, sideslip excursion (k by speed range), pitch frequency/damping/"
                       "delay/stick force, long period (A5); steady-sideslip senses, 1/T_theta1 > 0 (A4) and "
                       "aperiodic long-period divergence are every-level requirements, so breaking one is OOD",
        "realism": "lateral ratios inside the CR-2144 min-max envelope; N'_r < 0 and M_q < 0 required",
        "changes_from_v3": "see pinmo6/sampler_v4.py docstring and the 2026-10-31 review in the design note",
        "envelopes": envelopes,
        "envelopes_sha256": hashlib.sha256(Path(envelopes_path).read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return destination


load_v4 = load_v3
