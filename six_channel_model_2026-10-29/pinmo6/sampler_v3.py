"""v3 library: quality-sampled modes, real-aircraft coupling shape, joint grading.

The roll-axis modes (T_R, spiral, Dutch roll, 1 s sensitivity) and the pitch
short period are drawn per GJB level as in the original IV-A library and v1;
the coupling shape (Y_beta, N'_p/L'_p, L'_r/N'_r, aileron yaw, pedal ratios,
speed derivatives) is drawn from the NASA CR-2144 ranges.  The lateral
derivatives follow exactly from the modes, so the roll zeros omega_phi,
zeta_phi and every cross channel are results, not inputs.  Each candidate is
graded on all criteria; the library keeps equal level-1/2/3 quotas of the
joint (worst) level plus an OOD set, with the original split sizes.

Departures from v1/v2: zeta_sp <= 1.2 and omega_sp <= 12 rad/s (CR-2144
maxima 0.885 and 10.2); every object carries a flight envelope (V_min, V_max)
so the sideslip-excursion parameter k uses the GJB speed range of its V0.
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
from pinmo6.criteria import (dutch_roll_eigenvector, dutch_roll_level, long_period_level, psi_beta_deg,
                             roll_oscillation, sideslip_excursion, steady_sideslip)
from pinmo6.derivatives import lateral_from_modes, roll_zeros, solve_longitudinal
from pinmo6.library_v2 import ENVELOPES, load_envelopes
from pinmo6.longitudinal import (COMPONENTS as LON_COMPONENTS, LongitudinalParameters, _cap, _fs_per_g,
                                 _log_between, _tau, classify_longitudinal)
from pinmo6.plausibility import lateral_ratios, violation
from src.aircraft.gain_calibration import calibrate_l_fa_for_sensitivity, sensitivity_1s_deg_per_n
from src.aircraft.parameters import PChannelParameters
from src.aircraft.sampler import (IV_A_PLANT_DT_S, _iv_a_bucket_schedule,
                                  _level_one_dutch, _level_three_dutch, _level_two_dutch, classify_iv_a_quality)

UNIT_DIM = 33
SPLIT_COUNTS = {"train_core": 2100, "train_boundary": 900, "validation": 450, "id_test": 1050, "ood_test": 500}
ZETA_SP_MAX = 1.2
OMEGA_SP_MAX = 12.0
KT = 0.514444


def _scale(u: float, low: float, high: float) -> float:
    return low + u * (high - low)


def _log_scale(u: float, low: float, high: float) -> float:
    return math.exp(math.log(low) + u * (math.log(high) - math.log(low)))


def _quantile(u: float, table: list[float]) -> float:
    return float(np.interp(u, np.linspace(0.0, 1.0, len(table)), table))


def speed_range(v0: float, v_min: float, v_max: float) -> str:
    """MIL-STD-1797A PDF p.582: VL below V_min + 20 kt, L to 1.4 V_min, M to 0.7 V_max, H above."""
    if v0 < v_min + 20.0 * KT:
        return "VL"
    if v0 < 1.4 * v_min:
        return "L"
    return "M" if v0 < 0.7 * v_max else "H"


def _proposal(q: float) -> tuple[str, str | None]:
    """Proposed joint bucket and the anchored component (rest stay at level 1)."""
    lateral_l2, lateral_l3 = ("t_r", "spiral", "dutch_roll", "sensitivity_1s"), ("t_r", "spiral", "dutch_roll")
    # level-1 proposals are the most often lost to cross-axis criteria, so they get the largest share
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
    u_s, u_tr, u_lam, u_zeta, u_omega, u_tau = u
    sensitivity = _log_scale(u_s, 0.85, 3.38)
    t_r = _log_scale(u_tr, 0.18, 1.0)
    lam = _scale(u_lam, -0.15, math.log(2) / 12)
    zeta, omega = _level_one_dutch(u_zeta, u_omega)
    tau = _scale(u_tau, 0.001, 0.20)
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
    elif level == 3:
        if anchor == "t_r":
            t_r = _log_scale(u_tr, np.nextafter(1.4, math.inf), 10.0)
        elif anchor == "spiral":
            lam = _scale(u_lam, np.nextafter(math.log(2) / 8, math.inf), math.log(2) / 4)
        elif anchor == "dutch_roll":
            zeta, omega = _level_three_dutch(u_zeta, u_omega)
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
            tau = _scale(u_tau, np.nextafter(0.25, math.inf), 0.50)
    if abs(lam) < 1e-6:
        lam = -1e-6
    return {"lambda_s": lam, "t_r": t_r, "zeta_d": zeta, "omega_d": omega, "tau_p": tau, "sensitivity": sensitivity}


def _zeta_sp(level: int, u: float) -> float:
    """Table A6 bands below the CR-2144 maximum: level 1 to 1.2, levels 2-3 on the light side only."""
    return {1: _scale(u, 0.35 * 1.01, ZETA_SP_MAX), 2: _scale(u, 0.25, 0.35 * 0.99),
            3: _scale(u, 0.10, 0.25 * 0.99)}[level]


def _longitudinal(levels: dict[str, int], v0: float, u) -> LongitudinalParameters:
    u_t2, u_cap, b_cap, u_zeta, u_tau, u_fs, b_fs = u
    inv_t2 = _log_between(u_t2, 0.4, 2.5)
    n_alpha = v0 * inv_t2 / G
    cap = _cap(levels["frequency"], n_alpha, u_cap, b_cap)
    cap = min(cap, OMEGA_SP_MAX**2 / n_alpha)
    return LongitudinalParameters(
        omega_sp=math.sqrt(cap * n_alpha), zeta_sp=_zeta_sp(levels["damping"], u_zeta), n_alpha=n_alpha,
        v0=v0, fs_per_g=_fs_per_g(levels["stick_force"], n_alpha, u_fs, b_fs),
        tau_theta=_tau(levels["delay"], u_tau))


def evaluate_candidate(args) -> dict | None:
    """Build and grade one candidate; None when it cannot be a realistic aircraft."""
    index, unit, envelopes = args
    u = [float(x) for x in unit]
    bucket, anchor = _proposal(u[0])
    modes = _roll_modes(bucket, anchor, u[1:7])
    y_b = _scale(u[7], *envelopes["y_b_sampling"])
    c_np = _scale(u[8], *envelopes["n_p_over_l_p_sampling"])
    c_lr = _scale(u[9], *envelopes["l_r_over_n_r_sampling"])
    eta = _scale(u[10], *envelopes["eta_sampling"])
    v_min = _scale(u[15], *envelopes["v_min_m_s"])
    v_max = _scale(u[16], *envelopes["v_max_m_s"])
    v0 = _log_scale(u[17], max(60.0, 1.15 * v_min), min(250.0, v_max))
    roll_modes = (modes["lambda_s"], modes["t_r"], modes["zeta_d"], modes["omega_d"])
    candidates = []
    for d in lateral_from_modes(roll_modes, v0, y_b, c_np, c_lr):
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
    d, probe = candidates[min(int(u[32] * len(candidates)), len(candidates) - 1)]
    roll = calibrate_l_fa_for_sensitivity(probe, modes["sensitivity"], dt=IV_A_PLANT_DT_S)
    measured = sensitivity_1s_deg_per_n(roll, dt=IV_A_PLANT_DT_S)
    n_rp = math.radians(_log_scale(u[11], *envelopes["sideslip_per_force_deg_per_n"])) * roll.omega_d**2
    # the fin sets both the rudder's moments and the airframe's sideslip derivatives, so the two
    # rudder ratios are drawn relative to them from real-aircraft quantiles (inverse CDF)
    dihedral = -d.l_b / d.n_b
    rudder_roll = (-_quantile(u[12], envelopes["rudder_roll_over_dihedral_quantiles"]) * dihedral
                   if dihedral > 0 else _scale(u[12], *envelopes["l_rp_over_n_rp"]))
    rudder_side = -_quantile(u[13], envelopes["rudder_side_force_ratio_quantiles"]) * abs(d.y_b) / d.n_b
    lateral = replace(d, l_a=roll.l_fa, n_a=eta * roll.l_fa, n_rp=n_rp, l_rp=rudder_roll * n_rp,
                      y_rp=rudder_side * n_rp)
    tau_rp = _scale(u[14], 0.001, 0.20)

    lon_levels = dict.fromkeys(LON_COMPONENTS, 1)
    if anchor in LON_COMPONENTS:
        lon_levels[anchor] = {"L2": 2, "L3": 3}[bucket]
    loes = _longitudinal(lon_levels, v0, u[18:25])
    longitudinal = solve_longitudinal(
        loes, x_u=-2.0 * G / (v0 * _scale(u[25], *envelopes["lift_to_drag"])),
        x_a_aero=-G * _scale(u[26], *envelopes["drag_slope_over_lift"]), m_u=_scale(u[27], *envelopes["m_u"]))

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
    long_period = long_period_level(longitudinal)

    components = {**{f"roll_{k}": v for k, v in roll_grade["gjb_component_levels"].items()},
                  "roll_dutch_roll": dutch["level"], "roll_oscillation": oscillation_level,
                  "sideslip_excursion": excursion.get("level"),
                  "steady_sideslip_sense": 1 if steady["dihedral_sense_ok"] and steady["pedal_sense_ok"] else 3,
                  **{f"pitch_{k}": v for k, v in lon_grade["component_levels"].items()},
                  "long_period": long_period["level"]}
    # roll components beyond level 3 already appear in the roll classifier's reasons
    ood = list(roll_grade["ood_reasons"]) + [f"{k}_beyond_level_3" for k, v in components.items()
                                             if v is None and not k.startswith("roll_")]
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
        "inertia": {"iy": (iy := _scale(u[28], *envelopes["iy_over_ix"])),
                    "iz": _scale(u[29], *envelopes["iz_fraction_of_ix_plus_iy"]) * (1.0 + iy),
                    "ixz": _scale(u[30], *envelopes["ixz_over_ix"])},
        "span_m": _scale(u[31], *envelopes["span_m"]),
        "criteria": {"dutch_roll_shape": shape, "psi_beta_deg": psi, "dutch_roll": dutch,
                     "roll_oscillation": oscillation, "sideslip_excursion": excursion,
                     "steady_sideslip": steady, "long_period": long_period},
    }


def binding_group(row: dict) -> str:
    """What sets the object's level: its designed anchor if that binds, else the binding criterion."""
    if row["sampling_bucket"] == "OOD":
        return sorted(row["ood_reasons"])[0]
    if row["joint_level"] == 1:
        return "level_1"
    binding = sorted(k for k, v in row["component_levels"].items() if v == row["joint_level"])
    anchor = row["proposal_anchor"]
    for name in binding:
        if anchor and name.endswith(anchor):
            return name
    return binding[0]


def _round_robin(rows: list[dict]):
    """Yield rows cycling over binding groups (as the original library rotated its anchors)."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(binding_group(row), []).append(row)
    iterators = [iter(groups[name]) for name in sorted(groups)]
    while iterators:
        alive = []
        for iterator in iterators:
            row = next(iterator, None)
            if row is not None:
                alive.append(iterator)
                yield row
        iterators = alive


def generate_candidates(count: int, seed: int, envelopes: dict, workers: int) -> list[dict]:
    unit = qmc.Sobol(d=UNIT_DIM, scramble=True, seed=seed).random_base2(math.ceil(math.log2(count)))[:count]
    jobs = [(index, row, envelopes) for index, row in enumerate(unit)]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(workers) as pool:
        results = pool.map(evaluate_candidate, jobs, chunksize=64)
    return [row for row in results if row is not None]


def build_v3_library(output_dir: Path, seed: int = 20261031, candidate_count: int = 65536,
                     workers: int = 16, envelopes_path: Path = ENVELOPES) -> Path:
    destination = Path(output_dir)
    if (destination / "plants.jsonl").exists():
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
        "schema_version": "six-channel-v3", "seed": seed, "candidate_count": candidate_count,
        "valid_candidates": len(candidates),
        "pool_sizes": {bucket: len(rows) for bucket, rows in pools.items()},
        "selection": "within each level, round-robin over the binding component (designed anchor if it binds)",
        "proposal_counts": {bucket: sum(r["proposal_bucket"] == bucket for r in candidates) for bucket in pools},
        "split_counts": SPLIT_COUNTS,
        "level_basis": "worst of roll T_R/spiral/S1s (A18/A19/A31), Dutch roll with |phi/beta| correction, "
                       "roll oscillation, sideslip excursion (k by speed range), steady-sideslip sense, "
                       "pitch frequency/damping/delay/stick force, long period",
        "envelopes": envelopes,
        "envelopes_sha256": hashlib.sha256(Path(envelopes_path).read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return destination


def object_parts(record: dict) -> dict:
    """Rebuild the typed parameter objects of one v3 record."""
    from pinmo6.coupled import CoupledAircraft, InertiaMoments
    from pinmo6.derivatives import LateralDerivatives, LongitudinalDerivatives
    roll = PChannelParameters(**{k: record["roll"][k] for k in PChannelParameters.__dataclass_fields__})
    loes = LongitudinalParameters(**record["longitudinal_loes"])
    lateral = LateralDerivatives(**record["lateral"])
    lon_fields = {k: v for k, v in record["longitudinal"].items() if k in LongitudinalDerivatives.__dataclass_fields__}
    lon_fields["long_period"] = tuple(lon_fields["long_period"])
    longitudinal = LongitudinalDerivatives(**lon_fields)
    inertia = InertiaMoments(**record["inertia"])

    def aircraft(dt: float = 0.001, engine_lag_s: float = 1.0) -> CoupledAircraft:
        return CoupledAircraft(loes, roll, record["tau_rp"], lateral, longitudinal, inertia,
                               engine_lag_s=engine_lag_s, dt=dt)

    return {"roll": roll, "loes": loes, "lateral": lateral, "longitudinal": longitudinal,
            "inertia": inertia, "aircraft": aircraft}


def load_v3(path: str | Path) -> list[tuple[dict, dict]]:
    return [(record, object_parts(record)) for record in
            (json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines())]
