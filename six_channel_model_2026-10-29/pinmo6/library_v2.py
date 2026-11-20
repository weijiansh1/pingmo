"""Coupled library v2: derivatives, cross-axis grades and long period for every v1 object.

The v1 objects (frozen roll channel, GJB short period, pedal sensitivity and
delays) are kept as they are.  Each gains
- lateral derivatives from the closed-form inversion, with (Y_b, L'_r) chosen on
  a grid so the dimensionless ratios fall inside real-aircraft envelopes;
  among such choices, one that does not lower the object's level on the
  cross-axis checks is preferred;
- pedal, inertia and wing-span values drawn inside real-aircraft ranges;
- longitudinal derivatives from L/D (X_u = -2g/(V L/D)), a drag-slope term and
  M_u, solved so the GJB short period stays exact;
- grades: Dutch roll with the |phi/beta| correction, roll oscillation,
  sideslip excursion, steady-sideslip sense and the long period.
Envelopes come from literature/envelopes.json (sources in the file).
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pinmo6 import G
from pinmo6.criteria import (dutch_roll_eigenvector, dutch_roll_level, long_period_level, psi_beta_deg,
                             roll_oscillation, sideslip_excursion, steady_sideslip)
from pinmo6.derivatives import LateralDerivatives, solve_longitudinal
from pinmo6.plausibility import LateralEnvelope, lateral_candidates
from pinmo6.sampler import load_joint_library

HERE = Path(__file__).resolve().parents[1]
ENVELOPES = HERE / "literature/envelopes.json"


def load_envelopes(path: Path = ENVELOPES) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _uniform(rng: np.random.Generator, span: list[float]) -> float:
    return float(rng.uniform(*span))


def cross_axis_grades(d: LateralDerivatives, roll) -> dict[str, object]:
    shape = dutch_roll_eigenvector(d, roll)
    psi = psi_beta_deg(d, roll)
    return {
        "dutch_roll_shape": shape,
        "psi_beta_deg": psi,
        "dutch_roll_corrected": dutch_roll_level(roll, shape["phi_over_beta"]),
        "roll_oscillation": roll_oscillation(roll, psi, shape["p_lead_beta_deg"], shape["phi_over_beta"]),
        "sideslip_excursion": sideslip_excursion(d, roll, psi, speed_range="M"),
        "sideslip_excursion_low_speed": sideslip_excursion(d, roll, psi, speed_range="L"),
        "steady_sideslip": steady_sideslip(d),
    }


def worst_cross_level(grades: dict[str, object]) -> int | None:
    levels = [grades["dutch_roll_corrected"]["level"], grades["roll_oscillation"]["level"],
              grades["sideslip_excursion"].get("level")]
    if any(level is None for level in levels):
        return None
    return max(levels)


def choose_object(record: dict, base, envelopes: dict, rng: np.random.Generator,
                  max_checks: int = 12) -> dict:
    roll, lon = base.roll, base.longitudinal
    lat_env = LateralEnvelope(tuple(envelopes["y_b"]), tuple(envelopes["l_r"]),
                              {k: tuple(v) for k, v in envelopes["lateral_ratios"].items()})
    n_rp = math.radians(base.directional.sideslip_per_force_deg_per_n) * roll.omega_d**2
    pedal = {"n_rp": n_rp,
             "l_rp_over_n_rp": _uniform(rng, envelopes["l_rp_over_n_rp"]),
             "y_rp_over_n_rp": _uniform(rng, envelopes["y_rp_over_n_rp"])}
    feasible, best = lateral_candidates(roll, lon.v0, lat_env, pedal)
    target = record["roll"]["gjb_level"]
    chosen, grades, selection = None, None, None
    if feasible:
        order = rng.permutation(len(feasible))
        for index in order[:max_checks]:
            candidate_grades = cross_axis_grades(feasible[index], roll)
            worst = worst_cross_level(candidate_grades)
            if chosen is None:
                chosen, grades = feasible[index], candidate_grades
            if target is not None and worst is not None and worst <= target:
                chosen, grades, selection = feasible[index], candidate_grades, "within_envelope_no_downgrade"
                break
        selection = selection or "within_envelope"
    else:
        chosen = best[2]
        grades = cross_axis_grades(chosen, roll)
        selection = "closest_to_envelope"
    lift_drag = _uniform(rng, envelopes["lift_to_drag"])
    drag_slope = _uniform(rng, envelopes["drag_slope_over_lift"])      # (dC_D/dalpha) / C_L
    m_u = _uniform(rng, envelopes["m_u"])
    longitudinal = solve_longitudinal(lon, x_u=-2.0 * G / (lon.v0 * lift_drag), x_a_aero=-G * drag_slope, m_u=m_u)
    iy = _uniform(rng, envelopes["iy_over_ix"])
    inertia = {"iy": iy, "iz": _uniform(rng, envelopes["iz_fraction_of_ix_plus_iy"]) * (1.0 + iy),
               "ixz": _uniform(rng, envelopes["ixz_over_ix"])}
    return {
        "lateral": asdict(chosen),
        "lateral_selection": selection,
        "lateral_violated": [] if feasible else best[1],
        "lateral_feasible_fraction": len(feasible) / (9 * 81),
        "cross_axis": grades,
        "cross_axis_worst_level": worst_cross_level(grades),
        "longitudinal": {**asdict(longitudinal), "lift_to_drag": lift_drag, "drag_slope_over_lift": drag_slope,
                         "phugoid": longitudinal.phugoid},
        "long_period": long_period_level(longitudinal),
        "inertia": inertia,
        "span_m": _uniform(rng, envelopes["span_m"]),
    }


def build_coupled_library(source: Path, output_dir: Path, seed: int = 20261030,
                          envelopes_path: Path = ENVELOPES) -> Path:
    destination = Path(output_dir)
    if (destination / "plants.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    envelopes = load_envelopes(envelopes_path)
    rng = np.random.default_rng(seed)
    lines = []
    for record, base in load_joint_library(source):
        extra = choose_object(record, base, envelopes, rng)
        lines.append(json.dumps({"plant_id": record["plant_id"], **extra}, ensure_ascii=False,
                                sort_keys=True, default=float))
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plants.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"schema_version": "six-channel-coupled-v2", "seed": seed,
                "v1_library": str(Path(source)), "v1_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                "envelopes": envelopes, "envelopes_sha256": hashlib.sha256(envelopes_path.read_bytes()).hexdigest()}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    return destination
