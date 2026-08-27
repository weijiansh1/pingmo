"""Scenario-first, joint Sobol generation for the local plant library."""

from dataclasses import asdict, dataclass
import math
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from src.aircraft.parameters import PChannelParameters
from src.aircraft.p_channel import PChannel
from src.aircraft.gain_calibration import calibrate_l_fa_for_sensitivity, sensitivity_1s_deg_per_n
from src.benchmark.time_domain import evaluate_roll_response


@dataclass(frozen=True, slots=True)
class PlantRecord:
    plant_id: str
    split: str
    quality_region: str
    aircraft_class: str
    flight_phase: str
    parameters: PChannelParameters


def _scale(unit: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + unit * (high - low)


def generate_plant_library(seed: int, split_counts: dict[str, int]) -> list[PlantRecord]:
    """Generate deterministic labeled candidates without independent derived sampling."""
    total = sum(split_counts.values())
    unit = qmc.Sobol(d=7, scramble=True, seed=seed).random_base2(math.ceil(math.log2(max(total, 1))))[:total]
    records: list[PlantRecord] = []
    index = 0
    for split, count in split_counts.items():
        region = "boundary" if "boundary" in split else "ood" if split == "ood_test" else "extreme" if split == "extreme_test" else "core"
        for row in unit[index:index + count]:
            t_r_low, t_r_high = (3.0, 10.0) if region == "extreme" else (0.08, 2.0 if region == "core" else 3.0)
            t_r = float(np.exp(_scale(row[0], np.log(t_r_low), np.log(t_r_high))))
            lam = float(_scale(row[1], -0.10, -0.003) if region == "core" else _scale(row[1], -0.15, 0.04))
            if abs(lam) < 0.003:
                lam = -0.003 if lam <= 0 else 0.003
            omega_d = float(_scale(row[2], 0.5 if region != "extreme" else 0.4, 3.5 if region == "core" else 8.0))
            zeta_d = float(_scale(row[3], 0.08 if region == "core" else 0.02, 0.5 if region == "core" else 0.8))
            r_omega = float(_scale(row[4], 0.8 if region == "core" else 0.65, 1.15 if region == "core" else 1.35))
            r_zeta = float(_scale(row[5], 0.8 if region == "core" else 0.5, 1.4 if region == "core" else 2.0))
            tau_p = float(_scale(row[6], 0.01 if region != "extreme" else 0.2, 0.1 if region == "core" else 0.25))
            kp_over_f = 1.0
            l_fa = kp_over_f / (t_r * r_omega**2)
            records.append(PlantRecord(
                plant_id=f"{split}-{index:04d}", split=split, quality_region=region,
                aircraft_class="mixed_fixed_wing", flight_phase="research_roll",
                parameters=PChannelParameters(l_fa, lam, t_r, zeta_d, omega_d, r_omega, r_zeta, tau_p),
            ))
            index += 1
    return records


def persist_plant_library(output_dir: str | Path, seed: int, split_counts: dict[str, int]) -> Path:
    """Persist a reproducible, human-auditable plant table and manifest."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    records = generate_plant_library(seed, split_counts)
    plant_rows = []
    for record in records:
        parameters = asdict(record.parameters)
        parameters["omega_phi"] = record.parameters.omega_phi
        parameters["zeta_phi"] = record.parameters.zeta_phi
        plant_rows.append({"plant_id": record.plant_id, "split": record.split, "quality_region": record.quality_region, "aircraft_class": record.aircraft_class, "flight_phase": record.flight_phase, "parameters": parameters})
    (destination / "plants.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in plant_rows) + "\n", encoding="utf-8")
    manifest = {"schema_version": "1.0", "seed": seed, "total_plants": len(records), "split_counts": split_counts, "sampler": "scrambled_sobol_scenario_first", "parameter_order": ["l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p"]}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _conditioned_parameters(unit: np.ndarray) -> tuple[str, PChannelParameters]:
    """Map Sobol latents to region-conditioned, correlated physical parameters."""
    q = unit[0]
    region = "core" if q < .55 else "boundary" if q < .78 else "ood" if q < .92 else "extreme"
    severity, class_latent, u_tr, u_noise, u_omega, u_zeta, u_ratio, u_gain = unit[1:9]
    if region == "core":
        t_r = float(np.exp(_scale(u_tr, np.log(.08), np.log(.35 if class_latent > .55 else 2.0))))
        tau = float(np.clip(.01 + .09 * (.2 + .58 * (np.log(t_r)-np.log(.08))/(np.log(2)-np.log(.08)) + .22*u_noise), .01, .1))
        omega = float(_scale(u_omega, 1.8, 6.0) if class_latent > .55 else _scale(u_omega, .5, 3.5))
        zeta = float(np.clip(.08 + .42 * (.2 + .58*u_zeta - .12*severity), .08, .5))
        r_omega, r_zeta, kp = .8+.35*u_ratio, .8+.6*(.35*u_ratio+.65*u_zeta), .7+.6*u_gain
    elif region == "boundary":
        t_r = float(1.0 + .8*(severity-.5) + .25*(u_tr-.5))
        tau = float(np.clip(.075 + .125*(.6*severity+.4*u_noise), .04, .2))
        omega, zeta = float(.5+5.5*u_omega), float(.04+.7*(.25*u_zeta+.75*severity))
        r_omega, r_zeta, kp = .65+.7*u_ratio, .5+1.5*(.3*u_ratio+.7*severity), 1.0+1.1*severity+.25*u_gain
    elif region == "ood":
        t_r = float(np.exp(_scale(u_tr, np.log(.05), np.log(3.0))))
        tau = float(np.clip(.02+.22*(.55*severity+.45*u_noise), .01, .25))
        omega, zeta = float(.4+7.6*u_omega), float(.02+.78*u_zeta)
        r_omega, r_zeta, kp = .65+.7*u_ratio, .5+1.5*u_zeta, .55+1.7*u_gain
    else:
        t_r = float(np.exp(_scale(u_tr, np.log(.04), np.log(.28))))
        tau = float(.20+.05*u_noise)
        omega, zeta = float(.4+7.6*u_omega), float(.02+.78*u_zeta)
        r_omega, r_zeta, kp = .65+.7*u_ratio, .5+1.5*u_zeta, 1.8+1.2*u_gain
    lam = float(-.10+.097*class_latent if region == "core" else -.15+.19*(.65*severity+.35*u_noise))
    if abs(lam) < .003: lam = -.003 if lam <= 0 else .003
    l_fa = float(kp/(t_r*r_omega**2))
    return region, PChannelParameters(l_fa, lam, t_r, zeta, omega, r_omega, r_zeta, tau)


def _raw_benchmark(parameters: PChannelParameters) -> dict[str, float | bool]:
    plant = PChannel(parameters, dt=.005)
    # Candidate screening uses the initial 0.75 s transient at the mandated
    # 200 Hz plant rate; full-horizon reports are reserved for the 3000 finalists.
    response = np.empty(150)
    for index in range(len(response)):
        response[index] = plant.step(1.0)[0]
    valid = bool(np.isfinite(response).all() and np.max(np.abs(response)) < 1e5)
    if not valid:
        return {"valid": False, "rho_osc": float("inf"), "peak": float("inf")}
    metrics = evaluate_roll_response(np.arange(len(response))*.005, response)
    return {"valid": True, "rho_osc": metrics["p_osc_over_p_av"], "peak": metrics["peak"], "settling_time_s": metrics["settling_time_s"]}


def build_stratified_library(output_dir: str | Path, seed: int, candidate_count: int = 8192, target_counts: dict[str, int] | None = None) -> Path:
    """Candidate → constraints/derived → raw benchmark → stratified final library."""
    targets = target_counts or {"train_core": 1200, "train_boundary": 600, "validation": 300, "id_test": 450, "ood_test": 300, "extreme_test": 150}
    unit = qmc.Sobol(d=9, scramble=True, seed=seed).random_base2(math.ceil(math.log2(candidate_count)))[:candidate_count]
    candidates=[]
    for index, row in enumerate(unit):
        region, parameters = _conditioned_parameters(row)
        metrics = _raw_benchmark(parameters)
        candidates.append({"candidate_id": f"candidate-{index:05d}", "proposal_region": region, "aircraft_class": "small_uas" if row[2]>.55 else "manned_fixed_wing", "flight_phase": "provisional_mixed_roll", "parameters": {**asdict(parameters), "omega_phi": parameters.omega_phi, "zeta_phi": parameters.zeta_phi}, "raw_metrics": metrics, "gjb_label_status": "provisional_mixed_fixed_wing"})
    pools={region:[row for row in candidates if row["proposal_region"]==region and row["raw_metrics"]["valid"]] for region in ("core","boundary","ood","extreme")}
    for pool in pools.values(): pool.sort(key=lambda row: (row["raw_metrics"]["rho_osc"], row["candidate_id"]))
    selected=[]
    def take(region: str, split: str, count: int, offset: int) -> None:
        pool=pools[region]
        if len(pool) < offset+count: raise ValueError(f"not enough valid {region} candidates")
        for row in pool[offset:offset+count]:
            selected.append({**row, "plant_id": f"{split}-{len(selected):04d}", "split": split, "quality_region": region, "selection_reason": f"{region}_stratified_by_raw_rho_osc"})
    take("core","train_core",targets["train_core"],0); core_offset=targets["train_core"]
    take("boundary","train_boundary",targets["train_boundary"],0)
    take("core","validation",targets["validation"],core_offset); core_offset+=targets["validation"]
    take("core","id_test",targets["id_test"],core_offset)
    take("ood","ood_test",targets["ood_test"],0); take("extreme","extreme_test",targets["extreme_test"],0)
    destination=Path(output_dir); destination.mkdir(parents=True,exist_ok=True)
    for name, rows in (("candidates.jsonl",candidates),("plants.jsonl",selected)):
        (destination/name).write_text("\n".join(json.dumps(row,ensure_ascii=False,sort_keys=True,allow_nan=False) for row in rows)+"\n",encoding="utf-8")
    manifest={"schema_version":"2.0","seed":seed,"candidate_count":candidate_count,"total_plants":len(selected),"split_counts":targets,"pipeline":["conditioned_sobol_candidates","constraints_and_derived_parameters","raw_p_channel_benchmark","stratified_selection"],"gjb_label_status":"provisional_mixed_fixed_wing"}
    (destination/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return destination


def _log_scale(value: float, low: float, high: float) -> float:
    return float(np.exp(_scale(value, np.log(low), np.log(high))))


def _iv_a_candidate(unit: np.ndarray) -> tuple[str, float, PChannelParameters]:
    """Generate IV-A coordinates; ``L_Fa`` is calibrated from proposed S1s."""
    q, u_s, u_tr, u_lam, u_zeta, u_omega, u_rw, u_rz, u_tau = unit
    if q < 0.50:
        region = "core"
        sensitivity = _log_scale(u_s, .85, 5.62)
        t_r, lam = _log_scale(u_tr, .18, 1.40), _scale(u_lam, -.15, .0578)
        zeta, omega = _scale(u_zeta, .15, .70), _log_scale(u_omega, 1.5, 6.0)
        r_omega, r_zeta, tau = _log_scale(u_rw, .70, 1.40), _log_scale(u_rz, .50, 2.0), _scale(u_tau, 0.0, .18)
    elif q < 0.76:
        region = "boundary"
        sensitivity = _log_scale(u_s, 2.7, 6.0)
        t_r = _log_scale(u_tr, .12, .35) if u_lam < .5 else _log_scale(u_tr, .85, 1.70)
        lam, zeta, omega = _scale(u_lam, .045, .10), _scale(u_zeta, .05, .25), _log_scale(u_omega, .8, 8.0)
        r_omega, r_zeta, tau = _log_scale(u_rw, .45, 2.20), _log_scale(u_rz, .25, 3.0), _scale(u_tau, .08, .27)
    elif q < .91:
        region = "ood"
        sensitivity = _log_scale(u_s, .4, 8.43)
        t_r, lam = _log_scale(u_tr, .08, 3.0), _scale(u_lam, -.30, .173)
        zeta, omega = _scale(u_zeta, .02, 1.0), _log_scale(u_omega, .5, 10.0)
        r_omega = _log_scale(u_rw, .30, .45) if u_rw < .5 else _log_scale(u_rw, 2.20, 3.0)
        r_zeta, tau = _log_scale(u_rz, .15, 4.0), _scale(u_tau, .25, .40)
    else:
        region = "extreme"
        sensitivity = _log_scale(u_s, .4, 8.43)
        t_r, lam = _log_scale(u_tr, .08, 10.0), _scale(u_lam, .173, .30)
        zeta, omega = _scale(u_zeta, .02, .10), _log_scale(u_omega, .5, 10.0)
        r_omega = _log_scale(u_rw, .30, .45) if u_rw < .5 else _log_scale(u_rw, 2.20, 3.0)
        r_zeta, tau = _log_scale(u_rz, .15, 4.0), _scale(u_tau, .40, .50)
    omega_phi = float(np.clip(r_omega * omega, .1, 10.0))
    r_omega = omega_phi / omega
    zeta_phi = float(np.clip(r_zeta * zeta, .03, 1.5))
    r_zeta = zeta_phi / zeta
    uncalibrated = PChannelParameters(1.0, float(lam), t_r, zeta, omega, r_omega, r_zeta, tau)
    return region, sensitivity, calibrate_l_fa_for_sensitivity(uncalibrated, sensitivity)


def build_iv_a_library(output_dir: str | Path, seed: int, candidate_count: int = 16384, target_counts: dict[str, int] | None = None) -> Path:
    """Build the manual-v1 IV-A bank with response-calibrated gain provenance."""
    targets = target_counts or {"train_core": 1200, "train_boundary": 600, "validation": 300, "id_test": 450, "ood_test": 300, "extreme_test": 150}
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    unit = qmc.Sobol(d=9, scramble=True, seed=seed).random_base2(math.ceil(math.log2(candidate_count)))[:candidate_count]
    candidates: list[dict[str, object]] = []
    for index, row in enumerate(unit):
        region, target_sensitivity, parameters = _iv_a_candidate(row)
        measured_sensitivity = sensitivity_1s_deg_per_n(parameters)
        metrics = _raw_benchmark(parameters)
        candidates.append({
            "candidate_id": f"candidate-{index:05d}", "proposal_region": region,
            "aircraft_class": "IV", "flight_phase": "A", "profile": "IV-A",
            "sensitivity_target_deg_per_n": target_sensitivity,
            "sensitivity_measured_deg_per_n": measured_sensitivity,
            "parameters": {**asdict(parameters), "omega_phi": parameters.omega_phi, "zeta_phi": parameters.zeta_phi},
            "raw_metrics": metrics,
        })
    pools = {region: [row for row in candidates if row["proposal_region"] == region and bool(row["raw_metrics"]["valid"])] for region in ("core", "boundary", "ood", "extreme")}
    selected: list[dict[str, object]] = []

    def take(region: str, split: str, count: int, offset: int = 0) -> int:
        pool = pools[region]
        if len(pool) < offset + count:
            raise ValueError(f"not enough valid IV-A {region} candidates")
        for row in pool[offset:offset + count]:
            selected.append({**row, "plant_id": f"{split}-{len(selected):04d}", "split": split, "quality_region": region, "selection_reason": "iv_a_response_calibrated_sobol"})
        return offset + count

    core_offset = take("core", "train_core", targets["train_core"])
    take("boundary", "train_boundary", targets["train_boundary"])
    core_offset = take("core", "validation", targets["validation"], core_offset)
    take("core", "id_test", targets["id_test"], core_offset)
    take("ood", "ood_test", targets["ood_test"])
    take("extreme", "extreme_test", targets["extreme_test"])
    destination = Path(output_dir); destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidates.jsonl", candidates), ("plants.jsonl", selected)):
        (destination / name).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows) + "\n", encoding="utf-8")
    manifest = {"schema_version": "4.0", "model_version": "GJB_s1_corrected", "profile": "IV-A", "seed": seed, "candidate_count": candidate_count, "total_plants": len(selected), "split_counts": targets, "gain_calibration": "S1s_deg_per_n", "a116_boundary_status": "digitized_A_C_levels_1_2"}
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
