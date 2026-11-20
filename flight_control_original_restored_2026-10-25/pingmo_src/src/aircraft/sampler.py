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


IV_A_LEVELS = ("L1", "L2", "L3")
IV_A_L_FA_EVIDENCE_MIN = 0.04
IV_A_L_FA_EVIDENCE_MAX = 0.4708387494022242
IV_A_TRANSPORT_DELAY_RESEARCH_MAX_S = 0.25
IV_A_TRANSPORT_DELAY_MIN_S = 0.001
IV_A_PLANT_DT_S = 0.001
IV_A_DEFAULT_SPLIT_COUNTS = {
    "train_core": 1200,
    "train_boundary": 600,
    "validation": 300,
    "id_test": 600,
    "ood_test": 300,
}


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
            tau_p = float(_scale(row[6], IV_A_TRANSPORT_DELAY_MIN_S if region != "extreme" else 0.2, 0.1 if region == "core" else 0.25))
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
        tau = float(np.clip(.001 + .099 * (.2 + .58 * (np.log(t_r)-np.log(.08))/(np.log(2)-np.log(.08)) + .22*u_noise), .001, .1))
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
        tau = float(np.clip(.02+.22*(.55*severity+.45*u_noise), .001, .25))
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
    plant = PChannel(parameters, dt=IV_A_PLANT_DT_S)
    # Candidate screening uses the initial 0.75 s transient on the same 1 kHz
    # grid as training; full-horizon reports are reserved for the finalists.
    response = np.empty(int(round(0.75 / IV_A_PLANT_DT_S)))
    for index in range(len(response)):
        response[index] = plant.step(1.0)[0]
    valid = bool(np.isfinite(response).all() and np.max(np.abs(response)) < 1e5)
    if not valid:
        return {"valid": False, "rho_osc": float("inf"), "peak": float("inf")}
    metrics = evaluate_roll_response(np.arange(len(response)) * IV_A_PLANT_DT_S, response)
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


def _upper_bound_level(value: float, bounds: tuple[tuple[int, float], ...]) -> int | None:
    for level, upper_bound in bounds:
        if value <= upper_bound:
            return level
    return None


def classify_iv_a_quality(parameters: PChannelParameters, sensitivity_deg_per_n: float) -> dict[str, object]:
    """Assign the worst directly assessable IV-A level and separate research OOD gates."""
    if not math.isfinite(sensitivity_deg_per_n) or sensitivity_deg_per_n <= 0:
        raise ValueError("sensitivity_deg_per_n must be positive and finite")
    t_r_level = _upper_bound_level(parameters.t_r, ((1, 1.0), (2, 1.4), (3, 10.0)))
    spiral_level = 1 if parameters.lambda_s <= 0 else _upper_bound_level(
        parameters.lambda_s,
        ((1, math.log(2) / 12), (2, math.log(2) / 8), (3, math.log(2) / 4)),
    )
    dutch_product = parameters.zeta_d * parameters.omega_d
    if parameters.zeta_d >= .19 and dutch_product >= .35 and parameters.omega_d >= 1.0:
        dutch_level = 1
    elif parameters.zeta_d >= .02 and dutch_product >= .05 and parameters.omega_d >= .4:
        dutch_level = 2
    elif parameters.zeta_d >= 0 and parameters.omega_d >= .4:
        dutch_level = 3
    else:
        dutch_level = None
    sensitivity_level = _upper_bound_level(sensitivity_deg_per_n, ((1, 3.38), (2, 5.62)))
    component_levels = {
        "t_r": t_r_level,
        "spiral": spiral_level,
        "dutch_roll": dutch_level,
        "sensitivity_1s": sensitivity_level,
    }
    ood_reasons: list[str] = []
    for name, level in component_levels.items():
        if level is None:
            suffix = "above_level_2_ungraded" if name == "sensitivity_1s" else "beyond_level_3"
            ood_reasons.append(f"{name}_{suffix}")
    if not IV_A_L_FA_EVIDENCE_MIN <= parameters.l_fa <= IV_A_L_FA_EVIDENCE_MAX:
        ood_reasons.append("l_fa_outside_iv_force_input_evidence")
    if not .65 <= parameters.r_omega <= 1.35:
        ood_reasons.append("r_omega_outside_evidence_envelope")
    if not .1 <= parameters.zeta_phi <= .7:
        ood_reasons.append("zeta_phi_outside_research_envelope")
    if parameters.tau_p > IV_A_TRANSPORT_DELAY_RESEARCH_MAX_S:
        ood_reasons.append("transport_delay_above_research_envelope")
    gjb_level = None if any(level is None for level in component_levels.values()) else max(component_levels.values())
    sampling_bucket = "OOD" if ood_reasons else f"L{gjb_level}"
    return {
        "sampling_bucket": sampling_bucket,
        "gjb_level": gjb_level,
        "gjb_component_levels": component_levels,
        "ood_reasons": ood_reasons,
    }


def _level_one_dutch(u_zeta: float, u_omega: float) -> tuple[float, float]:
    omega = _log_scale(u_omega, 1.0, 6.0)
    zeta_min = max(.19, .35 / omega)
    return float(_scale(u_zeta, zeta_min, .70)), omega


def _level_two_dutch(u_zeta: float, u_omega: float) -> tuple[float, float]:
    omega = _log_scale(u_omega, .4, .99)
    zeta_min = max(.02, .05 / omega)
    return float(_scale(u_zeta, zeta_min, .70)), omega


def _level_three_dutch(u_zeta: float, u_omega: float) -> tuple[float, float]:
    omega = _log_scale(u_omega, .4, 2.0)
    zeta_max = min(.0199, .049 / omega)
    return _log_scale(u_zeta, .005, zeta_max), omega


def _iv_a_candidate(unit: np.ndarray) -> tuple[str, float, PChannelParameters]:
    """Generate a level-targeted IV-A candidate; final labels are always recomputed."""
    q, u_s, u_tr, u_lam, u_zeta, u_omega, u_rw, u_rz, u_tau = unit
    if q < .3:
        proposed_bucket, anchor = "L1", 0
    elif q < .6:
        proposed_bucket, anchor = "L2", min(int(((q - .3) / .3) * 4), 3)
    elif q < .9:
        proposed_bucket, anchor = "L3", min(int(((q - .6) / .3) * 3), 2)
    else:
        proposed_bucket, anchor = "OOD", min(int(((q - .9) / .1) * 6), 5)

    sensitivity = _log_scale(u_s, .85, 3.38)
    t_r = _log_scale(u_tr, .18, 1.0)
    lam = float(_scale(u_lam, -.15, math.log(2) / 12))
    zeta, omega = _level_one_dutch(u_zeta, u_omega)
    r_omega = _log_scale(u_rw, .65, 1.35)
    zeta_phi = _scale(u_rz, .1, .7)
    tau = _scale(u_tau, IV_A_TRANSPORT_DELAY_MIN_S, .20)

    if proposed_bucket == "L2":
        if anchor == 0:
            t_r = _log_scale(u_tr, np.nextafter(1.0, math.inf), 1.4)
        elif anchor == 1:
            lam = float(_scale(u_lam, np.nextafter(math.log(2) / 12, math.inf), math.log(2) / 8))
        elif anchor == 2:
            zeta, omega = _level_two_dutch(u_zeta, u_omega)
        else:
            sensitivity = _log_scale(u_s, np.nextafter(3.38, math.inf), 5.62)
    elif proposed_bucket == "L3":
        if anchor == 0:
            t_r = _log_scale(u_tr, np.nextafter(1.4, math.inf), 10.0)
        elif anchor == 1:
            lam = float(_scale(u_lam, np.nextafter(math.log(2) / 8, math.inf), math.log(2) / 4))
        else:
            zeta, omega = _level_three_dutch(u_zeta, u_omega)
    elif proposed_bucket == "OOD":
        if anchor == 0:
            t_r = _log_scale(u_tr, np.nextafter(10.0, math.inf), 15.0)
        elif anchor == 1:
            lam = float(_scale(u_lam, np.nextafter(math.log(2) / 4, math.inf), .30))
        elif anchor == 2:
            omega = _log_scale(u_omega, .20, np.nextafter(.4, 0.0))
        elif anchor == 3:
            sensitivity = _log_scale(u_s, np.nextafter(5.62, math.inf), 8.43)
        elif anchor == 4:
            r_omega = _log_scale(u_rw, .30, .60) if u_rw < .5 else _log_scale(u_rw, 1.50, 3.0)
        else:
            tau = _scale(u_tau, np.nextafter(.25, math.inf), .50)

    if abs(lam) < 1e-6:
        lam = -1e-6
    r_zeta = float(zeta_phi / zeta)
    uncalibrated = PChannelParameters(1.0, float(lam), t_r, zeta, omega, r_omega, r_zeta, tau)
    return proposed_bucket, sensitivity, calibrate_l_fa_for_sensitivity(uncalibrated, sensitivity, dt=IV_A_PLANT_DT_S)


def _iv_a_bucket_schedule(targets: dict[str, int]) -> dict[str, list[str]]:
    schedule: dict[str, list[str]] = {}
    level_cursor = 0
    for split, count in targets.items():
        if count < 0:
            raise ValueError("target counts must be non-negative")
        if "ood" in split or "extreme" in split:
            schedule[split] = ["OOD"] * count
            continue
        schedule[split] = [IV_A_LEVELS[(level_cursor + index) % len(IV_A_LEVELS)] for index in range(count)]
        level_cursor += count
    return schedule


def build_iv_a_library(output_dir: str | Path, seed: int, candidate_count: int = 16384, target_counts: dict[str, int] | None = None) -> Path:
    """Build a level-balanced IV-A bank with response-calibrated gain provenance."""
    targets = target_counts or IV_A_DEFAULT_SPLIT_COUNTS
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    unit = qmc.Sobol(d=9, scramble=True, seed=seed).random_base2(math.ceil(math.log2(candidate_count)))[:candidate_count]
    candidates: list[dict[str, object]] = []
    for index, row in enumerate(unit):
        proposed_bucket, target_sensitivity, parameters = _iv_a_candidate(row)
        measured_sensitivity = sensitivity_1s_deg_per_n(parameters, dt=IV_A_PLANT_DT_S)
        metrics = _raw_benchmark(parameters)
        assessment = classify_iv_a_quality(parameters, measured_sensitivity)
        candidates.append({
            "candidate_id": f"candidate-{index:05d}", "proposal_bucket": proposed_bucket,
            "aircraft_class": "IV", "flight_phase": "A", "profile": "IV-A",
            "task_subtype": "ordinary_not_CO_GA",
            "inceptor_assumption": "stick_controlled_for_A31_gate",
            "input_definition": "equivalent_roll_control_force_N",
            "delay_definition": "pure_transport_delay_before_p_channel",
            "sensitivity_target_deg_per_n": target_sensitivity,
            "sensitivity_measured_deg_per_n": measured_sensitivity,
            "parameters": {**asdict(parameters), "omega_phi": parameters.omega_phi, "zeta_phi": parameters.zeta_phi},
            "raw_metrics": metrics,
            **assessment,
        })
    pools = {
        bucket: [row for row in candidates if row["sampling_bucket"] == bucket and bool(row["raw_metrics"]["valid"])]
        for bucket in (*IV_A_LEVELS, "OOD")
    }
    selected: list[dict[str, object]] = []
    offsets = {bucket: 0 for bucket in pools}
    split_level_counts: dict[str, dict[str, int]] = {}
    for split, buckets in _iv_a_bucket_schedule(targets).items():
        split_level_counts[split] = {bucket: buckets.count(bucket) for bucket in (*IV_A_LEVELS, "OOD") if bucket in buckets}
        for bucket in buckets:
            offset = offsets[bucket]
            if offset >= len(pools[bucket]):
                raise ValueError(f"not enough valid IV-A {bucket} candidates for requested quotas")
            row = pools[bucket][offset]
            offsets[bucket] += 1
            quality_region = "ood" if bucket == "OOD" else f"level_{row['gjb_level']}"
            selected.append({
                **row,
                "plant_id": f"{split}-{len(selected):04d}",
                "split": split,
                "quality_region": quality_region,
                "selection_reason": "uniform_static_gjb_level_quota" if bucket != "OOD" else "held_out_research_ood_quota",
            })
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidates.jsonl", candidates), ("plants.jsonl", selected)):
        (destination / name).write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) for row in rows) + "\n", encoding="utf-8")
    level_counts = {bucket: sum(row["sampling_bucket"] == bucket for row in selected) for bucket in (*IV_A_LEVELS, "OOD")}
    manifest = {
        "schema_version": "6.0",
        "model_version": "GJB_s1_1khz",
        "profile": "IV-A",
        "seed": seed,
        "candidate_count": candidate_count,
        "total_plants": len(selected),
        "split_counts": targets,
        "level_counts": level_counts,
        "split_level_counts": split_level_counts,
        "sampling_policy": "equal_L1_L2_L3_plus_held_out_OOD",
        "level_basis": "worst_of_A18_A19_A35_and_A31_when_defined",
        "task_subtype": "ordinary_not_CO_GA",
        "inceptor_assumption": "stick_controlled_for_A31_gate",
        "input_definition": "equivalent_roll_control_force_N",
        "delay_definition": "pure_transport_delay_before_p_channel",
        "plant_dt_s": IV_A_PLANT_DT_S,
        "transport_delay_sampling_lower_bound_s": IV_A_TRANSPORT_DELAY_MIN_S,
        "gjb_label_scope": "static_appendix_A_recommended_boundaries_not_formal_compliance",
        "ungraded_parameters": ["r_omega", "zeta_phi", "tau_p"],
        "l_fa_evidence_range": [IV_A_L_FA_EVIDENCE_MIN, IV_A_L_FA_EVIDENCE_MAX],
        "gain_calibration": "S1s_deg_per_n_with_evidence_gate",
        "a116_boundary_status": "digitized_not_applied_to_level_quota",
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination
