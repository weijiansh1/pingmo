"""Read-only audit and corrected metadata overlay for the frozen 5,000 objects.

Historical labels remain identifiable. This does not reinterpret old training
results, mutate plant parameters, or claim full GJB flying-quality compliance.
"""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "flight_control_corrected_2026-10-24"
BANK = ROOT / "pingmo_src/data/aircraft/generated/p_channel_library_iv_a_5000_v1"
PARAMS = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")
COMPONENTS = ("t_r", "spiral", "dutch_roll", "sensitivity_1s")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(values):
    return dict(sorted(collections.Counter(map(str, values)).items()))


def split_stats(rows):
    return {
        "n": len(rows),
        "historical_partial_level_counts": counts(r["gjb_level"] for r in rows),
        "historical_component_signatures": counts(
            ",".join(str(r["gjb_component_levels"][k]) for k in COMPONENTS) for r in rows),
        "degraded_component_count_histogram": counts(
            sum(v is None or v > 1 for v in r["gjb_component_levels"].values()) for r in rows),
        "simultaneous_degradation_ge_2": sum(
            sum(v is None or v > 1 for v in r["gjb_component_levels"].values()) >= 2 for r in rows),
        "dutch_L2_omega_ge_1": sum(r["gjb_component_levels"]["dutch_roll"] == 2
                                    and r["parameters"]["omega_d"] >= 1 for r in rows),
        "transport_delay_min_max_s": [min(r["parameters"]["tau_p"] for r in rows),
                                       max(r["parameters"]["tau_p"] for r in rows)],
        "transport_delay_in_0p20_to_0p25_s": sum(.20 < r["parameters"]["tau_p"] <= .25 for r in rows),
        "positive_spiral_poles": sum(r["parameters"]["lambda_s"] > 0 for r in rows),
        "roll_time_constant_gt_8s": sum(r["parameters"]["t_r"] > 8 for r in rows),
        "dutch_decay_time_gt_8s": sum(1/(r["parameters"]["zeta_d"]*r["parameters"]["omega_d"]) > 8 for r in rows),
        "ood_reasons_nonexclusive": counts(v for r in rows for v in r["ood_reasons"]),
        "ood_reason_combinations": counts(";".join(sorted(r["ood_reasons"])) for r in rows),
    }


def main():
    path = BANK / "plants.jsonl"
    original_sha = sha(path)
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    old_review = json.loads((BANK / "quality_label_review.json").read_text())
    assert old_review["original_bank_sha256"] == original_sha
    corrected = {r["plant_id"]: r for r in old_review["corrected_boundary_cases"]}
    output = []
    for row in rows:
        p = row["parameters"]
        levels = dict(row["gjb_component_levels"])
        patch = corrected.get(row["plant_id"])
        if patch:
            levels["sensitivity_1s"] = patch["aligned_grid_level"]
        partial = None if None in levels.values() else max(levels.values())
        output.append({
            "plant_id": row["plant_id"],
            "historical_split": row["split"],
            "historical_sampling_bucket": row["sampling_bucket"],
            "historical_partial_screen_level": row["gjb_level"],
            "reviewed_partial_screen_level": partial,
            "reviewed_partial_components": levels,
            "sensitivity_boundary_correction": patch,
            "quality_label_scope": "A18_A19_A31_and_A35_base_only_IV_A_not_CO_GA_stick_assumption",
            "full_gjb_level": None,
            "missing_for_full_grade": ["A35_phi_beta_correction", "A116_response_and_beta_phase",
                                       "closed_loop_input_applicability_and_LOES_delay", "other_applicable_clauses"],
            "parameters_sha256": hashlib.sha256(json.dumps({k:p[k] for k in PARAMS}, sort_keys=True).encode()).hexdigest(),
            "split_semantics": ("same_partial_level_pool_no_boundary_distance_guarantee"
                                if row["split"] == "train_boundary" else row["split"]),
            "historical_ood_reasons": row["ood_reasons"],
            "is_ood_a_formal_quality_level": False,
            "raw_screen": {
                "duration_s": .75,
                "last_output_timestamp_s": .749,
                "finite_response_valid": row["raw_metrics"]["valid"],
                "legacy_oscillation_ratio_diagnostic": row["raw_metrics"]["rho_osc"],
                "gjb_oscillation_ratio": None,
                "gjb_oscillation_status": "not_computed_from_valid_standard_peak_definition",
                "legacy_last_threshold_exceedance_s": row["raw_metrics"]["settling_time_s"],
                "settling_time_s": None,
                "settling_status": "not_established_by_legacy_last_exceedance_diagnostic",
            },
            "derived_time_scales": {
                "roll_s": p["t_r"],
                "spiral_exponent_per_s": p["lambda_s"],
                "spiral_double_or_half_s": math.log(2)/abs(p["lambda_s"]),
                "dutch_decay_s": 1/(p["zeta_d"]*p["omega_d"]),
                "dutch_period_s": 2*math.pi/(p["omega_d"]*math.sqrt(1-p["zeta_d"]**2)),
                "cycles_in_screen_after_delay": max(0,.75-p["tau_p"])*p["omega_d"]*math.sqrt(1-p["zeta_d"]**2)/(2*math.pi),
            },
        })
    (HERE / "reviewed_objects.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in output))
    summary = {
        "source_bank": str(path), "source_bank_sha256": original_sha,
        "statistics_basis": "frozen_historical_labels; two sensitivity corrections shown only in overlay",
        "component_signature_order": COMPONENTS,
        "total": split_stats(rows),
        "splits": {s:split_stats([r for r in rows if r["split"] == s]) for s in sorted({r["split"] for r in rows})},
        "unique_parameter_vectors": len({tuple(r["parameters"][k] for k in PARAMS) for r in rows}),
        "legacy_raw_ratio_zero": sum(r["raw_metrics"]["rho_osc"] == 0 for r in rows),
        "legacy_raw_ratio_one": sum(r["raw_metrics"]["rho_osc"] == 1 for r in rows),
        "legacy_last_threshold_at_screen_end": sum(r["raw_metrics"]["settling_time_s"] == .749 for r in rows),
        "objects_with_less_than_one_dutch_cycle_in_screen": sum(r["derived_time_scales"]["cycles_in_screen_after_delay"] < 1 for r in output),
        "corrected_sensitivity_boundary_objects": list(corrected),
    }
    (HERE / "coverage.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    # Correct a cross-phase field in the evidence register; keep all historical evidence untouched.
    registry_path = ROOT / "webpage/evidence/gjb/quality_parameter_registry.csv"
    with registry_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, registry = reader.fieldnames, list(reader)
    for r in registry:
        if r["parameter"] == "ER_Rs":
            r.update(level1="A阶段不允许耦合模态", level2="A阶段不允许耦合模态", level3="A阶段不允许耦合模态",
                     source="表A20：PDF232/印刷228", note="A阶段各等级均不允许；B/C列0.5、0.3、0.15不可移到A列；快速转弯C阶段亦有例外。原扫描表题与相邻理论讨论存在差异，不自行改写B/C不等号。")
        if r["parameter"] in {"zeta_d", "omega_d", "zeta_d_omega_d"}:
            r["status"] = "partial"
            r["note"] += "；只评估A35基准列，phi/beta修正未验证，不能作为完整荷兰滚评级"
        if r["parameter"] == "P_osc_over_P_av":
            r["note"] += "；正文3.6/PDF12还区分zeta_d<=0.2三极值式与>0.2两极值式；峰缺失不补零"
    with (HERE / "quality_parameter_registry_reviewed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(registry)
    assert sha(path) == original_sha
    print(json.dumps({k:summary[k] for k in ("unique_parameter_vectors", "legacy_raw_ratio_zero", "legacy_raw_ratio_one", "legacy_last_threshold_at_screen_end", "objects_with_less_than_one_dutch_cycle_in_screen")}, indent=2))
    print("Wrote coverage.json, reviewed_objects.jsonl and quality_parameter_registry_reviewed.csv; frozen bank unchanged.")


if __name__ == "__main__":
    main()
