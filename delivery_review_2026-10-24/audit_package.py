"""Read-only recomputation of supplied JSON summaries; no policy execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "package"


def read(relative):
    return json.loads((PACKAGE / relative).read_text(encoding="utf-8"))


def qualifies(row):
    fit = row["fit"]
    return (bool(fit["ok"]) and fit["vaf_mag"] >= 0.95
            and fit["phase_rmse_deg"] <= 10.0)


report = {"scope": "Reaggregate supplied JSON only; training and rollouts not rerun."}
files = [p for p in PACKAGE.rglob("*") if p.is_file()]
report["inventory"] = {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
                       "mat_records": len(list((PACKAGE / "data").glob("*.mat")))}

library_file = PACKAGE / "pingmo_src/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"
library = [json.loads(s) for s in library_file.read_text().splitlines() if s.strip()]
report["library"] = {"n": len(library), "splits": dict(Counter(r["split"] for r in library))}

metas = read("experiments/sae_delay_20261023/results_large/plants.json")
report["sae_dataset"] = {"n_plants": len(metas), "windows": sum(m["n_windows"] for m in metas),
    "split_bins": dict(Counter(f'{m["split"]}/{m["delay_bin"]}' for m in metas))}
report["policies"] = {}
checks = []
for tag in ("all_dagger2", "cross", "seed2"):
    data = read(f"experiments/sae_delay_20261023/results_large/policies_{tag}/policy_results.json")
    results = {}
    for policy, value in data.items():
        results[policy] = {}
        for delay_bin in ("short", "medium", "long"):
            rows = [r for r in value["rows"] if r["delay_bin"] == delay_bin]
            if not rows:
                continue
            triples = [v for r in rows for v in zip(r["metrics"]["rmse"], r["metrics"]["peak"], r["metrics"]["tv"])]
            n_pass = sum(r <= 1.5 and p <= 5.0 and t <= 12.0 for r, p, t in triples)
            result = {"plants": len(rows), "episodes": len(triples), "passed_episodes": n_pass,
                      "pass_rate": n_pass / len(triples), "rmse_median": median(t[0] for t in triples)}
            results[policy][delay_bin] = result
            given = value["eval"][delay_bin]
            for metric in ("pass_rate", "rmse_median"):
                checks.append({"source": f"{tag}/{policy}/{delay_bin}/{metric}",
                               "match": abs(result[metric] - given[metric]) < 1e-10})
    report["policies"][tag] = results

report["tau_e"] = {}
all_fits = {}
for batch, amplitude, seed in (("batch", 1.0, 1), ("batch_amp050", 0.5, 1),
                               ("batch_amp200", 2.0, 1), ("batch_seed2", 1.0, 2)):
    filename = "fits_gate10.json" if batch == "batch" else "fits.json"
    fits = read(f"experiments/gjb_tau_e_20261023/results/{batch}/{filename}")
    all_fits[batch] = fits
    batch_result = {"amplitude_scale": amplitude, "policy_seed_index": seed, "policies": {}}
    for policy in ("raw_ft", "aux_topk4", "topk4"):
        rows = [r for r in fits.values() if r["policy"] == policy]
        passed = [r for r in rows if qualifies(r)]
        batch_result["policies"][policy] = {
            "n": len(rows), "assessable": len(passed),
            "tau_e_median": median(r["fit"]["params"]["tau_e"] for r in passed) if passed else None,
            "levels": dict(Counter(r["a21"] for r in passed)),
            "by_bin": {b: sum(r["delay_bin"] == b for r in passed) for b in ("short", "medium", "long")},
        }
    report["tau_e"][batch] = batch_result

given = read("webpage/evidence/gjb_tau_e/summary_gate10.json")
for policy, result in report["tau_e"]["batch"]["policies"].items():
    for metric in ("n", "assessable", "tau_e_median"):
        checks.append({"source": f"tau_e/{policy}/{metric}", "match": result[metric] == given[policy][metric]})

# Same-object comparison prevents changing assessability from silently changing the sample.
base = all_fits["batch"]
paired = []
for row in base.values():
    if row["policy"] != "raw_ft" or not qualifies(row):
        continue
    aux = base.get("aux_topk4__" + row["plant_id"])
    if aux is not None and qualifies(aux):
        paired.append((row, aux))
report["tau_e_raw_aux_common"] = {
    "n": len(paired),
    "raw_median": median(a["fit"]["params"]["tau_e"] for a, b in paired),
    "aux_median": median(b["fit"]["params"]["tau_e"] for a, b in paired),
    "median_paired_delta_aux_minus_raw": median(b["fit"]["params"]["tau_e"] - a["fit"]["params"]["tau_e"] for a, b in paired),
    "note": "Descriptive only; no uncertainty interval or causal claim.",
}
report["teacher_legacy_summary"] = read("webpage/evidence/teacher_admission_summary.json")
probe = read("experiments/sae_delay_20261023/results_large/probe_episode.json")
report["probe_1_2s"] = probe["probes"]["1.2s"]

pairs = (
    ("webpage/evidence/sae_delay/policy_all.json", "experiments/sae_delay_20261023/results_large/policies_all_dagger2/policy_results.json"),
    ("webpage/evidence/sae_delay/policy_cross.json", "experiments/sae_delay_20261023/results_large/policies_cross/policy_results.json"),
    ("webpage/evidence/gjb_tau_e/summary_gate10.json", "experiments/gjb_tau_e_20261023/results/batch/summary_gate10.json"),
)
report["duplicate_evidence"] = [{"paths": pair, "same_json": read(pair[0]) == read(pair[1])} for pair in pairs]
report["aggregation_checks"] = {"total": len(checks), "passed": sum(c["match"] for c in checks),
                                 "mismatches": [c for c in checks if not c["match"]]}
report["important_missing_paths"] = [str(p.relative_to(PACKAGE)) for p in (
    PACKAGE / "paper_source/_report_review/results_summary.json",
    PACKAGE / "paper_source/_report_review/remote/student_library_eval.json",
) if not p.exists()]
report["non_sae_weights"] = [str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*.pt") if "sae_delay" not in str(p)]
out = ROOT / "audit_metrics.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "inventory": report["inventory"],
                  "aggregation_checks": report["aggregation_checks"],
                  "tau_e_raw_aux_common": report["tau_e_raw_aux_common"],
                  "duplicate_evidence": report["duplicate_evidence"]}, ensure_ascii=False, indent=2))
