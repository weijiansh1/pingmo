"""Compare tau_e between the two policy seeds (seed1 vs seed2)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "results"

def load(path: Path, gate_key: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        ok = bool(value.get(gate_key, value.get("gate", False)))
        tau = value.get("tau_e")
        if tau is None and value.get("fit", {}).get("ok"):
            tau = value["fit"]["params"]["tau_e"]
        out[key] = {"policy": value["policy"], "plant_id": value["plant_id"],
                    "delay_bin": value["delay_bin"], "tau_p": value["tau_p"],
                    "ok": ok, "tau_e": tau,
                    "vaf": value["fit"].get("vaf_mag"),
                    "phase": value["fit"].get("phase_rmse_deg")}
    return out

def level(tau: float) -> str:
    if tau <= 0.10:
        return "level1"
    if tau <= 0.20:
        return "level2"
    if tau <= 0.25:
        return "level3"
    return "exceeds_level3"

def summarize(data: dict, policy: str) -> dict:
    group = [v for v in data.values() if v["policy"] == policy]
    ok = [v for v in group if v["ok"]]
    t = [v["tau_e"] for v in ok]
    return {
        "n": len(group), "assessable": len(ok),
        "tau_e_median": float(np.median(t)) if t else None,
        "by_bin": {b: {"n": len([v for v in ok if v["delay_bin"] == b]),
                       "tau_e_median": float(np.median([v["tau_e"] for v in ok
                                                        if v["delay_bin"] == b]))
                       if any(v["delay_bin"] == b for v in ok) else None}
                   for b in ("short", "medium", "long")},
        "levels": {k: sum(1 for v in ok if level(v["tau_e"]) == k)
                   for k in ("level1", "level2", "level3", "exceeds_level3")},
    }

def main() -> None:
    seed1 = load(BASE / "batch/fits_gate10.json", "gate10")
    seed2 = load(BASE / "batch_seed2/fits.json", "gate")
    summary = {}
    for policy in ("raw_ft", "aux_topk4", "topk4"):
        keys = {k for k, v in seed1.items() if v["policy"] == policy} & \
               {k for k, v in seed2.items() if v["policy"] == policy}
        common = [k for k in keys if seed1[k]["ok"] and seed2[k]["ok"]]
        summary[policy] = {
            "seed1": summarize(seed1, policy),
            "seed2": summarize(seed2, policy),
            "common_n": len(common),
            "common_median": {
                "seed1": float(np.median([seed1[k]["tau_e"] for k in common])) if common else None,
                "seed2": float(np.median([seed2[k]["tau_e"] for k in common])) if common else None,
            },
        }
    (BASE / "seed_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
