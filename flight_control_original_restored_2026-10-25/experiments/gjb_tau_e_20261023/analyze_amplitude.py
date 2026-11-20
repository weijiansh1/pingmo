"""Compare tau_e across the 0.5 / 1.0 / 2.0 deg/s excitation amplitudes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "results"

def load(tag: str, primary: bool) -> dict:
    name = "fits_gate10.json" if primary else "fits.json"
    data = json.loads((BASE / tag / name).read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        ok = bool(value.get("gate10", value.get("gate", False)))
        tau = value.get("tau_e")
        if tau is None and value.get("fit", {}).get("ok"):
            tau = value["fit"]["params"]["tau_e"]
        out[key] = {"policy": value["policy"], "plant_id": value["plant_id"],
                    "delay_bin": value["delay_bin"], "tau_p": value["tau_p"],
                    "ok": ok, "tau_e": tau,
                    "vaf": value["fit"].get("vaf_mag"),
                    "phase": value["fit"].get("phase_rmse_deg")}
    return out

def main() -> None:
    sets = {
        "0.5": load("batch_amp050", False),
        "1.0": load("batch", True),
        "2.0": load("batch_amp200", False),
    }
    policies = ("raw_ft", "aux_topk4", "topk4")
    summary = {}
    for policy in policies:
        keys = set.intersection(*[
            {k for k, v in s.items() if v["policy"] == policy} for s in sets.values()
        ])
        common = [k for k in keys if all(sets[a][k]["ok"] for a in sets)]
        rows = {}
        for amp, data in sets.items():
            group = [data[k] for k in keys if data[k]["policy"] == policy]
            ok = [v for v in group if v["ok"]]
            t = [v["tau_e"] for v in ok]
            rows[amp] = {
                "n": len(group), "assessable": len(ok),
                "tau_e_median": float(np.median(t)) if t else None,
                "vaf_median": float(np.median([v["vaf"] for v in group
                                                if v["vaf"] is not None])),
                "phase_median": float(np.median([v["phase"] for v in group
                                                  if v["phase"] is not None])),
            }
        common_tau = {
            amp: float(np.median([sets[amp][k]["tau_e"] for k in common]))
            if common else None for amp in sets
        }
        summary[policy] = {"by_amplitude": rows, "common_n": len(common),
                           "common_tau_e_median": common_tau}
    (BASE / "amplitude_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
