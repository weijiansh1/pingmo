"""Regression checks for scientific failure modes found during this audit."""
import collections
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from gjb_metric_corrections import roll_oscillation_ratio, settling_with_censoring

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "flight_control_corrected_2026-10-24"
sys.path.insert(0,str(ROOT/"pingmo_src"))
from src.aircraft.parameters import PChannelParameters
from src.aircraft.gain_calibration import sensitivity_1s_deg_per_n
from src.aircraft.sampler import classify_iv_a_quality


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    t = np.arange(7.)
    p = np.array([0.,3.,1.,2.5,1.2,2.,1.5])
    low = roll_oscillation_ratio(t,p,.2,spiral_removed=True)
    high = roll_oscillation_ratio(t,p,.201,spiral_removed=True)
    check("GJB 3.6 low-damping signed peak/valley/peak formula",math.isclose(low["ratio"],3.5/7.5))
    check("GJB 3.6 high-damping peak/valley branch",math.isclose(high["ratio"],.5))
    check("negative maneuver has identical normalized ratio",roll_oscillation_ratio(t,-p,.2,spiral_removed=True,command_sign=-1)["ratio"] == low["ratio"])
    check("missing peaks do not become zero oscillation",roll_oscillation_ratio(t,t,.1,spiral_removed=True)["ratio"] is None)
    check("two extrema insufficient for low-damping branch",roll_oscillation_ratio(np.arange(5.),[0,3,1,1.5,2],.2,spiral_removed=True)["ratio"] is None)
    check("two extrema sufficient for high-damping branch",roll_oscillation_ratio(np.arange(5.),[0,3,1,1.5,2],.3,spiral_removed=True)["ratio"] == .5)
    check("spiral prerequisite enforced",roll_oscillation_ratio(t,p,.1,spiral_removed=False)["ratio"] is None)
    check("ratio alone is not A116 grade",low["A116_grade"] is None and high["A116_grade"] is None)
    check("unsettled final sample is censored",settling_with_censoring(t,np.ones(7),.1,minimum_tail_s=1)["censored"])
    check("insufficient final dwell is censored",settling_with_censoring(t,[1,1,1,1,1,.05,.05],.1,minimum_tail_s=2)["censored"])
    check("settling starts after last exceedance",settling_with_censoring(t,[1,1,1,.05,.05,.05,.05],.1,minimum_tail_s=2)["settling_time_s"] == 3)
    old = ROOT/"pingmo_src/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"
    check("historical bank hash unchanged",hashlib.sha256(old.read_bytes()).hexdigest() == "1daaccbef44b36274d2ce53612a3f46e54bd5c691d7f7bc8d498f1b903273fde")
    rows = [json.loads(s) for s in (HERE/"pilot_joint_v1/plants.jsonl").read_text().splitlines()]
    keys = ("l_fa","lambda_s","t_r","zeta_d","omega_d","r_omega","r_zeta","tau_p")
    check("pilot parameter vectors are distinct across splits",len({tuple(r["parameters"][k] for k in keys) for r in rows}) == len(rows))
    order = ("t_r","spiral","dutch_roll","sensitivity_1s")
    for split in ("train_joint","validation_joint","id_test_joint"):
        subset = [r for r in rows if r["split"] == split]
        check(f"{split}: all 54 joint component cells",len({tuple(r["partial_components"][k] for k in order) for r in subset}) == 54)
    errors = []
    for row in rows:
        p = PChannelParameters(**{k:row["parameters"][k] for k in keys})
        sensitivity = sensitivity_1s_deg_per_n(p,dt=.001)
        errors.append(abs(sensitivity-row["sensitivity_target_deg_per_n"]))
        check(f"recomputed partial grade: {row['plant_id']}",classify_iv_a_quality(p,sensitivity)["gjb_component_levels"] == row["partial_components"])
    check("all 190 gain calibrations agree with 1 ms simulation",max(errors)<1e-9)
    pairs = collections.defaultdict(list)
    for r in rows:
        if r["split"] == "boundary_probe":
            pairs[(r["boundary_axis"],r["boundary_value"])].append(r)
            axis = r["boundary_axis"]
            p = r["parameters"]
            measured = (r["sensitivity_measured_deg_per_n"] if axis == "sensitivity_1s" else
                        p["omega_d"] if axis == "dutch_omega" else
                        p["zeta_d"] if axis == "dutch_zeta" else
                        p["zeta_d"]*p["omega_d"] if axis == "dutch_product" else p[axis])
            check(f"actual 2% boundary distance: {r['plant_id']}",math.isclose(abs(measured/r["boundary_value"]-1),.02,abs_tol=1e-10))
    check("14 boundary pairs cross actual grade transitions",len(pairs)==14 and all(len(v)==2 and v[0]["partial_screen_level"] != v[1]["partial_screen_level"] for v in pairs.values()))
    check("high-frequency level-2 Dutch region covered",any(r["partial_components"]["dutch_roll"]==2 and r["parameters"]["omega_d"]>=1 for r in rows))
    check("0.20..0.25 s research delay region covered",any(.20<r["parameters"]["tau_p"]<=.25 for r in rows))
    result={"passed":len(checks),"failures":0,"max_sensitivity_error_deg_per_n":max(errors),"checks":checks,
            "scope":"artifact/data/math checks; no policy training and no improvement claim"}
    (HERE/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},indent=2))


if __name__ == "__main__":
    main()
