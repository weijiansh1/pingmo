"""Versioned construction pilot: joint partial-grade cells and actual boundaries.

This is a new research distribution, not a regenerated historical experiment or
proof of aircraft realizability. It imports the frozen single-s plant and aligned
1 ms gain calibration, records their hashes, and refuses to overwrite a bank.
"""
import argparse
from dataclasses import asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "flight_control_corrected_2026-10-24"
sys.path.insert(0, str(ROOT / "pingmo_src"))
from src.aircraft.parameters import PChannelParameters
from src.aircraft.gain_calibration import calibrate_l_fa_for_sensitivity, sensitivity_1s_deg_per_n
from src.aircraft.sampler import classify_iv_a_quality

COMPONENTS = ("t_r", "spiral", "dutch_roll", "sensitivity_1s")
CELLS = list(itertools.product((1,2,3), (1,2,3), (1,2,3), (1,2)))


def logscale(u, lo, hi):
    return math.exp(math.log(lo)+u*math.log(hi/lo))


def candidate(signature, u):
    tr_level, spiral_level, dutch_level, sensitivity_level = signature
    tr_bounds = {1:(.18,.98), 2:(1.02,1.38), 3:(1.43,10.0)}[tr_level]
    tr = logscale(u[0], *tr_bounds)
    spiral_bounds = {1:(-.15,math.log(2)/12*.98),
                     2:(math.log(2)/12*1.02,math.log(2)/8*.98),
                     3:(math.log(2)/8*1.02,math.log(2)/4*.98)}[spiral_level]
    lam = spiral_bounds[0]+u[1]*(spiral_bounds[1]-spiral_bounds[0])
    if abs(lam) < 1e-6:
        lam = -1e-6
    if dutch_level == 1:
        omega = logscale(u[2],1.05,6)
        zmin = max(.20,.36/omega)
        zeta = zmin+u[3]*(.70-zmin)
        dutch_route = "all_base_L1_conditions"
    elif dutch_level == 2:
        branch = min(int(u[4]*3),2)
        if branch == 0:
            omega = logscale(u[2],.42,.98)
            zmin = max(.03,.055/omega)
            zeta = zmin+u[3]*(.70-zmin)
            dutch_route = "frequency_below_L1"
        elif branch == 1:
            zeta = .04+u[3]*.14
            omega = logscale(u[2],max(1.05,.055/zeta),6)
            dutch_route = "damping_below_L1_at_frequency_ge_1"
        else:
            omega = logscale(u[2],1.05,1.70)
            zeta = .195+u[3]*(min(.32,.345/omega)-.195)
            dutch_route = "product_below_L1_at_frequency_ge_1"
    else:
        omega = logscale(u[2],.42,6)
        zeta = logscale(u[3],.005,min(.019,.045/omega))
        dutch_route = "damping_or_product_below_L2"
    sensitivity = logscale(u[5],*( (.85,3.30) if sensitivity_level == 1 else (3.50,5.50)))
    rw = logscale(u[6],.65,1.35)
    zphi = .1+u[7]*.6
    tau = .001+u[8]*.249
    return dict(l_fa=1.,lambda_s=lam,t_r=tr,zeta_d=zeta,omega_d=omega,
                r_omega=rw,r_zeta=zphi/zeta,tau_p=tau), sensitivity, dutch_route


BOUNDARIES = [("t_r",x) for x in (1.,1.4,10.)] + [
    ("lambda_s",math.log(2)/x) for x in (12.,8.,4.)] + [
    ("sensitivity_1s",x) for x in (3.38,5.62)] + [
    ("dutch_omega",1.),("dutch_omega",.4),
    ("dutch_zeta",.19),("dutch_zeta",.02),
    ("dutch_product",.35),("dutch_product",.05)]


def set_boundary(params, sensitivity, axis, value, side):
    """Put one active grade constraint at exactly +/-2%, holding others suitable."""
    p = dict(params)
    target = value*(1+side*.02)
    zphi = p["zeta_d"]*p["r_zeta"]
    if axis in {"t_r","lambda_s"}:
        p[axis] = target
    elif axis == "sensitivity_1s":
        sensitivity = target
    elif axis == "dutch_omega":
        p.update(omega_d=target,zeta_d=.4)
    elif axis == "dutch_zeta":
        p.update(omega_d=3.,zeta_d=target)
    else:
        p["omega_d"] = 1.2 if value == .35 else .8
        p["zeta_d"] = target/p["omega_d"]
    p["r_zeta"] = zphi/p["zeta_d"]
    return p, sensitivity, target


def construct(params, sensitivity):
    p = calibrate_l_fa_for_sensitivity(PChannelParameters(**params), sensitivity, dt=.001)
    measured = sensitivity_1s_deg_per_n(p, dt=.001)
    result = classify_iv_a_quality(p, measured)
    # Grade overflow is deliberately retained for boundary probes, while this
    # pilot still conditions on the original gain and zero-location envelopes.
    if not .04 <= p.l_fa <= .4708387494022242:
        return None
    return {"parameters":{**asdict(p),"omega_phi":p.omega_phi,"zeta_phi":p.zeta_phi},
            "sensitivity_target_deg_per_n":sensitivity,"sensitivity_measured_deg_per_n":measured,
            "partial_components":result["gjb_component_levels"],
            "partial_screen_level":result["gjb_level"],
            "research_ood_reasons":result["ood_reasons"],"full_gjb_level":None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE/"pilot_joint_v1")
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--per-cell", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new versioned output directory")
    if args.per_cell < 1:
        raise ValueError("--per-cell must be positive")
    sampler = qmc.Sobol(d=9, scramble=True, seed=args.seed)
    unit = sampler.random_base2(18)
    cursor = 0
    rows = []
    rejection_count = 0
    def draw():
        nonlocal cursor
        if cursor >= len(unit):
            raise RuntimeError("Candidate budget exhausted; do not silently reduce coverage")
        u = unit[cursor]
        cursor += 1
        return u
    for split in ("train_joint","validation_joint","id_test_joint"):
        for signature in CELLS:
            for rep in range(args.per_cell):
                for _ in range(2048):
                    raw, s, route = candidate(signature,draw())
                    r = construct(raw,s)
                    if r is not None:
                        assert tuple(r["partial_components"][k] for k in COMPONENTS) == signature
                        rows.append({**r,"plant_id":f"{split}-{len(rows):05d}","split":split,
                                     "requested_component_signature":signature,"dutch_sampling_route":route})
                        break
                    rejection_count += 1
                else:
                    raise RuntimeError(f"Cannot populate declared joint cell {signature}")
        print(f"Completed {split}: {len(CELLS)*args.per_cell} objects",flush=True)
    for axis, value in BOUNDARIES:
        # A matched pair changes only this constraint and recalibrates gain to
        # hold the declared sensitivity; actual Lfa is therefore not held fixed.
        for _ in range(2048):
            raw, s, _ = candidate((1,1,1,1),draw())
            pair = []
            for side in (-1,1):
                modified, ms, target = set_boundary(raw,s,axis,value,side)
                r = construct(modified,ms)
                if r is not None:
                    pair.append({**r,"split":"boundary_probe","boundary_axis":axis,
                                 "boundary_value":value,"boundary_side":side,
                                 "boundary_relative_distance":.02,"boundary_target":target})
            if len(pair) == 2:
                assert pair[0]["partial_screen_level"] != pair[1]["partial_screen_level"]
                for r in pair:
                    r["plant_id"] = f"boundary_probe-{len(rows):05d}"
                    rows.append(r)
                break
            rejection_count += 1
        else:
            raise RuntimeError(f"Cannot populate boundary {axis}:{value}")
    args.output.mkdir(parents=True)
    bank = args.output/"plants.jsonl"
    bank.write_text("".join(json.dumps(r,ensure_ascii=False,allow_nan=False)+"\n" for r in rows))
    source_files = [Path(__file__), ROOT/"pingmo_src/src/aircraft/sampler.py",
                    ROOT/"pingmo_src/src/aircraft/gain_calibration.py",
                    ROOT/"pingmo_src/src/aircraft/p_channel.py", ROOT/"pingmo_src/src/aircraft/parameters.py"]
    manifest = {
        "version":"joint_partial_iv_a_construction_pilot_v1", "seed":args.seed,
        "purpose":"construction_and_metric_validation_only; not yet policy-trained or evidence of efficacy",
        "n":len(rows),"per_cell_per_split":args.per_cell,"component_order":COMPONENTS,
        "joint_cells_per_split":54,"boundary_pairs":len(BOUNDARIES),"candidates_consumed":cursor,
        "rejected_gain_envelope_candidates_or_pairs":rejection_count,
        "model":"single_s_roll_LOES_family; no claim every parameter combination is a real aircraft",
        "input":"equivalent_N_before_added_actuator_and_transport; not identified physical surface command",
        "standard_scope":"A18_A19_A31_and_A35_base_only_IV_A_not_CO_GA_stick_assumption",
        "priors":"research_choices; neither GJB-prescribed sampling distributions nor fleet statistics",
        "changes":"54 joint signatures; L2 Dutch frequency/damping/product routes; actual active-boundary pairs; transport delay 0.001..0.25s",
        "boundary_distance_definition":"2% of the named positive numerical threshold, on each side",
        "boundary_probe_role":"construction diagnostic only; assign a training/test role before a new frozen protocol",
        "omitted":"A35 phi/beta correction, A116 phase grade, true-airframe embedding, OOD suite, task feasibility and policy validation",
        "bank_sha256":hashlib.sha256(bank.read_bytes()).hexdigest(),
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    }
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("n","joint_cells_per_split","boundary_pairs","candidates_consumed","rejected_gain_envelope_candidates_or_pairs")},indent=2))


if __name__ == "__main__":
    main()
