"""Generate the manual-v1 G2 raw/reference/oracle authority diagnostic."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aircraft.parameters import PChannelParameters
from src.experiments.reference_oracle_check import simulate_reference_oracle
from src.gjb.profile import load_roll_profile


def _load_parameters(library: Path, plant_id: str) -> PChannelParameters:
    for line in (library / "plants.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["plant_id"] == plant_id:
            parameters = {key: value for key, value in row["parameters"].items() if key in PChannelParameters.__dataclass_fields__}
            return PChannelParameters(**parameters)
    raise ValueError(f"plant_id not found: {plant_id}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="G2 same-input reference/oracle diagnostic")
    parser.add_argument("--library", type=Path, default=root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1")
    parser.add_argument("--plant-id", default="train_core-0000")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--authority", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=root / "img/手册G2_原始参考Oracle受限响应.png")
    args = parser.parse_args()

    profile = load_roll_profile(root / "data/gjb_roll_spec.yaml", "IV-A")
    parameters = _load_parameters(args.library, args.plant_id)
    trace = simulate_reference_oracle(
        parameters,
        pilot_force_n=float(profile["pilot_force_scale_n"]),
        duration_s=args.duration,
        correction_ratio=args.authority,
        normalized_rate_limit_s_inv=float(profile["normalized_rate_limit_s_inv"]),
    )

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, (axes_response, axes_force) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, layout="constrained")
    axes_response.plot(trace.time_s, trace.p_raw, label="原始飞机 raw", linewidth=1.8)
    axes_response.plot(trace.time_s, trace.p_ref, label="参考模型 ref", linewidth=1.8)
    axes_response.plot(trace.time_s, trace.p_oracle, label="无约束 Oracle", linewidth=1.5)
    axes_response.plot(trace.time_s, trace.p_constrained, label="受限 Oracle", linewidth=1.8)
    axes_response.set_ylabel("滚转响应 p")
    axes_response.set_title(f"G2 同输入对照：{args.plant_id}，IV-A，动作权限 {args.authority:.1f} × 22 N")
    axes_response.grid(alpha=0.3)
    axes_response.legend(ncol=2)
    axes_force.step(trace.time_s, trace.f_pilot, where="post", label="驾驶员输入 F_pilot")
    axes_force.step(trace.time_s, trace.delta_constrained, where="post", label="受限修正 ΔF")
    axes_force.step(trace.time_s, trace.f_eq_constrained, where="post", label="受限等效输入 F_eq")
    axes_force.set_xlabel("时间 (s)")
    axes_force.set_ylabel("力 (N)")
    axes_force.grid(alpha=0.3)
    axes_force.legend(ncol=3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    report = {
        "gate": "G2_reference_oracle_authority_diagnostic",
        "profile": "IV-A",
        "plant_id": args.plant_id,
        "pilot_force_n": float(profile["pilot_force_scale_n"]),
        "augmentation_ratio": args.authority,
        "augmentation_limit_n": args.authority * float(profile["pilot_force_scale_n"]),
        "normalized_rate_limit_s_inv": float(profile["normalized_rate_limit_s_inv"]),
        "duration_s": args.duration,
        "metrics": trace.metrics,
        "automatic_gate_status": "evidence_only_no_fabricated_GJB_threshold",
        "plot": str(args.output),
    }
    report_path = root / "results/手册G2_参考与Oracle训练前检查.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
