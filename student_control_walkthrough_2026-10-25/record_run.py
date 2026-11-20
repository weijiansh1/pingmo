"""Record one actual frozen original-student rollout for an explanatory viewer."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import math
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / "flight_control_original_restored_2026-10-25"
sys.path.insert(0, str(ORIGINAL / "pingmo_src"))
import numpy as np
import torch
# The original bundle uses NumPy 2's spelling of the trapezoidal rule.
# Keep its source unchanged when replaying under this NumPy 1.26 environment.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
from src.envs.batched_p_channel import BatchedRollRateEnv, DT_POLICY, OBS_DIM
from src.envs.roll_rate_commands import RollRateCommandProfile

common_path = ORIGINAL / "experiments/sae_delay_20261023/common.py"
spec = importlib.util.spec_from_file_location("original_student_walkthrough", common_path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    web = ROOT / "web"
    web.mkdir(parents=True, exist_ok=True)
    if (web / "run.json").exists():
        raise FileExistsError("Preserve the recorded demonstration")
    weights = ORIGINAL / "evidence/weights/student_tiny.pt"
    library = ORIGINAL / "pingmo_src/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"
    row = next(json.loads(line) for line in library.read_text().splitlines()
               if json.loads(line)["plant_id"] == "train_core-0421")
    before_hash = sha(weights)
    payload = torch.load(weights, map_location="cpu", weights_only=False)
    student = common.TinyStudent().eval()
    student.load_state_dict(payload["student"])
    n_parameters = sum(p.numel() for p in student.parameters())
    assert n_parameters == 3856 and OBS_DIM == 176
    command = RollRateCommandProfile("walkthrough-step-plus15", "step", amplitude_deg_s=15,
                                     onset_s=1, duration_s=8)
    env = BatchedRollRateEnv(common.plant_parameters(row), 1, commands=[command], seed=0)
    env.command_index[:] = 0
    cycles = []
    for k in range(399):
        obs = env.observation()
        assert obs.shape == (1, 176)
        p0 = float((env.state @ env.plant.c.squeeze(0)).item())
        cmd = float(env.command.item())
        ref = float(env.reference.item())
        force0 = float(env.requested.item())
        applied0 = float(env.applied.item())
        action = float(student(obs).item())
        expected_force = float(np.clip(force0 + 88 * DT_POLICY * action, -22, 22))
        observation = obs[0].cpu().tolist()
        _, reward, response = env.step(torch.tensor([[action]], dtype=torch.float32))
        force1 = float(env.requested.item())
        assert abs(force1 - expected_force) < 1e-5
        cycles.append({"index": k, "t0": k * DT_POLICY, "t1": (k + 1) * DT_POLICY,
            "command_deg_s": math.degrees(cmd), "target_deg_s": math.degrees(ref),
            "p_before_deg_s": math.degrees(p0), "error_before_deg_s": math.degrees(ref - p0),
            "action": action, "requested_before_n": force0, "requested_after_n": force1,
            "requested_change_n": force1 - force0, "unclipped_change_n": 88 * DT_POLICY * action,
            "commanded_after_n": float(env.commanded.item()),
            "applied_before_n": applied0, "applied_after_n": float(env.applied.item()),
            "p_after_deg_s": math.degrees(float(response.item())),
            "next_command_deg_s": math.degrees(float(env.command.item())),
            "next_target_deg_s": math.degrees(float(env.reference.item())),
            "reward": float(reward.item()), "observation_normalized": observation})
    assert sha(weights) == before_hash
    data = {"object_label": "对象 1", "quality_level": row["gjb_level"],
        "student_parameters": n_parameters, "observation_dim": OBS_DIM,
        "policy_period_s": DT_POLICY, "plant_period_s": .001,
        "command_onset_s": 1, "command_amplitude_deg_s": 15,
        "object_parameters": {k: row["parameters"][k] for k in common.KEYS},
        "original_reference_retained": True, "reference_uses_true_transport_delay": True,
        "student_receives_explicit_theta": False, "weights_changed": False,
        "cycles": cycles}
    (web / "run.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    columns = [k for k in cycles[0] if k != "observation_normalized"]
    with (web / "cycles.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cycles)
    provenance = {"purpose": "One original frozen-student walkthrough; no training or paper-result replacement",
        "raw_object_id": row["plant_id"], "object_label": "对象 1", "weights": str(weights),
        "weights_sha256": before_hash, "library_sha256": sha(library),
        "student_source": str(common_path), "student_source_sha256": sha(common_path),
        "environment_source_sha256": sha(ORIGINAL / "pingmo_src/src/envs/batched_p_channel.py"),
        "trajectory_alignment": "Each cycle stores inputs at t0 and response at t1=t0+20ms; no terminal auto-reset included",
        "observation_contract": "9 current + 32x5 history + 7 previous actions, no explicit object parameter vector",
        "historical_reference": "The archived reference generator uses true plant tau_p, as in the original implementation",
        "cycles": len(cycles), "device": "cpu",
        "compatibility": "NumPy 1.26 np.trapz supplies the original np.trapezoid spelling; original source unchanged"}
    (ROOT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"object_label": "对象 1", "parameters": row["parameters"],
        "student_parameters": n_parameters, "cycles": len(cycles), "weights_unchanged": True}, ensure_ascii=False))
    for index in [50, 51, 55, 60, 75, 100, 150, 200, 300, 398]:
        r = cycles[index]
        print(json.dumps({k: round(r[k], 5) for k in ["t0", "command_deg_s", "target_deg_s", "p_before_deg_s",
            "action", "requested_before_n", "requested_after_n", "applied_after_n", "p_after_deg_s"]}))


if __name__ == "__main__":
    main()
