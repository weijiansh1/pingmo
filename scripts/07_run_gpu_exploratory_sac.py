"""Run one non-formal GPU SAC diagnostic and save matched response plots."""

import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import SAC
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.exploratory_sac import (
    build_fixed_env,
    collect_response_trace,
    save_response_comparison,
    train_short_experiment,
)


class _ZeroPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool):
        return np.zeros(1, dtype=np.float32), None


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("this exploratory run requires CUDA")
    root = Path(__file__).resolve().parents[1]
    library = root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
    plant_id = "train_core-0000"
    correction_ratio = 0.3
    checkpoint_dir = root / "checkpoints/gpu_sac_exploratory_train_core_0000"
    report = train_short_experiment(
        library,
        plant_id,
        checkpoint_dir,
        timesteps=40_000,
        seed=20260827,
        device="cuda",
        correction_ratio=correction_ratio,
    )
    model = SAC.load(checkpoint_dir / "fixed_plant_sac", device="cuda")
    raw_env = build_fixed_env(library, plant_id, horizon_steps=500, correction_ratio=correction_ratio, pilot_signal="step")
    sac_env = build_fixed_env(library, plant_id, horizon_steps=500, correction_ratio=correction_ratio, pilot_signal="step")
    raw_trace = collect_response_trace(_ZeroPolicy(), raw_env, seed=20260827)
    sac_trace = collect_response_trace(model, sac_env, seed=20260827)
    plot = save_response_comparison(raw_trace, sac_trace, root / "img/GPU探索SAC_原始参考控制阶跃响应.png")
    report.update({
        "gpu": torch.cuda.get_device_name(0),
        "library": str(library),
        "plot": str(plot),
        "raw_tracking_rmse": float(np.sqrt(np.mean(np.square(raw_trace["p"] - raw_trace["p_ref"]))),),
        "sac_tracking_rmse": float(np.sqrt(np.mean(np.square(sac_trace["p"] - sac_trace["p_ref"]))),),
        "sac_delta_f_rms_n": float(np.sqrt(np.mean(np.square(sac_trace["delta_f"]))),),
    })
    (checkpoint_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
