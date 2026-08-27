"""Run the explicitly non-formal sequential SAC screening batch on one GPU."""

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import SAC
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.envs.reward import RewardWeights
from src.experiments.exploratory_sac import (
    build_fixed_env,
    collect_response_trace,
    load_completed_screening_report,
    load_persisted_records,
    response_metrics,
    summarize_held_out_metrics,
    train_short_experiment,
)


class _ZeroPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool):
        return np.zeros(1, dtype=np.float32), None


def _evenly_spaced_train_ids(library: Path, count: int = 16) -> list[str]:
    rows = [json.loads(line) for line in library.read_text(encoding="utf-8").splitlines()]
    plant_ids = [row["plant_id"] for row in rows if row["split"] == "train_core"]
    indices = np.linspace(0, len(plant_ids) - 1, count, dtype=int)
    return [plant_ids[index] for index in indices]


def _evaluate(model: SAC, library: Path, plant_ids: list[str], correction_ratio: float, reward_weights: RewardWeights) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for plant_id in plant_ids:
        raw_env = build_fixed_env(library, plant_id, horizon_steps=500, correction_ratio=correction_ratio, pilot_signal="step", reward_weights=reward_weights)
        sac_env = build_fixed_env(library, plant_id, horizon_steps=500, correction_ratio=correction_ratio, pilot_signal="step", reward_weights=reward_weights)
        raw = collect_response_trace(_ZeroPolicy(), raw_env, seed=20260827)
        sac = collect_response_trace(model, sac_env, seed=20260827)
        records.append({"plant_id": plant_id, "raw": response_metrics(raw), "sac": response_metrics(sac)})
    return records


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("this screening batch requires CUDA")
    root = Path(__file__).resolve().parents[1]
    library = root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
    correction_ratio = 0.3
    seeds = (20260828, 20260829)
    multi_ids = _evenly_spaced_train_ids(library)
    held_out_ids = [f"id_test-{index:04d}" for index in range(2100, 2106)]
    load_persisted_records(library, held_out_ids)
    current = RewardWeights()
    strong = RewardWeights(action_energy=0.20, command_delta=1.50, applied_delta=0.30, late_error=0.50)
    configurations = (
        {"id": "single-40k", "plant_ids": ["train_core-0000"], "timesteps": 40_000, "weights": current},
        {"id": "single-120k", "plant_ids": ["train_core-0000"], "timesteps": 120_000, "weights": current},
        {"id": "multi-40k", "plant_ids": multi_ids, "timesteps": 40_000, "weights": current},
        {"id": "multi-strong-40k", "plant_ids": multi_ids, "timesteps": 40_000, "weights": strong},
    )
    output_root = root / "checkpoints/gpu_sac_screening_batch"
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for configuration in configurations:
        for seed in seeds:
            run_id = f"{configuration['id']}-seed-{seed}"
            run_dir = output_root / run_id
            completed = load_completed_screening_report(run_dir)
            if completed is not None:
                runs.append(completed)
                print(json.dumps({"event": "run_skipped", "run_id": run_id}, ensure_ascii=False), flush=True)
                continue
            print(json.dumps({"event": "run_started", "run_id": run_id}, ensure_ascii=False), flush=True)
            train_report = train_short_experiment(
                library,
                configuration["plant_ids"][0],
                run_dir,
                timesteps=configuration["timesteps"],
                seed=seed,
                device="cuda",
                correction_ratio=correction_ratio,
                plant_ids=configuration["plant_ids"],
                reward_weights=configuration["weights"],
            )
            model = SAC.load(run_dir / "fixed_plant_sac", device="cuda")
            held_out = _evaluate(model, library, held_out_ids, correction_ratio, configuration["weights"])
            run = {
                "run_id": run_id,
                "configuration_id": configuration["id"],
                "seed": seed,
                "timesteps": configuration["timesteps"],
                "training_plant_ids": configuration["plant_ids"],
                "reward_weights": asdict(configuration["weights"]),
                "train_report": train_report,
                "held_out": held_out,
                "held_out_summary": summarize_held_out_metrics(held_out),
            }
            (run_dir / "screening_report.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            runs.append(run)
            print(json.dumps({"event": "run_finished", "run_id": run_id, "held_out_summary": run["held_out_summary"]}, ensure_ascii=False), flush=True)
    summary = {"gpu": torch.cuda.get_device_name(0), "library": str(library), "correction_ratio": correction_ratio, "held_out_plant_ids": held_out_ids, "runs": runs}
    destination = root / "results/GPU批量SAC筛选报告.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "batch_finished", "report": str(destination), "run_count": len(runs)}, ensure_ascii=False), flush=True)
