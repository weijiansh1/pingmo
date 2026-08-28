"""Run the explicitly non-formal sequential SAC screening batch on one GPU."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
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


SCREENING_SEEDS = (20260828, 20260829)
HELD_OUT_PLANT_IDS = [f"id_test-{index:04d}" for index in range(2100, 2106)]
CORRECTION_RATIO = 0.3


@dataclass(frozen=True)
class ScreeningRun:
    run_id: str
    configuration_id: str
    seed: int
    timesteps: int
    plant_ids: list[str]
    reward_weights: RewardWeights


def _evenly_spaced_train_ids(library: Path, count: int = 16) -> list[str]:
    rows = [json.loads(line) for line in library.read_text(encoding="utf-8").splitlines()]
    plant_ids = [row["plant_id"] for row in rows if row["split"] == "train_core"]
    indices = np.linspace(0, len(plant_ids) - 1, count, dtype=int)
    return [plant_ids[index] for index in indices]


def screening_configurations(library: Path) -> tuple[dict[str, object], ...]:
    multi_ids = _evenly_spaced_train_ids(library)
    current = RewardWeights()
    strong = RewardWeights(action_energy=0.20, action_delta=0.15)
    return (
        {"id": "single-40k", "plant_ids": ["train_core-0000"], "timesteps": 40_000, "weights": current},
        {"id": "single-120k", "plant_ids": ["train_core-0000"], "timesteps": 120_000, "weights": current},
        {"id": "multi-40k", "plant_ids": multi_ids, "timesteps": 40_000, "weights": current},
        {"id": "multi-strong-40k", "plant_ids": multi_ids, "timesteps": 40_000, "weights": strong},
    )


def resolve_screening_run(run_id: str, library: Path) -> ScreeningRun:
    for configuration in screening_configurations(library):
        for seed in SCREENING_SEEDS:
            if run_id == f"{configuration['id']}-seed-{seed}":
                return ScreeningRun(
                    run_id=run_id,
                    configuration_id=str(configuration["id"]),
                    seed=seed,
                    timesteps=int(configuration["timesteps"]),
                    plant_ids=list(configuration["plant_ids"]),
                    reward_weights=configuration["weights"],
                )
    raise ValueError(f"unknown screening run ID: {run_id}")


def _evaluate(model: object, library: Path, plant_ids: list[str], correction_ratio: float, reward_weights: RewardWeights) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for plant_id in plant_ids:
        raw_env = build_fixed_env(library, plant_id, horizon_steps=500, correction_ratio=correction_ratio, pilot_signal="step", reward_weights=reward_weights)
        sac_env = build_fixed_env(library, plant_id, horizon_steps=500, correction_ratio=correction_ratio, pilot_signal="step", reward_weights=reward_weights)
        raw = collect_response_trace(_ZeroPolicy(), raw_env, seed=20260827)
        sac = collect_response_trace(model, sac_env, seed=20260827)
        records.append({"plant_id": plant_id, "raw": response_metrics(raw), "sac": response_metrics(sac)})
    return records


def execute_screening_run(run: ScreeningRun, library: Path, output_root: Path) -> tuple[dict[str, object], bool]:
    run_dir = output_root / run.run_id
    completed = load_completed_screening_report(run_dir)
    if completed is not None:
        return completed, True

    from stable_baselines3 import SAC

    train_report = train_short_experiment(
        library,
        run.plant_ids[0],
        run_dir,
        timesteps=run.timesteps,
        seed=run.seed,
        device="cuda",
        correction_ratio=CORRECTION_RATIO,
        plant_ids=run.plant_ids,
        reward_weights=run.reward_weights,
    )
    model = SAC.load(run_dir / "fixed_plant_sac", device="cuda")
    held_out = _evaluate(model, library, HELD_OUT_PLANT_IDS, CORRECTION_RATIO, run.reward_weights)
    report = {
        "run_id": run.run_id,
        "configuration_id": run.configuration_id,
        "seed": run.seed,
        "timesteps": run.timesteps,
        "training_plant_ids": run.plant_ids,
        "reward_weights": asdict(run.reward_weights),
        "train_report": train_report,
        "held_out": held_out,
        "held_out_summary": summarize_held_out_metrics(held_out),
    }
    (run_dir / "screening_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report, False


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("this screening batch requires CUDA")
    root = Path(__file__).resolve().parents[1]
    library = root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
    load_persisted_records(library, HELD_OUT_PLANT_IDS)
    configurations = screening_configurations(library)
    output_root = root / "checkpoints/gpu_sac_screening_batch"
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for configuration in configurations:
        for seed in SCREENING_SEEDS:
            run_id = f"{configuration['id']}-seed-{seed}"
            run = resolve_screening_run(run_id, library)
            if load_completed_screening_report(output_root / run_id) is not None:
                report, skipped = execute_screening_run(run, library, output_root)
                assert skipped
                runs.append(report)
                print(json.dumps({"event": "run_skipped", "run_id": run_id}, ensure_ascii=False), flush=True)
                continue
            print(json.dumps({"event": "run_started", "run_id": run_id}, ensure_ascii=False), flush=True)
            report, skipped = execute_screening_run(run, library, output_root)
            runs.append(report)
            if skipped:
                print(json.dumps({"event": "run_skipped", "run_id": run_id}, ensure_ascii=False), flush=True)
                continue
            print(json.dumps({"event": "run_finished", "run_id": run_id, "held_out_summary": report["held_out_summary"]}, ensure_ascii=False), flush=True)
    summary = {"gpu": torch.cuda.get_device_name(0), "library": str(library), "correction_ratio": CORRECTION_RATIO, "held_out_plant_ids": HELD_OUT_PLANT_IDS, "runs": runs}
    destination = root / "results/GPU批量SAC筛选报告.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "batch_finished", "report": str(destination), "run_count": len(runs)}, ensure_ascii=False), flush=True)
