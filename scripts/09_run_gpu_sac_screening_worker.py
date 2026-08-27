"""Run one frozen GPU SAC screening experiment without writing the aggregate report."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def _load_batch_module(root: Path):
    script = root / "scripts/08_run_gpu_sac_screening_batch.py"
    spec = importlib.util.spec_from_file_location("gpu_sac_screening_batch", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load screening batch script: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    batch = _load_batch_module(root)
    library = root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
    try:
        run = batch.resolve_screening_run(args.run_id, library)
    except ValueError as error:
        parser.error(str(error))
    if not torch.cuda.is_available():
        raise RuntimeError("this screening worker requires CUDA")

    batch.load_persisted_records(library, batch.HELD_OUT_PLANT_IDS)
    output_root = root / "checkpoints/gpu_sac_screening_batch"
    output_root.mkdir(parents=True, exist_ok=True)
    report, skipped = batch.execute_screening_run(run, library, output_root)
    event = "run_skipped" if skipped else "run_finished"
    print(json.dumps({"event": event, "run_id": run.run_id, "held_out_summary": report["held_out_summary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
