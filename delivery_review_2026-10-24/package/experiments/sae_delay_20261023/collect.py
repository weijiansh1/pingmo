"""Collect closed-loop history windows from the tiny Student.

    cd pingmo
    python experiments/sae_delay_20261023/collect.py --n-train 30 --n-val 15 --n-test 20

The script writes one compressed ``.npz`` per plant plus a combined
``dataset/`` directory of ``.npy`` arrays.  Splits are plant-disjoint:
``train_core`` / ``validation`` / ``id_test`` from the IV-A library.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

_MODEL = None


def _init_worker() -> None:
    global _MODEL
    torch.set_num_threads(1)
    _MODEL = common.load_tiny()


def _work(args: tuple[dict, str, int]) -> dict:
    row, out_dir, seed = args
    result = common.run_rollout(_MODEL, row, seed=seed, record=True)
    arrays = {k: result[k] for k in
              ("history", "current", "older", "action", "step", "command")}
    tau = float(row["parameters"]["tau_p"])
    arrays["tau"] = np.full(len(result["action"]), tau, dtype=np.float32)
    path = Path(out_dir) / f"{row['plant_id']}.npz"
    np.savez_compressed(path, **arrays)
    return {
        "plant_id": row["plant_id"],
        "split": row["split"],
        "level": int(row["gjb_level"]),
        "tau_p": tau,
        "delay_bin": common.delay_bin(tau),
        "n_windows": int(len(result["action"])),
        **common.summarize_metrics(result["metrics"]),
    }


def _combine(out_dir: Path, metas: list[dict]) -> dict:
    metas = sorted(metas, key=lambda m: m["plant_id"])
    arrays = {name: [] for name in
              ("history", "current", "older", "action", "tau", "plant_index",
               "command", "step")}
    for i, meta in enumerate(metas):
        data = np.load(out_dir / "plants" / f"{meta['plant_id']}.npz")
        n = len(data["action"])
        arrays["history"].append(data["history"])
        arrays["current"].append(data["current"])
        arrays["older"].append(data["older"])
        arrays["action"].append(data["action"])
        arrays["tau"].append(data["tau"])
        arrays["plant_index"].append(np.full(n, i, dtype=np.int32))
        arrays["command"].append(data["command"])
        arrays["step"].append(data["step"])
    combined = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    ds = out_dir / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    for name, value in combined.items():
        np.save(ds / f"{name}.npy", value)
    (out_dir / "plants.json").write_text(json.dumps(metas, ensure_ascii=False, indent=1),
                                         encoding="utf-8")

    # Baseline summary of the frozen tiny Student on this sample.
    by_bin: dict[str, list[dict]] = {}
    for meta in metas:
        by_bin.setdefault(meta["delay_bin"], []).append(meta)
    summary = {}
    for name, group in by_bin.items():
        rm = np.array([m["rmse_by_command"] for m in group])
        pk = np.array([m["peak_by_command"] for m in group])
        tv = np.array([m["tv_by_command"] for m in group])
        ok = (rm <= 1.5) & (pk <= 5.0) & (tv <= 12.0)
        summary[name] = {
            "n_plants": len(group),
            "tau_median": float(np.median([m["tau_p"] for m in group])),
            "rmse_median": float(np.median(rm)),
            "rmse_p90": float(np.percentile(rm, 90)),
            "peak_median": float(np.median(pk)),
            "tv_median": float(np.median(tv)),
            "pass_rate": float(ok.mean()),
        }
    summary["all"] = {
        "n_plants": len(metas),
        "n_windows": int(len(combined["action"])),
        "rmse_median": float(np.median([m["rmse_median"] for m in metas])),
        "pass_rate": float(np.mean([o for m in metas for o in
                                    (np.array(m["rmse_by_command"]) <= 1.5) &
                                    (np.array(m["peak_by_command"]) <= 5.0) &
                                    (np.array(m["tv_by_command"]) <= 12.0)])),
    }
    (out_dir / "baseline.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=30, help="plants per delay bin")
    ap.add_argument("--n-val", type=int, default=15, help="plants per delay bin")
    ap.add_argument("--n-test", type=int, default=20, help="plants per delay bin")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    rows = common.load_library()
    selected: list[dict] = []
    for split, n in (("train_core", args.n_train),
                     ("validation", args.n_val),
                     ("id_test", args.n_test)):
        selected += common.sample_plants(rows, split, n, args.seed)
    out_dir = args.out.resolve()
    (out_dir / "plants").mkdir(parents=True, exist_ok=True)
    print(f"selected {len(selected)} plants; writing to {out_dir}", flush=True)

    t0 = time.time()
    jobs = [(row, str(out_dir / "plants"), args.seed) for row in selected]
    metas: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as pool:
        for i, meta in enumerate(pool.map(_work, jobs), 1):
            metas.append(meta)
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} plants, {time.time() - t0:.1f}s", flush=True)
    summary = _combine(out_dir, metas)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
