"""Probe transport delay from causal episode-level averages of each representation.

    cd pingmo
    python experiments/sae_delay_20261023/probe_episode.py

A policy cannot use the whole episode mean at t=1 s, so the probe only averages
windows up to the decision time (``step <= threshold``).  The target is tau_p;
splits are plant-disjoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe as P  # noqa: E402
import common  # noqa: E402

DATA = HERE / "results_large"
FEAT = DATA / "features"
TIMES = (1.2, 1.6, 2.0, 3.0, 4.0, 6.0, 7.5)


def episode_index(plant_index: np.ndarray, command: np.ndarray) -> np.ndarray:
    return plant_index.astype(np.int64) * 8 + command.astype(np.int64)


def aggregate(features: np.ndarray, ep: np.ndarray, n_ep: int,
              mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((n_ep, features.shape[1]), dtype=np.float64)
    count = np.zeros(n_ep, dtype=np.float64)
    np.add.at(out, ep[mask], features[mask])
    np.add.at(count, ep[mask], 1.0)
    return out / np.maximum(count[:, None], 1.0), count


def load_features() -> dict[str, np.ndarray]:
    meta = json.loads((FEAT / "meta.json").read_text(encoding="utf-8"))
    feats = {}
    for name in meta["models"]:
        feats[name] = np.load(FEAT / f"{name}.npy").astype(np.float64)
    return feats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()

    data = P.load_dataset(args.data)
    arrays, metas = data["arrays"], data["metas"]
    n_plants = len(metas)
    n_ep = n_plants * 8
    plant_index = arrays["plant_index"]
    command = arrays["command"]
    step = arrays["step"]
    ep = episode_index(plant_index, command)
    tau = np.array([m["tau_p"] for m in metas])
    split = np.array([m["split"] for m in metas])
    ep_tau = np.repeat(tau, 8)
    ep_split = np.repeat(split, 8)
    tr_plants = split == "train_core"
    va_plants = split == "validation"
    te_plants = split == "id_test"

    feats = load_features()
    # Add the raw history and a 5-channel mean as references.
    hist = P.flatten_history(arrays["history"]).astype(np.float64)
    feats["raw_history"] = hist
    feats["mean5"] = arrays["history"].astype(np.float64).mean(1)
    feats["handcrafted_xcorr"] = P.handcrafted_features(arrays["history"]).astype(np.float64)

    results = {"times_s": list(TIMES), "n_episodes": int(n_ep),
               "n_train_plants": int(tr_plants.sum()),
               "n_test_plants": int(te_plants.sum()), "probes": {}}
    onset_step = 60
    first_recorded = int(step[step >= onset_step].min())
    for t in TIMES:
        thr = max(min(int(round(t / 0.02)), int(step.max())), first_recorded)
        mask = (step >= onset_step) & (step <= thr)
        if not mask.any():
            continue
        results["probes"][f"{t:.1f}s"] = {}
        for name, x in feats.items():
            agg, count = aggregate(x, ep, n_ep, mask)
            valid = count > 0
            tr = np.repeat(tr_plants, 8) & valid
            va = np.repeat(va_plants, 8) & valid
            te = np.repeat(te_plants, 8) & valid
            alpha, _ = P.choose_alpha(agg[tr], ep_tau[tr], agg[va], ep_tau[va])
            params = P.ridge_fit(agg[tr], ep_tau[tr], alpha)
            pred = P.ridge_predict(agg[te], params)
            y = ep_tau[te]
            bins = common.delay_bin_index(y)
            pbins = common.delay_bin_index(pred)
            results["probes"][f"{t:.1f}s"][name] = {
                "dim": int(x.shape[1]), "alpha": alpha,
                "test_r2": P.r2_score(y, pred),
                "test_mae_ms": P.mae(y, pred) * 1000.0,
                "test_spearman": P.spearman(y, pred),
                "bin_accuracy": float((bins == pbins).mean()),
                "n_train": int(tr.sum()), "n_test": int(te.sum()),
            }
        best = sorted(results["probes"][f"{t:.1f}s"].items(),
                      key=lambda kv: kv[1]["test_r2"], reverse=True)[:3]
        print(f"{t:.1f}s  " + "  ".join(f"{k}={v['test_r2']:.3f}" for k, v in best))

    # Which SAE units are delay-tuned?  Correlate each episode-mean unit at 2 s
    # with tau on training plants only (descriptive, not a probe).
    thr = int(round(2.0 / 0.02))
    mask = (step >= onset_step) & (step <= thr)
    tuning = {}
    for name in ("topk4", "l1_3e-03", "l1_3e-02", "pca16", "dense16"):
        if name not in feats:
            continue
        agg, count = aggregate(feats[name], ep, n_ep, mask)
        valid = np.repeat(tr_plants, 8) & (count > 0)
        corr = []
        for j in range(agg.shape[1]):
            c = np.corrcoef(agg[valid, j], ep_tau[valid])[0, 1]
            corr.append(float(c) if np.isfinite(c) else 0.0)
        order = np.argsort(np.abs(corr))[::-1][:8]
        tuning[name] = [{"unit": int(j), "pearson_r": corr[j]} for j in order]
    results["delay_tuning_at_2s"] = tuning

    (args.data / "probe_episode.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.data / 'probe_episode.json'}")


if __name__ == "__main__":
    main()
