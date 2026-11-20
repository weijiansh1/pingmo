"""Fast linear / MLP delay probes, without training an autoencoder.

    cd pingmo
    python experiments/sae_delay_20261023/probe_linear.py --data .../results_large

This is the gate for the SAE experiment: if neither the raw 32x5 history nor
scale-invariant cross-correlation features can recover tau_p on held-out plants,
a sparse code of the same window is unlikely to do better.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe as P  # noqa: E402
import common  # noqa: E402

torch.set_num_threads(8)


class ProbeMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp_probe(x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray,
                    seed: int = 0, epochs: int = 80) -> tuple[ProbeMLP, dict]:
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    model = ProbeMLP(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = 512
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(epochs):
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), batch):
            idx = perm[start:start + batch]
            opt.zero_grad()
            loss = (model(xt[idx]) - yt[idx]).pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val = float((model(xv) - yv).pow(2).mean())
        if val < best - 1e-9:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"val_mse": best}


def predict_mlp(model: ProbeMLP, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy()


def plant_aggregate(x: np.ndarray, plant_index: np.ndarray, n_plants: int) -> np.ndarray:
    out = np.zeros((n_plants, x.shape[1]), dtype=np.float64)
    count = np.zeros(n_plants, dtype=np.float64)
    np.add.at(out, plant_index, x)
    np.add.at(count, plant_index, 1.0)
    return out / np.maximum(count[:, None], 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "results_large")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    data = P.load_dataset(args.data)
    arrays, metas = data["arrays"], data["metas"]
    masks = P.split_masks(arrays, metas)
    history = P.flatten_history(arrays["history"])
    mu, sd = P.standardizer(history[masks["train"]])
    x_all = P.apply_standardizer(history, mu, sd)
    components, eigenvalues = P.pca_fit(x_all[masks["train"]], 16)

    features = {
        "current": arrays["current"].astype(np.float32),
        "raw_history": history.astype(np.float32),
        "handcrafted_xcorr": P.handcrafted_features(arrays["history"]),
        "pca16": (x_all @ components.T).astype(np.float32),
    }
    y = arrays["tau"].astype(np.float64)
    logy = np.log(np.maximum(y, 1e-6))

    out = {"n_train": int(masks["train"].sum()), "n_val": int(masks["val"].sum()),
           "n_test": int(masks["test"].sum()), "linear": {}, "mlp": {},
           "plant_level": {}}
    for target_name, target in (("tau", y), ("log_tau", logy)):
        table = P.probe_table(features, arrays, masks, target=target_name)
        out["linear"][target_name] = {
            name: {k: v for k, v in row.items() if not k.startswith("_")}
            for name, row in table.items()
        }
        for name in ("raw_history", "handcrafted_xcorr", "pca16", "current"):
            x = features[name].astype(np.float64)
            alpha, _ = P.choose_alpha(x[masks["train"]], target[masks["train"]],
                                      x[masks["val"]], target[masks["val"]])
            params = P.ridge_fit(x[masks["train"]], target[masks["train"]], alpha)
            pred = P.ridge_predict(x[masks["test"]], params)
            yt = target[masks["test"]]
            out["linear"][target_name][name].update({
                "test_r2": P.r2_score(yt, pred),
                "test_mae_ms": P.mae(yt, pred) * 1000.0,
                "test_spearman": P.spearman(yt, pred),
            })
            print(f"linear {target_name:8s} {name:20s} R2={out['linear'][target_name][name]['test_r2']:.3f} "
                  f"MAE={out['linear'][target_name][name]['test_mae_ms']:.1f}ms")

    # MLP probe on the three high-dimensional representations.
    if not args.skip_mlp:
        for name in ("raw_history", "handcrafted_xcorr", "pca16", "current"):
            x = features[name].astype(np.float64)
            xm, xs = x[masks["train"]].mean(0), x[masks["train"]].std(0) + 1e-6
            xn = (x - xm) / xs
            model, stats = train_mlp_probe(xn[masks["train"]], y[masks["train"]],
                                           xn[masks["val"]], y[masks["val"]],
                                           seed=args.seed, epochs=args.epochs)
            pred = predict_mlp(model, xn[masks["test"]])
            out["mlp"][name] = {
                "dim": int(x.shape[1]), "val_mse": stats["val_mse"],
                "test_r2": P.r2_score(y[masks["test"]], pred),
                "test_mae_ms": P.mae(y[masks["test"]], pred) * 1000.0,
                "test_spearman": P.spearman(y[masks["test"]], pred),
            }
            print(f"mlp    tau      {name:20s} R2={out['mlp'][name]['test_r2']:.3f} "
                  f"MAE={out['mlp'][name]['test_mae_ms']:.1f}ms")

    # Plant-level means: one row per plant, evaluated with leave-one-out-style
    # ridge on held-out plants.  This removes the within-plant window weighting
    # that can make window-level R2 look better or worse than the plant ranking.
    n_plants = len(metas)
    plant_index = arrays["plant_index"]
    plant_tau = np.array([m["tau_p"] for m in metas])
    plant_split = np.array([m["split"] for m in metas])
    for name in ("raw_history", "handcrafted_xcorr", "pca16"):
        agg = plant_aggregate(features[name].astype(np.float64), plant_index, n_plants)
        tr = np.isin(plant_split, ("train_core", "validation"))
        te = plant_split == "id_test"
        alpha, _ = P.choose_alpha(agg[plant_split == "train_core"], plant_tau[plant_split == "train_core"],
                                  agg[plant_split == "validation"], plant_tau[plant_split == "validation"])
        params = P.ridge_fit(agg[tr], plant_tau[tr], alpha)
        pred = P.ridge_predict(agg[te], params)
        out["plant_level"][name] = {
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "test_r2": P.r2_score(plant_tau[te], pred),
            "test_mae_ms": P.mae(plant_tau[te], pred) * 1000.0,
            "test_spearman": P.spearman(plant_tau[te], pred),
        }
        print(f"plant  tau      {name:20s} R2={out['plant_level'][name]['test_r2']:.3f} "
              f"MAE={out['plant_level'][name]['test_mae_ms']:.1f}ms")

    out["pca_explained"] = float(eigenvalues.sum() / np.var(x_all[masks["train"]], axis=0).sum())
    (args.data / "probe_linear.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"wrote {args.data / 'probe_linear.json'}")


if __name__ == "__main__":
    main()
