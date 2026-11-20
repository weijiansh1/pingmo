"""Train sparse autoencoders on the 32x5 history and probe transport delay.

    cd pingmo
    python experiments/sae_delay_20261023/probe.py --data experiments/sae_delay_20261023/results

Outputs ``probe_results.json`` next to the dataset.  The question is deliberately
narrow: can a linear read-out recover ``tau_p`` from a 16-D history code, and does
sparsity cost anything relative to PCA / a dense autoencoder / the raw history?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402

torch.set_num_threads(8)


# --------------------------------------------------------------------------- data
def load_dataset(data_dir: Path) -> dict:
    ds = data_dir / "dataset"
    arrays = {name: np.load(ds / f"{name}.npy") for name in
              ("history", "current", "older", "action", "tau", "plant_index",
               "command", "step")}
    metas = json.loads((data_dir / "plants.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "metas": metas, "data_dir": data_dir}


def split_masks(arrays: dict, metas: list[dict]) -> dict[str, np.ndarray]:
    plant_split = np.array([m["split"] for m in metas])
    idx = arrays["plant_index"]
    return {
        "train": plant_split[idx] == "train_core",
        "val": plant_split[idx] == "validation",
        "test": plant_split[idx] == "id_test",
    }


def flatten_history(history: np.ndarray) -> np.ndarray:
    return history.reshape(len(history), -1)


def standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(0)
    sd = x.std(0) + 1e-6
    return mu, sd


def apply_standardizer(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (x - mu) / sd


# ------------------------------------------------------------------- handcrafted
def handcrafted_features(history: np.ndarray) -> np.ndarray:
    """Cross-correlation summaries between requested force and roll-rate output."""
    req = history[:, :, 4]
    out = history[:, :, 2]
    cmd = history[:, :, 0]
    feats = []
    for lag in range(0, 21):
        a = req[:, :32 - lag]
        b = out[:, lag:]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        denom = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-8
        feats.append((a * b).sum(1) / denom)
    corr = np.stack(feats, axis=1)
    cmd_corr = []
    for lag in range(0, 11):
        a = cmd[:, :32 - lag]
        b = out[:, lag:]
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        denom = np.sqrt((a * a).sum(1) * (b * b).sum(1)) + 1e-8
        cmd_corr.append((a * b).sum(1) / denom)
    cmd_corr = np.stack(cmd_corr, axis=1)
    argmax = corr.argmax(1).astype(np.float32)
    max_corr = corr.max(1)
    lag0 = corr[:, 0]
    return np.concatenate([
        corr,
        cmd_corr,
        argmax[:, None] / 20.0,
        max_corr[:, None],
        lag0[:, None],
        (max_corr - lag0)[:, None],
    ], axis=1).astype(np.float32)


# ----------------------------------------------------------------------- models
class MLPAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 160, hidden: int = 64, latent: int = 16,
                 mode: str = "dense", topk: int = 4) -> None:
        super().__init__()
        self.mode = mode
        self.topk = topk
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.mode == "l1":
            return torch.relu(z)
        if self.mode == "topk":
            z = torch.relu(z)
            values, indices = torch.topk(z, self.topk, dim=1)
            out = torch.zeros_like(z)
            out.scatter_(1, indices, values)
            return out
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


def train_autoencoder(x_train: np.ndarray, x_val: np.ndarray, mode: str,
                      lam: float = 0.0, topk: int = 4, seed: int = 0,
                      epochs: int = 30) -> tuple[MLPAutoencoder, dict]:
    torch.manual_seed(seed)
    xt = torch.tensor(x_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)
    model = MLPAutoencoder(mode=mode, topk=topk)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    batch = 512
    n = len(xt)
    best = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    history = []
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            xb = xt[idx]
            recon, z = model(xb)
            loss = (recon - xb).pow(2).mean()
            if mode == "l1":
                loss = loss + lam * z.abs().sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        with torch.no_grad():
            recon, z = model(xv)
            val_mse = float((recon - xv).pow(2).mean())
            active = (z.abs() > 1e-6).float().mean().item()
        history.append({"epoch": epoch + 1, "train_loss": total / n,
                        "val_mse": val_mse, "active_fraction": active})
        if val_mse < best - 1e-6:
            best = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon, z = model(xt)
        train_mse = float((recon - xt).pow(2).mean())
        active = (z.abs() > 1e-6).float().mean().item()
        l0 = (z.abs() > 1e-6).float().sum(1).mean().item()
    stats = {"mode": mode, "lambda": lam, "topk": topk, "train_mse": train_mse,
             "val_mse": best, "active_fraction": active, "l0": l0,
             "history": history}
    return model, stats


def encode(model: MLPAutoencoder, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        z = model.encode(torch.tensor(x, dtype=torch.float32))
    return z.numpy()


def pca_fit(x_train: np.ndarray, k: int = 16) -> tuple[np.ndarray, np.ndarray]:
    cov = np.cov(x_train, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][:k]
    components = vectors[:, order].T.astype(np.float32)
    return components, values[order].astype(np.float32)


# ------------------------------------------------------------------------ ridge
def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    xm = x.mean(0)
    ym = float(y.mean())
    xc = x - xm
    a = xc.T @ xc + alpha * np.eye(x.shape[1], dtype=np.float64)
    w = np.linalg.solve(a, xc.T @ (y - ym))
    return w.astype(np.float64), xm, ym


def ridge_predict(x: np.ndarray, params: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    w, xm, ym = params
    return (x - xm) @ w + ym


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(y - pred).mean())


def spearman(y: np.ndarray, pred: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(y, pred).statistic)


def choose_alpha(x_train: np.ndarray, y_train: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray) -> tuple[float, list[dict]]:
    rows = []
    best = (None, float("inf"))
    for alpha in np.logspace(-3, 3, 13):
        params = ridge_fit(x_train, y_train, float(alpha))
        pred = ridge_predict(x_val, params)
        mse = float(np.mean((y_val - pred) ** 2))
        rows.append({"alpha": float(alpha), "val_mse": mse})
        if mse < best[1]:
            best = (float(alpha), mse)
    return best[0], rows


def probe_table(features: dict[str, np.ndarray], arrays: dict, masks: dict,
                target: str = "tau") -> dict:
    y = arrays["tau"].astype(np.float64)
    if target == "log_tau":
        y = np.log(np.maximum(y, 1e-6))
    out = {}
    for name, x in features.items():
        x = x.astype(np.float64)
        alpha, scan = choose_alpha(x[masks["train"]], y[masks["train"]],
                                   x[masks["val"]], y[masks["val"]])
        params = ridge_fit(x[masks["train"]], y[masks["train"]], alpha)
        pred_val = ridge_predict(x[masks["val"]], params)
        pred_test = ridge_predict(x[masks["test"]], params)
        y_test = y[masks["test"]]
        out[name] = {
            "dim": int(x.shape[1]),
            "alpha": alpha,
            "val_r2": r2_score(y[masks["val"]], pred_val),
            "test_r2": r2_score(y_test, pred_test),
            "test_mae_ms": mae(y_test, pred_test) * 1000.0,
            "test_rmse_ms": float(np.sqrt(np.mean((y_test - pred_test) ** 2))) * 1000.0,
            "test_spearman": spearman(y_test, pred_test),
            "_pred_test": pred_test,
            "_y_test": y_test,
        }
    return out


# ------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    data = load_dataset(args.data)
    arrays, metas = data["arrays"], data["metas"]
    masks = split_masks(arrays, metas)
    history = flatten_history(arrays["history"])
    print(f"windows train/val/test = {masks['train'].sum()}/{masks['val'].sum()}/"
          f"{masks['test'].sum()}", flush=True)

    mu, sd = standardizer(history[masks["train"]])
    x_train = apply_standardizer(history[masks["train"]], mu, sd)
    x_val = apply_standardizer(history[masks["val"]], mu, sd)
    x_test = apply_standardizer(history[masks["test"]], mu, sd)

    features: dict[str, np.ndarray] = {
        "current": arrays["current"],
        "raw_history": history,
        "handcrafted_xcorr": handcrafted_features(arrays["history"]),
    }

    # PCA
    components, eigenvalues = pca_fit(x_train, 16)
    features["pca16"] = x_train @ components.T
    pca_test = x_test @ components.T
    pca_val = x_val @ components.T

    # Autoencoders.  Sweep L1 strength, then keep the sparsest model whose
    # validation reconstruction is within 5% of the dense autoencoder.
    t0 = time.time()
    ae_models: dict[str, tuple[MLPAutoencoder, dict]] = {}
    dense, dense_stats = train_autoencoder(x_train, x_val, "dense", epochs=args.epochs,
                                           seed=args.seed)
    ae_models["dense16"] = (dense, dense_stats)
    print(f"dense16 val_mse={dense_stats['val_mse']:.4f} l0={dense_stats['l0']:.2f}",
          flush=True)

    l1_stats = []
    for lam in (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        model, stats = train_autoencoder(x_train, x_val, "l1", lam=lam,
                                         epochs=args.epochs, seed=args.seed)
        l1_stats.append(stats)
        print(f"l1 lam={lam:.0e} val_mse={stats['val_mse']:.4f} l0={stats['l0']:.2f}",
              flush=True)
    eligible = [s for s in l1_stats if s["val_mse"] <= dense_stats["val_mse"] * 1.05]
    if eligible:
        # Prefer L0 nearest to four active units, then lower reconstruction error.
        chosen = min(eligible, key=lambda s: (abs(s["l0"] - 4.0), s["val_mse"]))
    else:
        chosen = min(l1_stats, key=lambda s: s["val_mse"])
    for model, stats in ae_models.values():
        pass
    # Re-train the chosen lambda to keep a handle on the model (cheap, one run).
    chosen_model, chosen_stats = train_autoencoder(
        x_train, x_val, "l1", lam=chosen["lambda"], epochs=args.epochs, seed=args.seed)
    ae_models["sae16"] = (chosen_model, chosen_stats)
    print(f"chosen SAE lambda={chosen['lambda']:.0e} l0={chosen_stats['l0']:.2f}",
          flush=True)

    for name, mode, topk in (("topk4", "topk", 4), ("topk8", "topk", 8)):
        model, stats = train_autoencoder(x_train, x_val, mode, topk=topk,
                                         epochs=args.epochs, seed=args.seed)
        ae_models[name] = (model, stats)
        print(f"{name} val_mse={stats['val_mse']:.4f} l0={stats['l0']:.2f}",
              flush=True)

    x_all = apply_standardizer(history, mu, sd)
    for name, (model, _) in ae_models.items():
        features[name] = encode(model, x_all).astype(np.float32)
    features["pca16"] = (x_all @ components.T).astype(np.float32)

    results = {
        "dataset": str(args.data),
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test": int(masks["test"].sum()),
        "seed": args.seed,
        "pca_explained": float(eigenvalues.sum() / np.var(x_train, axis=0).sum()),
        "autoencoders": {name: {k: v for k, v in stats.items() if k != "history"}
                         for name, (_, stats) in ae_models.items()},
        "probes_tau": {},
        "probes_log_tau": {},
    }
    for target in ("tau", "log_tau"):
        table = probe_table(features, arrays, masks, target=target)
        cleaned = {}
        for name, row in table.items():
            cleaned[name] = {k: v for k, v in row.items() if not k.startswith("_")}
            if target == "tau":
                pred = row["_pred_test"]
                y = row["_y_test"]
                bins = common.delay_bin_index(y)
                pred_bins = common.delay_bin_index(pred)
                cleaned[name]["bin_accuracy"] = float((bins == pred_bins).mean())
                cleaned[name]["bin_confusion"] = [
                    [int(((bins == i) & (pred_bins == j)).sum()) for j in range(3)]
                    for i in range(3)
                ]
        results[f"probes_{target}"] = cleaned
        print(f"\nprobe target={target}")
        for name, row in cleaned.items():
            print(f"  {name:20s} d={row['dim']:3d} R2={row['test_r2']:.3f} "
                  f"MAE={row['test_mae_ms']:.1f}ms rho={row['test_spearman']:.3f}")
    results["elapsed_s"] = time.time() - t0
    (args.data / "probe_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {args.data / 'probe_results.json'} in {results['elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
