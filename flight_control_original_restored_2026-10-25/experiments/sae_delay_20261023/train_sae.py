"""Train sparse / dense autoencoders on closed-loop history windows.

    cd pingmo
    python experiments/sae_delay_20261023/train_sae.py --data .../results_large --suite

The encoder is trained only on `train_core` plants; `validation` plants select
the early-stopping epoch and the L1 strength.  `id_test` plants are never seen.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe as P  # noqa: E402
import common  # noqa: E402

torch.set_num_threads(8)


class SAE(nn.Module):
    def __init__(self, in_dim: int = 160, hidden: int = 64, latent: int = 16,
                 mode: str = "l1", topk: int = 4) -> None:
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


def load_xy(data_dir: Path, max_train: int = 180_000, seed: int = 0):
    data = P.load_dataset(data_dir)
    arrays, metas = data["arrays"], data["metas"]
    masks = P.split_masks(arrays, metas)
    history = P.flatten_history(arrays["history"]).astype(np.float32)
    mu, sd = P.standardizer(history[masks["train"]])
    x = P.apply_standardizer(history, mu, sd).astype(np.float32)
    rng = np.random.default_rng(seed)
    train_idx = np.where(masks["train"])[0]
    if len(train_idx) > max_train:
        train_idx = rng.choice(train_idx, size=max_train, replace=False)
    val_idx = np.where(masks["val"])[0]
    return {
        "data": data, "masks": masks, "x": x, "mu": mu.astype(np.float32),
        "sd": sd.astype(np.float32), "train_idx": train_idx, "val_idx": val_idx,
    }


def train_one(x: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
              mode: str, lam: float, topk: int, latent: int = 16,
              hidden: int = 64, epochs: int = 20, batch: int = 1024,
              seed: int = 0) -> tuple[SAE, dict]:
    torch.manual_seed(seed)
    model = SAE(mode=mode, topk=topk, latent=latent, hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    xt = torch.tensor(x[train_idx], dtype=torch.float32)
    xv = torch.tensor(x[val_idx], dtype=torch.float32)
    best = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve = []
    for epoch in range(epochs):
        perm = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(xt), batch):
            idx = perm[start:start + batch]
            xb = xt[idx]
            recon, z = model(xb)
            loss = (recon - xb).pow(2).mean()
            if mode == "l1":
                loss = loss + lam * z.abs().sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        with torch.no_grad():
            recon, z = model(xv)
            val_mse = float((recon - xv).pow(2).mean())
            l0 = float((z.abs() > 1e-6).float().sum(1).mean())
        curve.append({"epoch": epoch + 1, "train_loss": total / len(xt),
                      "val_mse": val_mse, "l0": l0})
        if val_mse < best - 1e-7:
            best = val_mse
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        recon, z = model(xt)
        train_mse = float((recon - xt).pow(2).mean())
        l0 = float((z.abs() > 1e-6).float().sum(1).mean())
        active_fraction = float((z.abs() > 1e-6).float().mean())
        # Per-latent variance and the fraction of windows in which each unit fires.
        fire = (z.abs() > 1e-6).float().mean(0)
        var = z.var(0)
    stats = {
        "mode": mode, "lambda": lam, "topk": topk, "latent": latent,
        "hidden": hidden, "epochs": epochs, "best_epoch": best_epoch,
        "train_mse": train_mse, "val_mse": best, "l0": l0,
        "active_fraction": active_fraction,
        "fire_fraction": [float(v) for v in fire],
        "latent_variance": [float(v) for v in var],
        "curve": curve,
    }
    return model, stats


def save_model(path: Path, model: SAE, stats: dict, mu: np.ndarray,
               sd: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "stats": stats, "mu": mu, "sd": sd},
               path)
    print(f"  saved {path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HERE / "results_large")
    ap.add_argument("--suite", action="store_true",
                    help="train dense, L1 sweep, and top-k variants")
    ap.add_argument("--mode", choices=("dense", "l1", "topk"), default="l1")
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-train", type=int, default=180_000)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    t0 = time.time()
    xy = load_xy(args.data, max_train=args.max_train, seed=args.seed)
    model_dir = args.data / "models"
    summary = {}
    if args.suite:
        jobs = [("dense16", "dense", 0.0, 0)]
        for lam in (3e-4, 3e-3, 3e-2):
            jobs.append((f"l1_{lam:.0e}", "l1", lam, 0))
        jobs += [("topk4", "topk", 0.0, 4), ("topk8", "topk", 0.0, 8)]
    else:
        jobs = [(f"{args.mode}_{args.lam:.0e}" if args.mode == "l1" else
                 f"{args.mode}{args.topk}", args.mode, args.lam, args.topk)]
    for name, mode, lam, topk in jobs:
        print(f"training {name} mode={mode} lam={lam} topk={topk}", flush=True)
        model, stats = train_one(xy["x"], xy["train_idx"], xy["val_idx"],
                                 mode=mode, lam=lam, topk=topk,
                                 epochs=args.epochs, seed=args.seed)
        stats["name"] = name
        save_model(model_dir / f"{name}.pt", model, stats, xy["mu"], xy["sd"])
        summary[name] = stats
        print(f"  val_mse={stats['val_mse']:.6f} l0={stats['l0']:.2f} "
              f"best_epoch={stats['best_epoch']}", flush=True)
    (args.data / "sae_training.json").write_text(
        json.dumps({"n_train_windows": int(len(xy["train_idx"])),
                    "n_val_windows": int(len(xy["val_idx"])),
                    "seed": args.seed, "models": summary},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
