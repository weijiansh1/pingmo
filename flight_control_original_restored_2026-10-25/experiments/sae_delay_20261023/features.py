"""Precompute frozen history features for the policy experiment."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import probe as P  # noqa: E402
import common  # noqa: E402
from train_sae import SAE  # noqa: E402

torch.set_num_threads(8)


def main() -> None:
    data_dir = HERE / "results_large"
    data = P.load_dataset(data_dir)
    arrays = data["arrays"]
    hist = P.flatten_history(arrays["history"]).astype(np.float32)
    hist_neg = -hist
    models_dir = data_dir / "models"
    feat_dir = data_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(models_dir / "dense16.pt", map_location="cpu",
                         weights_only=False)
    mu = np.asarray(payload["mu"], dtype=np.float32)
    sd = np.asarray(payload["sd"], dtype=np.float32)
    np.savez(feat_dir / "standardizer.npz", mu=mu, sd=sd)

    # PCA is fit on train_core only.
    masks = P.split_masks(arrays, data["metas"])
    x_train = ((hist[masks["train"]] - mu) / sd).astype(np.float32)
    cov = np.cov(x_train, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    components = vectors[:, np.argsort(values)[::-1][:16]].T.astype(np.float32)
    np.save(feat_dir / "pca16_components.npy", components)

    meta = {"models": {}, "pca": {"n_components": 16,
                                  "explained": float(values[np.argsort(values)[::-1][:16]].sum()
                                                     / np.var(x_train, axis=0).sum())}}
    x_all = (hist - mu) / sd
    x_neg = (hist_neg - mu) / sd
    z = (x_all @ components.T).astype(np.float32)
    z_neg = (x_neg @ components.T).astype(np.float32)
    np.save(feat_dir / "pca16.npy", z)
    np.save(feat_dir / "pca16_neg.npy", z_neg)
    meta["models"]["pca16"] = {"kind": "pca", "dim": 16,
                               "components": "pca16_components.npy"}

    for path in sorted(models_dir.glob("*.pt")):
        name = path.stem
        if name == "dense16":
            pass
        payload = torch.load(path, map_location="cpu", weights_only=False)
        stats = payload["stats"]
        model = SAE(mode=stats["mode"], topk=stats["topk"],
                    latent=stats["latent"], hidden=stats["hidden"])
        model.load_state_dict(payload["state"])
        model.eval()
        with torch.no_grad():
            z = model.encode(torch.tensor(x_all, dtype=torch.float32)).numpy().astype(np.float32)
            z_neg = model.encode(torch.tensor(x_neg, dtype=torch.float32)).numpy().astype(np.float32)
        np.save(feat_dir / f"{name}.npy", z)
        np.save(feat_dir / f"{name}_neg.npy", z_neg)
        meta["models"][name] = {
            "kind": stats["mode"], "dim": int(stats["latent"]),
            "model": str(path.relative_to(data_dir)),
            "l0": float(stats["l0"]), "val_mse": float(stats["val_mse"]),
        }
        print(f"features {name}: {z.shape} l0={stats['l0']:.2f}", flush=True)

    (feat_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"wrote {feat_dir}")


if __name__ == "__main__":
    main()
