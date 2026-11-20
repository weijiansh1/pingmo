"""Train and evaluate policies that read raw history vs frozen SAE features.

    cd pingmo
    python experiments/sae_delay_20261023/train_policy.py train --variants pca16 topk4 aux_topk4

The policy body and head are identical across variants; only the 16-D history
front end changes.  This isolates the representation from policy capacity.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import probe as P  # noqa: E402
from train_sae import SAE  # noqa: E402
from src.teacher.ppo.networks import HistoryConv  # noqa: E402
from src.teacher.residual import ResidualMLPTrunk  # noqa: E402

torch.set_num_threads(8)

DATA = HERE / "results_large"
FEAT = DATA / "features"


# --------------------------------------------------------------------- models
class Frontend(nn.Module):
    def __init__(self, kind: str, mu: np.ndarray, sd: np.ndarray,
                 encoder: nn.Module | None = None,
                 components: np.ndarray | None = None) -> None:
        super().__init__()
        self.kind = kind
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.encoder = encoder
        if components is not None:
            self.register_buffer("components", torch.tensor(components, dtype=torch.float32))
        if encoder is not None:
            for p in encoder.parameters():
                p.requires_grad_(False)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        x = (hist.reshape(-1, 160) - self.mu) / self.sd
        if self.kind == "pca":
            return x @ self.components.T
        return self.encoder.encode(x)


class BodyPolicy(nn.Module):
    def __init__(self, feature_dim: int = 16) -> None:
        super().__init__()
        self.body = ResidualMLPTrunk(feature_dim + 9 + 7, 32, 1)
        self.head = nn.Linear(32, 1, bias=False)

    def forward(self, feat: torch.Tensor, feat_neg: torch.Tensor,
                current: torch.Tensor, older: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.body(torch.cat([feat, current, older], dim=-1)))
        logits_neg = self.head(self.body(torch.cat([feat_neg, -current, -older], dim=-1)))
        return torch.tanh(0.5 * (logits - logits_neg).float())


class OnlinePolicy(nn.Module):
    def __init__(self, body: BodyPolicy, frontend: Frontend) -> None:
        super().__init__()
        self.body = body
        self.frontend = frontend

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        cur, hist, old = common.split_observation(obs)
        feat = self.frontend(hist)
        feat_neg = self.frontend(-hist)
        return self.body(feat, feat_neg, cur, old)


class AuxPolicy(nn.Module):
    """Tiny Student with frozen SAE features appended to the conv features."""

    def __init__(self, frontend: Frontend) -> None:
        super().__init__()
        self.encoder = HistoryConv(width=8)
        self.frontend = frontend
        self.body = ResidualMLPTrunk(16 + 16 + 9 + 7, 32, 1)
        self.head = nn.Linear(32, 1, bias=False)

    def _logits(self, obs: torch.Tensor) -> torch.Tensor:
        cur, hist, old = common.split_observation(obs)
        conv = self.encoder(hist)
        with torch.no_grad():
            z = self.frontend(hist)
        return self.head(self.body(torch.cat([conv, z, cur, old], dim=-1)))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(0.5 * (self._logits(obs) - self._logits(-obs)).float())


def init_aux_from_tiny(aux: AuxPolicy) -> None:
    ck = torch.load(common.TINY_CKPT, map_location="cpu", weights_only=False)
    sd = ck["student"]
    aux.encoder.load_state_dict({k.split(".", 1)[1]: v for k, v in sd.items()
                                 if k.startswith("encoder.")})
    old_w = sd["body.projection.weight"]
    new_w = aux.body.projection.weight.data
    new_w[:, :16] = old_w[:, :16]
    new_w[:, 32:41] = old_w[:, 16:25]
    new_w[:, 41:48] = old_w[:, 25:32]
    new_w[:, 16:32] = 0.0
    aux.body.projection.bias.data.copy_(sd["body.projection.bias"])
    for name, param in aux.body.named_parameters():
        if name.startswith("projection."):
            continue
        param.data.copy_(sd[f"body.{name}"])
    for name, buf in aux.body.named_buffers():
        buf.data.copy_(sd[f"body.{name}"])
    aux.head.weight.data.copy_(sd["head.weight"])


class DaggerPolicy(nn.Module):
    """Mixture of learner and tiny labels used only to collect DAgger states."""

    def __init__(self, learner: nn.Module, labels: nn.Module, beta: float) -> None:
        super().__init__()
        self.learner = learner
        self.labels = labels
        self.beta = beta

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.beta) * self.labels(obs) + self.beta * self.learner(obs)


# ----------------------------------------------------------------------- data
def build_arrays(data_dir: Path) -> dict:
    data = P.load_dataset(data_dir)
    arrays = data["arrays"]
    metas = data["metas"]
    split = np.array([m["split"] for m in metas])
    tau = np.array([m["tau_p"] for m in metas], dtype=np.float64)
    plant_index = arrays["plant_index"]
    window_split = split[plant_index]
    window_tau = tau[plant_index]
    obs = np.concatenate([
        arrays["current"],
        arrays["history"].reshape(len(arrays["history"]), -1),
        arrays["older"],
    ], axis=1).astype(np.float32)
    idx = {
        "train": np.where(window_split == "train_core")[0],
        "val": np.where(window_split == "validation")[0],
        "test": np.where(window_split == "id_test")[0],
    }
    return {"arrays": arrays, "metas": metas, "split": split, "tau": tau,
            "plant_index": plant_index, "obs": obs, "idx": idx,
            "window_tau": window_tau, "action": arrays["action"]}


def load_feature(name: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(FEAT / f"{name}.npy").astype(np.float32)
    zn = np.load(FEAT / f"{name}_neg.npy").astype(np.float32)
    if len(z) != n:
        raise ValueError(f"feature {name} has {len(z)} rows, expected {n}")
    return z, zn


def make_frontend(name: str) -> Frontend:
    meta = json.loads((FEAT / "meta.json").read_text(encoding="utf-8"))
    entry = meta["models"][name]
    std = np.load(FEAT / "standardizer.npz")
    mu, sd = std["mu"], std["sd"]
    if entry["kind"] == "pca":
        components = np.load(FEAT / "pca16_components.npy")
        return Frontend("pca", mu, sd, components=components)
    payload = torch.load(DATA / entry["model"], map_location="cpu", weights_only=False)
    stats = payload["stats"]
    sae = SAE(mode=stats["mode"], topk=stats["topk"], latent=stats["latent"],
              hidden=stats["hidden"])
    sae.load_state_dict(payload["state"])
    sae.eval()
    return Frontend(stats["mode"], mu, sd, encoder=sae)


# -------------------------------------------------------------------- training
def _train_loop(model: nn.Module, batches: dict, train_idx: np.ndarray,
                val_idx: np.ndarray, epochs: int, batch: int = 1024,
                lr: float = 1e-3) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    forward_keys = [k for k in batches if k != "action"]

    def run(indices):
        perm = torch.randperm(len(indices))
        total = 0.0
        for start in range(0, len(indices), batch):
            take = indices[perm[start:start + batch]]
            pred = model(*[batches[k][take] for k in forward_keys]).squeeze(-1)
            target = batches["action"][take]
            loss = (pred - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(take)
        return total / len(indices)
    best = float("inf")
    best_epoch = -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve = []
    for epoch in range(epochs):
        tr = run(train_idx)
        with torch.no_grad():
            pred = model(*[batches[k][val_idx] for k in forward_keys]).squeeze(-1)
            va = float((pred - batches["action"][val_idx]).pow(2).mean())
        curve.append({"epoch": epoch + 1, "train_mse": tr, "val_mse": va})
        if va < best - 1e-9:
            best = va
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return {"best_epoch": best_epoch, "val_mse": best, "curve": curve}


def train_variant(variant: str, data: dict, epochs: int, source_only: bool = False,
                  init_tiny: bool = False) -> tuple[nn.Module, dict, str]:
    idx = data["idx"]
    train_idx, val_idx = idx["train"], idx["val"]
    if source_only:
        keep_tr = data["window_tau"][train_idx] <= 0.15
        keep_va = data["window_tau"][val_idx] <= 0.15
        train_idx, val_idx = train_idx[keep_tr], val_idx[keep_va]
    obs = torch.tensor(data["obs"])
    action = torch.tensor(data["action"], dtype=torch.float32)
    if variant in ("raw", "raw_ft"):
        model = common.TinyStudent()
        if variant == "raw_ft" or init_tiny:
            ck = torch.load(common.TINY_CKPT, map_location="cpu", weights_only=False)
            model.load_state_dict(ck["student"])
        stats = _train_loop(model, {"obs": obs, "action": action}, train_idx, val_idx,
                            epochs=epochs, batch=1024,
                            lr=5e-4 if variant == "raw_ft" else 1e-3)
        return model, stats, "tiny"
    if variant.startswith("aux_"):
        feature = variant[4:]
        frontend = make_frontend(feature)
        model = AuxPolicy(frontend)
        if init_tiny:
            init_aux_from_tiny(model)
        stats = _train_loop(model, {"obs": obs, "action": action}, train_idx, val_idx,
                            epochs=epochs, batch=1024, lr=5e-4)
        return model, stats, "aux"
    # Frozen feature variants: precompute features and train the body only.
    feature = variant
    z, zn = load_feature(feature, len(obs))
    body = BodyPolicy()
    # The loop indexes batches with global indices, so build full-size tensors.
    full = {
        "feat": torch.tensor(z), "feat_neg": torch.tensor(zn),
        "current": torch.tensor(data["arrays"]["current"]),
        "older": torch.tensor(data["arrays"]["older"]),
        "action": action,
    }
    stats = _train_loop(body, full, train_idx, val_idx, epochs=epochs, batch=2048)
    return body, stats, "feature"


# ------------------------------------------------------------------ evaluation
_EVAL_MODEL = None
_EVAL_CONFIG = None


def _init_eval(config: dict) -> None:
    global _EVAL_MODEL, _EVAL_CONFIG
    torch.set_num_threads(1)
    _EVAL_CONFIG = config
    _EVAL_MODEL = build_model(config)


def build_model(config: dict) -> nn.Module:
    kind = config["kind"]
    if kind == "tiny":
        model = common.TinyStudent()
        model.load_state_dict(config["state_dict"])
        return model.eval()
    if kind == "feature":
        body = BodyPolicy()
        body.load_state_dict(config["state_dict"])
        frontend = make_frontend(config["feature"])
        return OnlinePolicy(body, frontend).eval()
    if kind == "aux":
        frontend = make_frontend(config["feature"])
        model = AuxPolicy(frontend)
        missing = model.load_state_dict(config["state_dict"], strict=False)
        if any(not k.startswith("frontend.") for k in missing.missing_keys):
            raise ValueError(f"aux checkpoint missing keys: {missing.missing_keys}")
        return model.eval()
    raise ValueError(kind)


def _eval_row(row: dict) -> dict:
    result = common.run_rollout(_EVAL_MODEL, row, record=False)
    return {"plant_id": row["plant_id"], "split": row["split"],
            "level": int(row["gjb_level"]),
            "tau_p": float(row["parameters"]["tau_p"]),
            "delay_bin": common.delay_bin(float(row["parameters"]["tau_p"])),
            "metrics": {k: [float(x) for x in v]
                        for k, v in result["metrics"].items()}}


def evaluate(config: dict, rows: list[dict], workers: int) -> list[dict]:
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_eval,
                             initargs=(config,)) as pool:
        return list(pool.map(_eval_row, rows))


def summarize(rows: list[dict]) -> dict:
    out = {}
    for name in ("short", "medium", "long"):
        group = [r for r in rows if r["delay_bin"] == name]
        if not group:
            continue
        rm = np.array([r["metrics"]["rmse"] for r in group])
        pk = np.array([r["metrics"]["peak"] for r in group])
        tv = np.array([r["metrics"]["tv"] for r in group])
        ok = (rm <= 1.5) & (pk <= 5.0) & (tv <= 12.0)
        out[name] = {
            "n_plants": len(group),
            "n_episodes": int(rm.size),
            "rmse_median": float(np.median(rm)),
            "rmse_p90": float(np.percentile(rm, 90)),
            "peak_median": float(np.median(pk)),
            "tv_median": float(np.median(tv)),
            "pass_rate": float(ok.mean()),
        }
    return out


def save_config(path: Path, model: nn.Module, kind: str, feature: str | None) -> dict:
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()
             if not k.startswith("frontend.")}
    config = {"kind": kind, "feature": feature, "state_dict": state,
              "data_dir": str(DATA)}
    torch.save(config, path)
    return config


def load_config(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------- DAgger
_DAGGER_POLICY = None
_DAGGER_LABEL = None


def _init_dagger(config: dict, beta: float) -> None:
    global _DAGGER_POLICY, _DAGGER_LABEL
    torch.set_num_threads(1)
    learner = build_model(config)
    _DAGGER_LABEL = common.load_tiny()
    _DAGGER_POLICY = DaggerPolicy(learner, _DAGGER_LABEL, beta)


def _dagger_row(row: dict) -> dict:
    return common.run_rollout(_DAGGER_POLICY, row, record=True,
                              label_model=_DAGGER_LABEL)


def collect_dagger(config: dict, rows: list[dict], beta: float,
                   workers: int) -> dict:
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_dagger,
                             initargs=(config, beta)) as pool:
        outs = list(pool.map(_dagger_row, rows))
    keys = ("history", "current", "older", "action", "step", "command")
    return {k: np.concatenate([o[k] for o in outs], axis=0) for k in keys}


def dagger_retrain(data: dict, model: nn.Module, kind: str, feature: str | None,
                   dagger: dict, epochs: int = 8,
                   lr: float = 3e-4) -> tuple[nn.Module, dict]:
    """Retrain on the union of the original BC states and DAgger states."""
    tr, va = data["idx"]["train"], data["idx"]["val"]
    act = torch.tensor(data["action"], dtype=torch.float32)
    dagger_act = torch.tensor(dagger["action"], dtype=torch.float32)
    n_d = len(dagger_act)
    if kind == "feature":
        frontend = make_frontend(feature)
        hist_d = torch.tensor(dagger["history"], dtype=torch.float32)
        with torch.no_grad():
            f_d = frontend(hist_d)
            fn_d = frontend(-hist_d)
        f = torch.tensor(np.load(FEAT / f"{feature}.npy"), dtype=torch.float32)
        fn = torch.tensor(np.load(FEAT / f"{feature}_neg.npy"), dtype=torch.float32)
        cur = torch.tensor(data["arrays"]["current"], dtype=torch.float32)
        old = torch.tensor(data["arrays"]["older"], dtype=torch.float32)
        batches = {
            "feat": torch.cat([f[tr], f_d, f[va]], dim=0),
            "feat_neg": torch.cat([fn[tr], fn_d, fn[va]], dim=0),
            "current": torch.cat([cur[tr], torch.tensor(dagger["current"]), cur[va]], dim=0),
            "older": torch.cat([old[tr], torch.tensor(dagger["older"]), old[va]], dim=0),
            "action": torch.cat([act[tr], dagger_act, act[va]], dim=0),
        }
        n_train = len(tr) + n_d
        train_idx = np.arange(n_train)
        val_idx = np.arange(n_train, len(batches["action"]))
        stats = _train_loop(model, batches, train_idx, val_idx,
                            epochs=epochs, batch=2048, lr=lr)
        return model, stats
    # Observation-level models: tiny (raw) and aux.
    obs = torch.tensor(data["obs"], dtype=torch.float32)
    hist_d = torch.tensor(dagger["history"], dtype=torch.float32)
    obs_d = torch.cat([
        torch.tensor(dagger["current"], dtype=torch.float32),
        hist_d.reshape(len(hist_d), -1),
        torch.tensor(dagger["older"], dtype=torch.float32),
    ], dim=1)
    batches = {
        "obs": torch.cat([obs[tr], obs_d, obs[va]], dim=0),
        "action": torch.cat([act[tr], dagger_act, act[va]], dim=0),
    }
    n_train = len(tr) + n_d
    train_idx = np.arange(n_train)
    val_idx = np.arange(n_train, len(batches["action"]))
    stats = _train_loop(model, batches, train_idx, val_idx,
                        epochs=epochs, batch=1024, lr=lr)
    return model, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("train", "eval"))
    ap.add_argument("--variants", nargs="+",
                    default=["raw_ft", "pca16", "dense16", "topk4", "aux_topk4"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--source-only", action="store_true")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--no-dagger", action="store_true")
    ap.add_argument("--dagger-beta", type=float, default=0.5)
    ap.add_argument("--dagger-epochs", type=int, default=8)
    ap.add_argument("--dagger-plants", type=int, default=96)
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    data = build_arrays(DATA)
    rows = common.load_library()
    metas = data["metas"]
    tau = data["tau"]
    split = data["split"]
    if args.source_only:
        test_rows = [r for r in rows if r["split"] == "id_test"
                     and float(r["parameters"]["tau_p"]) > 0.15]
    else:
        test_rows = [r for r in rows if r["split"] == "id_test"
                     and r["gjb_level"] in (1, 2, 3)]
    # Keep only the sampled test plants, not the whole split.
    test_ids = {m["plant_id"] for m in metas if m["split"] == "id_test"}
    test_rows = [r for r in test_rows if r["plant_id"] in test_ids]
    print(f"test plants: {len(test_rows)}", flush=True)

    out_dir = DATA / f"policies_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # DAgger states are collected on a fixed subset of source plants; the
    # labels always come from the deployed tiny Student.
    rows_by_id = {r["plant_id"]: r for r in rows}
    train_ids = [m["plant_id"] for m in metas if m["split"] == "train_core"]
    rng = np.random.default_rng(args.seed)
    take = min(args.dagger_plants, len(train_ids))
    dagger_rows = [rows_by_id[i] for i in rng.choice(train_ids, size=take, replace=False)]
    if args.source_only:
        dagger_rows = [r for r in dagger_rows
                       if float(r["parameters"]["tau_p"]) <= 0.15]
    print(f"DAgger plants: {len(dagger_rows)}", flush=True)

    results = {}
    for variant in args.variants:
        print(f"\n=== {variant} ===", flush=True)
        variant_seed = args.seed + sum(ord(ch) for ch in variant)
        torch.manual_seed(variant_seed)
        np.random.seed(variant_seed)
        t0 = time.time()
        model, stats, kind = train_variant(
            variant, data, epochs=args.epochs, source_only=args.source_only,
            init_tiny=(variant == "raw_ft" or variant.startswith("aux_")))
        feature = variant[4:] if variant.startswith("aux_") else (
            variant if kind == "feature" else None)
        ckpt = out_dir / f"{variant}.pt"
        config = save_config(ckpt, model, kind, feature)
        bc_stats = stats
        if not args.no_dagger and variant != "raw_ft":
            print(f"  DAgger beta={args.dagger_beta:.2f}", flush=True)
            dagger = collect_dagger(config, dagger_rows, args.dagger_beta,
                                    args.workers)
            model, stats = dagger_retrain(data, model, kind, feature, dagger,
                                          epochs=args.dagger_epochs)
            config = save_config(ckpt, model, kind, feature)
        print(f"trained in {time.time() - t0:.1f}s val_mse={stats['val_mse']:.2e}",
              flush=True)
        metrics = evaluate(config, test_rows, args.workers)
        summary = summarize(metrics)
        results[variant] = {"kind": kind, "feature": feature,
                            "train": {k: v for k, v in stats.items() if k != "curve"},
                            "curve": stats["curve"],
                            "bc_val_mse": bc_stats["val_mse"],
                            "dagger": not args.no_dagger and variant != "raw_ft",
                            "eval": summary, "rows": metrics}
        (out_dir / "policy_results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)

    (out_dir / "policy_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out_dir / 'policy_results.json'}", flush=True)


if __name__ == "__main__":
    main()
