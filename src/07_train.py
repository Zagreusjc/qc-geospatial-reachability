"""Phase 7 -- Train the 3x2x3 ablation.

Trains every cell of the ablation grid: {W1,W2,W3} x {Siamese, MLP} x {seed 0,1,2}
= 18 runs (labelled A-W1-seed0 .. B-W3-seed2 via config.run_id).

Training: MSE loss vs Dijkstra labels, Adam (lr=1e-3, weight_decay=1e-5), cosine
annealing over 100 epochs, early stopping on validation MAE (patience 10), z-score
label standardization (inverted at eval).

Inputs  : outputs/embeddings/Z_*.npy, data/processed/pairs_*_*.parquet
Outputs : outputs/models/<run_id>.pt (best checkpoint), training-curve figures.
Next    : 08_evaluate.py

CLI examples:
  python src/07_train.py                       # all 18 runs
  python src/07_train.py --condition W3         # only W3 cells (6 runs)
  python src/07_train.py --arch siamese --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402
from importlib import import_module  # noqa: E402

models_mod = import_module("06_models")

log = get_logger("07_train")


def selected_runs(args: argparse.Namespace):
    for weight, arch, seed in config.ablation_runs():
        if args.condition and weight != args.condition:
            continue
        if args.arch and arch != args.arch:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        yield weight, arch, seed


def _load_pairs(weight: str, split: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"pairs_{split}_{weight}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run 05_sampling_labels.py first.")
    return pd.read_parquet(path)


def _batches(n: int, batch_size: int, generator: torch.Generator, device: str):
    perm = torch.randperm(n, generator=generator).to(device)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def _train_one_run(weight: str, arch: str, seed: int) -> dict:
    run_id = config.run_id(weight, arch, seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config.DEVICE

    Z = np.load(config.EMBEDDINGS_DIR / f"Z_{weight}.npy").astype(np.float32)
    Z_t = torch.from_numpy(Z).to(device)

    train_df = _load_pairs(weight, "train")
    val_df = _load_pairs(weight, "val")

    mu = float(train_df["dist"].mean())
    sigma = float(train_df["dist"].std())
    sigma = sigma if sigma > 1e-8 else 1.0

    def to_tensors(df):
        u_idx = torch.from_numpy(df["u_idx"].to_numpy(dtype=np.int64)).to(device)
        v_idx = torch.from_numpy(df["v_idx"].to_numpy(dtype=np.int64)).to(device)
        y = torch.from_numpy(df["dist"].to_numpy(dtype=np.float32)).to(device)
        y_std = (y - mu) / sigma
        return u_idx, v_idx, y_std, y

    train_u, train_v, train_y_std, _ = to_tensors(train_df)
    val_u, val_v, val_y_std, val_y_raw = to_tensors(val_df)

    model = models_mod.build_model(arch).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    best_val_mae = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    n_train = len(train_df)
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_idx in _batches(n_train, config.BATCH_SIZE, gen, device):
            zu = Z_t[train_u[batch_idx]]
            zv = Z_t[train_v[batch_idx]]
            target = train_y_std[batch_idx]

            optimizer.zero_grad()
            pred = model(zu, zv)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred_std = model(Z_t[val_u], Z_t[val_v])
            val_pred_raw = val_pred_std * sigma + mu
            val_mae = torch.mean(torch.abs(val_pred_raw - val_y_raw)).item()

        history.append({"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_mae": val_mae})

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            # Saved on CPU regardless of training device, so any machine can load it.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                log.info("%s: early stopping at epoch %d (best val MAE=%.4f)", run_id, epoch, best_val_mae)
                break

    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "arch": arch,
        "weight_condition": weight,
        "seed": seed,
        "mu": mu,
        "sigma": sigma,
        "best_val_mae": best_val_mae,
        "n_epochs_trained": len(history),
    }
    out_path = config.MODELS_DIR / f"{run_id}.pt"
    torch.save(checkpoint, out_path)
    log.info("%s: best val MAE=%.4f after %d epochs -> %s", run_id, best_val_mae, len(history), out_path)

    _plot_training_curve(run_id, history)
    return checkpoint


def _plot_training_curve(run_id: str, history: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train MSE (standardized)", color="#3b6ea5")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train MSE (standardized)")
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["val_mae"] for h in history], label="val MAE (raw units)", color="#d97642")
    ax2.set_ylabel("val MAE (raw units)")
    fig.legend(loc="upper right")
    ax1.set_title(f"Training curve: {run_id}")
    fig.tight_layout()
    fig_path = config.FIGURES_DIR / f"{run_id}_training_curve.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    runs = list(selected_runs(args))
    log.info("Selected %d ablation run(s):", len(runs))
    for weight, arch, seed in runs:
        log.info("  %s  (weight=%s: %s | arch=%s | seed=%d)",
                 config.run_id(weight, arch, seed), weight,
                 config.WEIGHT_CONDITION_LABELS[weight], arch, seed)

    for weight, arch, seed in runs:
        run_id = config.run_id(weight, arch, seed)
        log.info(">>> Now training %s -- weight condition %s: %s",
                 run_id, weight, config.WEIGHT_CONDITION_LABELS[weight])
        with timed(log, f"train {run_id}"):
            _train_one_run(weight, arch, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--condition", choices=config.WEIGHT_CONDITIONS, default=None,
        help="friction-weight condition to train (default: all three). "
             + " | ".join(f"{k}={v}" for k, v in config.WEIGHT_CONDITION_LABELS.items()),
    )
    parser.add_argument(
        "--arch", choices=config.ARCHITECTURES, default=None,
        help="architecture to train (default: both). siamese=A, mlp=B",
    )
    parser.add_argument("--seed", type=int, default=None,
                         help="single seed to train (default: all of config.SEEDS)")
    main(parser.parse_args())
