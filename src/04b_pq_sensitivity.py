"""Phase 4b -- Node2Vec+ p/q sensitivity sweep (manuscript secondary finding).

Chapter III: "The sensitivity of results to p and q values is evaluated by
training oracle variants under p in {0.5, 1.0, 2.0} and q in {0.25, 0.5, 1.0}
and reporting validation MAE for each combination, as a secondary finding."

Reuses the W3 (composite, primary) train/val pairs and Dijkstra labels from
Phase 5, since ground truth depends on the friction-weighted graph, not on the
Node2Vec+ walk parameters -- only the embedding differs per (p, q). Trains a
single Siamese oracle (Architecture A, one seed) per combination: this is the
manuscript's secondary finding, not the primary 3x2x3 ablation, so it does not
repeat that design's 3-seed structure.

Inputs  : outputs/embeddings/Z_W3.npy and Z_W3_p{p}_q{q}.npy
          (from `04_embeddings.py --sweep`),
          data/processed/pairs_{train,val}_W3.parquet.
Outputs : outputs/tables/pq_sensitivity.csv (p, q, val_mae, n_epochs_trained)
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

log = get_logger("04b_pq_sensitivity")

SEED = config.SAMPLING_SEED  # fixed single seed -- secondary finding, not the 3-seed primary ablation


def _load_pairs(split: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"pairs_{split}_W3.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run 05_sampling_labels.py (condition W3) first.")
    return pd.read_parquet(path)


def _batches(n: int, batch_size: int, generator: torch.Generator, device: str):
    perm = torch.randperm(n, generator=generator).to(device)
    for i in range(0, n, batch_size):
        yield perm[i:i + batch_size]


def _train_and_validate(Z: np.ndarray, train_df: pd.DataFrame, val_df: pd.DataFrame, seed: int) -> tuple[float, int]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = config.DEVICE
    Z_t = torch.from_numpy(Z.astype(np.float32)).to(device)

    mu = float(train_df["dist"].mean())
    sigma = float(train_df["dist"].std())
    sigma = sigma if sigma > 1e-8 else 1.0

    def to_tensors(df):
        u_idx = torch.from_numpy(df["u_idx"].to_numpy(dtype=np.int64)).to(device)
        v_idx = torch.from_numpy(df["v_idx"].to_numpy(dtype=np.int64)).to(device)
        y = torch.from_numpy(df["dist"].to_numpy(dtype=np.float32)).to(device)
        return u_idx, v_idx, (y - mu) / sigma, y

    train_u, train_v, train_y_std, _ = to_tensors(train_df)
    val_u, val_v, _, val_y_raw = to_tensors(val_df)

    model = models_mod.build_model("siamese").to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)

    best_val_mae = float("inf")
    epochs_without_improvement = 0
    n_train = len(train_df)
    n_epochs_trained = 0

    for epoch in range(config.EPOCHS):
        model.train()
        for batch_idx in _batches(n_train, config.BATCH_SIZE, gen, device):
            optimizer.zero_grad()
            pred = model(Z_t[train_u[batch_idx]], Z_t[train_v[batch_idx]])
            loss = loss_fn(pred, train_y_std[batch_idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        n_epochs_trained += 1

        model.eval()
        with torch.no_grad():
            val_pred_raw = model(Z_t[val_u], Z_t[val_v]) * sigma + mu
            val_mae = torch.mean(torch.abs(val_pred_raw - val_y_raw)).item()

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                break

    return best_val_mae, n_epochs_trained


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("p/q sensitivity sweep: %d x %d combinations, seed=%d",
              len(config.N2V_P_SWEEP), len(config.N2V_Q_SWEEP), SEED)

    train_df = _load_pairs("train")
    val_df = _load_pairs("val")

    rows = []
    for p in config.N2V_P_SWEEP:
        for q in config.N2V_Q_SWEEP:
            is_primary = (p == config.N2V_P and q == config.N2V_Q)
            embed_path = config.EMBEDDINGS_DIR / ("Z_W3.npy" if is_primary else f"Z_W3_p{p}_q{q}.npy")
            if not embed_path.exists():
                raise FileNotFoundError(
                    f"{embed_path} not found; run `python 04_embeddings.py --sweep` first."
                )
            with timed(log, f"train+validate p={p} q={q}"):
                Z = np.load(embed_path)
                val_mae, n_epochs = _train_and_validate(Z, train_df, val_df, seed=SEED)
            log.info("p=%.2f q=%.2f -> val MAE=%.4f (%d epochs)", p, q, val_mae, n_epochs)
            rows.append({"p": p, "q": q, "val_mae": val_mae, "n_epochs_trained": n_epochs})

    out_path = config.TABLES_DIR / "pq_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    main(parser.parse_args())
