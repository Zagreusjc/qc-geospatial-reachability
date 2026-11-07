"""Phase 10 -- Directional asymmetry check (supplementary finding, Chapter III p.98).

The road graph is directed (one-way streets), so the true distance d*(u, v) can
differ from d*(v, u). This script measures (1) how often that happens in Quezon
City and (2) how well each trained oracle reproduces the difference.

A symmetric Siamese (|h_u - h_v|) gives the same prediction in both directions by
construction, so its predicted asymmetry is always zero. The order-aware Siamese
(concat[h_u, h_v]) and the MLP (concat[z_u, z_v]) can predict different values per
direction.

Procedure (per weight condition W1, W2, W3; same node pairs for all three):
  - pick N_SOURCES random source nodes s and N_TARGETS_PER_SOURCE random targets t
  - forward Dijkstra from s gives d*(s, t); Dijkstra on the reversed graph from s
    gives d*(t, s) -- so every sampled pair has its true distance in BOTH directions
  - every trained model predicts both directions; we compare predicted asymmetry
    pred(s,t) - pred(t,s) against true asymmetry d*(s,t) - d*(t,s)

Nothing is retrained or overwritten. Reads existing graph, weights, embeddings and
model checkpoints (outputs/models/*.pt, plus outputs/models_archived_A/*.pt if present).

Output : outputs/tables/asymmetry_results.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

import torch  # must load before numpy/pandas on Windows
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402
from importlib import import_module  # noqa: E402

models_mod = import_module("06_models")
log = get_logger("10_asymmetry")

N_SOURCES = 200
N_TARGETS_PER_SOURCE = 25          # -> up to 5,000 pairs, each with both directions
ASYM_REL_THRESHOLD = 0.01          # a pair counts as "asymmetric" if directions differ by >1%


def _annotate_condition_weight(G, weights_df: pd.DataFrame, condition: str) -> None:
    lookup = {
        (int(r.u), int(r.v), int(r.key)): float(getattr(r, condition))
        for r in weights_df.itertuples()
    }
    for u, v, key, data in G.edges(keys=True, data=True):
        data["w"] = lookup[(u, v, key)]


def _sample_pairs(nodes: list, rng: random.Random) -> list:
    sources = rng.sample(nodes, N_SOURCES)
    pairs = []
    for s in sources:
        targets = rng.sample(nodes, N_TARGETS_PER_SOURCE + 1)
        targets = [t for t in targets if t != s][:N_TARGETS_PER_SOURCE]
        pairs.extend((s, t) for t in targets)
    return pairs


def _true_distances_both_directions(G, pairs: list, condition: str) -> pd.DataFrame:
    G_rev = G.reverse(copy=False)
    by_source = {}
    for s, t in pairs:
        by_source.setdefault(s, []).append(t)

    rows = []
    for i, (s, targets) in enumerate(by_source.items(), 1):
        fwd = nx.single_source_dijkstra_path_length(G, s, weight="w")      # d*(s, t)
        rev = nx.single_source_dijkstra_path_length(G_rev, s, weight="w")  # d*(t, s)
        for t in targets:
            rows.append((s, t, float(fwd[t]), float(rev[t])))
        if i % 25 == 0 or i == len(by_source):
            log.info("  %s: Dijkstra both directions for %d/%d sources", condition, i, len(by_source))

    df = pd.DataFrame(rows, columns=["s", "t", "d_fwd", "d_rev"])
    df = df[(df["d_fwd"] > 0) & (df["d_rev"] > 0)].reset_index(drop=True)
    df["true_asym"] = df["d_fwd"] - df["d_rev"]
    df["rel_asym"] = df["true_asym"].abs() / df[["d_fwd", "d_rev"]].max(axis=1)
    df["is_asym"] = df["rel_asym"] > ASYM_REL_THRESHOLD
    return df


def _checkpoint_paths() -> list:
    paths = sorted(config.MODELS_DIR.glob("*.pt"))
    archived = config.OUTPUTS_DIR / "models_archived_A"
    if archived.exists():
        paths += sorted(archived.glob("*.pt"))
    return paths


def _predict_both(checkpoint, Z_t, u_idx, v_idx):
    device = config.DEVICE
    model = models_mod.build_model(checkpoint["arch"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        fwd = model(Z_t[u_idx], Z_t[v_idx]) * checkpoint["sigma"] + checkpoint["mu"]
        rev = model(Z_t[v_idx], Z_t[u_idx]) * checkpoint["sigma"] + checkpoint["mu"]
    return fwd.cpu().numpy(), rev.cpu().numpy()


def _run_metrics(df: pd.DataFrame, pred_fwd: np.ndarray, pred_rev: np.ndarray) -> dict:
    pred_asym = pred_fwd - pred_rev
    m = df["is_asym"].to_numpy()
    true_asym = df["true_asym"].to_numpy()

    out = {
        "mae_forward": float(np.mean(np.abs(pred_fwd - df["d_fwd"].to_numpy()))),
        "mae_reverse": float(np.mean(np.abs(pred_rev - df["d_rev"].to_numpy()))),
        "mean_abs_pred_asym": float(np.mean(np.abs(pred_asym))),
    }
    if m.sum() >= 3:
        pa, ta = pred_asym[m], true_asym[m]
        out["asym_mae"] = float(np.mean(np.abs(pa - ta)))
        out["asym_sign_agreement"] = float(np.mean(np.sign(pa) == np.sign(ta)))
        out["asym_spearman"] = float(spearmanr(pa, ta)[0]) if np.std(pa) > 1e-12 else float("nan")
    else:
        out["asym_mae"] = out["asym_sign_agreement"] = out["asym_spearman"] = float("nan")
    return out


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    device = config.DEVICE

    with timed(log, "load graph + weights + node index"):
        with open(config.PROCESSED_DIR / "G_scc.gpickle", "rb") as f:
            G = pickle.load(f)
        weights_df = pd.read_parquet(config.PROCESSED_DIR / "friction_weights.parquet")
        node_index_df = pd.read_parquet(config.PROCESSED_DIR / "node_index.parquet")
        node_to_idx = {int(nid): i for i, nid in enumerate(node_index_df["node_id"])}

    rng = random.Random(config.SAMPLING_SEED)
    pairs = _sample_pairs(list(G.nodes), rng)
    log.info("Sampled %d node pairs (%d sources x %d targets)", len(pairs), N_SOURCES, N_TARGETS_PER_SOURCE)

    checkpoints = []
    for p in _checkpoint_paths():
        ck = torch.load(p, map_location=device, weights_only=False)
        checkpoints.append((p.stem, ck))
    log.info("Found %d trained models", len(checkpoints))

    rows = []
    for condition in config.WEIGHT_CONDITIONS:
        log.info("=== Condition %s ===", condition)
        with timed(log, f"{condition}: true distances in both directions"):
            _annotate_condition_weight(G, weights_df, condition)
            df = _true_distances_both_directions(G, pairs, condition)

        asym = df[df["is_asym"]]
        rows.append({
            "row_type": "condition_summary", "weight": condition, "n_pairs": len(df),
            "n_asymmetric": int(df["is_asym"].sum()),
            "pct_asymmetric": round(100 * df["is_asym"].mean(), 3),
            "median_rel_asym_among_asymmetric": float(asym["rel_asym"].median()) if len(asym) else float("nan"),
            "mean_abs_true_asym": float(df["true_asym"].abs().mean()),
        })
        log.info("%s: %d/%d pairs (%.1f%%) differ by direction by more than %.0f%%",
                 condition, rows[-1]["n_asymmetric"], len(df), rows[-1]["pct_asymmetric"],
                 100 * ASYM_REL_THRESHOLD)

        Z = np.load(config.EMBEDDINGS_DIR / f"Z_{condition}.npy").astype(np.float32)
        Z_t = torch.from_numpy(Z).to(device)
        u_idx = torch.tensor([node_to_idx[int(s)] for s in df["s"]], dtype=torch.long, device=device)
        v_idx = torch.tensor([node_to_idx[int(t)] for t in df["t"]], dtype=torch.long, device=device)

        for run_id, ck in checkpoints:
            if ck["weight_condition"] != condition:
                continue
            pred_fwd, pred_rev = _predict_both(ck, Z_t, u_idx, v_idx)
            metrics = _run_metrics(df, pred_fwd, pred_rev)
            rows.append({"row_type": "run", "run_id": run_id, "weight": condition,
                         "arch": ck["arch"], "seed": ck["seed"], **metrics})
            log.info("  %s: asym_spearman=%.4f sign_agree=%.3f mean|pred_asym|=%.4f",
                     run_id, metrics["asym_spearman"], metrics["asym_sign_agreement"],
                     metrics["mean_abs_pred_asym"])

    results = pd.DataFrame(rows)
    run_rows = results[results["row_type"] == "run"]
    agg_cols = ["mae_forward", "mae_reverse", "mean_abs_pred_asym", "asym_mae",
                "asym_sign_agreement", "asym_spearman"]
    agg_rows = []
    for (weight, arch), grp in run_rows.groupby(["weight", "arch"]):
        row = {"row_type": "arch_aggregate", "run_id": f"{arch}-{weight}-mean",
               "weight": weight, "arch": arch, "seed": "mean"}
        for c in agg_cols:
            row[f"{c}_mean"] = grp[c].mean()
            row[f"{c}_std"] = grp[c].std()
        agg_rows.append(row)
    results = pd.concat([results, pd.DataFrame(agg_rows)], ignore_index=True)

    out_path = config.TABLES_DIR / "asymmetry_results.csv"
    results.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    main(parser.parse_args())
