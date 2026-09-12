"""Phase 3 -- Friction weight assignment (three ablation conditions).

Assigns the three edge-weight conditions that define the three friction-weighted
graphs used in the ablation:
  W1 = length / speed(road_class)                     (traversal cost only)
  W2 = normalized EBC                                  (structural load only)
  W3 = W1 * (1 + alpha * normalized EBC), alpha=1.0    (composite, primary)

Inputs  : data/processed/G_scc.gpickle, data/processed/ebc.parquet
Outputs : data/processed/friction_weights.parquet (all conditions + alpha
          sensitivity variants, per edge); data/interim/edgelists/{W1,W2,W3}.edg
          (src\\tdst\\tweight, for PecanPy); weight-distribution figure.
Next    : 04_embeddings.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402

log = get_logger("03_friction_weights")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Weight conditions: %s (alpha=%s)", config.WEIGHT_CONDITIONS, config.ALPHA)

    g_path = config.PROCESSED_DIR / "G_scc.gpickle"
    ebc_path = config.PROCESSED_DIR / "ebc.parquet"
    if not g_path.exists():
        raise FileNotFoundError(f"{g_path} not found; run 01_graph_construction.py first.")
    if not ebc_path.exists():
        raise FileNotFoundError(f"{ebc_path} not found; run 02_ebc.py first.")

    with timed(log, "load graph + EBC"):
        with open(g_path, "rb") as f:
            G: nx.MultiDiGraph = pickle.load(f)
        ebc_df = pd.read_parquet(ebc_path)
        ebc_lookup = {
            (int(r.u), int(r.v), int(r.key)): float(r.ebc_norm) for r in ebc_df.itertuples()
        }
        log.info("Loaded graph (%d edges) and EBC table (%d edges)", G.number_of_edges(), len(ebc_df))

    with timed(log, "compute W1/W2/W3 + alpha-sensitivity variants"):
        rows = []
        n_missing_ebc = 0
        for u, v, key, data in G.edges(keys=True, data=True):
            length_m = data.get("length", 0.0)
            highway = data.get("highway", config.DEFAULT_ROAD_CLASS)
            speed_mps = config.speed_mps(highway)
            w1 = length_m / speed_mps

            ebc_norm = ebc_lookup.get((u, v, key))
            if ebc_norm is None:
                ebc_norm = 0.0
                n_missing_ebc += 1
            w2 = ebc_norm
            w3 = w1 * (1 + config.ALPHA * ebc_norm)

            row = {
                "u": u, "v": v, "key": key, "highway": highway, "length_m": length_m,
                "ebc_norm": ebc_norm, "W1": w1, "W2": w2, "W3": w3,
            }
            for a in config.ALPHA_SENSITIVITY:
                row[f"W3_alpha{a}"] = w1 * (1 + a * ebc_norm)
            rows.append(row)
        weights_df = pd.DataFrame(rows)
        if n_missing_ebc:
            log.warning("%d/%d edges had no EBC entry; defaulted ebc_norm=0.0", n_missing_ebc, len(weights_df))

    out_path = config.PROCESSED_DIR / "friction_weights.parquet"
    weights_df.to_parquet(out_path)
    log.info("Wrote %s (%d edges)", out_path, len(weights_df))

    edgelist_dir = config.INTERIM_DIR / "edgelists"
    edgelist_dir.mkdir(parents=True, exist_ok=True)
    with timed(log, "write .edg edgelists for PecanPy"):
        # PecanPy's edge-list reader drops any edge with weight <= 0 (W2 =
        # normalized EBC is exactly 0 for edges that never sit on a sampled
        # shortest path). Dropped edges would change the node order PecanPy
        # assigns while parsing the file, and all three conditions must share
        # one node_index.parquet -- so only this PecanPy-facing copy floors
        # such weights to a tiny positive epsilon. friction_weights.parquet
        # above keeps the true values, including real zeros, for Phase 5's
        # Dijkstra ground truth.
        edg_epsilon = 1e-6
        for condition in config.WEIGHT_CONDITIONS:
            edg_path = edgelist_dir / f"{condition}.edg"
            edg_df = weights_df[["u", "v", condition]].copy()
            n_floored = int((edg_df[condition] <= 0).sum())
            edg_df[condition] = edg_df[condition].clip(lower=edg_epsilon)
            edg_df.to_csv(edg_path, sep="\t", header=False, index=False)
            log.info("Wrote %s (%d edges, %d floored to %g)", edg_path, len(edg_df), n_floored, edg_epsilon)

    with timed(log, "write weight-distribution figure"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, cond in zip(axes, config.WEIGHT_CONDITIONS):
            ax.hist(weights_df[cond], bins=50, color="#3b6ea5")
            ax.set_title(cond)
            ax.set_xlabel("edge weight")
            ax.set_ylabel("edge count")
        fig.tight_layout()
        fig_path = config.FIGURES_DIR / "friction_weight_distributions.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", fig_path)

    for condition in config.WEIGHT_CONDITIONS:
        s = weights_df[condition]
        log.info("%s summary: min=%.6g max=%.6g mean=%.6g median=%.6g",
                 condition, s.min(), s.max(), s.mean(), s.median())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
