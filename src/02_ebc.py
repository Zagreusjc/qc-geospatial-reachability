"""Phase 2 -- Edge Betweenness Centrality.

Computes EBC on the study graph using unweighted shortest paths (weight=None, per
the manuscript, so centrality reflects topology rather than the friction it informs),
then min-max normalizes to [0, 1].

Modes   : config.EBC_MODE == "exact" (full Brandes) or "approx" (k sampled pivots,
          config.EBC_K, config.EBC_SEED) for tractability on limited CPU.
Inputs  : data/processed/G_scc.gpickle
Outputs : data/processed/ebc.parquet (edge -> ebc_norm), distribution figure.
Next    : 03_friction_weights.py
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

log = get_logger("02_ebc")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    mode = args.mode or config.EBC_MODE
    log.info("EBC mode=%s k=%s seed=%s", mode, config.EBC_K, config.EBC_SEED)

    g_path = config.PROCESSED_DIR / "G_scc.gpickle"
    if not g_path.exists():
        raise FileNotFoundError(f"{g_path} not found; run 01_graph_construction.py first.")

    with timed(log, "load G_scc"):
        with open(g_path, "rb") as f:
            G: nx.MultiDiGraph = pickle.load(f)
        log.info("Loaded graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    k = None if mode == "exact" else min(config.EBC_K, G.number_of_nodes())
    with timed(log, f"compute edge betweenness centrality (mode={mode}, k={k})"):
        # weight=None per manuscript: EBC reflects unweighted topological structure,
        # kept independent of the friction weights it will later help construct.
        raw_ebc = nx.edge_betweenness_centrality(
            G, k=k, normalized=False, weight=None, seed=config.EBC_SEED
        )

    max_ebc = max(raw_ebc.values()) if raw_ebc else 0.0
    with timed(log, "write ebc.parquet"):
        rows = [
            {"u": u, "v": v, "key": key, "ebc_raw": val,
             "ebc_norm": (val / max_ebc) if max_ebc > 0 else 0.0}
            for (u, v, key), val in raw_ebc.items()
        ]
        df = pd.DataFrame(rows)
        out_path = config.PROCESSED_DIR / "ebc.parquet"
        df.to_parquet(out_path)
        log.info("Wrote %s (%d edges); max_raw_ebc=%.6g", out_path, len(df), max_ebc)

    with timed(log, "write EBC distribution figure"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["ebc_norm"], bins=50, color="#3b6ea5")
        ax.set_xlabel("Normalized Edge Betweenness Centrality")
        ax.set_ylabel("Edge count")
        ax.set_title(f"EBC distribution (mode={mode}, k={k})")
        fig_path = config.FIGURES_DIR / "ebc_distribution.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Wrote %s", fig_path)

    log.info(
        "EBC summary: min=%.6g max=%.6g mean=%.6g median=%.6g",
        df["ebc_norm"].min(), df["ebc_norm"].max(), df["ebc_norm"].mean(), df["ebc_norm"].median(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["exact", "approx"], default=None,
                        help="override config.EBC_MODE")
    main(parser.parse_args())
