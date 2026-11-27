"""Phase 4 -- Node2Vec+ embeddings.

Generates 128-dimensional node embeddings per weight condition using PecanPy's
weight-aware Node2Vec+ (extend=True). q<1 biases walks toward global structure,
which suits long-range distance approximation.

Params  : dim=128, walk_length=80, num_walks=10, p=1.0, q=0.5, window=10 (config).
Inputs  : per-condition .edg edgelists from Phase 3 (data/interim/edgelists/).
Outputs : outputs/embeddings/Z_W1.npy, Z_W2.npy, Z_W3.npy;
          data/processed/node_index.parquet (embedding row index <-> graph node id);
          --sweep: outputs/embeddings/Z_W3_p{p}_q{q}.npy for each (p,q) in the
          config.N2V_P_SWEEP x N2V_Q_SWEEP grid. This generates embeddings only;
          scoring p/q sensitivity end-to-end requires re-running Phases 5/7/8
          against each variant separately.
Next    : 05_sampling_labels.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from pecanpy.pecanpy import SparseOTF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402

log = get_logger("04_embeddings")


def _embed_condition(edg_path, p, q, seed) -> tuple[np.ndarray, list[int]]:
    """Embed one weighted edgelist with Node2Vec+ and return (Z, node_ids).

    Takes the .edg path, the walk parameters p and q, and a random seed. Reads
    the edgelist as weighted and directed, runs PecanPy SparseOTF with the
    config.N2V_* settings, and returns the embedding matrix together with the
    graph node ids in matrix row order.
    """
    g = SparseOTF(
        p=p, q=q, workers=config.N2V_WORKERS, extend=config.N2V_EXTEND,
        random_state=seed, verbose=False,
    )
    g.read_edg(str(edg_path), weighted=True, directed=True)
    Z = g.embed(
        dim=config.N2V_DIM, num_walks=config.N2V_NUM_WALKS,
        walk_length=config.N2V_WALK_LENGTH, window_size=config.N2V_WINDOW,
        epochs=1, verbose=False,
    )
    node_ids = [int(nid) for nid in g.nodes]
    return Z, node_ids


def main(args: argparse.Namespace) -> None:
    """Run Phase 4: embed each weight condition, then optionally sweep p/q on W3.

    Steps: (1) for each condition (all of config.WEIGHT_CONDITIONS, or the one
    given by --condition), read data/interim/edgelists/{condition}.edg and
    embed it with config.N2V_P and config.N2V_Q; (2) save the matrix to
    outputs/embeddings/Z_{condition}.npy; (3) write the node ordering of the
    first condition processed to data/processed/node_index.parquet and warn if a
    later condition yields a different ordering; (4) with --sweep, embed W3.edg
    for every (p, q) in config.N2V_P_SWEEP x config.N2V_Q_SWEEP except the
    primary pair, saving outputs/embeddings/Z_W3_p{p}_q{q}.npy.
    """
    config.ensure_dirs()
    log.info("Node2Vec+ dim=%d L=%d r=%d p=%.2f q=%.2f window=%d extend=%s",
             config.N2V_DIM, config.N2V_WALK_LENGTH, config.N2V_NUM_WALKS,
             config.N2V_P, config.N2V_Q, config.N2V_WINDOW, config.N2V_EXTEND)

    # Locate the Phase 3 edgelists and choose which conditions to embed.
    edgelist_dir = config.INTERIM_DIR / "edgelists"
    conditions = [args.condition] if args.condition else config.WEIGHT_CONDITIONS

    node_index_path = config.PROCESSED_DIR / "node_index.parquet"
    reference_node_ids = None

    # Primary pass: one embedding per weight condition, using the primary p and q.
    for condition in conditions:
        edg_path = edgelist_dir / f"{condition}.edg"
        if not edg_path.exists():
            raise FileNotFoundError(f"{edg_path} not found; run 03_friction_weights.py first.")

        with timed(log, f"embed {condition} (p={config.N2V_P}, q={config.N2V_Q})"):
            Z, node_ids = _embed_condition(edg_path, config.N2V_P, config.N2V_Q, seed=config.SAMPLING_SEED)
            log.info("%s embedding shape: %s", condition, Z.shape)

        out_path = config.EMBEDDINGS_DIR / f"Z_{condition}.npy"
        np.save(out_path, Z)
        log.info("Wrote %s", out_path)

        # The first condition defines node_index.parquet; later ones are only checked against it.
        if reference_node_ids is None:
            reference_node_ids = node_ids
            pd.DataFrame({"node_id": node_ids}).to_parquet(node_index_path)
            log.info("Wrote %s (%d nodes)", node_index_path, len(node_ids))
        elif node_ids != reference_node_ids:
            log.warning(
                "%s produced a different node ordering than the reference condition; "
                "node_index.parquet reflects the first condition processed only. "
                "Downstream phases must index embeddings by this file's row order.",
                condition,
            )

    # Optional sensitivity sweep: embeddings only, on the W3 edgelist.
    if args.sweep:
        log.info("Running p/q sensitivity sweep on W3 (embeddings only, %d combinations)",
                  len(config.N2V_P_SWEEP) * len(config.N2V_Q_SWEEP))
        w3_edg = edgelist_dir / "W3.edg"
        for p in config.N2V_P_SWEEP:
            for q in config.N2V_Q_SWEEP:
                if p == config.N2V_P and q == config.N2V_Q:
                    continue  # already generated above as the primary W3 embedding
                with timed(log, f"embed W3 sweep p={p} q={q}"):
                    Z, _ = _embed_condition(w3_edg, p, q, seed=config.SAMPLING_SEED)
                out_path = config.EMBEDDINGS_DIR / f"Z_W3_p{p}_q{q}.npy"
                np.save(out_path, Z)
                log.info("Wrote %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=config.WEIGHT_CONDITIONS, default=None,
                        help="embed a single weight condition (default: all)")
    parser.add_argument("--sweep", action="store_true", help="run p/q sensitivity sweep")
    main(parser.parse_args())
