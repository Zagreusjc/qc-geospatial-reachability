"""Phase 5 -- Stratified pair sampling and Dijkstra ground-truth labels.

Builds the train/val/test node-pair sets and their exact friction-weighted
shortest-path distances (the supervised regression targets).

Procedure:
  - pilot 10k random pairs -> Dijkstra -> 10 equal-count (quantile) distance deciles
  - train (500k) + val (50k): stratified across deciles (disjoint)
  - test (50k): uniform random (natural distance skew)
  - labels: networkx single_source_dijkstra_path_length per weight condition
            (directed pairs preserved; drop d*=0; dedupe)

Because the study graph is a single strongly connected component, one
single_source_dijkstra_path_length call reaches every other node, so a
modest number of random source nodes yields a large, deduplicated pool of
(source, target, distance) pairs far faster than sampling pairs one at a time.

Inputs  : data/processed/G_scc.gpickle, data/processed/friction_weights.parquet,
          data/processed/node_index.parquet.
Outputs : data/processed/pairs_{train,val,test}_{W1,W2,W3}.parquet (u_idx, v_idx, dist)
Next    : 07_train.py
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402

log = get_logger("05_sampling_labels")


def _annotate_condition_weight(G: nx.MultiDiGraph, weights_df: pd.DataFrame, condition: str) -> None:
    """Set the edge attribute "w" on every edge of G to the given condition's weight.

    Weights are looked up in weights_df by (u, v, key) from the column named
    after the condition (W1, W2 or W3). Modifies G in place and returns None;
    Dijkstra later reads this attribute via weight="w".
    """
    lookup = {
        (int(r.u), int(r.v), int(r.key)): float(getattr(r, condition))
        for r in weights_df.itertuples()
    }
    for u, v, key, data in G.edges(keys=True, data=True):
        data["w"] = lookup[(u, v, key)]


def _pilot_deciles(G: nx.MultiDiGraph, nodes: list, rng: random.Random) -> np.ndarray:
    """Sample PILOT_PAIRS random (u,v) pairs and Dijkstra them, then return the 9
    interior decile cutoffs.

    The graph is a single strongly connected component, so one
    single_source_dijkstra_path_length call reaches every other node -- drawing
    fresh random (u,v) pairs one at a time would pick close to PILOT_PAIRS
    distinct sources (birthday-paradox math over ~32k nodes), each needing its
    own full Dijkstra run. Instead, run from a small number of random sources and
    sample many targets per source, matching the strategy _collect_pair_pool
    uses for the same reason.
    """
    targets_per_source = 500
    n_sources = max(1, -(-config.PILOT_PAIRS // targets_per_source))  # ceil division
    sources = rng.sample(nodes, min(n_sources, len(nodes)))

    dists = []
    for src in sources:
        dist_dict = nx.single_source_dijkstra_path_length(G, src, weight="w")
        candidates = [t for t, d in dist_dict.items() if t != src and d > 0]
        sample_size = min(len(candidates), targets_per_source)
        for t in rng.sample(candidates, sample_size):
            dists.append(dist_dict[t])

    dists = np.array(dists[:config.PILOT_PAIRS])
    cutoffs = np.quantile(dists, np.linspace(0.1, 0.9, config.N_DECILES - 1))
    log.info("Pilot: %d valid distances, decile cutoffs=%s", len(dists), np.round(cutoffs, 3))
    return cutoffs


def _decile_bucket(d: float, cutoffs: np.ndarray) -> int:
    """Return the index of the decile bucket that distance d falls into, given the pilot cutoffs."""
    return int(np.searchsorted(cutoffs, d))


def _collect_pair_pool(G: nx.MultiDiGraph, nodes: list, target_total: int, rng: random.Random) -> dict:
    """Run single_source_dijkstra_path_length from many random sources, capping
    targets per source so that the pool draws from a wide spread of source nodes
    rather than exhausting the target space from just ~34 sources.

    With TARGETS_PER_SOURCE=200 and ~32k nodes, we need ~5,400 sources to fill
    a pool of 1,080,000 -- giving genuine (u,v) diversity across the SCC.
    """
    TARGETS_PER_SOURCE = 200  # cap targets per source for wide node coverage
    pool: dict = {}
    shuffled = list(nodes)
    rng.shuffle(shuffled)
    for src in shuffled:
        if len(pool) >= target_total:
            break
        dist_dict = nx.single_source_dijkstra_path_length(G, src, weight="w")
        candidates = [(t, d) for t, d in dist_dict.items() if t != src and d > 0]
        if not candidates:
            continue
        sample_size = min(len(candidates), TARGETS_PER_SOURCE)
        for t, d in rng.sample(candidates, sample_size):
            pool[(src, t)] = d
    log.info(
        "Pool: %d deduped directed pairs from %d+ unique source nodes",
        len(pool), min(len(shuffled), target_total // TARGETS_PER_SOURCE),
    )
    return pool

def _stratified_split(pool: dict, cutoffs: np.ndarray, rng: random.Random) -> dict:
    """Draw test set FIRST uniformly at random from the full pool (natural distance
    skew, shared node-pair identity across conditions), then stratify train/val
    from what remains.

    Fix 1: test set is now a true uniform random draw -- not biased leftover residual.
    Fix 2: caller must pass the same pool keys across conditions so the same (u,v)
           pairs appear in every condition's test set (label them separately per condition).
    """
    # Shuffle the whole pool so every slice below is a random sample.
    all_pairs = list(pool.items())
    rng.shuffle(all_pairs)

    # Draw test set FIRST -- uniform random, reflects natural distance skew
    n_test = min(config.N_TEST, len(all_pairs))
    test_items = all_pairs[:n_test]
    remaining = all_pairs[n_test:]

    # Stratify train/val from what remains
    # Group the remaining pairs by distance decile and shuffle within each decile.
    buckets: dict = {i: [] for i in range(config.N_DECILES)}
    for (u, v), d in remaining:
        buckets[_decile_bucket(d, cutoffs)].append((u, v, d))
    for b in buckets.values():
        rng.shuffle(b)

    # Equal per-decile quotas for train and val; train takes its share first, val the next slice.
    n_train_per_bucket = config.N_TRAIN // config.N_DECILES
    n_val_per_bucket = config.N_VAL // config.N_DECILES

    train_rows, val_rows = [], []
    for bucket_idx, items in buckets.items():
        n_train_here = min(n_train_per_bucket, len(items))
        n_val_here = min(n_val_per_bucket, len(items) - n_train_here)
        if n_train_here < n_train_per_bucket or n_val_here < n_val_per_bucket:
            log.warning(
                "Decile bucket %d short of quota: has %d, needed %d (train)+%d (val)",
                bucket_idx, len(items), n_train_per_bucket, n_val_per_bucket,
            )
        train_rows.extend(items[:n_train_here])
        val_rows.extend(items[n_train_here:n_train_here + n_val_here])

    # Flatten the test items to (u, v, d) rows like the other splits.
    test_rows = [(u, v, d) for (u, v), d in test_items]

    log.info(
        "Split: train=%d val=%d test=%d (test drawn first, uniform random)",
        len(train_rows), len(val_rows), len(test_rows),
    )
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def _write_split(rows: list, node_to_idx: dict, out_path) -> None:
    """Write (u, v, dist) rows to a parquet file with graph node ids mapped to embedding row indices.

    node_to_idx maps node id -> row index (from node_index.parquet). The
    output keeps only the columns u_idx, v_idx and dist.
    """
    df = pd.DataFrame(rows, columns=["u", "v", "dist"])
    df["u_idx"] = df["u"].map(node_to_idx)
    df["v_idx"] = df["v"].map(node_to_idx)
    df[["u_idx", "v_idx", "dist"]].to_parquet(out_path)
    log.info("Wrote %s (%d pairs)", out_path, len(df))


def main(args: argparse.Namespace) -> None:
    """Run Phase 5: sample node pairs and label them with Dijkstra distances per condition.

    Steps: (1) load data/processed/G_scc.gpickle, friction_weights.parquet and
    node_index.parquet (FileNotFoundError if any is missing); (2) for each
    condition (all of config.WEIGHT_CONDITIONS, or --condition), with an RNG
    seeded by config.SAMPLING_SEED: annotate edge weights, estimate decile
    cutoffs from a pilot sample, collect a pool of deduplicated directed
    (source, target, distance) pairs, and split it into test, train and val;
    (3) write data/processed/pairs_{train,val,test}_{condition}.parquet with
    columns u_idx, v_idx and dist.
    """
    config.ensure_dirs()
    log.info("Sampling train=%d val=%d test=%d (deciles=%d, pilot=%d)",
             config.N_TRAIN, config.N_VAL, config.N_TEST,
             config.N_DECILES, config.PILOT_PAIRS)

    # Check that all earlier-phase inputs exist before doing any work.
    g_path = config.PROCESSED_DIR / "G_scc.gpickle"
    weights_path = config.PROCESSED_DIR / "friction_weights.parquet"
    node_index_path = config.PROCESSED_DIR / "node_index.parquet"
    for p in (g_path, weights_path, node_index_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; run the earlier phases first.")

    # Load the graph, per-condition weights and the node-id -> embedding-row map.
    with timed(log, "load graph + weights + node index"):
        with open(g_path, "rb") as f:
            G: nx.MultiDiGraph = pickle.load(f)
        weights_df = pd.read_parquet(weights_path)
        node_index_df = pd.read_parquet(node_index_path)
        node_to_idx = {int(nid): idx for idx, nid in enumerate(node_index_df["node_id"])}
        nodes = list(G.nodes)

    conditions = [args.condition] if args.condition else config.WEIGHT_CONDITIONS
    # Oversample the raw pool beyond N_TRAIN+N_VAL+N_TEST to absorb uneven decile
    # representation and still leave enough leftover for the uniform-random test draw.
    target_pool_size = int((config.N_TRAIN + config.N_VAL + config.N_TEST) * 1.8)

    # Per-condition pipeline: weights -> pilot cutoffs -> pair pool -> split -> parquet.
    for condition in conditions:
        rng = random.Random(config.SAMPLING_SEED)
        log.info("=== Condition %s ===", condition)

        with timed(log, f"{condition}: annotate edge weights"):
            _annotate_condition_weight(G, weights_df, condition)

        with timed(log, f"{condition}: pilot decile estimation"):
            cutoffs = _pilot_deciles(G, nodes, rng)

        with timed(log, f"{condition}: collect pair pool (target={target_pool_size})"):
            pool = _collect_pair_pool(G, nodes, target_pool_size, rng)
            log.info("Pool size: %d deduped directed pairs", len(pool))

        with timed(log, f"{condition}: stratified train/val/test split"):
            splits = _stratified_split(pool, cutoffs, rng)

        for split_name, rows in splits.items():
            out_path = config.PROCESSED_DIR / f"pairs_{split_name}_{condition}.parquet"
            _write_split(rows, node_to_idx, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=config.WEIGHT_CONDITIONS, default=None,
                        help="label a single weight condition (default: all)")
    main(parser.parse_args())
