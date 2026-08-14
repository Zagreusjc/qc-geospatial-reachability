"""Phase 5 -- Stratified pair sampling and Dijkstra ground-truth labels.

Builds the train/val/test node-pair sets and their exact friction-weighted
shortest-path distances (the supervised regression targets).

Procedure:
  - pilot 10k random pairs -> Dijkstra -> 10 equal-width distance deciles
  - train (500k) + val (50k): stratified across deciles (disjoint)
  - test (50k): uniform random (natural distance skew)
  - labels: networkx single_source_dijkstra_path_length per weight condition
            (directed pairs preserved; drop d*=0; dedupe)

Inputs  : data/processed/G_scc.gpickle with per-condition weights, node_index.
Outputs : data/processed/pairs_{train,val,test}_{W1,W2,W3}.parquet (u_idx, v_idx, dist)
Next    : 07_train.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("05_sampling_labels")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Sampling train=%d val=%d test=%d (deciles=%d, pilot=%d)",
             config.N_TRAIN, config.N_VAL, config.N_TEST,
             config.N_DECILES, config.PILOT_PAIRS)
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=config.WEIGHT_CONDITIONS, default=None,
                        help="label a single weight condition (default: all)")
    main(parser.parse_args())
