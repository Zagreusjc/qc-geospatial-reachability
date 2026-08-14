"""Phase 4 -- Node2Vec+ embeddings.

Generates 128-dimensional node embeddings per weight condition using PecanPy's
weight-aware Node2Vec+ (extend=True). q<1 biases walks toward global structure,
which suits long-range distance approximation.

Params  : dim=128, walk_length=80, num_walks=10, p=1.0, q=0.5, window=10 (config).
Inputs  : per-condition .edg edgelists from Phase 3.
Outputs : outputs/embeddings/Z_W1.npy, Z_W2.npy, Z_W3.npy;
          data/processed/node_index.parquet (PecanPy id <-> OSM node id);
          optional p/q sensitivity table.
Next    : 05_sampling_labels.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("04_embeddings")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Node2Vec+ dim=%d L=%d r=%d p=%.2f q=%.2f window=%d extend=%s",
             config.N2V_DIM, config.N2V_WALK_LENGTH, config.N2V_NUM_WALKS,
             config.N2V_P, config.N2V_Q, config.N2V_WINDOW, config.N2V_EXTEND)
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=config.WEIGHT_CONDITIONS, default=None,
                        help="embed a single weight condition (default: all)")
    parser.add_argument("--sweep", action="store_true", help="run p/q sensitivity sweep")
    main(parser.parse_args())
