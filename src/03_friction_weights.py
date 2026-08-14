"""Phase 3 -- Friction weight assignment (three ablation conditions).

Assigns the three edge-weight conditions that define the three friction-weighted
graphs used in the ablation:
  W1 = length / speed(road_class)                     (traversal cost only)
  W2 = normalized EBC                                  (structural load only)
  W3 = W1 * (1 + alpha * normalized EBC), alpha=1.0    (composite, primary)

Inputs  : data/processed/G_scc.gpickle, data/processed/ebc.parquet
Outputs : per-condition edge weights on the graph, per-condition .edg edgelists
          (src dst weight) for PecanPy, weight-distribution figures, plus an
          alpha-sensitivity variant set.
Next    : 04_embeddings.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("03_friction_weights")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Weight conditions: %s (alpha=%s)", config.WEIGHT_CONDITIONS, config.ALPHA)
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
