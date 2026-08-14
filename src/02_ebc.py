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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("02_ebc")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    mode = args.mode or config.EBC_MODE
    log.info("EBC mode=%s k=%s seed=%s", mode, config.EBC_K, config.EBC_SEED)
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["exact", "approx"], default=None,
                        help="override config.EBC_MODE")
    main(parser.parse_args())
