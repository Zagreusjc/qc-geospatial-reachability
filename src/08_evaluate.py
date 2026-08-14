"""Phase 8 -- Evaluation and ablation results.

Evaluates every trained run on the shared held-out test set and assembles the
ablation results table.

Metrics : Spearman rho (primary), MAE, RMSE, RMSE/MAE ratio, multiplicative
          distortion (mean/worst/std; exclude d*=0) vs Thorup-Zwick (2k-1) at
          k=2,3. Euclidean great-circle baseline on the same pairs. High-divergence
          pairs = Euclidean distortion above the 95th percentile (saved for maps).
          Ablation: mean +/- std across seeds, Cohen's d (H3: W1 vs W3; H4: Siamese
          vs MLP), bootstrap confidence intervals.

Inputs  : outputs/models/*.pt, embeddings, test pairs, node coordinates.
Outputs : outputs/tables/ablation_results.csv,
          data/processed/high_divergence_pairs.parquet
Next    : 09_isolation_choropleth.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("08_evaluate")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Evaluating ablation; primary metric = Spearman rho")
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
