"""Phase 7 -- Train the 3x2x3 ablation.

Trains every cell of the ablation grid: {W1,W2,W3} x {Siamese, MLP} x {seed 0,1,2}
= 18 runs (labelled A-W1-seed0 .. B-W3-seed2 via config.run_id).

Training: MSE loss vs Dijkstra labels, Adam (lr=1e-3, weight_decay=1e-5), cosine
annealing over 100 epochs, early stopping on validation MAE (patience 10), z-score
label standardization (inverted at eval).

Inputs  : outputs/embeddings/Z_*.npy, data/processed/pairs_*_*.parquet
Outputs : outputs/models/<run_id>.pt (best checkpoint), training-curve figures.
Next    : 08_evaluate.py

CLI examples:
  python src/07_train.py                       # all 18 runs
  python src/07_train.py --condition W3         # only W3 cells (6 runs)
  python src/07_train.py --arch siamese --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("07_train")


def selected_runs(args: argparse.Namespace):
    for weight, arch, seed in config.ablation_runs():
        if args.condition and weight != args.condition:
            continue
        if args.arch and arch != args.arch:
            continue
        if args.seed is not None and seed != args.seed:
            continue
        yield weight, arch, seed


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    runs = list(selected_runs(args))
    log.info("Selected %d ablation run(s):", len(runs))
    for weight, arch, seed in runs:
        log.info("  %s", config.run_id(weight, arch, seed))
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", choices=config.WEIGHT_CONDITIONS, default=None)
    parser.add_argument("--arch", choices=config.ARCHITECTURES, default=None)
    parser.add_argument("--seed", type=int, default=None)
    main(parser.parse_args())
