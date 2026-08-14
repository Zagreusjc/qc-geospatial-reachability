"""Phase 9 -- Structural Distance Maps.

Turns the best oracle's distance estimates into the barangay-level output.

Steps   : per-node distance estimates from the best oracle (A-W3) -> spatial join
          nodes to barangay polygons -> mean structural isolation score per
          barangay -> Folium choropleth of the 142 barangays, with the
          high-divergence pairs on a second layer.

Inputs  : outputs/models/ best run, data/processed/qc_barangays.gpkg,
          data/processed/high_divergence_pairs.parquet, node coordinates.
Outputs : outputs/maps/structural_distance_map.html,
          outputs/tables/barangay_isolation.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("09_isolation_choropleth")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Building Structural Distance Maps for %d barangays", config.N_BARANGAYS)
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="oracle run to map (default: best)")
    main(parser.parse_args())
