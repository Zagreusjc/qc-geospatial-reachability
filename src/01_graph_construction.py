"""Phase 1 -- Graph construction, cleaning, and SCC extraction.

Builds the directed road-network multigraph for Quezon City and cleans it into
the study graph G_scc.

Inputs  : data/raw/ HOTOSM roads, data/processed/qc_barangays.gpkg (for logging
          the barangay distribution of excluded nodes).
Steps   : build (OSMnx) -> highway-class filter -> missing-tag default (+log %)
          -> project to UTM 51N for length (keep WGS84 copy) -> simplify /
          consolidate -> dedupe self-loops & parallel edges -> largest SCC.
Outputs : data/processed/G_scc.gpickle, node/edge GeoDataFrames,
          data/processed/graph_stats.json (counts, class distribution,
          % defaulted tags, SCC-excluded node count/%/barangay distribution).
Next    : 02_ebc.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("01_graph_construction")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Study area: %s", config.PLACE_NAME)
    log.info("Highway whitelist: %s", ", ".join(config.HIGHWAY_WHITELIST))
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
