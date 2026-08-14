"""Phase 1b -- Barangay boundary resolver.

Resolves the 142 Quezon City barangay polygons via a fallback ladder and caches
a single canonical file that every downstream spatial step reads. This isolates
the "which source" question so the pipeline never has to change if the source does.

Fallback ladder (config.BARANGAY_SOURCES), first that resolves wins:
  1. HDX COD-AB Philippines ADM4 (filter to Quezon City)
  2. Curated PSA/NAMRIA GeoJSON (bendlikeabamboo/barangay-boundaries-repository)
  3. OpenStreetMap barangay relations (admin_level=10) via OSMnx
  4. geoBoundaries PHL ADM4
  5. Synthesized Voronoi proxy zones around barangay centroids (documented limitation)

Outputs : data/processed/qc_barangays.gpkg (validated to ~142 features).
Next    : used by 01_graph_construction.py and 09_isolation_choropleth.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("01b_barangays")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Resolving barangay boundaries; expecting ~%d features", config.N_BARANGAYS)
    for i, src in enumerate(config.BARANGAY_SOURCES, 1):
        log.info("  ladder[%d]: %s (%s)", i, src["name"], src["url"])
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None,
                        help="force a specific ladder source by name")
    main(parser.parse_args())
