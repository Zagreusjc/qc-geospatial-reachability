"""Phase 0 -- Data acquisition.

Downloads the open datasets and freezes static snapshots with declared retrieval
dates for reproducibility.

Inputs  : HDX HOTOSM Philippines Roads, PSA/NAMRIA COD-AB boundaries (config URLs).
Outputs : data/raw/ cached files; data/DATA_MANIFEST.md updated with dates/versions.
Next    : 01_graph_construction.py, 01b_barangays.py
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, not_implemented  # noqa: E402

log = get_logger("00_download_data")


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Would download HOTOSM roads from %s", config.HOTOSM_ROADS_HDX)
    log.info("Would download COD-AB boundaries from %s", config.COD_AB_HDX)
    not_implemented(__file__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    main(parser.parse_args())
