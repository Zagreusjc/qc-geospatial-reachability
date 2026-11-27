"""Phase 0 -- Data acquisition.

Downloads/verifies the open datasets and freezes static snapshots with declared
retrieval dates for reproducibility.

Inputs  : HDX HOTOSM Philippines Roads, PSA/NAMRIA COD-AB boundaries (config URLs).
Outputs : data/raw/ cached files; data/DATA_MANIFEST.md updated with dates/versions.
Next    : 01_graph_construction.py, 01b_barangays.py

Notes on how each dataset actually gets into data/raw/:
  - COD-AB admin boundaries (~344MB zipped) are large enough that a scripted
    re-download on every run is wasteful and fragile (the concrete HDX resource
    URL changes per dataset version). This script therefore *verifies* the
    already-extracted geodatabase at config.ADMIN_GDB_RAW rather than fetching it,
    and records its retrieval date in the manifest from the HDX resource metadata
    JSON saved alongside it.
  - HOTOSM roads is fetched with a real HTTP download once a concrete resource
    URL is known (pass --roads-url; the HDX dataset landing page in
    config.HOTOSM_ROADS_HDX is not itself a downloadable file).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed, update_manifest_row  # noqa: E402

log = get_logger("00_download_data")

MANIFEST_PATH = config.DATA_DIR / "DATA_MANIFEST.md"
ADMIN_METADATA_JSON = config.RAW_DIR / "metadata-phl_admin_boundaries-gdb-zip.json"


def _verify_admin_boundaries() -> None:
    """Check that the extracted admin-boundaries geodatabase exists and log it.

    Takes no arguments and returns nothing. If config.ADMIN_GDB_RAW is missing,
    logs an error with download instructions and stops. Otherwise it updates the
    "COD-AB" row of data/DATA_MANIFEST.md, using the "created" date and "size"
    from the HDX metadata JSON when that file exists and today's UTC date if not.
    """
    if not config.ADMIN_GDB_RAW.exists():
        log.error(
            "Admin boundaries not found at %s. Download the COD-AB Philippines "
            "geodatabase from %s, extract it, and place the resulting "
            "'phl_admin_boundaries.gdb' folder at that path.",
            config.ADMIN_GDB_RAW, config.COD_AB_HDX,
        )
        return

    # Default to today's date; overridden below by the HDX metadata if present.
    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    version_notes = "COD-AB PHL geodatabase (ADM0-ADM4), NAMRIA geometry / PSA PSGC codes"
    if ADMIN_METADATA_JSON.exists():
        meta = json.loads(ADMIN_METADATA_JSON.read_text(encoding="utf-8"))
        created = meta.get("created")
        if created:
            retrieved = created.split("T")[0]
        size = meta.get("size")
        if size:
            version_notes += f"; resource size {size}"
        log.info("Admin boundaries verified at %s (HDX metadata: created=%s)",
                 config.ADMIN_GDB_RAW, created)
    else:
        log.info("Admin boundaries verified at %s (no metadata JSON found alongside it)",
                  config.ADMIN_GDB_RAW)

    # Record the retrieval date and version notes in the data manifest.
    update_manifest_row(MANIFEST_PATH, "COD-AB", retrieved, version_notes)


def _download_roads(url: str | None, force: bool) -> None:
    """Download the HOTOSM Philippines roads file into data/raw/ unless cached.

    Takes the direct resource URL (or None) and a force flag. If a
    hotosm_phl_roads* file with a known extension is already present and force is
    False, only its manifest row is refreshed (date from the file modification
    time). With no file and no URL it logs a warning and returns. Otherwise it
    streams the URL to disk, rejects files under 1024 bytes, and records the
    download in data/DATA_MANIFEST.md.
    """
    # Reuse a cached roads file when one exists, unless --force was given.
    dest = config.RAW_DIR / "hotosm_phl_roads.geojson"
    existing = [
        p for p in config.RAW_DIR.glob("hotosm_phl_roads*")
        if p.suffix.lower() in (".gpkg", ".geojson", ".json", ".shp", ".osm", ".xml", ".pbf")
    ]
    if existing and not force:
        log.info("HOTOSM roads already present at %s (use --force to re-download)", existing[0])
        retrieved = datetime.fromtimestamp(existing[0].stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        update_manifest_row(MANIFEST_PATH, "HOTOSM Philippines Roads", retrieved, existing[0].name)
        return

    # Nothing cached and no URL: tell the user how to obtain one.
    if not url:
        log.warning(
            "No HOTOSM roads file present and no --roads-url given. Visit %s, "
            "find the current 'Roads' resource, and re-run with "
            "--roads-url <direct file URL>.",
            config.HOTOSM_ROADS_HDX,
        )
        return

    # Stream the download to disk; the response content type picks the file suffix.
    with timed(log, f"download HOTOSM roads from {url}"):
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            content_type = r.headers.get("content-type", "")
            if "json" in content_type:
                dest = dest.with_suffix(".geojson")
            elif "zip" in content_type or url.endswith(".zip"):
                dest = dest.with_suffix(".zip")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)

    # Sanity-check the saved file, then record it in the manifest.
    size_bytes = dest.stat().st_size
    if size_bytes < 1024:
        raise RuntimeError(f"Downloaded HOTOSM roads file is suspiciously small ({size_bytes}B)")
    log.info("Saved HOTOSM roads to %s (%.1f MB)", dest, size_bytes / (1024 * 1024))

    retrieved = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    update_manifest_row(MANIFEST_PATH, "HOTOSM Philippines Roads", retrieved, url)


def main(args: argparse.Namespace) -> None:
    """Run Phase 0: verify admin boundaries, then fetch or register the roads file.

    Steps in order: create the project directories (config.ensure_dirs), verify
    the COD-AB geodatabase at config.ADMIN_GDB_RAW, then download or register
    the HOTOSM roads file using args.roads_url and args.force. Reads
    data/raw/metadata-phl_admin_boundaries-gdb-zip.json when present; writes
    data/raw/hotosm_phl_roads.* (only on a real download) and updates the rows
    in data/DATA_MANIFEST.md.
    """
    config.ensure_dirs()
    _verify_admin_boundaries()
    _download_roads(args.roads_url, args.force)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--roads-url", default=None,
                        help="direct download URL for the HOTOSM Philippines Roads resource")
    main(parser.parse_args())
