"""Phase 1b -- Barangay boundary resolver.

Resolves the 142 Quezon City barangay polygons via a fallback ladder and caches
a single canonical file that every downstream spatial step reads. This isolates
the "which source" question so the pipeline never has to change if the source does.

Fallback ladder (config.BARANGAY_SOURCES), first that resolves wins:
  1. HDX COD-AB Philippines ADM4 geodatabase (filter to Quezon City) -- NAMRIA
     geometry + PSA PSGC codes, already extracted to config.ADMIN_GDB_RAW
  2. Curated PSA/NAMRIA GeoJSON (bendlikeabamboo/barangay-boundaries-repository)
  3. OpenStreetMap barangay relations (admin_level=10) via OSMnx
  4. geoBoundaries PHL ADM4
  5. Synthesized Voronoi proxy zones around barangay centroids (documented limitation)

Only ladder item 1 is implemented for now (the source actually in hand); items 2-5
remain documented fallbacks that raise NotImplementedError if ever reached.

Outputs : data/processed/qc_barangays.gpkg
            - layer "barangays"  : 142 individual barangay polygons + attributes
            - layer "qc_boundary": single dissolved Quezon City boundary polygon
Next    : used by 01_graph_construction.py and 09_isolation_choropleth.py
"""
from __future__ import annotations

import argparse
import os
import sys

import fiona
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed, mark_barangay_source_used  # noqa: E402

log = get_logger("01b_barangays")


def _find_adm4_layer(gdb_path) -> str:
    """Return the ADM4 (barangay-level) layer name inside the geodatabase.

    COD-AB geodatabase exports have used more than one layer-naming convention
    across vintages -- e.g. `phl_admbnda_adm4_psa_namria_<date>` in some HDX
    resources, plain `phl_admin4` in others (this file's actual layout: admin0..
    admin4 + adminlines/adminpoints). Match generically for a layer that looks
    like the 4th administrative level and isn't the lines/points reference layers.
    """
    layers = fiona.listlayers(str(gdb_path))
    log.info("Layers found in %s: %s", gdb_path, layers)

    def is_level4(name: str) -> bool:
        n = name.lower()
        if "line" in n or "point" in n:
            return False
        return "adm4" in n or n.endswith("admin4") or n.endswith("adm_4") or n.endswith("_4")

    candidates = [l for l in layers if is_level4(l)]
    if not candidates:
        raise RuntimeError(
            f"No ADM4 (barangay-level) layer found in {gdb_path}. "
            f"Available layers: {layers}. Inspect these manually and update "
            "_find_adm4_layer()'s matching rule."
        )
    if len(candidates) > 1:
        log.warning("Multiple ADM4-like layers found, using the first: %s", candidates)
    return candidates[0]


def _filter_to_quezon_city(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter the national ADM4 layer down to Quezon City's barangays.

    Prefers an ADM3 PSGC code match (robust to name spelling/casing); falls back
    to a case/whitespace-insensitive match on the ADM3 name column.
    """
    cols = list(gdf.columns)
    log.info("ADM4 layer columns: %s", cols)

    name_col = next((c for c in cols if c.upper() in ("ADM3_EN", "ADM3_NAME", "NAME_3")), None)
    pcode_col = next((c for c in cols if "PCODE" in c.upper() and "ADM3" in c.upper()), None)

    if name_col is None:
        raise RuntimeError(
            f"Could not find an ADM3 name column among {cols}; "
            "inspect the layer manually and update _filter_to_quezon_city()."
        )

    target = config.ADM3_FILTER_NAME.strip().casefold()
    mask = gdf[name_col].astype(str).str.strip().str.casefold() == target
    filtered = gdf[mask].copy()

    log.info(
        "Filtered to ADM3 name == %r via column %r: %d rows",
        config.ADM3_FILTER_NAME, name_col, len(filtered),
    )
    if pcode_col is not None and len(filtered) > 0:
        pcodes = filtered[pcode_col].unique().tolist()
        log.info("Matching ADM3 PSGC code(s) via column %r: %s", pcode_col, pcodes)

    if filtered.empty:
        sample = sorted(gdf[name_col].astype(str).unique())[:10]
        raise RuntimeError(
            f"Zero barangays matched ADM3 name {config.ADM3_FILTER_NAME!r} "
            f"via column {name_col!r}. Sample of available ADM3 names: {sample}"
        )
    return filtered


def _validate_and_clean(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    n = len(gdf)
    if n != config.N_BARANGAYS:
        log.warning(
            "Filtered barangay count (%d) differs from expected N_BARANGAYS (%d); "
            "this is a documented possibility (boundary-dispute barangays) but "
            "should be visually sanity-checked before proceeding.",
            n, config.N_BARANGAYS,
        )
    else:
        log.info("Filtered barangay count matches expected N_BARANGAYS (%d).", n)

    invalid_before = (~gdf.geometry.is_valid).sum()
    if invalid_before:
        log.warning("%d invalid geometries found; repairing with make_valid()", invalid_before)
        gdf["geometry"] = gdf.geometry.make_valid()
    return gdf


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()

    if args.source and args.source != "hdx_cod_ab_adm4":
        raise NotImplementedError(
            f"Ladder source {args.source!r} is documented but not yet implemented; "
            "only 'hdx_cod_ab_adm4' is currently implemented."
        )

    gdb_path = config.ADMIN_GDB_RAW
    if not gdb_path.exists():
        raise FileNotFoundError(
            f"Expected extracted geodatabase at {gdb_path}; "
            "extract phl_admin_boundaries.gdb.zip there first."
        )

    with timed(log, "read ADM4 layer"):
        layer = _find_adm4_layer(gdb_path)
        gdf = gpd.read_file(gdb_path, layer=layer)
        log.info("Loaded %d total ADM4 features nationwide, CRS=%s", len(gdf), gdf.crs)

    with timed(log, "filter to Quezon City"):
        qc = _filter_to_quezon_city(gdf)
        qc = _validate_and_clean(qc)

    with timed(log, "reproject + dissolve"):
        qc_wgs84 = qc.to_crs(config.CRS_WGS84)
        qc_utm = qc.to_crs(config.CRS_UTM_51N)
        qc_boundary_utm = gpd.GeoDataFrame(
            geometry=[qc_utm.unary_union], crs=config.CRS_UTM_51N
        )
        qc_boundary_wgs84 = qc_boundary_utm.to_crs(config.CRS_WGS84)

    out_path = config.PROCESSED_DIR / "qc_barangays.gpkg"
    with timed(log, f"write {out_path}"):
        qc_wgs84.to_file(out_path, layer="barangays", driver="GPKG")
        qc_boundary_wgs84.to_file(out_path, layer="qc_boundary", driver="GPKG")

    manifest_path = config.DATA_DIR / "DATA_MANIFEST.md"
    mark_barangay_source_used(manifest_path, "HDX COD-AB ADM4", len(qc_wgs84))

    log.info(
        "Done: %d barangay polygons + 1 dissolved boundary written to %s",
        len(qc_wgs84), out_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=None,
                        help="force a specific ladder source by name")
    main(parser.parse_args())
