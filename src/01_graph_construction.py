"""Phase 1 -- Graph construction, cleaning, and SCC extraction.

Builds the directed road-network multigraph for Quezon City and cleans it into
the study graph G_scc.

Inputs  : data/raw/hotosm_phl_roads.* (HOTOSM roads, whatever format HDX serves --
          OSM XML/PBF extract or a line-vector file), data/processed/qc_barangays.gpkg
          (both the "barangays" layer, for logging the barangay distribution of
          excluded nodes, and the "qc_boundary" layer, for clipping).
Steps   : build (OSMnx if OSM XML/PBF, else a manual GeoDataFrame->graph builder for
          plain vector line files) -> truncate to the exact Quezon City boundary
          polygon -> highway-class filter -> missing-tag default (+log %) -> project
          to UTM 51N for length (keep WGS84 copy) -> simplify/consolidate -> dedupe
          self-loops & parallel edges -> largest SCC.
Outputs : data/processed/G_scc.gpickle, data/processed/nodes_scc.parquet,
          data/processed/edges_scc.parquet, data/processed/graph_stats.json
          (counts, class distribution, % defaulted tags, SCC-excluded node
          count/%/barangay distribution).
Next    : 02_ebc.py
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import MultiPoint, Point
from shapely.ops import split as shapely_split
from shapely.ops import unary_union
from shapely.strtree import STRtree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402

log = get_logger("01_graph_construction")

ONEWAY_TRUE = {"yes", "true", "1", "-1"}


_ROADS_DATA_EXTS = (".gpkg", ".geojson", ".json", ".shp", ".osm", ".xml", ".pbf")


def _find_roads_file():
    candidates = sorted(
        p for p in config.RAW_DIR.glob("hotosm_phl_roads*")
        if p.suffix.lower() in _ROADS_DATA_EXTS
    )
    # Prefer the "lines" resource over "polygons" (HOTOSM ships both; only lines
    # are the road network) if a filename distinguishes them.
    lines_only = [p for p in candidates if "polygon" not in p.stem.lower()]
    if lines_only:
        candidates = lines_only
    if not candidates:
        raise FileNotFoundError(
            f"No HOTOSM roads file found under {config.RAW_DIR} "
            "(expected a file named hotosm_phl_roads*, e.g. "
            "hotosm_phl_roads_lines_gpkg.gpkg); run 00_download_data.py first."
        )
    if len(candidates) > 1:
        log.warning("Multiple HOTOSM roads files found, using the first: %s", candidates)
    return candidates[0]


def _build_graph_from_osm_xml(path) -> nx.MultiDiGraph:
    log.info("Building graph from OSM XML/PBF via osmnx.graph_from_xml: %s", path)
    return ox.graph_from_xml(path, simplify=False, retain_all=True)


def _read_and_classify_lines(path, boundary) -> tuple[gpd.GeoDataFrame, dict]:
    """Read the roads file (spatially pre-filtered to `boundary`), default missing
    highway tags, and drop non-whitelisted road classes -- all at the GeoDataFrame
    level, before any topology is built.

    HOTOSM's "Philippines Roads" export is nationwide, not QC-specific, so
    `boundary` is passed as a `mask` to `gpd.read_file()`: this pushes the
    intersects-filter down to the OGR layer (GPKG spatial index), so only
    features touching Quezon City are ever materialized in memory instead of
    loading every road in the country first.
    """
    log.info("Reading with spatial mask pre-filter (nationwide file -> QC only)")
    mask = gpd.GeoSeries([boundary], crs=config.CRS_WGS84)
    lines = gpd.read_file(path, mask=mask)
    log.info("Loaded %d road features after spatial pre-filter", len(lines))
    if lines.crs is None:
        log.warning("Roads file has no CRS declared; assuming %s", config.CRS_WGS84)
        lines = lines.set_crs(config.CRS_WGS84)
    else:
        lines = lines.to_crs(config.CRS_WGS84)

    highway_col = next((c for c in lines.columns if c.lower() == "highway"), None)
    if highway_col is None:
        raise RuntimeError(f"No 'highway' column found among {list(lines.columns)}")
    if highway_col != "highway":
        lines = lines.rename(columns={highway_col: "highway"})
    oneway_col = next((c for c in lines.columns if c.lower() == "oneway"), None)
    if oneway_col is None:
        lines["oneway"] = None
    elif oneway_col != "oneway":
        lines = lines.rename(columns={oneway_col: "oneway"})

    missing = lines["highway"].isna() | (lines["highway"].astype(str).str.strip() == "")
    n_defaulted = int(missing.sum())
    lines.loc[missing, "highway"] = config.DEFAULT_ROAD_CLASS

    whitelist = set(config.HIGHWAY_WHITELIST)
    class_counts = lines["highway"].value_counts().to_dict()
    n_total = len(lines)
    lines = lines[lines["highway"].isin(whitelist)].copy().reset_index(drop=True)
    n_removed = n_total - len(lines)

    stats = {
        "n_edges_removed_non_whitelisted": n_removed,
        "n_edges_defaulted_road_class": n_defaulted,
        "pct_edges_defaulted": round(100 * n_defaulted / max(len(lines), 1), 3),
        "road_class_distribution": {k: v for k, v in class_counts.items() if k in whitelist},
    }
    log.info(
        "Whitelist filter: kept %d/%d features (%d defaulted to '%s')",
        len(lines), n_total, n_defaulted, config.DEFAULT_ROAD_CLASS,
    )
    return lines, stats


def _node_line_network(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split every line at every point it truly intersects/touches another line,
    not just at line endpoints, so T-junctions become shared graph nodes.

    Raw OSM/HOTOSM line exports frequently digitize a long road as one continuous
    LineString that side streets terminate against midway (a T-junction). Naive
    endpoint-only topology-building misses these entirely, fragmenting the
    network into disconnected pieces. `unary_union` performs GEOS "noding": it
    splits the whole input line set at every mutual intersection point, which is
    used here purely to *discover* junction points; original per-feature
    attributes (highway, oneway) are then reattached by splitting each original
    line at the subset of junction points that lie on its interior.
    """
    log.info("Noding network topology (%d input features)...", len(lines))
    geoms = list(lines.geometry.values)
    noded = unary_union(geoms)
    pieces = [noded] if noded.geom_type == "LineString" else list(noded.geoms)

    junction_pts = set()
    for piece in pieces:
        coords = list(piece.coords)
        junction_pts.add(coords[0])
        junction_pts.add(coords[-1])

    jp_geoms = [Point(p) for p in junction_pts]
    tree = STRtree(jp_geoms)

    out_rows = []
    n_split = 0
    for _, row in lines.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        endpoints = {geom.coords[0], geom.coords[-1]}
        candidate_idxs = tree.query(geom)
        split_pts = [
            jp_geoms[i] for i in candidate_idxs
            if (jp_geoms[i].x, jp_geoms[i].y) not in endpoints
            and geom.distance(jp_geoms[i]) < 1e-7
        ]
        if not split_pts:
            out_rows.append(row)
            continue
        n_split += 1
        result = shapely_split(geom, MultiPoint(split_pts))
        for sub in result.geoms:
            new_row = row.copy()
            new_row.geometry = sub
            out_rows.append(new_row)

    noded_gdf = gpd.GeoDataFrame(out_rows, crs=lines.crs).reset_index(drop=True)
    log.info(
        "Noding split %d/%d original lines into %d total segments (%d junction points found)",
        n_split, len(lines), len(noded_gdf), len(junction_pts),
    )
    return noded_gdf


def _build_graph_from_line_gdf(path, boundary) -> tuple[nx.MultiDiGraph, dict]:
    """Graph builder for a plain line-vector file (GeoJSON/shp/gpkg) -- reads +
    whitelist-filters at the GeoDataFrame level, nodes the network to capture
    T-junctions, then builds the graph from exact-endpoint topology (now valid
    since every true junction is guaranteed to be a segment endpoint).
    """
    lines, stats = _read_and_classify_lines(path, boundary)
    lines = _node_line_network(lines)

    node_id_of: dict = {}
    node_coords: list = []

    def node_id_for(pt):
        if pt not in node_id_of:
            node_id_of[pt] = len(node_coords)
            node_coords.append(pt)
        return node_id_of[pt]

    lines_utm = lines.to_crs(config.CRS_UTM_51N)

    G = nx.MultiDiGraph(crs=config.CRS_WGS84)
    for (_, row), length_m in zip(lines.iterrows(), lines_utm.geometry.length):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        coords = list(geom.coords)
        u = node_id_for(coords[0])
        v = node_id_for(coords[-1])
        oneway_raw = str(row["oneway"]).strip().lower() if row["oneway"] else "no"
        oneway = oneway_raw in ONEWAY_TRUE

        attrs = dict(highway=row["highway"], oneway=oneway, length=length_m, geometry=geom)
        G.add_edge(u, v, **attrs)
        if not oneway:
            G.add_edge(v, u, **{**attrs, "geometry": geom.reverse()})

    for nid, (x, y) in enumerate(node_coords):
        G.add_node(nid, x=x, y=y)

    log.info("Manual graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G, stats


def _build_raw_graph(roads_path, boundary) -> tuple[nx.MultiDiGraph, dict | None]:
    suffix = roads_path.suffix.lower()
    if suffix in (".osm", ".xml"):
        return _build_graph_from_osm_xml(roads_path), None
    if suffix == ".pbf":
        raise NotImplementedError(
            f"{roads_path} is a PBF extract; osmnx.graph_from_xml only reads OSM "
            "XML. Convert it first, e.g. `osmium cat roads.pbf -o roads.osm`, then "
            "re-run this script."
        )
    return _build_graph_from_line_gdf(roads_path, boundary)


def _filter_highway_whitelist(G: nx.MultiDiGraph) -> tuple[nx.MultiDiGraph, dict]:
    whitelist = set(config.HIGHWAY_WHITELIST)
    to_remove = []
    n_defaulted = 0
    class_counts: dict[str, int] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        if isinstance(hw, list):
            hw = hw[0] if hw else None
        if not hw:
            hw = config.DEFAULT_ROAD_CLASS
            n_defaulted += 1
            data["highway"] = hw
        if hw not in whitelist:
            to_remove.append((u, v, k))
        else:
            class_counts[hw] = class_counts.get(hw, 0) + 1
    G.remove_edges_from(to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    stats = {
        "n_edges_removed_non_whitelisted": len(to_remove),
        "n_edges_defaulted_road_class": n_defaulted,
        "pct_edges_defaulted": round(100 * n_defaulted / max(G.number_of_edges(), 1), 3),
        "road_class_distribution": class_counts,
    }
    return G, stats


def _sanitize_multivalued_edge_attrs(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """ox.simplify_graph merges chains of degree-2 nodes into one edge, aggregating
    any attribute that differs across the merged segments into a list (e.g.
    highway=['residential','tertiary']). Downstream phases (friction weights,
    parquet export) expect a single scalar per edge, so collapse to the first
    value -- the merged edge's dominant/originating road class.
    """
    for _, _, data in G.edges(data=True):
        for key, val in list(data.items()):
            if isinstance(val, list):
                data[key] = val[0] if val else None
    return G


def _dedupe_parallel_and_self_loops(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    G.remove_edges_from([(u, v, k) for u, v, k in G.edges(keys=True) if u == v])
    for u, v in list(set((u, v) for u, v, _ in G.edges(keys=True))):
        parallel = list(G.get_edge_data(u, v).items())
        if len(parallel) > 1:
            parallel.sort(key=lambda kd: kd[1].get("length", float("inf")))
            for k, _ in parallel[1:]:
                if G.has_edge(u, v, k):
                    G.remove_edge(u, v, k)
    return G


def _largest_scc_with_exclusions(G: nx.MultiDiGraph):
    components = list(nx.strongly_connected_components(G))
    largest = max(components, key=len)
    excluded = set(G.nodes) - largest
    G_scc = G.subgraph(largest).copy()
    return G_scc, excluded


def _excluded_barangay_distribution(G: nx.MultiDiGraph, excluded: set, barangays_path) -> dict:
    if not excluded or not barangays_path.exists():
        return {}
    pts = gpd.GeoDataFrame(
        {"node": list(excluded)},
        geometry=gpd.points_from_xy(
            [G.nodes[n]["x"] for n in excluded], [G.nodes[n]["y"] for n in excluded]
        ),
        crs=config.CRS_WGS84,
    )
    barangays = gpd.read_file(barangays_path, layer="barangays")
    name_col = next((c for c in barangays.columns if c.upper() in ("ADM4_EN", "ADM4_NAME", "NAME_4")), None)
    joined = gpd.sjoin(pts, barangays, how="left", predicate="within")
    if name_col is None:
        return {"_note": "no ADM4 name column found for barangay breakdown"}
    return joined[name_col].value_counts(dropna=False).to_dict()


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Study area: %s", config.PLACE_NAME)
    log.info("Highway whitelist: %s", ", ".join(config.HIGHWAY_WHITELIST))

    barangays_path = config.PROCESSED_DIR / "qc_barangays.gpkg"
    if not barangays_path.exists():
        raise FileNotFoundError(f"{barangays_path} not found; run 01b_barangays.py first.")

    roads_path = _find_roads_file()
    boundary = gpd.read_file(barangays_path, layer="qc_boundary").geometry.iloc[0]

    with timed(log, "build raw graph"):
        G, class_stats = _build_raw_graph(roads_path, boundary=boundary)
        log.info("Raw graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    with timed(log, "truncate to Quezon City boundary"):
        G = ox.truncate.truncate_graph_polygon(G, boundary, truncate_by_edge=True)
        log.info("Truncated graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if class_stats is None:
        # OSM XML/PBF path: whitelist filtering happens post-hoc on the nx graph
        # (the line-vector path already filtered at the GeoDataFrame level, pre-topology).
        with timed(log, "highway whitelist filter"):
            G, class_stats = _filter_highway_whitelist(G)
            log.info("After whitelist filter: %d nodes, %d edges (%.2f%% edges defaulted)",
                     G.number_of_nodes(), G.number_of_edges(), class_stats["pct_edges_defaulted"])

    with timed(log, "project to UTM 51N"):
        G_wgs84 = G.copy()
        G = ox.project_graph(G, to_crs=config.CRS_UTM_51N)

    with timed(log, "simplify + dedupe"):
        if not G.graph.get("simplified"):
            G = ox.simplify_graph(G)
        G = _sanitize_multivalued_edge_attrs(G)
        G = _dedupe_parallel_and_self_loops(G)
        log.info("After simplify/dedupe: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    with timed(log, "largest SCC extraction"):
        G_scc, excluded = _largest_scc_with_exclusions(G)
        pct_excluded = round(100 * len(excluded) / max(G.number_of_nodes(), 1), 3)
        log.info("Largest SCC: %d/%d nodes kept (%d excluded, %.2f%%)",
                 G_scc.number_of_nodes(), G.number_of_nodes(), len(excluded), pct_excluded)

    with timed(log, "excluded-node barangay distribution"):
        G_wgs84_for_join = G_wgs84.subgraph(
            [n for n in excluded if n in G_wgs84.nodes]
        )
        barangay_dist = _excluded_barangay_distribution(G_wgs84_for_join, set(G_wgs84_for_join.nodes), barangays_path)

    out_gpickle = config.PROCESSED_DIR / "G_scc.gpickle"
    with timed(log, f"write {out_gpickle}"):
        with open(out_gpickle, "wb") as f:
            pickle.dump(G_scc, f)

    with timed(log, "write node/edge GeoDataFrames"):
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_scc)
        nodes_gdf.to_parquet(config.PROCESSED_DIR / "nodes_scc.parquet")
        edges_gdf.to_parquet(config.PROCESSED_DIR / "edges_scc.parquet")

    stats = {
        "n_nodes_scc": G_scc.number_of_nodes(),
        "n_edges_scc": G_scc.number_of_edges(),
        "n_nodes_pre_scc": G.number_of_nodes(),
        "n_nodes_excluded": len(excluded),
        "pct_nodes_excluded": pct_excluded,
        "excluded_barangay_distribution": barangay_dist,
        **class_stats,
    }
    stats_path = config.PROCESSED_DIR / "graph_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s", stats_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
