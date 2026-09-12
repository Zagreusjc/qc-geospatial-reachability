"""Phase 9 -- Structural Distance Maps.

Turns the best oracle's distance estimates into the barangay-level output.

Steps   : per-node isolation score from the best oracle (default: highest
          test-set Spearman rho in ablation_results.csv, expected A-W3-*) ->
          spatial join nodes to barangay polygons -> mean structural isolation
          score per barangay -> Folium choropleth of the 142 barangays, with
          the high-divergence pairs on a second layer.

Per-node isolation score: the exhaustive mean of the oracle's predicted
distance from that node to every other node in the graph (~31,778 x ~31,778
pairs total). Computed in nested batches (config.ISOLATION_SOURCE_BATCH x
config.ISOLATION_TARGET_BATCH) so the full pairwise set never has to be held
in memory/VRAM at once.

Inputs  : outputs/models/ best run, outputs/tables/ablation_results.csv,
          data/processed/qc_barangays.gpkg, data/processed/nodes_scc.parquet,
          data/processed/node_index.parquet, outputs/embeddings/Z_*.npy,
          data/processed/high_divergence_pairs.parquet.
Outputs : outputs/maps/structural_distance_map.html,
          outputs/tables/barangay_isolation.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402
from importlib import import_module  # noqa: E402

models_mod = import_module("06_models")

log = get_logger("09_isolation_choropleth")


def _pick_best_run(run_id: str | None):
    if run_id:
        path = config.MODELS_DIR / f"{run_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        return path

    results_path = config.TABLES_DIR / "ablation_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} not found; run 08_evaluate.py first.")
    df = pd.read_csv(results_path)
    runs = df[df["row_type"] == "run"]
    if runs.empty:
        raise RuntimeError(f"No per-run rows found in {results_path}")
    best = runs.loc[runs["spearman"].idxmax()]
    log.info("Best run selected: %s (spearman=%.4f)", best["run_id"], best["spearman"])
    return config.MODELS_DIR / f"{best['run_id']}.pt"


def _node_isolation_scores(checkpoint) -> pd.DataFrame:
    device = config.DEVICE
    weight, arch = checkpoint["weight_condition"], checkpoint["arch"]
    Z = np.load(config.EMBEDDINGS_DIR / f"Z_{weight}.npy").astype(np.float32)
    n_nodes = Z.shape[0]
    Z_t = torch.from_numpy(Z).to(device)

    model = models_mod.build_model(arch).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    sigma, mu = checkpoint["sigma"], checkpoint["mu"]
    src_batch = config.ISOLATION_SOURCE_BATCH
    tgt_batch = config.ISOLATION_TARGET_BATCH
    log.info("Exhaustive per-node isolation: %d x %d pairs (source batch=%d, target batch=%d, device=%s)",
              n_nodes, n_nodes, src_batch, tgt_batch, device)

    sums = torch.zeros(n_nodes, dtype=torch.float64, device=device)
    with torch.no_grad():
        # Sum of predicted distance to every other node, including self-pairs
        # for now -- the self-pair contribution is computed once and removed
        # below, which is simpler than masking it out of every batch.
        for s_start in range(0, n_nodes, src_batch):
            s_end = min(s_start + src_batch, n_nodes)
            n_src = s_end - s_start
            src_idx = torch.arange(s_start, s_end, device=device)

            for t_start in range(0, n_nodes, tgt_batch):
                t_end = min(t_start + tgt_batch, n_nodes)
                n_tgt = t_end - t_start
                tgt_idx = torch.arange(t_start, t_end, device=device)

                src_rep = src_idx.repeat_interleave(n_tgt)
                tgt_rep = tgt_idx.repeat(n_src)
                pred_std = model(Z_t[src_rep], Z_t[tgt_rep])
                pred = (pred_std.double() * sigma + mu).view(n_src, n_tgt)
                sums[s_start:s_end] += pred.sum(dim=1)

            log.info("  isolation: sources %d/%d done", s_end, n_nodes)

        self_pred = model(Z_t, Z_t).double() * sigma + mu  # d(u, u) for every node
        sums -= self_pred

    scores = (sums / (n_nodes - 1)).cpu().numpy()
    node_index_df = pd.read_parquet(config.PROCESSED_DIR / "node_index.parquet")
    return pd.DataFrame({"node_id": node_index_df["node_id"].to_numpy(), "isolation_score": scores})


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Building Structural Distance Maps for %d barangays", config.N_BARANGAYS)

    run_path = _pick_best_run(args.run_id)
    checkpoint = torch.load(run_path, map_location=config.DEVICE, weights_only=False)
    log.info("Using run %s (%s / %s / seed %s)",
             run_path.stem, checkpoint["weight_condition"], checkpoint["arch"], checkpoint["seed"])

    with timed(log, "compute per-node isolation scores"):
        node_scores = _node_isolation_scores(checkpoint)

    with timed(log, "spatial join nodes -> barangays"):
        # nodes_scc.parquet is indexed by "osmid" (our graph node id) and already
        # carries both UTM (x, y) and WGS84 (lon, lat) columns from Phase 1's
        # ox.project_graph step, so no reprojection is needed here.
        nodes_gdf = gpd.read_parquet(config.PROCESSED_DIR / "nodes_scc.parquet")
        nodes_gdf = nodes_gdf.reset_index().rename(columns={"osmid": "node_id"})
        nodes_gdf = gpd.GeoDataFrame(
            nodes_gdf.drop(columns=["geometry"]),
            geometry=gpd.points_from_xy(nodes_gdf["lon"], nodes_gdf["lat"]),
            crs=config.CRS_WGS84,
        )
        nodes_gdf = nodes_gdf.merge(node_scores, on="node_id", how="inner")

        barangays = gpd.read_file(config.PROCESSED_DIR / "qc_barangays.gpkg", layer="barangays")
        name_col = next((c for c in barangays.columns if c.lower() in ("adm4_en", "adm4_name", "name_4")), None)
        if name_col is None:
            raise RuntimeError(f"No barangay-name column found among {list(barangays.columns)}")

        joined = gpd.sjoin(nodes_gdf, barangays[[name_col, "geometry"]], how="left", predicate="within")
        barangay_scores = (
            joined.groupby(name_col)["isolation_score"]
            .agg(mean_isolation_score="mean", n_nodes="count")
            .reset_index()
            .rename(columns={name_col: "barangay_name"})
        )
        log.info("Aggregated isolation scores for %d/%d barangays", len(barangay_scores), config.N_BARANGAYS)

    out_csv = config.TABLES_DIR / "barangay_isolation.csv"
    barangay_scores.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    with timed(log, "build Folium choropleth"):
        barangays_scored = barangays.merge(
            barangay_scores, left_on=name_col, right_on="barangay_name", how="left"
        )
        centroid = barangays_scored.geometry.unary_union.centroid
        fmap = folium.Map(location=[centroid.y, centroid.x], zoom_start=11, tiles="cartodbpositron")

        folium.Choropleth(
            geo_data=barangays_scored.__geo_interface__,
            data=barangay_scores,
            columns=["barangay_name", "mean_isolation_score"],
            key_on=f"feature.properties.{name_col}",
            fill_color="YlOrRd",
            fill_opacity=0.75,
            line_opacity=0.3,
            legend_name=f"Mean structural isolation ({checkpoint['weight_condition']}/{checkpoint['arch']})",
            nan_fill_color="lightgrey",
        ).add_to(fmap)

        folium.GeoJson(
            barangays_scored,
            name="barangays",
            tooltip=folium.GeoJsonTooltip(fields=[name_col, "mean_isolation_score"]),
            style_function=lambda _: {"fillOpacity": 0, "weight": 0.5, "color": "#555555"},
        ).add_to(fmap)

        hd_path = config.PROCESSED_DIR / "high_divergence_pairs.parquet"
        if hd_path.exists():
            hd_df = pd.read_parquet(hd_path)
            node_index_df = pd.read_parquet(config.PROCESSED_DIR / "node_index.parquet")
            idx_to_id = node_index_df["node_id"].to_dict()
            id_to_geom = dict(zip(nodes_gdf["node_id"], nodes_gdf.geometry))
            layer = folium.FeatureGroup(name="High-divergence pairs (top 5%)")
            for _, row in hd_df.head(200).iterrows():
                u_id = idx_to_id.get(int(row["u_idx"]))
                v_id = idx_to_id.get(int(row["v_idx"]))
                gu, gv = id_to_geom.get(u_id), id_to_geom.get(v_id)
                if gu is None or gv is None:
                    continue
                folium.PolyLine(
                    [(gu.y, gu.x), (gv.y, gv.x)], color="#3b6ea5", weight=1.5, opacity=0.6,
                ).add_to(layer)
            layer.add_to(fmap)

        folium.LayerControl().add_to(fmap)

        map_path = config.MAPS_DIR / "structural_distance_map.html"
        fmap.save(str(map_path))
        log.info("Wrote %s", map_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="oracle run to map (default: best)")
    main(parser.parse_args())
