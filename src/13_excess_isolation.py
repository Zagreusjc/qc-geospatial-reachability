"""Phase 9b -- Excess (network-caused) isolation per barangay.

The isolation score of 09_isolation_choropleth.py is the mean oracle travel
cost from a node to every other node. Part of that score is just location:
a node at the city edge is far from everything, so its trips are long no
matter how good its roads are. This script removes the location part.

For every ordered pair (u, v) the oracle predicts the travel cost, and the
straight-line (haversine) distance is computed from node coordinates. Over all
~1.01 billion pairs we build the citywide curve "expected cost for a trip of
this straight-line length" (mean oracle cost in 100 m distance bins). Then,
for each node u:

    actual(u)   = mean oracle cost from u to every other node   (= 09's score)
    expected(u) = mean of the curve at the straight-line distances from u
    excess(u)   = actual(u) / expected(u) - 1          (as a fraction)

excess > 0: trips from u cost more than typical trips of the same straight-line
length elsewhere in QC, so the road network itself (detours, one-way streets,
missing links, bottlenecks) makes u isolated. excess < 0: u is better
connected than its location would suggest. Barangay values are the ratio of
the mean actual to the mean expected cost over the barangay's nodes.

Same model, embeddings and batching as 09 (exhaustive, all ordered pairs).

Inputs  : same as 09 (best run, Z_*.npy, node_index, nodes_scc, qc_barangays)
Outputs : outputs/tables/barangay_excess_isolation.csv
          outputs/tables/node_excess_isolation.csv
          outputs/tables/excess_isolation_curve.csv
          outputs/figures/Figure_excess_isolation.png
Usage   : python src/13_excess_isolation.py --run-id C-W3-seed0
"""
from __future__ import annotations

import argparse
import os
import sys

import torch  # must load before numpy/pandas/geopandas on Windows
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402
from importlib import import_module  # noqa: E402

models_mod = import_module("06_models")
iso_mod = import_module("09_isolation_choropleth")
log = get_logger("13_excess_isolation")

EARTH_R_KM = 6371.0088
BIN_KM = 0.1      # width of a straight-line distance bin
N_BINS = 400      # covers 0-40 km; QC's longest straight line is about 25 km


def _haversine_km(lat1, lon1, lat2, lon2):
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * torch.asin(torch.sqrt(a.clamp(0, 1)))


def _excess_scores(checkpoint) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = config.DEVICE
    weight, arch = checkpoint["weight_condition"], checkpoint["arch"]
    Z = np.load(config.EMBEDDINGS_DIR / f"Z_{weight}.npy").astype(np.float32)
    n = Z.shape[0]
    Z_t = torch.from_numpy(Z).to(device)

    node_index = pd.read_parquet(config.PROCESSED_DIR / "node_index.parquet")
    nodes = pd.read_parquet(config.PROCESSED_DIR / "nodes_scc.parquet", columns=["lon", "lat"])
    nodes = nodes.loc[node_index["node_id"].to_numpy()]  # same row order as Z
    lat = torch.tensor(np.radians(nodes["lat"].to_numpy()), dtype=torch.float64, device=device)
    lon = torch.tensor(np.radians(nodes["lon"].to_numpy()), dtype=torch.float64, device=device)

    model = models_mod.build_model(arch).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sigma, mu = checkpoint["sigma"], checkpoint["mu"]

    # per-node sum of predicted cost, and per-node count of targets in each distance bin
    cost_sum = torch.zeros(n, dtype=torch.float64, device=device)
    node_bins = torch.zeros(n, N_BINS, dtype=torch.float64, device=device)
    # citywide sum of predicted cost per distance bin (for the expected-cost curve)
    bin_cost = torch.zeros(N_BINS, dtype=torch.float64, device=device)

    sb, tb = config.ISOLATION_SOURCE_BATCH, config.ISOLATION_TARGET_BATCH
    log.info("Exhaustive excess isolation: %d x %d pairs (device=%s)", n, n, device)
    with torch.no_grad():
        for s0 in range(0, n, sb):
            s1 = min(s0 + sb, n)
            ns = s1 - s0
            src = torch.arange(s0, s1, device=device)
            for t0 in range(0, n, tb):
                t1 = min(t0 + tb, n)
                nt = t1 - t0
                tgt = torch.arange(t0, t1, device=device)
                s_rep, t_rep = src.repeat_interleave(nt), tgt.repeat(ns)
                pred = model(Z_t[s_rep], Z_t[t_rep]).double() * sigma + mu
                d_km = _haversine_km(lat[s_rep], lon[s_rep], lat[t_rep], lon[t_rep])
                b = (d_km / BIN_KM).long().clamp_(0, N_BINS - 1)
                cost_sum[s0:s1] += pred.view(ns, nt).sum(dim=1)
                bin_cost.index_add_(0, b, pred)
                flat = (s_rep - s0) * N_BINS + b
                node_bins[s0:s1] += torch.bincount(flat, minlength=ns * N_BINS).view(ns, N_BINS).double()
            log.info("  sources %d/%d done", s1, n)

        # remove self-pairs (distance 0 -> bin 0)
        self_pred = model(Z_t, Z_t).double() * sigma + mu
        cost_sum -= self_pred
        bin_cost[0] -= self_pred.sum()
        node_bins[:, 0] -= 1

    bin_count = node_bins.sum(dim=0)
    curve = torch.where(bin_count > 0, bin_cost / bin_count.clamp(min=1), torch.zeros_like(bin_cost))
    expected_sum = node_bins @ curve

    actual = (cost_sum / (n - 1)).cpu().numpy()
    expected = (expected_sum / (n - 1)).cpu().numpy()
    node_df = pd.DataFrame({
        "node_id": node_index["node_id"].to_numpy(),
        "lon": nodes["lon"].to_numpy(), "lat": nodes["lat"].to_numpy(),
        "actual_cost": actual, "expected_cost": expected,
        "excess": actual / expected - 1,
    })
    curve_df = pd.DataFrame({
        "bin_start_km": np.arange(N_BINS) * BIN_KM,
        "n_pairs": bin_count.cpu().numpy().astype(np.int64),
        "mean_oracle_cost": curve.cpu().numpy(),
    })
    curve_df = curve_df[curve_df["n_pairs"] > 0]
    return node_df, curve_df


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    run_path = iso_mod._pick_best_run(args.run_id)
    checkpoint = torch.load(run_path, map_location=config.DEVICE, weights_only=False)
    log.info("Using run %s (%s / %s / seed %s)", run_path.stem,
             checkpoint["weight_condition"], checkpoint["arch"], checkpoint["seed"])

    with timed(log, "compute per-node actual and expected costs"):
        node_df, curve_df = _excess_scores(checkpoint)
    node_df.to_csv(config.TABLES_DIR / "node_excess_isolation.csv", index=False)
    curve_df.to_csv(config.TABLES_DIR / "excess_isolation_curve.csv", index=False)

    barangays = gpd.read_file(config.PROCESSED_DIR / "qc_barangays.gpkg", layer="barangays")
    name_col = next(c for c in barangays.columns if c.lower() in ("adm4_en", "adm4_name", "name_4"))
    pts = gpd.GeoDataFrame(node_df, geometry=gpd.points_from_xy(node_df["lon"], node_df["lat"]),
                           crs=config.CRS_WGS84)
    joined = gpd.sjoin(pts, barangays[[name_col, "geometry"]], how="left", predicate="within")
    brgy = (joined.groupby(name_col)
            .agg(mean_isolation_score=("actual_cost", "mean"),
                 mean_expected_cost=("expected_cost", "mean"),
                 n_nodes=("actual_cost", "size"))
            .reset_index().rename(columns={name_col: "barangay_name"}))
    brgy["excess_pct"] = (brgy["mean_isolation_score"] / brgy["mean_expected_cost"] - 1) * 100
    brgy["isolation_rank"] = brgy["mean_isolation_score"].rank(ascending=False, method="min").astype(int)
    brgy["excess_rank"] = brgy["excess_pct"].rank(ascending=False, method="min").astype(int)
    brgy = brgy.sort_values("excess_rank")
    out = config.TABLES_DIR / "barangay_excess_isolation.csv"
    brgy.to_csv(out, index=False)
    log.info("Wrote %s (%d barangays)", out, len(brgy))

    rho_loc = brgy["mean_isolation_score"].corr(brgy["excess_pct"], method="spearman")
    log.info("Spearman(total isolation, excess isolation) across barangays = %.3f", rho_loc)
    log.info("Top 10 by excess:\n%s", brgy.head(10)[["barangay_name", "excess_pct", "isolation_rank"]].to_string(index=False))
    log.info("Bottom 5 by excess:\n%s", brgy.tail(5)[["barangay_name", "excess_pct", "isolation_rank"]].to_string(index=False))

    # side-by-side figure: total isolation (current figure) vs excess isolation
    m = barangays.to_crs(config.CRS_UTM_51N).merge(brgy, left_on=name_col, right_on="barangay_name", how="left")
    fig, ax = plt.subplots(1, 2, figsize=(13, 8.5))
    m.plot("mean_isolation_score", ax=ax[0], cmap="YlOrRd", edgecolor="white", linewidth=0.4, legend=True,
           missing_kwds={"color": "#dddddd"},
           legend_kwds={"label": f"Mean predicted travel cost ({checkpoint['weight_condition']} units)", "shrink": 0.6})
    lim = max(abs(brgy["excess_pct"].min()), abs(brgy["excess_pct"].max()))
    m.plot("excess_pct", ax=ax[1], cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), edgecolor="white",
           linewidth=0.4, legend=True, missing_kwds={"color": "#dddddd"},
           legend_kwds={"label": "Cost above (+) or below (−) a typical trip of the same straight-line length (%)",
                        "shrink": 0.6})
    ax[0].set_title("(a) Total isolation", fontsize=12)
    ax[1].set_title("(b) Excess isolation (location effect removed)", fontsize=12)
    label = list(brgy.head(6)["barangay_name"]) + list(brgy.tail(3)["barangay_name"])
    for name in label:
        p = m.loc[m["barangay_name"] == name].geometry.representative_point().iloc[0]
        ax[1].annotate(name, (p.x, p.y), fontsize=7.5, ha="center")
    for a in ax:
        a.set_axis_off()
    plt.tight_layout()
    fig_path = config.FIGURES_DIR / "Figure_excess_isolation.png"
    plt.savefig(fig_path, dpi=300)
    log.info("Wrote %s", fig_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="C-W3-seed0", help="oracle run to use (default: primary oracle)")
    main(parser.parse_args())
