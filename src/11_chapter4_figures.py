"""Phase 11 -- Chapter IV map figures.

Draws the three static map figures used in Chapter IV from outputs that the
earlier phases already wrote. Nothing is retrained or recomputed here except
simple filtering and the great-circle distance (same formula as 08_evaluate.py).

  Figure 4.3  Top 1% of road edges by normalized edge betweenness centrality
  Figure 4.6  Test pairs where Euclidean distance most misrepresents travel
              cost under W3 (top 5% and bottom 5% of the distance-to-cost ratio)
  Figure 4.7  Mean structural isolation score by barangay (choropleth)

Inputs  : data/processed/ebc.parquet, data/processed/edges_scc.parquet,
          data/processed/nodes_scc.parquet, data/processed/node_index.parquet,
          data/processed/pairs_test_W3.parquet, data/processed/qc_barangays.gpkg,
          outputs/tables/barangay_isolation.csv (from 09_isolation_choropleth.py)
Outputs : outputs/figures/Figure_4.3_top_ebc_corridors.png
          outputs/figures/Figure_4.6_euclidean_divergence.png
          outputs/figures/Figure_4.7_isolation_choropleth.png
          outputs/tables/euclidean_divergence_summary.csv

Run     : python src/11_chapter4_figures.py
"""
from __future__ import annotations

import argparse
import os
import sys

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402

log = get_logger("11_chapter4_figures")

EARTH_RADIUS_M = 6_371_000.0   # same value as 08_evaluate.py
TOP_EBC_SHARE = 0.01           # Figure 4.3: top 1% of edges by normalized EBC
DIVERGENCE_SHARE = 0.05        # Figure 4.6: top and bottom 5% of the ratio
PAIRS_PER_PANEL = 300          # Figure 4.6: pairs drawn per panel (for legibility)
PLOT_SEED = 42                 # fixed seed for the drawn subsample
DPI = 300

BLUE_DARK = "#0b2f6b"
BLUE_LINE = "#1f4e8c"
GREY_ROAD = "#cfcfcf"
GREY_FILL = "#f2f2f2"
GREY_EDGE = "#c8c8c8"


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------
def _load_barangays() -> tuple[gpd.GeoDataFrame, str]:
    path = config.PROCESSED_DIR / "qc_barangays.gpkg"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run 01b_barangays.py first.")
    brgy = gpd.read_file(path, layer="barangays")
    name_col = next((c for c in brgy.columns if c.lower() in ("adm4_en", "adm4_name", "name_4")), None)
    if name_col is None:
        raise RuntimeError(f"No barangay-name column found among {list(brgy.columns)}")
    return brgy.to_crs(config.CRS_UTM_51N), name_col


def _load_nodes() -> pd.DataFrame:
    """Node coordinates in node_index order (row i = embedding index i)."""
    node_index = pd.read_parquet(config.PROCESSED_DIR / "node_index.parquet")
    nodes = pd.read_parquet(config.PROCESSED_DIR / "nodes_scc.parquet", columns=["x", "y", "lon", "lat"])
    return nodes.loc[node_index["node_id"].to_numpy()].reset_index(drop=True)


def _finish_map(ax) -> None:
    ax.set_axis_off()
    ax.set_aspect("equal")


# ---------------------------------------------------------------------------
# Figure 4.3 -- top 1% EBC corridors
# ---------------------------------------------------------------------------
def figure_top_ebc() -> None:
    edges = gpd.read_parquet(config.PROCESSED_DIR / "edges_scc.parquet").to_crs(config.CRS_UTM_51N)
    ebc = pd.read_parquet(config.PROCESSED_DIR / "ebc.parquet").set_index(["u", "v", "key"])
    edges = edges.join(ebc[["ebc_norm"]], how="left")
    if edges["ebc_norm"].isna().any():
        raise RuntimeError("Some edges in edges_scc.parquet have no EBC value; rerun 02_ebc.py.")

    n_top = int(round(len(edges) * TOP_EBC_SHARE))
    top = edges.nlargest(n_top, "ebc_norm").sort_values("ebc_norm")
    threshold = top["ebc_norm"].min()
    top_km = top["length"].sum() / 1000
    all_km = edges["length"].sum() / 1000
    log.info("Top %.0f%%: %d edges, EBC_norm >= %.4f, %.1f km of %.1f km",
             TOP_EBC_SHARE * 100, n_top, threshold, top_km, all_km)

    cmap = LinearSegmentedColormap.from_list("ebc", ["#7fb3d8", BLUE_DARK])
    norm = Normalize(vmin=threshold, vmax=top["ebc_norm"].max())

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    edges.plot(ax=ax, color=GREY_ROAD, linewidth=0.3, zorder=1)
    top.plot(ax=ax, column="ebc_norm", cmap=cmap, norm=norm, linewidth=1.6, zorder=2)
    _finish_map(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Normalized EBC (top 1% of edges)")
    cbar.outline.set_visible(False)

    out = config.FIGURES_DIR / "Figure_4.3_top_ebc_corridors.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 4.6 -- where Euclidean distance misrepresents travel cost (W3)
# ---------------------------------------------------------------------------
def _great_circle_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def figure_euclidean_divergence() -> None:
    nodes = _load_nodes()
    pairs = pd.read_parquet(config.PROCESSED_DIR / "pairs_test_W3.parquet")
    u, v = pairs["u_idx"].to_numpy(), pairs["v_idx"].to_numpy()
    pairs["euclid_m"] = _great_circle_m(nodes["lon"].to_numpy()[u], nodes["lat"].to_numpy()[u],
                                        nodes["lon"].to_numpy()[v], nodes["lat"].to_numpy()[v])
    pairs = pairs[pairs["dist"] > 0].copy()
    pairs["ratio"] = pairs["euclid_m"] / pairs["dist"]

    hi_cut = pairs["ratio"].quantile(1 - DIVERGENCE_SHARE)
    lo_cut = pairs["ratio"].quantile(DIVERGENCE_SHARE)
    top = pairs[pairs["ratio"] >= hi_cut]
    bottom = pairs[pairs["ratio"] <= lo_cut]

    summary = pd.DataFrame([
        {"group": name, "n_pairs": len(df), "ratio_cutoff": cut,
         "median_euclid_km": df["euclid_m"].median() / 1000, "median_cost": df["dist"].median()}
        for name, df, cut in [("all", pairs, np.nan), ("top_5pct", top, hi_cut), ("bottom_5pct", bottom, lo_cut)]
    ])
    summary_path = config.TABLES_DIR / "euclidean_divergence_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("Median ratio %.2f; top cut %.2f; bottom cut %.2f; wrote %s",
             pairs["ratio"].median(), hi_cut, lo_cut, summary_path)

    brgy, _ = _load_barangays()
    rng = np.random.default_rng(PLOT_SEED)
    fig, axes = plt.subplots(1, 2, figsize=(11, 8))
    panels = [
        (top, "(a) Far apart, cheap to traverse\n(top 5% of distance-to-cost ratio)"),
        (bottom, "(b) Close together, costly to traverse\n(bottom 5% of distance-to-cost ratio)"),
    ]
    for ax, (df, title) in zip(axes, panels):
        brgy.plot(ax=ax, color=GREY_FILL, edgecolor=GREY_EDGE, linewidth=0.5)
        sample = df.iloc[rng.choice(len(df), size=min(PAIRS_PER_PANEL, len(df)), replace=False)]
        su, sv = sample["u_idx"].to_numpy(), sample["v_idx"].to_numpy()
        xs = np.stack([nodes["x"].to_numpy()[su], nodes["x"].to_numpy()[sv]])
        ys = np.stack([nodes["y"].to_numpy()[su], nodes["y"].to_numpy()[sv]])
        ax.plot(xs, ys, color=BLUE_LINE, linewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter(np.concatenate([xs[0], xs[1]]), np.concatenate([ys[0], ys[1]]),
                   s=6, color=BLUE_LINE, zorder=3)
        ax.set_title(title, fontsize=12)
        _finish_map(ax)
    fig.text(0.5, 0.04, f"Each line joins one sampled test pair ({PAIRS_PER_PANEL} per panel); W3 condition.",
             ha="center", fontsize=10, color="#555555")

    out = config.FIGURES_DIR / "Figure_4.6_euclidean_divergence.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s", out)


# ---------------------------------------------------------------------------
# Figure 4.7 -- isolation choropleth
# ---------------------------------------------------------------------------
def figure_isolation() -> None:
    scores_path = config.TABLES_DIR / "barangay_isolation.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"{scores_path} not found; run 09_isolation_choropleth.py first.")
    scores = pd.read_csv(scores_path)
    brgy, name_col = _load_barangays()
    merged = brgy.merge(scores, left_on=name_col, right_on="barangay_name", how="left")
    missing = merged.loc[merged["mean_isolation_score"].isna(), name_col].tolist()
    log.info("Isolation scores for %d/%d barangays; no score: %s",
             merged["mean_isolation_score"].notna().sum(), len(merged), missing)

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    merged.plot(ax=ax, column="mean_isolation_score", cmap="Blues", edgecolor="white", linewidth=0.4,
                legend=True, legend_kwds={"shrink": 0.6, "pad": 0.02,
                                          "label": "Mean predicted friction-weighted distance (W3 cost units)"},
                missing_kwds={"color": "#d9d9d9", "hatch": "///", "edgecolor": "white"})
    _finish_map(ax)
    if missing:
        fig.text(0.02, 0.02, f"Grey hatched: no road nodes ({', '.join(missing)})", fontsize=8, color="#555555")

    out = config.FIGURES_DIR / "Figure_4.7_isolation_choropleth.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log.info("Wrote %s", out)


FIGURES = {"4.3": figure_top_ebc, "4.6": figure_euclidean_divergence, "4.7": figure_isolation}


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    for key in args.only or FIGURES:
        with timed(log, f"Figure {key}"):
            FIGURES[key]()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), help="draw only these figures, e.g. --only 4.3")
    main(parser.parse_args())
