"""Phase 8 -- Evaluation and ablation results.

Evaluates every trained run on the shared held-out test set and assembles the
ablation results table.

Metrics : Spearman rho (primary, with bootstrap CI), MAE, RMSE, RMSE/MAE ratio,
          multiplicative distortion (mean/worst/std; excludes d*=0) vs
          Thorup-Zwick (2k-1) reference bounds at k=2,3. Euclidean baseline: the
          great-circle distance between nodes u and v, used as-is with no unit
          conversion against MAE/RMSE/Spearman/distortion. MAE/RMSE/distortion
          for this baseline mix units of length (meters) and time (seconds for
          W1/W3; normalized betweenness for W2), so only Spearman rho is a
          dimensionally meaningful comparison for it; treat the others as
          illustrative. Ablation: mean +/- std across seeds, Cohen's d (H3: W1
          vs W3 per arch; H4: Siamese vs MLP per weight) -- effect sizes and
          bootstrap CIs are reported in place of a t-test/ANOVA.

High-divergence pairs are defined directly from ground truth vs the Euclidean
baseline on the W3 (primary) test set (distortion = Euclidean distance /
Dijkstra distance) -- this comparison does not involve any trained oracle.

Inputs  : outputs/models/*.pt, embeddings, test pairs, node coordinates.
Outputs : outputs/tables/ablation_results.csv,
          data/processed/high_divergence_pairs.parquet
Next    : 09_isolation_choropleth.py
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from utils import get_logger, timed  # noqa: E402
from importlib import import_module  # noqa: E402

models_mod = import_module("06_models")

log = get_logger("08_evaluate")

TZ_BOUNDS = {k: 2 * k - 1 for k in config.THORUP_ZWICK_K}
EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius, for haversine great-circle distance


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rho, _ = spearmanr(y_true, y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    nonzero = y_true > 0
    distortion = y_pred[nonzero] / y_true[nonzero]
    return {
        "spearman": float(rho),
        "mae": mae,
        "rmse": rmse,
        "rmse_mae_ratio": rmse / mae if mae > 0 else float("nan"),
        "distortion_mean": float(distortion.mean()),
        "distortion_worst": float(distortion.max()),
        "distortion_std": float(distortion.std()),
    }


def _bootstrap_spearman_ci(y_true: np.ndarray, y_pred: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rhos = np.empty(config.BOOTSTRAP_RESAMPLES)
    for i in range(config.BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        rhos[i], _ = spearmanr(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / max(n1 + n2 - 2, 1)
    pooled_std = np.sqrt(pooled_var)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def _load_test_pairs(weight: str) -> pd.DataFrame:
    path = config.PROCESSED_DIR / f"pairs_test_{weight}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run 05_sampling_labels.py first.")
    return pd.read_parquet(path)


def _great_circle_distances_meters(test_df: pd.DataFrame) -> np.ndarray:
    """Great-circle (haversine) distance in meters between each test pair's
    nodes -- the Euclidean baseline, used as-is with no unit conversion."""
    node_index_df = pd.read_parquet(config.PROCESSED_DIR / "node_index.parquet")
    # nodes_scc.parquet is indexed by "osmid" (our graph node id) and already
    # carries WGS84 lon/lat columns from Phase 1's ox.project_graph step.
    nodes_df = pd.read_parquet(config.PROCESSED_DIR / "nodes_scc.parquet", columns=["lon", "lat"])

    idx_to_id = node_index_df["node_id"].to_numpy()
    id_to_lonlat = dict(zip(nodes_df.index.to_numpy(), zip(nodes_df["lon"], nodes_df["lat"])))

    u_lonlat = np.radians(np.array([id_to_lonlat[idx_to_id[i]] for i in test_df["u_idx"]]))
    v_lonlat = np.radians(np.array([id_to_lonlat[idx_to_id[i]] for i in test_df["v_idx"]]))

    lon1, lat1 = u_lonlat[:, 0], u_lonlat[:, 1]
    lon2, lat2 = v_lonlat[:, 0], v_lonlat[:, 1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def _evaluate_run(run_path) -> dict:
    checkpoint = torch.load(run_path, weights_only=False)
    weight, arch, seed = checkpoint["weight_condition"], checkpoint["arch"], checkpoint["seed"]
    run_id = config.run_id(weight, arch, seed)

    device = torch.device(config.DEVICE)
    Z = np.load(config.EMBEDDINGS_DIR / f"Z_{weight}.npy").astype(np.float32)
    Z_t = torch.from_numpy(Z).to(device)
    test_df = _load_test_pairs(weight)

    model = models_mod.build_model(arch).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    u_idx = torch.from_numpy(test_df["u_idx"].to_numpy(dtype=np.int64))
    v_idx = torch.from_numpy(test_df["v_idx"].to_numpy(dtype=np.int64))
    with torch.no_grad():
        pred_std = model(Z_t[u_idx].to(device), Z_t[v_idx].to(device))
    pred = (pred_std.cpu() * checkpoint["sigma"] + checkpoint["mu"]).numpy()
    y_true = test_df["dist"].to_numpy()

    metrics = _metrics(y_true, pred)
    ci_lo, ci_hi = _bootstrap_spearman_ci(y_true, pred, seed=config.SAMPLING_SEED)

    return {
        "row_type": "run", "run_id": run_id, "weight": weight, "arch": arch, "seed": seed,
        "n_test": len(test_df), "spearman_ci_lo": ci_lo, "spearman_ci_hi": ci_hi,
        "tz_k2_bound": TZ_BOUNDS.get(2), "tz_k3_bound": TZ_BOUNDS.get(3),
        **metrics,
    }


def _evaluate_euclidean_baseline(weight: str) -> dict:
    test_df = _load_test_pairs(weight)
    y_true = test_df["dist"].to_numpy()
    y_pred = _great_circle_distances_meters(test_df)
    metrics = _metrics(y_true, y_pred)
    return {
        "row_type": "euclidean_baseline", "run_id": f"euclidean-{weight}", "weight": weight,
        "arch": "n/a", "seed": "n/a", "n_test": len(test_df),
        "spearman_ci_lo": float("nan"), "spearman_ci_hi": float("nan"),
        "tz_k2_bound": TZ_BOUNDS.get(2), "tz_k3_bound": TZ_BOUNDS.get(3),
        **metrics,
    }


def _cell_aggregates_and_effects(run_rows: list) -> list:
    df = pd.DataFrame(run_rows)
    extra_rows = []

    for (weight, arch), grp in df.groupby(["weight", "arch"]):
        row = {"row_type": "cell_aggregate", "run_id": f"{arch}-{weight}-mean",
               "weight": weight, "arch": arch, "seed": "mean(0,1,2)", "n_test": grp["n_test"].iloc[0]}
        for col in ("spearman", "mae", "rmse", "distortion_mean", "distortion_worst"):
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std()
        extra_rows.append(row)

    for arch, grp in df.groupby("arch"):
        w1 = grp[grp["weight"] == "W1"]["spearman"]
        w3 = grp[grp["weight"] == "W3"]["spearman"]
        if len(w1) and len(w3):
            extra_rows.append({
                "row_type": "cohens_d_H3_W1_vs_W3", "run_id": f"cohend-H3-{arch}",
                "weight": "W1_vs_W3", "arch": arch, "seed": "n/a",
                "cohens_d": _cohens_d(w3.values, w1.values),
                "note": "positive d => W3 (composite) has higher Spearman than W1 (traversal-only)",
            })

    for weight, grp in df.groupby("weight"):
        siamese = grp[grp["arch"] == "siamese"]["spearman"]
        mlp = grp[grp["arch"] == "mlp"]["spearman"]
        if len(siamese) and len(mlp):
            extra_rows.append({
                "row_type": "cohens_d_H4_siamese_vs_mlp", "run_id": f"cohend-H4-{weight}",
                "weight": weight, "arch": "siamese_vs_mlp", "seed": "n/a",
                "cohens_d": _cohens_d(siamese.values, mlp.values),
                "note": "positive d => Siamese has higher Spearman than MLP",
            })

    return extra_rows


def _high_divergence_pairs(percentile: float) -> pd.DataFrame:
    """Distortion = Euclidean distance / Dijkstra distance; high-divergence
    pairs are those at/above the given percentile of that ratio."""
    test_df = _load_test_pairs("W3")
    y_true = test_df["dist"].to_numpy()
    euclid_m = _great_circle_distances_meters(test_df)
    with np.errstate(divide="ignore", invalid="ignore"):
        distortion = np.where(y_true > 0, euclid_m / y_true, np.nan)
    threshold = np.nanpercentile(distortion, percentile)
    out = test_df.copy()
    out["euclidean_distance_m"] = euclid_m
    out["distortion_vs_euclidean"] = distortion
    return out[out["distortion_vs_euclidean"] >= threshold].sort_values(
        "distortion_vs_euclidean", ascending=False
    )


def main(args: argparse.Namespace) -> None:
    config.ensure_dirs()
    log.info("Evaluating ablation; primary metric = Spearman rho")

    run_paths = sorted(config.MODELS_DIR.glob("*.pt"))
    if not run_paths:
        raise FileNotFoundError(f"No trained models found in {config.MODELS_DIR}; run 07_train.py first.")

    run_rows = []
    for run_path in run_paths:
        with timed(log, f"evaluate {run_path.stem}"):
            row = _evaluate_run(run_path)
            log.info("%s: spearman=%.4f mae=%.4f rmse=%.4f", row["run_id"], row["spearman"], row["mae"], row["rmse"])
            run_rows.append(row)

    euclidean_rows = []
    for weight in config.WEIGHT_CONDITIONS:
        with timed(log, f"evaluate Euclidean baseline ({weight})"):
            euclidean_rows.append(_evaluate_euclidean_baseline(weight))

    with timed(log, "cell aggregates + Cohen's d (H3, H4)"):
        extra_rows = _cell_aggregates_and_effects(run_rows)

    results_df = pd.DataFrame(run_rows + euclidean_rows + extra_rows)
    out_path = config.TABLES_DIR / "ablation_results.csv"
    results_df.to_csv(out_path, index=False)
    log.info("Wrote %s (%d rows)", out_path, len(results_df))

    with timed(log, "high-divergence pairs (W3, ground truth vs Euclidean)"):
        hd_df = _high_divergence_pairs(config.HIGH_DIVERGENCE_PERCENTILE)
        hd_path = config.PROCESSED_DIR / "high_divergence_pairs.parquet"
        hd_df.to_parquet(hd_path)
        log.info("Wrote %s (%d pairs, >= p%d)", hd_path, len(hd_df), config.HIGH_DIVERGENCE_PERCENTILE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(parser.parse_args())
