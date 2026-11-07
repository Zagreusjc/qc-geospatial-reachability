"""Phase 12 -- Sample system simulation (Chapter IV, Section C).

Traces one representative W3 test pair through the whole pipeline and prints
every intermediate value, so Chapter IV can show how a single query is
processed from raw road data to the oracle's answer.

The default pair is row 28680 of pairs_test_W3.parquet: an intersection in
Loyola Heights (u_idx 8223) to an intersection in Commonwealth (u_idx 26688).
Its exact W3 cost (1,420.6) is the median of the test set.

Inputs  : data/processed/{nodes_scc,node_index,edges_scc,ebc,pairs_test_W3}.parquet,
          data/processed/G_scc.gpickle, outputs/embeddings/Z_W3.npy,
          outputs/models/C-W3-seed0.pt
Outputs : printed trace; outputs/tables/sample_simulation.csv

Run     : python src/12_sample_simulation.py
          python src/12_sample_simulation.py --row 12345   # any other test row
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import torch  # must load before numpy/pandas on Windows
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from importlib import import_module  # noqa: E402

models_mod = import_module("06_models")
EARTH_RADIUS_M = 6_371_000.0


def _great_circle_m(a, b) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, (a["lon"], a["lat"], b["lon"], b["lat"]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(h), np.sqrt(1 - h)))


def main(args: argparse.Namespace) -> None:
    P = config.PROCESSED_DIR
    pairs = pd.read_parquet(P / "pairs_test_W3.parquet")
    row = pairs.loc[args.row]
    u_idx, v_idx, d_true = int(row["u_idx"]), int(row["v_idx"]), float(row["dist"])

    node_ids = pd.read_parquet(P / "node_index.parquet")["node_id"].to_numpy()
    nodes = pd.read_parquet(P / "nodes_scc.parquet")
    u, v = node_ids[u_idx], node_ids[v_idx]

    # --- Modules 1-3: graph, EBC and W3 weights along the true shortest path
    with open(P / "G_scc.gpickle", "rb") as f:
        G = pickle.load(f)
    weights = pd.read_parquet(P / "friction_weights.parquet", columns=["u", "v", "key", "W3"])
    w = {(int(r.u), int(r.v), int(r.key)): float(r.W3) for r in weights.itertuples()}

    def w3(a, b, data):
        return min(w[(a, b, k)] for k in data)

    cost, path = nx.single_source_dijkstra(G, u, v, weight=w3)
    cost_rev = nx.dijkstra_path_length(G, v, u, weight=w3)
    length_km = sum(min(d.get("length", 0) for d in G[a][b].values()) for a, b in zip(path[:-1], path[1:])) / 1000
    euclid = _great_circle_m(nodes.loc[u], nodes.loc[v])

    # --- Modules 4 and 6-7: embeddings and the trained oracle
    Z = np.load(config.EMBEDDINGS_DIR / "Z_W3.npy").astype(np.float32)
    ckpt = torch.load(config.MODELS_DIR / f"{args.run_id}.pt", map_location="cpu", weights_only=False)
    model = models_mod.build_model(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    zu, zv = torch.from_numpy(Z[[u_idx]]), torch.from_numpy(Z[[v_idx]])
    with torch.no_grad():
        pred_uv = float(model(zu, zv)) * ckpt["sigma"] + ckpt["mu"]
        pred_vu = float(model(zv, zu)) * ckpt["sigma"] + ckpt["mu"]

    rows = [
        ("Test row", args.row),
        ("Source node u (OSM id / embedding index)", f"{u} / {u_idx}"),
        ("Target node v (OSM id / embedding index)", f"{v} / {v_idx}"),
        ("u lon, lat", f"{nodes.loc[u, 'lon']:.5f}, {nodes.loc[u, 'lat']:.5f}"),
        ("v lon, lat", f"{nodes.loc[v, 'lon']:.5f}, {nodes.loc[v, 'lat']:.5f}"),
        ("Great-circle (Euclidean baseline) distance, m", f"{euclid:,.1f}"),
        ("Shortest path: edges / length, km", f"{len(path) - 1} / {length_km:.2f}"),
        ("Exact W3 cost d*(u,v) (label)", f"{d_true:,.2f}"),
        ("Recomputed Dijkstra W3 cost (check)", f"{cost:,.2f}"),
        ("Exact W3 cost d*(v,u)", f"{cost_rev:,.2f}"),
        ("z_u first 5 of 128 dims", np.array2string(Z[u_idx, :5], precision=3)),
        ("z_v first 5 of 128 dims", np.array2string(Z[v_idx, :5], precision=3)),
        (f"Oracle {args.run_id} prediction d^(u,v)", f"{pred_uv:,.2f}"),
        (f"Oracle {args.run_id} prediction d^(v,u)", f"{pred_vu:,.2f}"),
        ("Absolute error |d^(u,v) - d*(u,v)|", f"{abs(pred_uv - d_true):,.2f}"),
        ("Relative error, %", f"{100 * abs(pred_uv - d_true) / d_true:.2f}"),
        ("Multiplicative distortion", f"{max(pred_uv / d_true, d_true / pred_uv):.4f}"),
    ]
    width = max(len(k) for k, _ in rows)
    for k, val in rows:
        print(f"{k:<{width}}  {val}")
    out = config.TABLES_DIR / "sample_simulation.csv"
    pd.DataFrame(rows, columns=["step", "value"]).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", type=int, default=28680, help="row label in pairs_test_W3.parquet")
    parser.add_argument("--run-id", default="C-W3-seed0", help="trained model to query")
    main(parser.parse_args())
