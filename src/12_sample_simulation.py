"""Phase 12 -- Sample system simulation (Chapter IV, Section C).

Traces one representative W3 test pair through the whole pipeline and prints
every intermediate value step by step, so Chapter IV can show (with one
screenshot per step) how a single query is processed from raw road data to
the oracle's answer.

The default pair is row 28680 of pairs_test_W3.parquet: an intersection in
Loyola Heights (u_idx 8223) to an intersection in Commonwealth (u_idx 26688).
Its exact W3 cost (1,420.6) is the median of the test set.

Inputs  : data/processed/{nodes_scc,node_index,friction_weights,pairs_test_W3}.parquet,
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
LINE = "=" * 78


def _great_circle_m(a, b) -> float:
    lon1, lat1, lon2, lat2 = map(np.radians, (a["lon"], a["lat"], b["lon"], b["lat"]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(EARTH_RADIUS_M * 2 * np.arctan2(np.sqrt(h), np.sqrt(1 - h)))


def _pick(cols, *names):
    """Return the first column name that exists (case-insensitive), else None."""
    lower = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _step(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def _show(pairs_list, width: int = 46) -> None:
    for k, val in pairs_list:
        print(f"  {k:<{width}}  {val}")


def main(args: argparse.Namespace) -> None:
    """Run Phase 12: trace one W3 test pair through the pipeline and print each step.

    Steps in order: (1) read the pair at --row of pairs_test_W3.parquet and look
    up its nodes; (2) load G_scc.gpickle and the straight-line distance;
    (3) read friction_weights.parquet and find the cheapest W3 route with
    Dijkstra; (4) recompute the label in both directions; (5) read the
    embeddings of the two nodes from outputs/embeddings/Z_W3.npy; (6) load the
    checkpoint outputs/models/<--run-id>.pt and predict both directions;
    (7) compare prediction with the exact cost.

    Also reads data/processed/node_index.parquet and nodes_scc.parquet. Prints
    the trace and writes outputs/tables/sample_simulation.csv (one row per
    printed value) and outputs/tables/sample_simulation_route_edges.csv (the
    edges of the cheapest route).
    """
    P = config.PROCESSED_DIR
    rows: list[tuple[str, object]] = []  # everything also goes to the CSV

    # record() prints a block of (label, value) rows and keeps them for the CSV written at the end
    def record(step_rows):
        _show(step_rows)
        rows.extend(step_rows)

    # ------------------------------------------------------------------ STEP 1
    _step("STEP 1  INPUT: one test query (source u -> target v)")
    pairs = pd.read_parquet(P / "pairs_test_W3.parquet")
    row = pairs.loc[args.row]
    u_idx, v_idx, d_true = int(row["u_idx"]), int(row["v_idx"]), float(row["dist"])

    node_ids = pd.read_parquet(P / "node_index.parquet")["node_id"].to_numpy()
    nodes = pd.read_parquet(P / "nodes_scc.parquet")
    u, v = node_ids[u_idx], node_ids[v_idx]

    step1 = [
        ("Test set", f"pairs_test_W3.parquet ({len(pairs):,} pairs)"),
        ("Test row", args.row),
        ("Source node u (graph node ID / embedding index)", f"{u} / {u_idx}"),
        ("Target node v (graph node ID / embedding index)", f"{v} / {v_idx}"),
        ("u lon, lat", f"{nodes.loc[u, 'lon']:.5f}, {nodes.loc[u, 'lat']:.5f}"),
        ("v lon, lat", f"{nodes.loc[v, 'lon']:.5f}, {nodes.loc[v, 'lat']:.5f}"),
    ]
    # barangay columns are optional: use any node column whose name mentions brgy/barangay, if present
    brgy_col = next((c for c in nodes.columns if "brgy" in c.lower() or "barangay" in c.lower()), None)
    if brgy_col:
        step1 += [("u barangay", nodes.loc[u, brgy_col]), ("v barangay", nodes.loc[v, brgy_col])]
    record(step1)

    # ------------------------------------------------------------------ STEP 2
    _step("STEP 2  GRAPH CONSTRUCTION MODULE: load the directed road graph")
    with open(P / "G_scc.gpickle", "rb") as f:
        G = pickle.load(f)
    euclid = _great_circle_m(nodes.loc[u], nodes.loc[v])
    record([
        ("Graph", "G_scc.gpickle (largest strongly connected component)"),
        ("Intersections (nodes)", f"{G.number_of_nodes():,}"),
        ("One-way links (directed edges)", f"{G.number_of_edges():,}"),
        ("u found in graph", u in G),
        ("v found in graph", v in G),
        ("Straight-line (Euclidean baseline) distance, m", f"{euclid:,.1f}"),
    ])

    # ------------------------------------------------------------------ STEP 3
    _step("STEP 3  EBC + FRICTION WEIGHT MODULES: edge weights along the route")
    weights = pd.read_parquet(P / "friction_weights.parquet")
    # _pick tolerates differing column spellings/case in friction_weights.parquet; each
    # of W1/EBC/W3 may be missing (None) and is then left out of `keep`
    c_w1 = _pick(weights.columns, "W1")
    c_ebc = _pick(weights.columns, "EBC", "ebc_norm", "ebc", "W2")
    c_w3 = _pick(weights.columns, "W3")
    keep = [c for c in ("u", "v", "key", c_w1, c_ebc, c_w3) if c]
    # lookup: (u, v, key) -> tuple of the selected weight values (columns after u, v, key)
    wt = {(int(r[0]), int(r[1]), int(r[2])): r[3:] for r in weights[keep].itertuples(index=False)}
    i_w3 = keep.index(c_w3) - 3

    # Dijkstra weight function: G is a multigraph, so (a, b) may have parallel edges
    # (`data` maps edge key -> attributes); take the cheapest W3 among them
    def w3(a, b, data):
        return min(wt[(a, b, k)][i_w3] for k in data)

    cost, path = nx.single_source_dijkstra(G, u, v, weight=w3)

    # walk the route, re-picking the parallel edge Dijkstra effectively used, and collect
    # its attributes, total length, total W1 and the count of primary/secondary edges
    edge_rows, total_len, total_w1, main_road = [], 0.0, 0.0, 0
    for a, b in zip(path[:-1], path[1:]):
        k = min(G[a][b], key=lambda kk: wt[(a, b, kk)][i_w3])  # edge actually used
        vals = dict(zip(keep[3:], wt[(a, b, k)]))
        d = G[a][b][k]
        length = float(d.get("length", 0))
        hw = d.get("highway", "")
        hw = hw[0] if isinstance(hw, list) else str(hw)
        total_len += length
        if c_w1:
            total_w1 += float(vals[c_w1])
        if hw.startswith(("primary", "secondary")):
            main_road += 1
        edge_rows.append({"from": a, "to": b, "road": hw, "length_m": round(length, 1),
                          **{c: round(float(vals[c]), 4) for c in keep[3:]}})
    edges_df = pd.DataFrame(edge_rows)
    n_edges = len(path) - 1

    # print the first few route edges and the route totals
    print(f"  Weight columns used: {', '.join(keep[3:])}")
    print(f"  First 5 of {n_edges} edges on the cheapest W3 route:")
    print(edges_df.head(5).to_string(index=False))
    print(f"  ... ({n_edges - 5} more edges)\n")
    step3 = [
        ("Cheapest W3 route: edges / length, km", f"{n_edges} / {total_len / 1000:.2f}"),
        ("Share of route on primary/secondary roads, %", f"{100 * main_road / n_edges:.0f}"),
    ]
    if c_w1:
        step3.append(("Travel time along route (sum of W1), s", f"{total_w1:,.1f} (about {total_w1 / 60:.1f} min)"))
    record(step3)

    # ------------------------------------------------------------------ STEP 4
    _step("STEP 4  PAIR SAMPLING AND LABELING MODULE: exact Dijkstra cost (label)")
    cost_rev = nx.dijkstra_path_length(G, v, u, weight=w3)
    record([
        ("Exact W3 cost d*(u,v) stored as label", f"{d_true:,.2f}"),
        ("Recomputed Dijkstra W3 cost (check)", f"{cost:,.2f}"),
        ("Label matches recomputation", bool(np.isclose(cost, d_true))),
        ("Exact W3 cost d*(v,u) (return trip)", f"{cost_rev:,.2f}"),
    ])

    # ------------------------------------------------------------------ STEP 5
    _step("STEP 5  NODE EMBEDDING MODULE: Node2Vec+ vectors for u and v")
    Z = np.load(config.EMBEDDINGS_DIR / "Z_W3.npy").astype(np.float32)
    record([
        ("Embedding matrix Z_W3.npy shape", f"{Z.shape[0]:,} nodes x {Z.shape[1]} dims"),
        ("z_u first 5 of 128 dims", np.array2string(Z[u_idx, :5], precision=3)),
        ("z_v first 5 of 128 dims", np.array2string(Z[v_idx, :5], precision=3)),
    ])

    # ------------------------------------------------------------------ STEP 6
    _step(f"STEP 6  ORACLE TRAINING MODULE: query the trained oracle ({args.run_id})")
    # the checkpoint stores the architecture name and the target standardisation (mu, sigma);
    # model output is standardised, so it is mapped back to W3 cost units before reporting
    ckpt = torch.load(config.MODELS_DIR / f"{args.run_id}.pt", map_location="cpu", weights_only=False)
    model = models_mod.build_model(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    zu, zv = torch.from_numpy(Z[[u_idx]]), torch.from_numpy(Z[[v_idx]])
    with torch.no_grad():
        pred_uv = float(model(zu, zv)) * ckpt["sigma"] + ckpt["mu"]
        pred_vu = float(model(zv, zu)) * ckpt["sigma"] + ckpt["mu"]
    record([
        ("Model file", f"{args.run_id}.pt"),
        ("Architecture", ckpt["arch"]),
        ("Trainable parameters", f"{n_params:,}"),
        ("Oracle prediction d^(u,v)", f"{pred_uv:,.2f}"),
        ("Oracle prediction d^(v,u) (return trip)", f"{pred_vu:,.2f}"),
    ])

    # ------------------------------------------------------------------ STEP 7
    _step("STEP 7  EVALUATION MODULE: compare oracle with exact cost")
    record([
        ("Exact cost d*(u,v)", f"{d_true:,.2f}"),
        ("Oracle prediction d^(u,v)", f"{pred_uv:,.2f}"),
        ("Absolute error |d^(u,v) - d*(u,v)|", f"{abs(pred_uv - d_true):,.2f}"),
        ("Relative error, %", f"{100 * abs(pred_uv - d_true) / d_true:.2f}"),
        ("Multiplicative distortion", f"{max(pred_uv / d_true, d_true / pred_uv):.4f}"),
        ("Oracle knows return trip is cheaper", (pred_vu < pred_uv) == (cost_rev < d_true)),
    ])

    # write the CSV outputs
    out = config.TABLES_DIR / "sample_simulation.csv"
    pd.DataFrame(rows, columns=["step", "value"]).to_csv(out, index=False)
    edges_df.to_csv(config.TABLES_DIR / "sample_simulation_route_edges.csv", index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--row", type=int, default=28680, help="row label in pairs_test_W3.parquet")
    parser.add_argument("--run-id", default="C-W3-seed0", help="trained model to query")
    main(parser.parse_args())