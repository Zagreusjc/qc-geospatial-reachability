# qc-accessibility-oracle
# Neural Distance Oracles for Geospatial Reachability
### Graph Embedding and Metric Learning on Quezon City's Road Network

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Open%20Source-orange.svg)](https://data.humdata.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

> Undergraduate Thesis — B.S. Computer Science
> University of Santo Tomas, College of Information and Computing Sciences, 2025-2026
> Benter, J.C. · Magcaling, E.J. · Solano, S. · Vega, J.F.E. · Adviser: Estabillo, C.R.

---

## The Problem

In Quezon City — the most populous city in the Philippines, with 2.9 million residents across 142 barangays — **physical distance is a broken ruler.** Two barangays that look equally close to a hospital on a straight-line map can be worlds apart once you account for how the road network actually connects them: low-speed road classes, fragmented connectivity, and a handful of overloaded arterial corridors create permanent structural friction that a flat map cannot see.

The challenge is computational as much as geographic. Computing exact structural shortest-path distances across a road network of roughly 70,000 intersections with Dijkstra's algorithm is prohibitively expensive for repeated, citywide distance scoring. Local planners under the Revised Quezon City Comprehensive Development Plan 2026-2031 are mandated to make data-driven infrastructure decisions but lack a principled, structural distance measurement tool.

---

## What This Study Builds

A **neural distance oracle** — a Siamese neural network trained on **Node2Vec+** embeddings of Quezon City's **friction-weighted** road network graph — that approximates structural (friction-weighted) shortest-path distances in **O(1) query time** after a one-time preprocessing step.

No GPS data. No behavioral surveys. No proprietary sources. Everything is derived from open, static, freely available datasets.

```
Open geographic data (HDX / PSA / JICA speed table)
        |
Friction-weighted road graph  G_scc = (V, E)
Edge weight = (length / road-class speed) x (1 + alpha x normalized EBC)
        |
Node2Vec+ structural embeddings  (128-dim, weight-aware biased walks)
        |
Siamese network trained on Dijkstra friction distances (MSE regression)
        |
Neural Distance Oracle -- O(1) structural distance queries
        |
Mean structural-isolation score per barangay
        |
Barangay structural-isolation choropleth of Quezon City
```

---

## Research Questions

**RQ1 — Structural validity.** How can travel cost between QC intersections be structurally validated through a friction-weighted road graph built from open geographic data?

**RQ2 — Approximation quality.** To what extent can a Siamese network trained on Node2Vec+ embeddings approximate friction-weighted shortest-path distances, evaluated by rank preservation (Spearman rho) and approximation error (MAE, RMSE) against exact Dijkstra ground truth?

**RQ3 — Edge weight and architecture design.** Which combination of friction-weight design and neural architecture best captures structural road-network impedance for identifying congestion-prone corridors and structurally isolated areas?

**RQ4 — Broken ruler.** How does the oracle perform relative to Euclidean and exact Dijkstra baselines, and where does physical distance most severely misrepresent friction-weighted travel cost?

---

## Hypotheses

| ID | Alternative hypothesis (Ha) | Tested by |
|---|---|---|
| H1 | Node2Vec+ embeddings encode topological proximity well enough to support supervised distance regression (better than chance) | Any oracle vs chance |
| H2 | The Siamese oracle differs significantly in accuracy from a Euclidean baseline | Oracle vs Euclidean |
| H3 | Adding normalized EBC to edge weights lowers approximation error vs traversal cost alone | W1 vs W3, same architecture |
| H4 | A Siamese architecture outperforms a single-stream MLP under identical embeddings | Siamese vs MLP, same weights |

---

## Datasets

All datasets are pre-existing, freely available, and statically timestamped at download. No live APIs, no proprietary sources.

| Dataset | Source | Role |
|---|---|---|
| HOTOSM Philippines Roads | [HDX](https://data.humdata.org/dataset/hotosm_phl_roads) | Road network skeleton — nodes (intersections) and edges (segments), `highway` class, `oneway` |
| PSA / NAMRIA Administrative Boundaries (COD-AB, ADM3 + ADM4) | [HDX `cod-ab-phl`](https://data.humdata.org/dataset/cod-ab-phl) | QC clip boundary + the 142 barangay polygons for node assignment and choropleth |
| JICA 2015 Speed Lookup Table | JICA (2015), encoded in `src/config.py` | Estimated operating speed per road class (friction is measured in travel time) |

### Barangay boundaries (resolving the clipping concern)

Assigning each node to a barangay and drawing the 142-barangay choropleth requires **barangay-level polygons**, not just the QC city outline. Open barangay polygons do exist; the pipeline uses a **fallback ladder** (`src/01b_barangays.py`) and caches the first source that resolves to `data/processed/qc_barangays.gpkg`:

1. **HDX COD-AB Philippines (`cod-ab-phl`), Admin Level 4** — barangay polygons on the same platform already cited; filter to QC. *(primary)*
2. Curated PSA + NAMRIA GeoJSON (`bendlikeabamboo/barangay-boundaries-repository`, PSGC-annotated).
3. OpenStreetMap barangay relations (`admin_level=10`) via OSMnx.
4. geoBoundaries PHL ADM4 (CC-BY).
5. Last resort: synthesized barangay proxy zones via Voronoi tessellation of barangay centroids clipped to the QC boundary (documented as a limitation).

Everything downstream reads the single cached `qc_barangays.gpkg`, so the source is swappable without touching the pipeline.

---

## Methodology

The pipeline follows the manuscript's seven sequential phases (Chapter III, Fig 3.1). Each stage is a re-runnable script that checkpoints its artifacts.

### Stage 1 — Graph construction, cleaning, and SCC extraction
The HOTOSM roads layer is clipped to QC and converted into a directed multigraph with OSMnx (nodes = intersections, edges = segments). Cleaning, in order:
1. **Highway-class filter** — keep `motorway, trunk, primary, secondary, tertiary, unclassified, residential, service, living_street` (+ `_link`); drop footways, cycleways, pedestrian, construction.
2. **Missing-tag handling** — edges lacking a `highway` tag default to `residential` speed; the percentage defaulted is logged (descriptive finding).
3. **Projection** — reproject to UTM 51N (EPSG:32651) for metric lengths; keep an EPSG:4326 copy for maps and the Euclidean baseline.
4. **Simplification / consolidation**, then **self-loop and parallel-edge dedupe**.
5. **Strongly Connected Component extraction** (Tarjan) — keep only the largest SCC so all Dijkstra distances are finite; excluded-node count, percentage, and barangay distribution are logged.

### Stage 2 — Edge Betweenness Centrality
EBC is computed **unweighted** (`weight=None`) so it reflects topology, then normalized to [0, 1]. Exact Brandes is available, plus a `k`-sampled approximate mode for large-graph tractability. Result is cached to `data/processed/ebc.parquet`.

### Stage 3 — Friction weights (three ablation conditions)
```
W1 (traversal cost)     w1 = length / speed(road_class)
W2 (EBC only)           w2 = normalized_EBC
W3 (composite, primary) w3 = w1 x (1 + alpha x normalized_EBC),  alpha = 1.0
```
W3 adapts the Bureau of Public Roads link-cost form, substituting normalized EBC for the volume-to-capacity ratio in the absence of behavioral traffic counts.

### Stage 4 — Node2Vec+ embeddings
For each weight condition, PecanPy's weight-aware Node2Vec+ (`extend=True`) generates **128-dimensional** node embeddings via biased random walks (`L=80`, `r=10`, `p=1.0`, `q=0.5`, `window=10`). `q < 1` biases toward global (DFS-like) structure, better for long-range distances. Produces `Z_W1`, `Z_W2`, `Z_W3`.

### Stage 5 — Stratified pair sampling and Dijkstra labels
A 10k pilot sample defines 10 distance deciles. Training (500k) and validation (50k) are **stratified** across deciles for balanced short- and long-range coverage; the test set (50k) is **uniform random** to reflect the natural distance skew. Exact friction-weighted Dijkstra distances are the ground-truth labels, computed per weight condition (directed pairs preserved).

### Stage 6-7 — Oracle architectures and the 3x2 ablation
- **Architecture A — Siamese:** shared twin branch `FC(128->256->128->64)` with BatchNorm + ReLU; combine by absolute element-wise difference `|z_u - z_v|`; prediction head `FC(64->32->1)`. Weight sharing enforces symmetry by construction.
- **Architecture B — MLP baseline:** concatenated `[z_u; z_v]` (256-d) through five FC layers to a linear scalar; symmetry only encouraged via reversed-pair augmentation.

Both are trained with **MSE** against Dijkstra labels using Adam (`lr=1e-3`, `weight_decay=1e-5`), cosine annealing over 100 epochs, early stopping on validation MAE (patience 10). The **3 weights x 2 architectures x 3 seeds = 18 runs** isolate the EBC contribution (H3) and the Siamese contribution (H4).

### Stage 8 — Evaluation
On a shared held-out test set: **Spearman rho** (primary), **MAE**, **RMSE**, the RMSE/MAE ratio, and multiplicative **distortion** (mean, worst-case, std), compared against Thorup-Zwick `(2k-1)` stretch at k = 2, 3. A **Euclidean great-circle baseline** is evaluated on the same pairs, and **high-divergence "broken-ruler" pairs** (>95th percentile Euclidean distortion) are extracted for mapping. Ablation results report mean +/- std across seeds with Cohen's d effect sizes and bootstrap confidence intervals.

### Stage 9 — Barangay structural-isolation choropleth
The best oracle produces a per-node structural-isolation score; nodes are spatially joined to their barangay polygon and averaged, yielding one score per barangay. Rendered as an interactive Folium choropleth, with the broken-ruler pairs on a second layer.

---

## Ablation Study (3 x 2 factorial, 3 seeds each)

| | Architecture A — Siamese | Architecture B — MLP |
|---|---|---|
| **W1** Traversal cost only | A-W1 | B-W1 |
| **W2** EBC only | A-W2 | B-W2 |
| **W3** Composite (primary) | A-W3 | B-W3 |

- **H3 (EBC signal):** compare W1 vs W3 within an architecture.
- **H4 (architecture):** compare Siamese vs MLP within a weight condition.

---

## Validation Metrics

| Metric | Role |
|---|---|
| Spearman rho | Primary — rank preservation of node-pair distances |
| MAE, RMSE | Absolute error magnitude (same units as the weight condition) |
| Distortion (mean / worst / std) | Ties results to classical oracle stretch bounds |
| Euclidean baseline + broken-ruler pairs | Where physical distance most misrepresents structural cost |

---

## Tools and Frameworks

```
Graph construction    OSMnx, NetworkX, GeoPandas, Shapely
Graph embedding       Node2Vec+ (PecanPy)
ML framework          PyTorch
Evaluation            scikit-learn, SciPy (spearmanr)
Spatial output        GeoPandas, Folium
```

All versions are pinned in `requirements.txt` for full reproducibility.

---

## Repository Structure

```
├── data/
│   ├── raw/                     # downloaded snapshots
│   ├── interim/                 # intermediate artifacts
│   ├── processed/               # G_scc, ebc.parquet, qc_barangays.gpkg, pairs_*.parquet
│   └── DATA_MANIFEST.md         # dataset sources, versions, retrieval dates
├── src/
│   ├── config.py                # speed table, hyperparameters, paths
│   ├── 00_download_data.py      # HDX roads + admin boundaries
│   ├── 01_graph_construction.py # build, clean, SCC extraction
│   ├── 01b_barangays.py         # barangay boundary resolver (fallback ladder)
│   ├── 02_ebc.py                # edge betweenness centrality (exact / k-sampled)
│   ├── 03_friction_weights.py   # W1, W2, W3 edge-weight conditions
│   ├── 04_embeddings.py         # Node2Vec+ per condition -> Z_W1..W3
│   ├── 05_sampling_labels.py    # stratified sampling + Dijkstra ground truth
│   ├── 06_models.py             # Siamese (A) + MLP (B)
│   ├── 07_train.py              # 3x2 x 3-seed ablation training
│   ├── 08_evaluate.py           # metrics, Euclidean baseline, ablation table
│   └── 09_isolation_choropleth.py # per-barangay isolation map
├── outputs/
│   ├── models/  embeddings/  figures/  tables/  maps/
├── requirements.txt
└── README.md
```

---

## Status

- [x] Thesis proposal finalized
- [x] Methodology pipeline designed (revised manuscript)
- [ ] Repo skeleton, config, requirements
- [ ] Data acquisition + barangay boundary resolver
- [ ] Graph construction, cleaning, SCC
- [ ] Edge Betweenness Centrality
- [ ] Friction weights (W1, W2, W3)
- [ ] Node2Vec+ embeddings
- [ ] Stratified sampling + Dijkstra labels
- [ ] Siamese / MLP training (3x2 ablation)
- [ ] Evaluation + ablation table
- [ ] Barangay isolation choropleth

---

## Novelty

This is the **first graph embedding-based distance analysis of any Philippine city's road network.** The combination of Node2Vec+ on EBC-enriched friction weights, a Siamese metric-learning oracle for road-distance approximation, and a barangay-level structural-isolation application has not been previously demonstrated. The trained oracle and its friction-weighted graph are reusable research artifacts for future Philippine urban network studies, and the methodology directly advances the Revised QC CDP 2026-2031 and UN SDGs 9 and 11.

---

## License

MIT License. All datasets are open source under their respective licenses. See `data/DATA_MANIFEST.md` for full attribution.

---

## Contact

**John Carlo I. Benter** — B.S. Computer Science, University of Santo Tomas
johncarlo.benter.cics@ust.edu.ph
