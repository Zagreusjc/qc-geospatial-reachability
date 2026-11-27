# qc-geospatial-reachability
# Neural Distance Oracles for Geospatial Reachability
### Graph Embedding and Metric Learning on Quezon City's Road Network

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
![License](https://img.shields.io/badge/License-MIT-green.svg)
[![Data](https://img.shields.io/badge/Data-Open%20Source-orange.svg)](https://data.humdata.org/)
![Status](https://img.shields.io/badge/Status-Pipeline%20Complete-brightgreen.svg)

> Undergraduate Thesis — B.S. Computer Science
> University of Santo Tomas, College of Information and Computing Sciences, 2025-2026
> Benter, J.C. · Magcaling, E.J. · Solano, S. · Vega, J.F.E. · Adviser: Estabillo, C.R.

---

## The Problem

In Quezon City — the most populous city in the Philippines, with 2.9 million residents across 142 barangays — **physical distance is a broken ruler.** Two barangays that look equally close to a hospital on a straight-line map can be worlds apart once you account for how the road network actually connects them: low-speed road classes, fragmented connectivity, and a handful of overloaded arterial corridors create permanent structural friction that a flat map cannot see.

The challenge is computational as much as geographic. Computing exact structural shortest-path distances across a road network of roughly 32,000 intersections (31,778 nodes and 73,071 edges in the study graph) with Dijkstra's algorithm is prohibitively expensive for repeated, citywide distance scoring. Local planners under the Revised Quezon City Comprehensive Development Plan 2026-2031 are mandated to make data-driven infrastructure decisions but lack a principled, structural distance measurement tool.

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

Assigning each node to a barangay and drawing the 142-barangay choropleth requires **barangay-level polygons**, not just the QC city outline. Open barangay polygons do exist. `src/01b_barangays.py` documents a **fallback ladder** of sources, but **only the first is implemented**; the others are documented alternatives that raise `NotImplementedError` if requested. The result is cached to `data/processed/qc_barangays.gpkg`:

1. **HDX COD-AB Philippines (`cod-ab-phl`), Admin Level 4** — barangay polygons on the same platform already cited; filter to QC. *(primary; the only implemented source)*
2. *(not implemented)* Curated PSA + NAMRIA GeoJSON (`bendlikeabamboo/barangay-boundaries-repository`, PSGC-annotated).
3. *(not implemented)* OpenStreetMap barangay relations (`admin_level=10`) via OSMnx.
4. *(not implemented)* geoBoundaries PHL ADM4 (CC-BY).
5. *(not implemented)* Last resort: synthesized barangay proxy zones via Voronoi tessellation of barangay centroids clipped to the QC boundary (would be documented as a limitation).

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

> **Reproducibility note.** The embeddings (`outputs/embeddings/Z_W*.npy`) are *not* committed, and Node2Vec+ run with several workers is not bit-for-bit reproducible. Re-generating them produces slightly different coordinates, so the committed checkpoints in `outputs/models/` no longer give the stored accuracy against re-generated embeddings. After re-embedding, retrain (`07_train.py`) before evaluating. `04_embeddings.py --sweep` also re-embeds W1, W2 and W3 and overwrites the primary files.

### Stage 5 — Stratified pair sampling and Dijkstra labels
A 10k pilot sample defines 10 distance deciles. Training (500k) and validation (50k) are **stratified** across deciles for balanced short- and long-range coverage; the test set (50k) is **uniform random** to reflect the natural distance skew. Exact friction-weighted Dijkstra distances are the ground-truth labels, computed per weight condition (directed pairs preserved).

### Stage 6-7 — Oracle architectures and the 3x3x3 ablation
Three architectures are trained. **The letters below are the code letters, which are also the run-id prefixes (e.g. `C-W3-seed0`); they differ from the thesis letters.** The thesis calls the order-aware Siamese "Architecture A", the MLP "Architecture B", and reports the symmetric Siamese only in Appendix A.

| Code letter | Code name | What it is | Thesis name |
|---|---|---|---|
| **A** | `siamese` | Shared twin branch `FC(128->256->128->64)` with LayerNorm + ReLU; combine by absolute element-wise difference `\|h_u - h_v\|`; head `FC(64->32->1)`. Symmetric by construction, so it predicts the same distance in both directions. | Appendix A, "symmetric variant" |
| **B** | `mlp` | Concatenated `[z_u; z_v]` (256-d) through five FC layers to a linear scalar. Symmetry is neither enforced nor augmented. | Architecture B (MLP baseline) |
| **C** | `siamese_ordered` | The same shared twin branch as A, but the head takes the ordered concatenation `[h_u; h_v]`, so it can predict different distances for `u->v` and `v->u` (one-way streets). | Architecture A (the headline "Siamese") |

All are trained with **MSE** against Dijkstra labels (z-score standardized, inverted at evaluation) using Adam (`lr=1e-3`, `weight_decay=1e-5`), cosine annealing over 100 epochs, early stopping on validation MAE (patience 10). The **3 weights x 3 architectures x 3 seeds = 27 runs** isolate the EBC contribution (H3) and the Siamese contribution (H4, order-aware Siamese vs MLP).

### Stage 8 — Evaluation
On a shared held-out test set: **Spearman rho** (primary), **MAE**, **RMSE**, the RMSE/MAE ratio, and multiplicative **distortion** (mean, worst-case, std), compared against Thorup-Zwick `(2k-1)` stretch at k = 2, 3. A **Euclidean great-circle baseline** is evaluated on the same pairs, and **high-divergence "broken-ruler" pairs** (>95th percentile Euclidean distortion) are extracted for mapping. Ablation results report mean +/- std across seeds with Cohen's d effect sizes and bootstrap confidence intervals.

### Stage 9 — Barangay structural-isolation choropleth
An oracle produces a per-node structural-isolation score (the mean predicted distance to every other node); nodes are spatially joined to their barangay polygon and averaged, yielding one score per barangay. Rendered as an interactive Folium choropleth, with the broken-ruler pairs on a second layer. By default `09_isolation_choropleth.py` uses the run with the highest test Spearman rho, which is a W2 run, so its scores are in W2 units; pass `--run-id C-W3-seed0` for W3 cost units, as in the thesis maps.

### Stage 10-13 — Supplementary analyses
- **Asymmetry (`10_asymmetry.py`):** how often one-way streets make `d(u,v)` differ from `d(v,u)`, and how well each oracle reproduces the difference.
- **Chapter IV figures (`11_chapter4_figures.py`):** Figure 4.3 (top 1% of edges by EBC), Figure 4.6 (where straight-line distance misleads most), Figure 4.7 (isolation choropleth).
- **Sample simulation (`12_sample_simulation.py`):** traces one test query step by step through every module, from raw road data to the oracle's answer.
- **Excess isolation (`13_excess_isolation.py`):** removes the location effect from the isolation score by comparing each node's actual mean cost with the cost expected for trips of the same straight-line length.

---

## Ablation Study (3 x 3 factorial, 3 seeds each)

Run ids use the code letters (A = symmetric Siamese, B = MLP, C = order-aware Siamese; see Stage 6-7).

| | A — symmetric Siamese (Appendix A) | B — MLP | C — order-aware Siamese (thesis headline) |
|---|---|---|---|
| **W1** Traversal cost only | A-W1 | B-W1 | C-W1 |
| **W2** EBC only | A-W2 | B-W2 | C-W2 |
| **W3** Composite (primary) | A-W3 | B-W3 | C-W3 |

- **H3 (EBC signal):** compare W1 vs W3 within an architecture.
- **H4 (architecture):** compare the order-aware Siamese (C) vs MLP (B) within a weight condition. The symmetric variant (A) is reported separately in the thesis appendix.

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
│   ├── interim/                 # intermediate artifacts (PecanPy edge lists)
│   ├── processed/               # G_scc, ebc.parquet, qc_barangays.gpkg, pairs_*.parquet
│   └── DATA_MANIFEST.md         # dataset sources, versions, retrieval dates
├── src/
│   ├── config.py                # speed table, hyperparameters, paths
│   ├── utils.py                 # logger, timer, manifest helpers
│   ├── 00_download_data.py      # HDX roads + admin boundaries
│   ├── 01_graph_construction.py # build, clean, SCC extraction
│   ├── 01b_barangays.py         # barangay boundaries (HDX ADM4; other ladder sources not implemented)
│   ├── 02_ebc.py                # edge betweenness centrality (exact / k-sampled)
│   ├── 03_friction_weights.py   # W1, W2, W3 edge-weight conditions
│   ├── 04_embeddings.py         # Node2Vec+ per condition -> Z_W1..W3 (--sweep: p/q variants)
│   ├── 04b_pq_sensitivity.py    # Node2Vec+ p/q sensitivity study (needs 04 --sweep and 05)
│   ├── 05_sampling_labels.py    # stratified sampling + Dijkstra ground truth
│   ├── 06_models.py             # architectures A (symmetric Siamese), B (MLP), C (order-aware Siamese)
│   ├── 07_train.py              # 27-run ablation training (3 weights x 3 architectures x 3 seeds)
│   ├── 08_evaluate.py           # metrics, Euclidean baseline, ablation table
│   ├── 09_isolation_choropleth.py # per-barangay isolation map
│   ├── 10_asymmetry.py          # one-way asymmetry check
│   ├── 11_chapter4_figures.py   # Chapter IV figures 4.3, 4.6, 4.7
│   ├── 12_sample_simulation.py  # one query traced step by step
│   └── 13_excess_isolation.py   # excess isolation (location effect removed)
├── outputs/
│   ├── models/  embeddings/  figures/  tables/  maps/
├── Makefile                     # targets for the main pipeline (see its header for the rest)
├── requirements.txt
└── README.md
```

---

## Status

- [x] Thesis proposal finalized
- [x] Methodology pipeline designed (revised manuscript)
- [x] Repo skeleton, config, requirements
- [x] Data acquisition + barangay boundary resolver
- [x] Graph construction, cleaning, SCC
- [x] Edge Betweenness Centrality
- [x] Friction weights (W1, W2, W3)
- [x] Node2Vec+ embeddings
- [x] Stratified sampling + Dijkstra labels
- [x] Siamese / MLP training (3x3x3 ablation, 27 runs)
- [x] Evaluation + ablation table
- [x] Barangay isolation choropleth
- [x] Supplementary analyses: asymmetry, Chapter IV figures, sample simulation, excess isolation

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
