# qc-accessibility-oracle
# A Neural Distance Oracle for Structural Urban Accessibility
### Graph Embedding and Metric Learning on Quezon City's Road Network

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Data](https://img.shields.io/badge/Data-Open%20Source-orange.svg)](https://data.humdata.org/)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

> Undergraduate Thesis — B.S. Computer Science Major in Data Science  
> University of Santo Tomas · 2025-2026  

---

## The Problem

In Quezon City — the most populous city in the Philippines with 2.96 million residents across 142 barangays — **physical distance is a broken ruler.**

A resident of Batasan Hills and a resident of Blue Ridge may live the same Euclidean distance from the nearest hospital. But their *structural* accessibility — the ease with which the road network actually connects them to that hospital — differs dramatically. Fragmented road connectivity, a small number of overloaded arterial corridors, and deep residential interior areas create permanent mobility gaps that a flat map cannot see.

The challenge is computational as much as it is geographic. Computing exact structural shortest-path distances across a road network of ~70,000 intersections using Dijkstra's algorithm requires **O(N² log N)** time and **O(N²)** space — prohibitive for repeated citywide accessibility queries.

---

## What This Study Builds

A **neural distance oracle** — a Siamese neural network trained on Node2Vec+ embeddings of Quezon City's friction-weighted road network graph — that approximates structural shortest-path distances in **O(1) query time** after a one-time preprocessing step.

No GPS data. No behavioral observations. No proprietary sources. Everything is derived from open, static, freely available datasets.

```
HDX Roads Shapefile (static)
        ↓
Friction-Weighted Graph
Road Class Traversal Cost × Edge Betweenness Centrality
        ↓
Node2Vec+ Structural Embeddings (64-dim)
+ Node Feature Vector (6-dim)
= 70-dim Node Identity
        ↓
Siamese Network trained on Dijkstra friction distances
        ↓
Neural Distance Oracle — O(1) structural distance queries
        ↓
Structural Accessibility Scores × 142 QC Barangays
        ↓
Accessibility Desert Map of Quezon City
```

---

## Research Questions

**RQ1 — Approximation Quality**
Can a Siamese network trained on Node2Vec+ embeddings approximate exact Dijkstra friction distances with Spearman ρ > 0.75 and MAE < 8 minutes on held-out node pairs?

**RQ2 — Edge Weight Design**
Does incorporating Edge Betweenness Centrality as a global topological load multiplier produce statistically significantly better distance approximations than road-class traversal cost alone?

**RQ3 — Equity Application**
Does the trained oracle identify structurally underserved barangays that Euclidean distance-based analysis systematically misclassifies as accessible?

---

## Datasets

All datasets are pre-existing, freely available, and statically timestamped at download. No live APIs. No proprietary sources.

| Dataset | Source | Role |
|---|---|---|
| HOTOSM Philippines Roads | [HDX](https://data.humdata.org/dataset/hotosm_phl_roads) | Graph skeleton — nodes and edges |
| HOTOSM Philippines POI | [HDX](https://data.humdata.org/dataset/hotosm_phl_points_of_interest) | Destination nodes — hospitals, schools, markets, transit |
| HOTOSM Philippines Buildings | [HDX](https://data.humdata.org/dataset/hotosm_phl_buildings) | Node building density feature |
| PSA Admin Boundaries | [HDX](https://data.humdata.org/dataset/cod-ab-phl) | QC clip boundary + barangay aggregation |
| PSA 2020 Census | [PSA](https://psa.gov.ph/population-and-housing) | Population density weight |

---

## Methodology

### Stage 1 — Graph Construction
The HDX roads shapefile is clipped to QC's administrative boundary and converted into a directed weighted graph using OSMnx and NetworkX. Nodes are road intersections (~70,000). Edges are road segments (~180,000).

### Stage 2 — Structural Friction Weights
Each edge receives a two-component friction weight:

```
friction_weight(e) = traversal_cost(e) × (1 + α × normalized_EBC(e))
```

**Component A — Traversal Cost:** Length divided by road-class speed calibrated to Metro Manila's documented operating conditions (MMDA regulations, ScienceDirect 2019 observed speeds).

**Component B — Edge Betweenness Centrality (EBC):** The proportion of all-pairs shortest paths in the QC network that pass through each edge. Computed via NetworkX. Identifies permanent topological bottlenecks.

### Stage 3 — Node2Vec+ Embeddings
Node2Vec+ (Newaz et al., 2023) — a weighted-graph-aware extension of Node2Vec — generates 64-dimensional structural embeddings via biased random walks on the friction-weighted graph. Each embedding is concatenated with a 6-dimensional node feature vector (population density, building density, distance to nearest hospital/school/market/transit).

### Stage 4 — Siamese Network Training
A Siamese neural network (shared encoder: 70→128→64→32, L2-normalized output) is trained on 5,000 sampled node pairs using pairwise MSE loss against Dijkstra friction distances as ground truth. After training, structural distances between any two nodes are computed as the Euclidean distance between their 32-dimensional latent representations — **O(1)** query time.

### Stage 5 — Accessibility Scoring and Desert Mapping
Node-level Structural Accessibility Scores aggregate inverse latent distances to nearby POIs. Scores are population-weighted and aggregated to 142 barangays. DBSCAN clustering delineates Accessibility Desert zones. Results rendered as an interactive Folium choropleth map.

---

## Ablation Study

The oracle is trained and evaluated under four conditions to empirically determine the contribution of each design choice:

| Condition | Edge Weight | Architecture |
|---|---|---|
| A — Baseline | Physical length | MLP |
| B — Road class only | Traversal cost | MLP |
| C — Full friction + MLP | Traversal cost × EBC | MLP |
| D — Full friction + Siamese | Traversal cost × EBC | Siamese |

---

## Validation

Evaluated on 1,000 held-out node pairs against two baselines:

| Metric | Target | Baseline 1 | Baseline 2 |
|---|---|---|---|
| Spearman ρ | > 0.75 | Euclidean distance | Unweighted Dijkstra |
| MAE (min) | < 8 min | Euclidean (rescaled) | Unweighted Dijkstra |
| RMSE (min) | < 12 min | Euclidean (rescaled) | Unweighted Dijkstra |

**Broken Ruler Case Studies:** The 10 node pairs with the greatest discrepancy between Euclidean rank and latent distance rank are documented with their structural explanation.

---

## Tools and Frameworks

```
Graph Construction    OSMnx, NetworkX, GeoPandas
Graph Embedding       Node2Vec+ (Newaz et al., 2023)
ML Framework          PyTorch
Clustering            DBSCAN (Scikit-learn)
Spatial Output        GeoPandas, Folium
Validation            Spearman ρ, MAE, RMSE
```

---

## Key References

- Boeing, G. (2017). OSMnx. *Computers, Environment and Urban Systems*, 65, 126–139.
- Boeing, G. (2025). Modeling and Analyzing Urban Networks and Amenities with OSMnx. *Geographical Analysis*.
- Grover, A., & Leskovec, J. (2016). node2vec. *KDD 2016*.
- Newaz, A. et al. (2023). Node2Vec+. *Oxford Bioinformatics*.
- Kirkley, A. et al. (2018). Betweenness centrality in street networks. *Nature Communications*.
- Bromley, J. et al. (1993). Siamese neural network. *NIPS 1993*.
- Rizi, A. et al. (2018). Shortest path distance approximation using deep learning. *ASONAM 2018*.
- Zhao, Y. et al. (2022). RNE: Road network embedding. *VLDB Journal*.

---

## Novelty

This is the **first graph embedding study of any Philippine city's road network.** The specific combination of:

- Node2Vec+ on EBC-enriched structural friction weights
- Siamese metric learning architecture for road distance approximation
- Urban accessibility equity application of a neural distance oracle

has not been previously demonstrated in the literature.

---

## Repository Structure

```
├── data/
│   └── DATA_MANIFEST.txt        ← dataset download dates and sources
├── src/
│   ├── 01_graph_construction.py
│   ├── 02_friction_weights.py
│   ├── 03_node_features.py
│   ├── 04_node2vec_embeddings.py
│   ├── 05_siamese_training.py
│   ├── 06_accessibility_scoring.py
│   └── 07_desert_mapping.py
├── notebooks/
│   └── exploratory_analysis.ipynb
├── outputs/
│   └── (trained model + choropleth map generated here)
├── requirements.txt
└── README.md
```

---

## Status

- [x] Thesis proposal finalized
- [x] Dataset plan confirmed
- [x] Methodology pipeline designed
- [ ] Graph construction — in progress
- [ ] Friction weight computation — pending
- [ ] Node2Vec+ embeddings — pending
- [ ] Siamese network training — pending
- [ ] Accessibility scoring — pending
- [ ] Desert map output — pending

---

## License

MIT License. All datasets are open source under their respective licenses. See `data/DATA_MANIFEST.txt` for full attribution.

---

## Contact

**[Your Name]**  
B.S. Data Science, [University Name]  
[your.email@university.edu.ph]  
[LinkedIn URL] · [GitHub URL]