"""Central configuration for the Quezon City Neural Distance Oracle pipeline.

Every phase script imports this module so that all paths, dataset sources, and
hyperparameters live in exactly one place. Values follow the revised thesis
manuscript (Chapter III). Nothing here performs I/O on import except computing
paths; call `ensure_dirs()` to create the folder tree.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MAPS_DIR = OUTPUTS_DIR / "maps"

ALL_DIRS = [
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    MODELS_DIR, EMBEDDINGS_DIR, FIGURES_DIR, TABLES_DIR, MAPS_DIR,
]


def ensure_dirs() -> None:
    """Create the full data/outputs folder tree if it does not exist."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Study area & coordinate reference systems
# ---------------------------------------------------------------------------
PLACE_NAME = "Quezon City, Metro Manila, Philippines"
CRS_WGS84 = "EPSG:4326"        # lat/lon: mapping + Euclidean great-circle baseline
CRS_UTM_51N = "EPSG:32651"     # projected metres: edge length + geometry math
N_BARANGAYS = 142              # expected barangay count for validation

# ---------------------------------------------------------------------------
# Dataset sources (Phase 0 / Phase 1b)
# ---------------------------------------------------------------------------
# HOTOSM Philippines Roads (HDX). Resolve the concrete resource URL at download
# time; the dataset landing page is kept here for provenance.
HOTOSM_ROADS_HDX = "https://data.humdata.org/dataset/hotosm_phl_roads"

# PSA / NAMRIA COD-AB administrative boundaries (ADM3 city clip + ADM4 barangays).
COD_AB_HDX = "https://data.humdata.org/dataset/cod-ab-phl"

# Barangay-boundary fallback ladder (first that resolves wins; see 01b_barangays.py).
BARANGAY_SOURCES = [
    {"name": "hdx_cod_ab_adm4", "url": COD_AB_HDX},
    {"name": "curated_psa_namria",
     "url": "https://github.com/bendlikeabamboo/barangay-boundaries-repository"},
    {"name": "osm_admin_level_10", "url": None},  # fetched live via OSMnx/Overpass
    {"name": "geoboundaries_adm4", "url": "https://www.geoboundaries.org"},
    {"name": "voronoi_synthetic", "url": None},   # last resort, documented as a limitation
]

# ---------------------------------------------------------------------------
# Graph cleaning (Phase 1)
# ---------------------------------------------------------------------------
# Drivable surface road classes retained; everything else (footway, cycleway,
# pedestrian, path, steps, construction, ...) is dropped.
HIGHWAY_WHITELIST = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
    "tertiary", "tertiary_link",
    "unclassified",
    "residential",
    "service",
    "living_street",
]

# JICA (2015) estimated operating speed per road class, in km/h (manuscript table).
ROAD_CLASS_SPEED_KMH = {
    "motorway": 60, "motorway_link": 60,
    "trunk": 60, "trunk_link": 60,
    "primary": 40, "primary_link": 40,
    "secondary": 30, "secondary_link": 30,
    "tertiary": 20, "tertiary_link": 20,
    "unclassified": 15,
    "residential": 15,
    "service": 10,
    "living_street": 10,
}
# Edges with a missing/unknown highway tag default to this class (conservative).
DEFAULT_ROAD_CLASS = "residential"

# ---------------------------------------------------------------------------
# Edge Betweenness Centrality (Phase 2)
# ---------------------------------------------------------------------------
# "exact" = full Brandes (slow, hours+ on ~70k nodes);
# "approx" = k sampled source nodes (recommended for development / limited CPU).
EBC_MODE = "approx"
EBC_K = 3000            # number of sampled pivots when EBC_MODE == "approx"
EBC_SEED = 42           # fixed seed so approximate EBC is reproducible
# EBC uses unweighted shortest paths (weight=None) per the manuscript, then is
# min-max normalized to [0, 1].

# ---------------------------------------------------------------------------
# Friction weights (Phase 3)
# ---------------------------------------------------------------------------
ALPHA = 1.0                       # EBC load multiplier in W3 = W1 * (1 + alpha * EBC_norm)
ALPHA_SENSITIVITY = [0.5, 1.0, 2.0]  # supplementary sensitivity sweep
WEIGHT_CONDITIONS = ["W1", "W2", "W3"]

# ---------------------------------------------------------------------------
# Node2Vec+ embeddings (Phase 4)
# ---------------------------------------------------------------------------
N2V_DIM = 128           # embedding dimensionality
N2V_WALK_LENGTH = 80    # L
N2V_NUM_WALKS = 10      # r (walks per node)
N2V_P = 1.0             # return parameter
N2V_Q = 0.5             # in-out parameter (<1 biases toward global/DFS structure)
N2V_WINDOW = 10         # Skip-Gram window size
N2V_WORKERS = 4         # threads (matches available cores)
N2V_EXTEND = True       # extend=True enables the Node2Vec+ weight-aware mechanism

# Supplementary p/q sensitivity sweep (reports validation MAE per combination).
N2V_P_SWEEP = [0.5, 1.0, 2.0]
N2V_Q_SWEEP = [0.25, 0.5, 1.0]

# ---------------------------------------------------------------------------
# Stratified pair sampling + Dijkstra labels (Phase 5)
# ---------------------------------------------------------------------------
PILOT_PAIRS = 10_000    # pilot sample used to estimate distance deciles
N_TRAIN = 500_000       # stratified across deciles
N_VAL = 50_000          # stratified across deciles, disjoint from train
N_TEST = 50_000         # uniform random (natural distance skew)
N_DECILES = 10
SAMPLING_SEED = 42

# ---------------------------------------------------------------------------
# Oracle training (Phases 6-7)
# ---------------------------------------------------------------------------
ARCHITECTURES = ["siamese", "mlp"]   # A and B
SEEDS = [0, 1, 2]                    # three seeds per ablation cell

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10        # on validation MAE
BATCH_SIZE = 1024
LR_SCHEDULE = "cosine"              # cosine annealing over EPOCHS
STANDARDIZE_LABELS = True          # z-score targets during training, invert at eval
DEVICE = "cpu"                      # no GPU in this environment; "cuda" if available

# Layer widths shared by both architectures (per manuscript tables).
SIAMESE_BRANCH_DIMS = [128, 256, 128, 64]  # input -> ... -> latent
SIAMESE_HEAD_DIMS = [64, 32, 1]            # |z_u - z_v| -> scalar distance
MLP_DIMS = [256, 256, 128, 64, 32, 1]      # concat[z_u; z_v] -> scalar distance

# ---------------------------------------------------------------------------
# Evaluation (Phase 8)
# ---------------------------------------------------------------------------
HIGH_DIVERGENCE_PERCENTILE = 95    # high-divergence pairs threshold (Euclidean distortion)
THORUP_ZWICK_K = [2, 3]            # (2k-1) stretch reference points: 3 and 5
BOOTSTRAP_RESAMPLES = 1000
COHENS_D_THRESHOLD = 0.5           # substantive effect size for ablation comparisons


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------
def speed_mps(road_class: str) -> float:
    """Operating speed in metres/second for a road class (defaults applied)."""
    kmh = ROAD_CLASS_SPEED_KMH.get(road_class, ROAD_CLASS_SPEED_KMH[DEFAULT_ROAD_CLASS])
    return kmh * 1000.0 / 3600.0


def ablation_runs():
    """Yield every (weight_condition, architecture, seed) cell of the 3x2x3 design."""
    for weight in WEIGHT_CONDITIONS:
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                yield weight, arch, seed


def run_id(weight: str, arch: str, seed: int) -> str:
    """Canonical identifier for one ablation run, e.g. 'A-W3-seed1'."""
    arch_letter = "A" if arch == "siamese" else "B"
    return f"{arch_letter}-{weight}-seed{seed}"
