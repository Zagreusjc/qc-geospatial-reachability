# Neural Distance Oracle for Quezon City -- pipeline orchestration
#
# Phases run in order; each caches its outputs so later phases can be re-run
# without repeating earlier work. Run `make help` for the list of targets.
#
# Typical first run:
#   make setup            # install dependencies
#   make data             # download datasets + resolve barangay boundaries
#   make graph ebc weights embeddings labels
#   make ablation         # train all 18 runs, then evaluate
#   make maps             # Structural Distance Maps

PYTHON ?= python
SRC := src

.PHONY: help setup data graph ebc weights embeddings labels \
        train evaluate ablation maps all clean-outputs

help:
	@echo "Targets:"
	@echo "  setup       Install pinned dependencies (requirements.txt)"
	@echo "  data        Phase 0 + 1b: download datasets, resolve barangays"
	@echo "  graph       Phase 1 : build + clean graph, extract largest SCC"
	@echo "  ebc         Phase 2 : edge betweenness centrality"
	@echo "  weights     Phase 3 : friction weights W1/W2/W3"
	@echo "  embeddings  Phase 4 : Node2Vec+ embeddings per condition"
	@echo "  labels      Phase 5 : stratified sampling + Dijkstra labels"
	@echo "  train       Phase 7 : train all 18 ablation runs"
	@echo "  evaluate    Phase 8 : metrics + ablation_results.csv"
	@echo "  ablation    train + evaluate (the full ablation study)"
	@echo "  maps        Phase 9 : Structural Distance Maps"
	@echo "  all         Run the entire pipeline end to end"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) $(SRC)/00_download_data.py
	$(PYTHON) $(SRC)/01b_barangays.py

graph:
	$(PYTHON) $(SRC)/01_graph_construction.py

ebc:
	$(PYTHON) $(SRC)/02_ebc.py

weights:
	$(PYTHON) $(SRC)/03_friction_weights.py

embeddings:
	$(PYTHON) $(SRC)/04_embeddings.py

labels:
	$(PYTHON) $(SRC)/05_sampling_labels.py

# Phase 7: trains the full 3 x 2 x 3 grid (18 runs). Narrow it down with flags, e.g.
#   $(PYTHON) src/07_train.py --condition W3
#   $(PYTHON) src/07_train.py --arch siamese --seed 0
train:
	$(PYTHON) $(SRC)/07_train.py

evaluate:
	$(PYTHON) $(SRC)/08_evaluate.py

# The ablation study = train every cell then evaluate them against the shared test set.
ablation: train evaluate

maps:
	$(PYTHON) $(SRC)/09_isolation_choropleth.py

all: data graph ebc weights embeddings labels ablation maps

clean-outputs:
	rm -rf outputs/models/* outputs/embeddings/* outputs/figures/* \
	       outputs/tables/* outputs/maps/*
	@echo "Cleared outputs/ (kept .gitkeep files)."
