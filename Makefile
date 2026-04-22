# Makefile — FungalClassifier pipeline orchestration
# Usage: make help

SHELL       := /bin/bash
PYTHON      := python
CONFIG      := configs/default.yaml
CONFIG_DEEP := configs/deep_fusion.yaml
METADATA    := data/raw/metadata.tsv
TREE        := data/raw/phylogeny.nwk
GENOME_DIR  := data/raw/genomes
ANNOT_DIR   := data/raw/annotations
FEAT_DIR    := data/features
RESULTS     := results
N_JOBS      := 8

.PHONY: help env metadata features train train-deep evaluate predict test lint clean

# ── help ───────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "FungalClassifier — pipeline targets"
	@echo "──────────────────────────────────────────────────────"
	@echo "  make env            Create conda environment"
	@echo "  make metadata       Build data/raw/metadata.tsv from taxonomy + FunGuild"
	@echo "  make features       Build all feature matrices"
	@echo "  make train          Train XGBoost + stacking (taxonomy_order)"
	@echo "  make train-deep     Train deep fusion model"
	@echo "  make train-ecology  Train on ecological_niche target"
	@echo "  make evaluate       Evaluate trained models"
	@echo "  make predict        Predict on new genomes in data/new_genomes/"
	@echo "  make compare        Compare all targets and blocks"
	@echo "  make test           Run unit tests"
	@echo "  make lint           Run ruff linter"
	@echo "  make clean          Remove generated files"
	@echo ""

# ── environment ────────────────────────────────────────────────────────────────
env:
	conda env create -f environment.yml
	@echo "Activate with: conda activate fungal-classifier"

SAMPLES     := $(ANNOT_DIR)/taxonomy/samples.csv
FUNGUILD    := $(ANNOT_DIR)/funguild/species_funguild.csv

# ── metadata ───────────────────────────────────────────────────────────────────
metadata: $(METADATA)

$(METADATA): $(SAMPLES) $(FUNGUILD)
	$(PYTHON) scripts/00_build_metadata.py \
		--samples     $(SAMPLES) \
		--funguild    $(FUNGUILD) \
		--genome-dir  $(GENOME_DIR) \
		--output      $(METADATA)

# ── feature building ───────────────────────────────────────────────────────────
features: $(METADATA)
	$(PYTHON) scripts/01_build_features.py \
		--genome-dir $(GENOME_DIR) \
		--annotation-dir $(ANNOT_DIR) \
		--metadata $(METADATA) \
		--output-dir $(FEAT_DIR) \
		--config $(CONFIG) \
		--n-jobs $(N_JOBS)

features-kmer-only:
	$(PYTHON) scripts/01_build_features.py \
		--genome-dir $(GENOME_DIR) \
		--annotation-dir $(ANNOT_DIR) \
		--output-dir $(FEAT_DIR) \
		--config $(CONFIG) \
		--blocks kmer \
		--n-jobs $(N_JOBS)

# ── training ───────────────────────────────────────────────────────────────────
train: $(FEAT_DIR)
	$(PYTHON) scripts/02_train.py \
		--features-dir $(FEAT_DIR) \
		--metadata $(METADATA) \
		--tree $(TREE) \
		--target taxonomy_order \
		--config $(CONFIG) \
		--output-dir $(RESULTS)/taxonomy_order \
		--cv-strategy clade_holdout

train-deep: $(FEAT_DIR)
	$(PYTHON) scripts/02_train.py \
		--features-dir $(FEAT_DIR) \
		--metadata $(METADATA) \
		--tree $(TREE) \
		--target taxonomy_order \
		--config $(CONFIG_DEEP) \
		--output-dir $(RESULTS)/taxonomy_order_deep \
		--model-type deep

train-ecology: $(FEAT_DIR)
	$(PYTHON) scripts/02_train.py \
		--features-dir $(FEAT_DIR) \
		--metadata $(METADATA) \
		--tree $(TREE) \
		--target ecological_niche \
		--config $(CONFIG) \
		--output-dir $(RESULTS)/ecological_niche \
		--cv-strategy clade_holdout \
		--phylo-eigenvectors

train-lifestyle: $(FEAT_DIR)
	$(PYTHON) scripts/02_train.py \
		--features-dir $(FEAT_DIR) \
		--metadata $(METADATA) \
		--tree $(TREE) \
		--target lifestyle \
		--config $(CONFIG) \
		--output-dir $(RESULTS)/lifestyle \
		--cv-strategy clade_holdout \
		--phylo-eigenvectors

train-all: train train-ecology train-lifestyle

# ── evaluation ─────────────────────────────────────────────────────────────────
evaluate:
	$(PYTHON) scripts/03_evaluate.py \
		--model-dir $(RESULTS)/taxonomy_order \
		--features-dir $(FEAT_DIR) \
		--metadata $(METADATA) \
		--tree $(TREE) \
		--target taxonomy_order

evaluate-all:
	for target in taxonomy_order ecological_niche lifestyle; do \
		$(PYTHON) scripts/03_evaluate.py \
			--model-dir $(RESULTS)/$$target \
			--features-dir $(FEAT_DIR) \
			--metadata $(METADATA) \
			--tree $(TREE) \
			--target $$target; \
	done

# ── multi-target comparison ────────────────────────────────────────────────────
compare:
	$(PYTHON) scripts/05_compare_targets.py \
		--results-dir $(RESULTS) \
		--output-dir $(RESULTS)/comparison

# ── prediction ─────────────────────────────────────────────────────────────────
predict:
	$(PYTHON) scripts/04_predict.py \
		--model-dir $(RESULTS)/taxonomy_order \
		--genome-dir data/new_genomes \
		--annotation-dir data/new_annotations \
		--output $(RESULTS)/predictions.tsv

# ── embeddings ─────────────────────────────────────────────────────────────────
embeddings:
	$(PYTHON) scripts/06_export_embeddings.py \
		--model-dir $(RESULTS)/taxonomy_order \
		--features-dir $(FEAT_DIR) \
		--metadata $(METADATA) \
		--output-dir $(RESULTS)/embeddings

# ── testing & linting ──────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/ -v --tb=short -m "not slow"

lint:
	ruff check fungal_classifier/ scripts/ tests/
	ruff format --check fungal_classifier/ scripts/ tests/

format:
	ruff format fungal_classifier/ scripts/ tests/

# ── cleanup ────────────────────────────────────────────────────────────────────
clean:
	rm -rf data/features/*.parquet data/features/*.h5
	rm -rf results/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name ".pytest_cache" -type d -exec rm -rf {} +

clean-features:
	rm -rf data/features/*.parquet data/features/*.h5

clean-results:
	rm -rf results/
