# FungalClassifier

A phylogeny-aware, multi-feature machine learning framework for classifying fungal genomes by taxonomy, ecological niche, and life history traits.

---

## Overview

FungalClassifier integrates heterogeneous genomic features — k-mer composition, protein domains, functional pathways, repeat content, and sequence motifs — into a unified classification pipeline. It is designed for datasets of annotated fungal genomes with associated phylogenetic and taxonomic metadata.

**Key design principles:**
- Phylogeny-aware cross-validation to prevent clade leakage
- Modular feature blocks that can be trained and evaluated independently
- Late fusion architecture for combining feature types
- Full SHAP-based interpretability

---

## Repository Structure

```
fungal-classifier/
├── fungal_classifier/           # Core Python package
│   ├── features/                # Feature extraction modules
│   │   ├── kmer.py              # k-mer and oligonucleotide composition
│   │   ├── domains.py           # Pfam/InterPro domain vectors
│   │   ├── pathways.py          # KEGG/GO/CAZyme pathway aggregation
│   │   ├── repeats.py           # Repeat content feature extraction
│   │   ├── motifs.py            # Promoter motif enrichment
│   │   └── fusion.py            # Feature block fusion strategies
│   ├── models/                  # Model definitions
│   │   ├── block_classifier.py  # Per-block XGBoost/LightGBM classifiers
│   │   ├── fusion_model.py      # Late fusion stacking model
│   │   └── deep_fusion.py       # PyTorch multi-modal neural net
│   ├── evaluation/              # Evaluation and interpretation
│   │   ├── phylo_cv.py          # Phylogeny-aware cross-validation
│   │   ├── metrics.py           # Classification metrics
│   │   └── shap_analysis.py     # SHAP feature importance
│   └── utils/                   # Shared utilities
│       ├── io.py                # Data loading helpers
│       ├── phylo.py             # Phylogenetic tree utilities
│       └── preprocessing.py     # Normalization and sparse handling
├── scripts/                     # CLI entry points
│   ├── 01_build_features.py
│   ├── 02_train.py
│   ├── 03_evaluate.py
│   └── 04_predict.py
├── configs/                     # YAML experiment configs
│   ├── default.yaml
│   └── deep_fusion.yaml
├── notebooks/                   # Exploratory notebooks
│   ├── 01_feature_exploration.ipynb
│   ├── 02_phylo_signal_analysis.ipynb
│   └── 03_shap_interpretation.ipynb
├── data/
│   ├── raw/                     # Raw genome annotations (gitignored)
│   ├── processed/               # Processed feature matrices
│   └── features/                # Per-block feature files (HDF5/Parquet)
├── tests/
├── docs/
│   ├── feature_engineering.md
│   ├── phylo_cv.md
│   └── model_architecture.md
├── environment.yml
├── pyproject.toml
└── README.md
```

---

## Installation

```bash
git clone https://github.com/your-org/fungal-classifier.git
cd fungal-classifier
conda env create -f environment.yml
conda activate fungal-classifier
pip install -e .
```

---

## Quick Start

### 1. Build feature matrices

```bash
python scripts/01_build_features.py \
    --genome-dir data/raw/genomes/ \
    --annotation-dir data/raw/annotations/ \
    --output-dir data/features/ \
    --config configs/default.yaml
```

### 2. Train classifiers

```bash
# Train all feature blocks + fusion model
python scripts/02_train.py \
    --features-dir data/features/ \
    --tree data/raw/phylogeny.nwk \
    --metadata data/raw/metadata.tsv \
    --target taxonomy_order \
    --config configs/default.yaml \
    --output-dir results/

# Train deep fusion model
python scripts/02_train.py \
    --config configs/deep_fusion.yaml \
    --target ecological_niche
```

### 3. Evaluate with phylogeny-aware CV

```bash
python scripts/03_evaluate.py \
    --model-dir results/ \
    --tree data/raw/phylogeny.nwk \
    --cv-strategy clade_holdout \
    --n-folds 10
```

### 4. Predict on new genomes

```bash
python scripts/04_predict.py \
    --model-dir results/ \
    --genome-dir data/new_genomes/ \
    --output predictions.tsv
```

---

## Feature Blocks

| Block | Features | Tools Required |
|---|---|---|
| `kmer` | k-mer frequencies k=1–6, dinucleotide relative abundance | jellyfish, sourmash |
| `domains` | Pfam/InterPro domain copy numbers | hmmer, pfam_scan |
| `pathways` | KEGG pathway counts, GO term aggregation, CAZyme profiles, BGC counts | KEGG, dbCAN, antiSMASH |
| `repeats` | TE class proportions, simple repeat density | RepeatMasker |
| `motifs` | Promoter motif enrichment scores | FIMO, JASPAR fungal PWMs |

---

## Metadata File Format

`data/raw/metadata.tsv` must contain:

| Column | Description |
|---|---|
| `genome_id` | Unique genome identifier (matches filename) |
| `taxonomy_kingdom` | e.g. Fungi |
| `taxonomy_phylum` | e.g. Ascomycota |
| `taxonomy_class` | |
| `taxonomy_order` | |
| `taxonomy_family` | |
| `taxonomy_genus` | |
| `ecological_niche` | e.g. saprotrophic, mycorrhizal, pathogenic |
| `lifestyle` | e.g. obligate_biotroph, hemibiotroph, necrotroph |
| `tree_label` | Label matching tip in phylogeny .nwk file |

---

## Phylogeny-Aware Cross-Validation

Standard random CV inflates accuracy for phylogenetically structured data. This framework implements **clade holdout CV**: entire clades are held out as test sets, ensuring the model cannot use phylogenetic proximity to cheat.

See [`docs/phylo_cv.md`](docs/phylo_cv.md) for details.

---

## Citation

If you use this framework, please cite:

```
[manuscript in preparation]
```

---

## License

MIT License. See `LICENSE`.
