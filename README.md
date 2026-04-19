# FungalClassifier

A phylogeny-aware, multi-feature machine learning framework for classifying fungal genomes by taxonomy, ecological niche, and life history traits.

---

## Overview

FungalClassifier integrates heterogeneous genomic features — k-mer composition, protein domains, functional pathways, repeat content, sequence motifs, subcellular localisation, intrinsic disorder, protease repertoire, and codon usage — into a unified classification pipeline. It is designed for datasets of annotated fungal genomes with associated phylogenetic and taxonomic metadata.

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
│   │   ├── pathways.py          # KEGG/GO/CAZyme/BGC pathway aggregation
│   │   ├── repeats.py           # Repeat content feature extraction
│   │   ├── motifs.py            # Promoter motif enrichment (FIMO/JASPAR)
│   │   ├── subcellular.py       # TMHMM, SignalP, WolfPSORT, TargetP
│   │   ├── disorder.py          # Intrinsic disorder (AIUPred)
│   │   ├── proteases.py         # Protease repertoire (MEROPS BLAST)
│   │   ├── composition.py       # Codon usage, AA frequency, gene count
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
    --taxonomy data/raw/annotations/taxonomy/samples.csv \
    --config configs/default.yaml
```

Select specific blocks with `--blocks` (default: all):

```bash
python scripts/01_build_features.py \
    --genome-dir data/raw/genomes/ \
    --annotation-dir data/raw/annotations/ \
    --output-dir data/features/ \
    --blocks kmer domains pathways subcellular disorder composition
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

| Block | Features | Input files | Tools required |
|---|---|---|---|
| `kmer` | k-mer frequencies k=1–6, obs/expected ratios | genome FASTA | — |
| `domains` | Pfam/InterPro domain copy numbers | `pfam/*.domtblout[.gz]` | hmmer, pfam_scan |
| `pathways` | KEGG KO counts, GO terms, CAZyme profiles, BGC types | `kegg/`, `dbcan/`, `antismash/` | KEGG, dbCAN, antiSMASH |
| `repeats` | TE class proportions, mean divergence per class | `repeatmasker/*.out[.gz]` | RepeatMasker |
| `motifs` | Promoter motif enrichment scores | `gff/*.gff3`, JASPAR PWMs | FIMO, bedtools |
| `subcellular` | TM helix counts, signal peptide fraction, compartment fractions, targeting probabilities | `tmhmm/`, `signalp/`, `wolfpsort/`, `targetp/` | TMHMM2, SignalP-6, WolfPSORT, TargetP-2 |
| `disorder` | Mean/median disorder score, disordered residue fraction, long IDR fraction | `aiupred/*.aiupred.txt[.gz]` | AIUPred |
| `proteases` | Protease family/clan copy numbers, protease proteome fraction | `merops/*.merops.blasttab[.gz]` | BLAST vs MEROPS pepunit |
| `composition` | 61 codon frequencies, 20 AA frequencies, GC1/2/3, gene count | `cds/*.codon_freq.csv` or `cds/*.fa[.gz]` | — |

All annotation files are auto-discovered in gzip-compressed (`.gz`) or uncompressed form.

### Annotation directory layout

```
data/raw/annotations/
├── pfam/            # hmmscan domtblout:       {genome_id}.domtblout[.gz]
├── dbcan/           # dbCAN overview:           {genome_id}_overview.txt[.gz]
├── antismash/       # antiSMASH JSON:           {genome_id}.json[.gz]
├── repeatmasker/    # RepeatMasker out:         {genome_id}.out[.gz]
├── gff/             # Gene annotations:         {genome_id}.gff3[.gz]
├── kegg/            # KO annotations:           {genome_id}_ko.tsv[.gz]
├── tmhmm/           # TMHMM2 short format:      {genome_id}.tmhmm_short.tsv[.gz]
├── signalp/         # SignalP-6 results:        {genome_id}.signalp.results.txt[.gz]
├── wolfpsort/       # WolfPSORT results:        {genome_id}.wolfpsort.results.txt[.gz]
├── targetp/         # TargetP-2 summary:        {genome_id}_summary.targetp2[.gz]
├── aiupred/         # AIUPred disorder scores:  {genome_id}.aiupred.txt[.gz]
├── merops/          # MEROPS BLAST tabular:     {genome_id}.merops.blasttab[.gz]
├── cds/             # CDS sequences and codon tables:
│   ├── {genome_id}.cds-transcripts.fa[.gz]
│   └── {genome_id}.cds-transcripts.codon_freq.csv
└── taxonomy/
    └── samples.csv  # Assembly-to-taxonomy mapping
```

---

## Metadata File Format

`data/raw/metadata.tsv` must contain:

| Column | Description |
|---|---|
| `genome_id` | Unique genome identifier (matches filename stem) |
| `taxonomy_kingdom` | e.g. Fungi |
| `taxonomy_phylum` | e.g. Ascomycota |
| `taxonomy_class` | |
| `taxonomy_order` | |
| `taxonomy_family` | |
| `taxonomy_genus` | |
| `ecological_niche` | e.g. saprotrophic, mycorrhizal, pathogenic |
| `lifestyle` | e.g. obligate_biotroph, hemibiotroph, necrotroph |
| `tree_label` | Label matching tip in phylogeny .nwk file |

### Taxonomy CSV (`annotations/taxonomy/samples.csv`)

An optional assembly-level taxonomy table can supplement or replace the metadata TSV. It is loaded with `load_taxonomy()` and saved alongside feature matrices when `--taxonomy` is provided. Expected columns:

| Column | Description |
|---|---|
| `ASMID` | Assembly accession (index; e.g. `GCF_000001.1_Asm1`) |
| `SPECIESIN` | Full species + strain name |
| `PHYLUM` – `SPECIES` | Full taxonomic hierarchy |
| `NCBI_TAXONID` | NCBI Taxonomy ID |
| `BUSCO_LINEAGE` | BUSCO lineage used for QC |
| `LOCUSTAG` | Locus tag prefix linking to protein IDs in annotation files |

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
