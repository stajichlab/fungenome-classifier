# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
# Preferred: conda
conda env create -f environment.yml
conda activate fungal-classifier
pip install -e .

# Alternative: pixi (manages Python 3.11+ and CUDA 12.8)
pixi install
```

Dev extras: `pip install -e ".[dev]"` adds pytest, ruff, pre-commit.

## Commands

```bash
# Tests
make test-fast                      # unit tests only (excludes @pytest.mark.slow)
make test                           # full suite including integration tests
pytest tests/test_features.py -v   # single file

# Lint / format
make lint                           # ruff check + ruff format --check
make format                         # apply ruff fixes

# Pipeline
make features                       # 01_build_features.py (all blocks)
make train                          # 02_train.py → taxonomy_order, clade-holdout CV
make train-all                      # train + train-ecology + train-lifestyle
make evaluate                       # 03_evaluate.py → confusion matrices, Blomberg's K
make embeddings                     # 06_export_embeddings.py → PCA/UMAP TSVs
make predict                        # 04_predict.py on data/new_genomes/
```

`make train N_JOBS=16 CONFIG=configs/deep_fusion.yaml` to override defaults.

## Architecture

Two-stage **late-fusion** classifier:

1. **Feature extraction** (`scripts/01_build_features.py`) — Each feature block reads its annotation files and writes a genome × features DataFrame to `data/features/*.parquet`.
2. **Per-block classifiers** (`scripts/02_train.py`) — An independent XGBoost/LightGBM is trained per block using clade-holdout CV. Out-of-fold (OOF) probability vectors are collected.
3. **Fusion meta-learner** — OOF probability vectors from all blocks are stacked and fed to a logistic regression (or XGBoost) meta-learner.
4. **Deep fusion alternative** (`configs/deep_fusion.yaml`) — Per-block linear towers compress each block to 128-dim embeddings, then an attention layer weights blocks before a shared classification head. Auxiliary per-block losses (weight 0.2) keep towers meaningful.

## Feature Blocks

Each block lives in `fungal_classifier/features/<name>.py` and exposes a `build_<name>_matrix(annotation_paths, **kwargs) -> pd.DataFrame` returning a genome × features matrix indexed by `genome_id`.

| Block | Input suffix | Key output features |
|---|---|---|
| `kmer` | genome `.fna` | k=1–6 relative abundances (~5460 dims) |
| `domains` | `.domtblout` | Pfam copy numbers (binary or count) |
| `pathways` | KEGG `_ko.tsv`, dbCAN `overview.txt`, antiSMASH `.json` | Pathway/CAZyme/BGC counts |
| `repeats` | RepeatMasker `.out` | TE class fractions (7 classes) |
| `motifs` | `.gff3` + JASPAR PWMs | FIMO motif enrichment per TF |
| `subcellular` | `.tmhmm_short.tsv`, `.signalp.results.txt`, etc. | Fraction of proteins with TMH/SP/localization |
| `disorder` | `.aiupred.txt` | Mean disorder score, IDR fraction, IDR count |
| `proteases` | `.merops.blasttab` | MEROPS family copy numbers |
| `composition` | `.cds-transcripts.codon_freq.csv` or `.cds-transcripts.fa` | Codon usage, AA freq, gene count |
| `genomic` | genome `.fna` + `.faa` | Genome length, N50, GC%, protein length stats |
| `introns` | `.gff3` + genome `.fna` | Intron counts/lengths, donor/acceptor dinucleotides, PPT score |

## File Discovery Convention

`fungal_classifier/utils/io.py::discover_annotation_files(dir, suffix, genome_ids)` matches files by stem prefix = `genome_id`. All annotation files must be named `{genome_id}{suffix}`. Genome IDs are the FASTA stem (without `.fna`).

## Cross-Validation

`clade_holdout` (default) holds out all genomes from a taxonomic clade (order/family/genus) at once, preventing phylogenetic leakage. Random or stratified CV dramatically inflates performance estimates for fungi due to close relatives being split across folds. Always use `clade_holdout` for publication-quality results.

## Adding a New Feature Block

1. Create `fungal_classifier/features/<name>.py` with a `build_<name>_matrix(paths, **kwargs) -> pd.DataFrame` function.
2. Export it from `fungal_classifier/features/__init__.py`.
3. Add a config section under `features:` in `configs/default.yaml`.
4. Add a handler block in `scripts/01_build_features.py` (follow the existing pattern: discover files → call builder → `save_feature_matrix`).
5. Add `"<name>"` to the `--blocks` default list in `parse_args()`.

## Expected Data Layout

```
data/raw/
  genomes/           {genome_id}.fna
  annotations/
    pfam/            {genome_id}.domtblout[.gz]
    dbcan/           {genome_id}_overview.txt[.gz]
    antismash/       {genome_id}.json[.gz]
    repeatmasker/    {genome_id}.out[.gz]
    gff/             {genome_id}.gff3[.gz]
    kegg/            {genome_id}_ko.tsv[.gz]
    tmhmm/           {genome_id}.tmhmm_short.tsv[.gz]
    signalp/         {genome_id}.signalp.results.txt[.gz]
    wolfpsort/       {genome_id}.wolfpsort.results.txt[.gz]
    targetp/         {genome_id}_summary.targetp2[.gz]
    aiupred/         {genome_id}.aiupred.txt[.gz]
    merops/          {genome_id}.merops.blasttab[.gz]
    cds/             {genome_id}.cds-transcripts.fa[.gz]
                     {genome_id}.cds-transcripts.codon_freq.csv
    proteins/        {genome_id}.faa[.gz]
    # note: gff/ is shared between motifs and introns blocks
  phylogeny.nwk      (tip labels = genome_id)
  metadata.tsv       (index = genome_id; columns include taxonomy_order, ecological_niche, lifestyle)
```

## Code Standards

- Python 3.11+, 100-char line length (ruff)
- NumPy-style docstrings on public functions; type hints required
- `ruff` (linter) + `ruff format` (formatter) enforced via pre-commit
- Tests in `tests/`; mark slow integration tests with `@pytest.mark.slow`
- Coverage target: >80% for `fungal_classifier/` core modules
