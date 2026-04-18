# Feature Engineering

## Overview

FungalClassifier integrates five complementary feature types, each capturing a distinct genomic signal. Each block can be trained independently to assess its individual contribution before fusion.

---

## Block 1: K-mer Composition (`kmer`)

**What it captures:** Sequence composition, codon usage bias, nucleotide skew, and higher-order patterns in the genome or CDS.

**Key parameters:**
- `k_values`: Recommended [1, 2, 3, 4, 5, 6]. k=4–6 becomes computationally intensive; k=6 has 4096 features.
- `normalize`: Use `obs_exp` (observed/expected ratio) to correct for base composition bias. This is especially important for comparing genomes with different GC content.
- `seq_type`: `cds` sequences emphasize coding regions (codon usage); `genomic` captures whole-genome composition including repeats.

**Dimensionality:** k=1: 4, k=2: 16, k=3: 64, k=4: 256, k=5: 1024, k=6: 4096. Total ≈ 5460 features before filtering.

**Notes:**
- Trinucleotide (k=3) composition is strongly associated with codon usage bias, which varies by ecological niche and expression level.
- Dinucleotide obs/exp ratios (especially CpG suppression) can reflect methylation history.
- k-mer profiles are highly redundant; SVD reduction to 200 components typically retains >90% of variance.

---

## Block 2: Protein Domains (`domains`)

**What it captures:** The functional repertoire of the proteome — which protein families are present and at what copy number.

**Input:** Pfam `hmmscan` domtblout files, or InterProScan TSV files.

**Key parameters:**
- `representation`: `copy_number` is preferred over `binary` — gene family expansions are biologically meaningful (e.g., expanded CAZyme families in saprotrophic fungi).
- `min_genome_freq`: Drop domains present in <1% of genomes to reduce sparsity.
- E-value threshold: 1e-5 is standard for Pfam hmmscan.

**Dimensionality:** ~5000–18,000 Pfam domains total; after min_genome_freq=0.01 filtering typically 2000–6000 remain.

**Notes:**
- Domain copy number is highly informative for lifestyle prediction: obligate biotrophs have reduced secreted protease/lipase repertoires; necrotrophs have expanded.
- Consider log-transforming copy numbers before modeling (many domain families have heavy-tailed distributions).

---

## Block 3: Functional Pathways (`pathways`)

Pathway features are split into four sub-blocks:

### 3a: KEGG Ortholog / Pathway Counts
Maps KO terms to KEGG pathways. Aggregation reduces sparsity from ~20,000 KO terms to ~500 pathways.

### 3b: GO Term Counts (GO-slim)
Full GO annotations are extremely sparse. Recommend aggregating to GO-slim (Pfam2GO slim or a custom fungal slim) or to level-2 GO terms. This reduces ~40,000 GO terms to ~200–500 informative categories.

### 3c: CAZyme Profiles (dbCAN)
Carbohydrate-active enzyme families are among the most discriminating features for fungal ecology:
- **Saprotrophs** (wood/litter decomposers): large GH, PL, CE, CBM repertoires
- **Mycorrhizal fungi**: CAZyme-reduced genomes (streamlined for biotrophic lifestyle)
- **Plant pathogens**: expanded pectinases (PL), glucanases (GH), cutin-degrading esterases

### 3d: Biosynthetic Gene Clusters (antiSMASH)
BGC type counts (PKS-I, PKS-II, NRPS, terpene, RiPP, etc.) are informative for distinguishing fungal orders and ecological roles in chemical ecology.

---

## Block 4: Repeat Content (`repeats`)

**What it captures:** Transposable element load, TE class composition, repeat landscape (divergence distribution).

**Input:** RepeatMasker `.out` files.

**Key parameters:**
- `normalize_by`: Always normalize by genome size — larger genomes tend to have more repeats simply due to size.
- `include_families`: Family-level features increase dimensionality substantially; use with caution.

**Biological relevance:**
- TE content correlates with genome size, which varies enormously in fungi (10–200 Mb).
- LTR retrotransposons (Ty3/gypsy, Ty1/copia) are the dominant TE class in most Ascomycetes.
- Some obligate biotrophs show near-complete TE elimination ("TE-free" genomes).
- Repeat-induced point mutation (RIP) history can be inferred from dinucleotide CpA/TpA ratios.

---

## Block 5: Motifs (`motifs`)

**What it captures:** Transcription factor binding site enrichment in promoter regions.

**Input:** FIMO output scanning JASPAR fungal PWMs against upstream sequences (1 kb promoters).

**Notes:**
- Motif features are sparse and computationally expensive to generate.
- Most informative for distinguishing expression programs associated with ecological niches (e.g., stress response TF binding enrichment in xerophytes).
- Can be replaced by k-mer features from promoter sequences alone as a proxy.

---

## Feature Selection and Fusion

### Per-block dimensionality reduction

Before fusion, each block undergoes:
1. **Variance threshold filter**: removes near-zero-variance features
2. **Univariate selection**: keeps top-k features by F-statistic or mutual information with target labels
3. **Optional SVD**: reduces sparse blocks (k-mer, domains) to dense lower-dimensional representations

### Fusion strategies

**Concatenation:** All blocks merged column-wise after normalization. Simple but doesn't weight blocks by informativeness.

**Stacking (default):** Each block classifier produces class probability vectors. These are concatenated as meta-features for a logistic regression meta-learner. Allows the meta-learner to learn how much to trust each block for each class.

**Attention (deep model only):** A learned scalar attention weight per block is computed from the fused embedding, producing a weighted sum. Allows the model to focus on the most informative blocks per sample.

---

## Feature Importance and Interpretation

After training, SHAP values are computed for each block classifier independently. This allows:

1. **Global importance**: Which features drive predictions across the whole dataset?
2. **Per-class importance**: Which features are most discriminative for *Saccharomycetales* vs *Hypocreales*?
3. **Block attribution**: What fraction of total SHAP importance comes from each feature type?

See `docs/model_architecture.md` for details on the SHAP analysis pipeline.
