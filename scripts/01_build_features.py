#!/usr/bin/env python3
"""
scripts/01_build_features.py

Build all feature matrices from genome annotations.

Usage:
    python scripts/01_build_features.py \\
        --genome-dir data/raw/genomes/ \\
        --annotation-dir data/raw/annotations/ \\
        --output-dir data/features/ \\
        --config configs/default.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fungal_classifier.features.composition import (
    build_composition_matrix_from_csvs,
    build_composition_matrix_from_fasta,
)
from fungal_classifier.features.disorder import build_disorder_matrix
from fungal_classifier.features.domains import build_domain_matrix
from fungal_classifier.features.genomic import build_genomic_matrix
from fungal_classifier.features.introns import build_intron_matrix
from fungal_classifier.features.kmer import build_kmer_matrix
from fungal_classifier.features.motifs import build_motif_matrix_from_genomes
from fungal_classifier.features.pathways import (
    build_bgc_matrix,
    build_cazyme_matrix,
)
from fungal_classifier.features.proteases import build_merops_matrix
from fungal_classifier.features.repeats import build_repeat_matrix
from fungal_classifier.features.subcellular import build_subcellular_matrix
from fungal_classifier.utils.io import (
    discover_annotation_files,
    discover_genome_files,
    load_metadata,
    load_taxonomy,
    save_feature_matrix,
    validate_species_prefixes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build feature matrices for FungalClassifier")
    p.add_argument("--genome-dir", type=Path, required=True, help="Directory of genome FASTA files")
    p.add_argument(
        "--annotation-dir", type=Path, required=True, help="Directory of annotation files"
    )
    p.add_argument(
        "--metadata", type=Path, help="Metadata TSV file (optional, for genome ID filtering)"
    )
    p.add_argument(
        "--taxonomy", type=Path, help="Taxonomy CSV (samples.csv) for metadata enrichment"
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/features"), help="Output directory"
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config YAML")
    p.add_argument(
        "--blocks",
        nargs="+",
        default=[
            "kmer",
            "domains",
            "pathways",
            "repeats",
            "motifs",
            "subcellular",
            "disorder",
            "proteases",
            "composition",
            "genomic",
            "introns",
        ],
        help="Which feature blocks to build",
    )
    p.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs")
    p.add_argument("--format", choices=["parquet", "hdf5"], default="parquet")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover genome files
    genome_paths = discover_genome_files(args.genome_dir)
    logger.info(f"Found {len(genome_paths)} genomes")

    # Optionally filter to metadata genome IDs
    if args.metadata:
        meta = load_metadata(args.metadata)
        genome_paths = {gid: p for gid, p in genome_paths.items() if gid in meta.index}
        logger.info(f"After metadata filter: {len(genome_paths)} genomes")

    # Optionally load taxonomy table
    if args.taxonomy and args.taxonomy.exists():
        taxonomy = load_taxonomy(args.taxonomy)
        taxonomy.to_csv(args.output_dir / "taxonomy_metadata.csv")
        logger.info(f"Taxonomy saved to {args.output_dir / 'taxonomy_metadata.csv'}")
        validate_species_prefixes(args.taxonomy, args.annotation_dir)

    # ── k-mer features ──────────────────────────────────────────────────────
    if "kmer" in args.blocks:
        kmer_cfg = config["features"]["kmer"]
        logger.info("Building k-mer feature matrix...")
        kmer_matrix = build_kmer_matrix(
            fasta_paths=genome_paths,
            k_values=kmer_cfg["k_values"],
            normalize=kmer_cfg["normalize"],
            seq_type=kmer_cfg["sequence_type"],
            n_jobs=args.n_jobs,
        )
        save_feature_matrix(
            kmer_matrix,
            args.output_dir / f"kmer.{args.format}",
            format=args.format,
        )

    # ── domain features ──────────────────────────────────────────────────────
    if "domains" in args.blocks:
        dom_cfg = config["features"]["domains"]
        domain_paths = discover_annotation_files(
            args.annotation_dir / "pfam",
            suffix=".domtblout",
            genome_ids=list(genome_paths.keys()),
        )
        if domain_paths:
            logger.info(f"Building domain feature matrix from {len(domain_paths)} files...")
            domain_matrix = build_domain_matrix(
                annotation_paths=domain_paths,
                format="domtblout",
                representation=dom_cfg["representation"],
                min_genome_freq=dom_cfg["min_genome_freq"],
            )
            save_feature_matrix(
                domain_matrix,
                args.output_dir / f"domains.{args.format}",
                format=args.format,
            )
        else:
            logger.warning("No Pfam domtblout files found — skipping domain features")

    # ── pathway features ─────────────────────────────────────────────────────
    if "pathways" in args.blocks:
        pw_cfg = config["features"]["pathways"]

        if pw_cfg.get("cazyme"):
            dbcan_dir = args.annotation_dir / "dbcan"
            cazyme_paths = discover_annotation_files(
                dbcan_dir,
                suffix="overview.tsv",
                genome_ids=list(genome_paths.keys()),
            )
            substrate_paths = discover_annotation_files(
                dbcan_dir,
                suffix="substrates.tsv",
                genome_ids=list(genome_paths.keys()),
            )
            if substrate_paths:
                logger.info(f"Found {len(substrate_paths)} dbCAN substrate files")
            if cazyme_paths:
                logger.info(f"Building CAZyme matrix from {len(cazyme_paths)} files...")
                cazyme_matrix = build_cazyme_matrix(
                    annotation_paths=cazyme_paths,
                    substrate_paths=substrate_paths or None,
                    min_genome_freq=pw_cfg["min_genome_freq"],
                )
                save_feature_matrix(
                    cazyme_matrix,
                    args.output_dir / f"cazyme.{args.format}",
                    format=args.format,
                )

        if pw_cfg.get("bgc"):
            bgc_paths = discover_annotation_files(
                args.annotation_dir / "antismash",
                suffix=".json",
                genome_ids=list(genome_paths.keys()),
            )
            if bgc_paths:
                logger.info(f"Building BGC matrix from {len(bgc_paths)} files...")
                bgc_matrix = build_bgc_matrix(
                    annotation_paths=bgc_paths,
                    min_genome_freq=pw_cfg["min_genome_freq"],
                )
                save_feature_matrix(
                    bgc_matrix,
                    args.output_dir / f"bgc.{args.format}",
                    format=args.format,
                )

    # ── repeat features ──────────────────────────────────────────────────────
    if "repeats" in args.blocks:
        rpt_cfg = config["features"]["repeats"]
        repeat_paths = discover_annotation_files(
            args.annotation_dir / "repeatmasker",
            suffix=".out",
            genome_ids=list(genome_paths.keys()),
        )
        if repeat_paths:
            logger.info(f"Building repeat feature matrix from {len(repeat_paths)} files...")
            repeat_matrix = build_repeat_matrix(
                rmout_paths=repeat_paths,
                normalize_by=rpt_cfg["normalize_by"],
                classes_to_include=rpt_cfg["classes"],
            )
            save_feature_matrix(
                repeat_matrix,
                args.output_dir / f"repeats.{args.format}",
                format=args.format,
            )
        else:
            logger.warning("No RepeatMasker .out files found — skipping repeat features")

    # ── motif features ───────────────────────────────────────────────────────
    if "motifs" in args.blocks:
        motif_cfg = config["features"]["motifs"]
        pwm_db = (
            args.annotation_dir / "jaspar" / "JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt"
        )

        if not pwm_db.exists():
            logger.warning(
                f"JASPAR PWM database not found at {pwm_db}. "
                "Download with:\n"
                "  wget https://jaspar.elixir.lu/download/data/2024/CORE/"
                "JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt "
                f"-O {pwm_db}\n"
                "Skipping motif features."
            )
        else:
            gff_paths = discover_annotation_files(
                args.annotation_dir / "gff",
                suffix=".gff3",
                genome_ids=list(genome_paths.keys()),
            )
            if gff_paths:
                logger.info(f"Building motif feature matrix for {len(gff_paths)} genomes...")
                motif_matrix = build_motif_matrix_from_genomes(
                    genome_fastas={gid: genome_paths[gid] for gid in gff_paths},
                    gff_paths=gff_paths,
                    pwm_database=pwm_db,
                    work_dir=args.annotation_dir / "fimo_work",
                    upstream_bp=motif_cfg["upstream_bp"],
                    p_value_threshold=motif_cfg["p_value_threshold"],
                    n_jobs=args.n_jobs,
                )
                save_feature_matrix(
                    motif_matrix,
                    args.output_dir / f"motifs.{args.format}",
                    format=args.format,
                )
            else:
                logger.warning("No GFF3 files found — skipping motif features")

    # ── subcellular localisation features ────────────────────────────────────
    if "subcellular" in args.blocks:
        tmhmm_paths = discover_annotation_files(
            args.annotation_dir / "tmhmm",
            suffix=".tmhmm_short.tsv",
            genome_ids=list(genome_paths.keys()),
        )
        signalp_paths = discover_annotation_files(
            args.annotation_dir / "signalp",
            suffix=".signalp.results.txt",
            genome_ids=list(genome_paths.keys()),
        )
        wolfpsort_paths = discover_annotation_files(
            args.annotation_dir / "wolfpsort",
            suffix=".wolfpsort.results.txt",
            genome_ids=list(genome_paths.keys()),
        )
        targetp_paths = discover_annotation_files(
            args.annotation_dir / "targetp",
            suffix="_summary.targetp2",
            genome_ids=list(genome_paths.keys()),
        )

        if any([tmhmm_paths, signalp_paths, wolfpsort_paths, targetp_paths]):
            logger.info("Building subcellular feature matrix...")
            subcellular_matrix = build_subcellular_matrix(
                tmhmm_paths=tmhmm_paths or None,
                signalp_paths=signalp_paths or None,
                wolfpsort_paths=wolfpsort_paths or None,
                targetp_paths=targetp_paths or None,
            )
            if not subcellular_matrix.empty:
                save_feature_matrix(
                    subcellular_matrix,
                    args.output_dir / f"subcellular.{args.format}",
                    format=args.format,
                )
        else:
            logger.warning("No subcellular annotation files found — skipping subcellular features")

    # ── disorder features ────────────────────────────────────────────────────
    if "disorder" in args.blocks:
        dis_cfg = config["features"].get("disorder", {})
        aiupred_paths = discover_annotation_files(
            args.annotation_dir / "aiupred",
            suffix=".aiupred.txt",
            genome_ids=list(genome_paths.keys()),
        )
        if aiupred_paths:
            logger.info(f"Building disorder feature matrix from {len(aiupred_paths)} files...")
            disorder_matrix = build_disorder_matrix(
                annotation_paths=aiupred_paths,
                threshold=dis_cfg.get("threshold", 0.5),
                idr_min_length=dis_cfg.get("idr_min_length", 10),
            )
            save_feature_matrix(
                disorder_matrix,
                args.output_dir / f"disorder.{args.format}",
                format=args.format,
            )
        else:
            logger.warning("No AIUPred files found — skipping disorder features")

    # ── protease features ────────────────────────────────────────────────────
    if "proteases" in args.blocks:
        pro_cfg = config["features"].get("proteases", {})
        merops_paths = discover_annotation_files(
            args.annotation_dir / "merops",
            suffix=".merops.blasttab",
            genome_ids=list(genome_paths.keys()),
        )
        if merops_paths:
            logger.info(f"Building MEROPS feature matrix from {len(merops_paths)} files...")
            merops_matrix = build_merops_matrix(
                annotation_paths=merops_paths,
                min_identity=pro_cfg.get("min_identity", 30.0),
                min_align_len=pro_cfg.get("min_align_len", 50),
                max_evalue=pro_cfg.get("max_evalue", 1e-5),
                min_genome_freq=pro_cfg.get("min_genome_freq", 0.01),
            )
            save_feature_matrix(
                merops_matrix,
                args.output_dir / f"proteases.{args.format}",
                format=args.format,
            )
        else:
            logger.warning("No MEROPS BLAST files found — skipping protease features")

    # ── composition features (codon usage, AA freq, gene count) ─────────────
    if "composition" in args.blocks:
        cds_dir = args.annotation_dir / "cds"

        codon_csv_paths = discover_annotation_files(
            cds_dir,
            suffix=".cds-transcripts.codon_freq.csv",
            genome_ids=list(genome_paths.keys()),
        )
        cds_fasta_paths = discover_annotation_files(
            cds_dir,
            suffix=".cds-transcripts.fa",
            genome_ids=list(genome_paths.keys()),
        )

        if codon_csv_paths:
            logger.info(
                f"Building composition matrix from {len(codon_csv_paths)} codon-freq CSVs..."
            )
            comp_matrix = build_composition_matrix_from_csvs(
                codon_csv_paths=codon_csv_paths,
                cds_fasta_paths=cds_fasta_paths or None,
            )
        elif cds_fasta_paths:
            logger.info(f"Building composition matrix from {len(cds_fasta_paths)} CDS FASTAs...")
            comp_matrix = build_composition_matrix_from_fasta(
                cds_fasta_paths=cds_fasta_paths,
                n_jobs=args.n_jobs,
            )
        else:
            comp_matrix = None
            logger.warning("No CDS files found — skipping composition features")

        if comp_matrix is not None and not comp_matrix.empty:
            save_feature_matrix(
                comp_matrix,
                args.output_dir / f"composition.{args.format}",
                format=args.format,
            )

    # ── genomic size features (genome length, N50, protein lengths) ──────────
    if "genomic" in args.blocks:
        prot_dir = args.annotation_dir / "proteins"
        protein_fasta_paths = {}
        if prot_dir.exists():
            for ext in (
                ".proteins.faa",
                ".faa",
                ".pep",
                ".aa",
                ".proteins.faa.gz",
                ".faa.gz",
                ".pep.gz",
            ):
                protein_fasta_paths = discover_annotation_files(
                    prot_dir,
                    suffix=ext,
                    genome_ids=list(genome_paths.keys()),
                )
                if protein_fasta_paths:
                    break

        logger.info(
            f"Building genomic size matrix "
            f"({len(genome_paths)} genomes, {len(protein_fasta_paths)} protein FASTAs)..."
        )
        genomic_matrix = build_genomic_matrix(
            genome_fasta_paths=genome_paths,
            protein_fasta_paths=protein_fasta_paths or None,
            n_jobs=args.n_jobs,
        )
        if not genomic_matrix.empty:
            save_feature_matrix(
                genomic_matrix,
                args.output_dir / f"genomic.{args.format}",
                format=args.format,
            )

    # ── intron structure + splice site features ──────────────────────────────
    if "introns" in args.blocks:
        intron_cfg = config["features"].get("introns", {})
        gff_paths = discover_annotation_files(
            args.annotation_dir / "gff",
            suffix=".gff3",
            genome_ids=list(genome_paths.keys()),
        )
        if gff_paths:
            logger.info(
                f"Building intron feature matrix from {len(gff_paths)} GFF3 files "
                f"(genome FASTAs matched: "
                f"{sum(gid in genome_paths for gid in gff_paths)})..."
            )
            intron_matrix = build_intron_matrix(
                gff_paths=gff_paths,
                genome_fasta_paths={
                    gid: genome_paths[gid] for gid in gff_paths if gid in genome_paths
                }
                or None,
                feature_types=tuple(intron_cfg.get("feature_types", ["exon"])),
                ppt_window=intron_cfg.get("ppt_window", 20),
                n_jobs=args.n_jobs,
            )
            if not intron_matrix.empty:
                save_feature_matrix(
                    intron_matrix,
                    args.output_dir / f"introns.{args.format}",
                    format=args.format,
                )
        else:
            logger.warning("No GFF3 files found — skipping intron features")

    logger.info(f"\nAll feature matrices saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
