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

from fungal_classifier.features.kmer import build_kmer_matrix
from fungal_classifier.features.domains import build_domain_matrix
from fungal_classifier.features.pathways import (
    build_kegg_matrix, build_cazyme_matrix, build_bgc_matrix, build_go_matrix
)
from fungal_classifier.features.repeats import build_repeat_matrix
from fungal_classifier.features.motifs import build_motif_matrix_from_genomes
from fungal_classifier.utils.io import (
    discover_genome_files, discover_annotation_files,
    load_metadata, save_feature_matrix
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Build feature matrices for FungalClassifier")
    p.add_argument("--genome-dir", type=Path, required=True, help="Directory of genome FASTA files")
    p.add_argument("--annotation-dir", type=Path, required=True, help="Directory of annotation files")
    p.add_argument("--metadata", type=Path, help="Metadata TSV file (optional, for genome ID filtering)")
    p.add_argument("--output-dir", type=Path, default=Path("data/features"), help="Output directory")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"), help="Config YAML")
    p.add_argument("--blocks", nargs="+", default=["kmer", "domains", "pathways", "repeats", "motifs"],
                   help="Which feature blocks to build")
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
            cazyme_paths = discover_annotation_files(
                args.annotation_dir / "dbcan",
                suffix="overview.txt",
                genome_ids=list(genome_paths.keys()),
            )
            if cazyme_paths:
                logger.info(f"Building CAZyme matrix from {len(cazyme_paths)} files...")
                cazyme_matrix = build_cazyme_matrix(
                    annotation_paths=cazyme_paths,
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
        pwm_db = args.annotation_dir / "jaspar" / "JASPAR2024_CORE_fungi_non-redundant_pfms_meme.txt"

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

    logger.info(f"\nAll feature matrices saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
