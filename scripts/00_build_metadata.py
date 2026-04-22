#!/usr/bin/env python3
"""Build data/raw/metadata.tsv from taxonomy samples and FunGuild annotations.

Sources
-------
- data/raw/annotations/taxonomy/samples.csv  — genome taxonomy table
- data/raw/annotations/funguild/species_funguild.csv  — FunGuild lifestyle/guild

Join key: samples.LOCUSTAG == funguild.species_prefix

Output columns (tab-separated):
  genome_id         — FASTA stem (matched to genome dir when provided, else derived)
  taxonomy_phylum   — PHYLUM
  taxonomy_class    — CLASS
  taxonomy_order    — ORDER
  ecological_niche  — primary FunGuild guild (pipe-delimited primary term), normalised
  lifestyle         — FunGuild trophicMode, normalised (lowercase, dashes→underscores)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def _primary_guild(guild_str: str) -> str:
    """Return the primary guild term.

    FunGuild marks the most-likely guild in pipe characters, e.g.
    ``Plant Pathogen-|Wood Saprotroph|``. Return the pipe-bracketed term when
    present; otherwise return the first dash-separated term.
    """
    pipe_match = re.search(r"\|([^|]+)\|", guild_str)
    if pipe_match:
        return pipe_match.group(1).strip()
    return guild_str.split("-")[0].strip()


def _normalise(text: str) -> str:
    """Lowercase and replace spaces/dashes with underscores."""
    return re.sub(r"[\s\-]+", "_", text.strip().lower())


def _candidate_ids(speciesin: str, strain: str, species: str = "") -> list[str]:
    """Return genome_id candidates in preference order.

    FASTA stems follow `{name}_{strain}.scaffolds.fa` but the name base varies:
    - For formal species: name = SPECIESIN (strain already embedded) or SPECIES
    - For "sp." entries: strain is always appended even if already in SPECIESIN
    - Variety/forma-specialis qualifiers are stripped from formal species names

    Trying candidates in order handles all observed cases without hard-coded rules.
    """
    si = speciesin.strip().replace(" ", "_")
    sp = species.strip().replace(" ", "_")
    st = strain.strip().replace(" ", "_")

    seen: set[str] = set()
    result: list[str] = []

    def add(c: str) -> None:
        if c and c not in seen:
            seen.add(c)
            result.append(c)

    # 1. SPECIESIN + strain (always append — covers most "sp." entries)
    add(f"{si}_{st}" if st else si)
    # 2. SPECIESIN as-is (strain already embedded in SPECIESIN)
    add(si)
    # 3. SPECIES + strain (strips var./f.sp./str. qualifiers)
    add(f"{sp}_{st}" if st else sp)
    # 4. SPECIES alone (empty strain)
    add(sp)

    return result


def _resolve_genome_id(
    speciesin: str,
    strain: str,
    species: str,
    known_stems: set[str] | None,
) -> str:
    """Return the best genome_id for this sample.

    When *known_stems* is provided, pick the first candidate that matches a
    real FASTA file stem.  Otherwise use SPECIESIN + strain as the default
    (covers the majority of cases).
    """
    candidates = _candidate_ids(speciesin, strain, species)
    if known_stems is not None:
        for c in candidates:
            if c in known_stems:
                return c
    # Default: SPECIESIN + strain (always append)
    base = speciesin.strip().replace(" ", "_")
    strain_n = strain.strip().replace(" ", "_")
    return f"{base}_{strain_n}" if strain_n else base


# ASMID → genome_id for entries where derivation misses due to taxonomy corrections
# or irregular naming in the source data.
_GENOME_ID_OVERRIDES: dict[str, str] = {
    "GCF_000002855.4_ASM285v2": "Aspergillus_niger",           # file has no strain suffix
    "GCA_030572455.1_ASM3057245v1": "Candida_koratica__NBRC_103208",   # trailing space → double underscore
    "GCA_030570575.1_ASM3057057v1": "Candida_sp.__NRRL_Y-17713",       # trailing space → double underscore
}


def _load_genome_stems(genome_dir: Path, suffix: str = ".scaffolds.fa") -> set[str]:
    return {p.name[: -len(suffix)] for p in genome_dir.iterdir() if p.name.endswith(suffix)}


def build_metadata(
    samples_path: Path,
    funguild_path: Path,
    output_path: Path,
    genome_dir: Path | None = None,
) -> None:
    known_stems: set[str] | None = None
    if genome_dir is not None and genome_dir.is_dir():
        known_stems = _load_genome_stems(genome_dir)

    # Load FunGuild keyed by species_prefix (== LOCUSTAG in samples)
    funguild: dict[str, dict[str, str]] = {}
    with funguild_path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            funguild[row["species_prefix"].strip()] = row

    rows_written = 0
    rows_no_guild = 0

    with (
        samples_path.open(newline="") as src,
        output_path.open("w", newline="") as dst,
    ):
        reader = csv.DictReader(src)
        writer = csv.writer(dst, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "genome_id",
                "taxonomy_phylum",
                "taxonomy_class",
                "taxonomy_order",
                "ecological_niche",
                "lifestyle",
            ]
        )

        for row in reader:
            asmid = row["ASMID"].strip()
            if asmid in _GENOME_ID_OVERRIDES:
                genome_id = _GENOME_ID_OVERRIDES[asmid]
            else:
                genome_id = _resolve_genome_id(
                    row["SPECIESIN"], row["STRAIN"], row["SPECIES"], known_stems
                )
            locustag = row["LOCUSTAG"].strip()

            fg = funguild.get(locustag)
            if fg is None:
                ecological_niche = ""
                lifestyle = ""
                rows_no_guild += 1
            else:
                ecological_niche = _normalise(_primary_guild(fg["guild"]))
                lifestyle = _normalise(fg["trophicMode"])

            writer.writerow(
                [
                    genome_id,
                    row["PHYLUM"].strip(),
                    row["CLASS"].strip(),
                    row["ORDER"].strip(),
                    ecological_niche,
                    lifestyle,
                ]
            )
            rows_written += 1

    print(
        f"Wrote {rows_written} rows to {output_path}  "
        f"({rows_no_guild} without FunGuild match)",
        file=sys.stderr,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--samples",
        default="data/raw/annotations/taxonomy/samples.csv",
        help="Path to samples.csv (default: %(default)s)",
    )
    p.add_argument(
        "--funguild",
        default="data/raw/annotations/funguild/species_funguild.csv",
        help="Path to species_funguild.csv (default: %(default)s)",
    )
    p.add_argument(
        "--output",
        default="data/raw/metadata.tsv",
        help="Output metadata TSV path (default: %(default)s)",
    )
    p.add_argument(
        "--genome-dir",
        default="data/raw/genomes",
        help="Genome directory to anchor genome_id to actual FASTA stems (default: %(default)s)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gdir = Path(args.genome_dir) if args.genome_dir else None
    build_metadata(
        samples_path=Path(args.samples),
        funguild_path=Path(args.funguild),
        output_path=Path(args.output),
        genome_dir=gdir,
    )
