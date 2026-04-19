"""
fungal_classifier/features/subcellular.py

Protein subcellular localisation and membrane topology feature extraction.

Handles:
  - TMHMM2  : transmembrane topology (tmhmm_short format)
  - SignalP 6: signal peptide prediction
  - WolfPSORT: subcellular localisation
  - TargetP 2: subcellular targeting (SP / mTP / noTP)
"""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

WOLFPSORT_LOCS = [
    "nucl",
    "mito",
    "cyto",
    "extr",
    "plas",
    "E.R.",
    "golg",
    "vacu",
    "cysk",
    "pero",
    "cyto_nucl",
    "cyto_pero",
]


# ── TMHMM ────────────────────────────────────────────────────────────────────


def parse_tmhmm(path: Path) -> pd.DataFrame:
    """
    Parse TMHMM2 short-format output.

    Format (tab-separated):
        protein_id  len=N  ExpAA=N  First60=N  PredHel=N  Topology=str

    Returns tidy DataFrame with columns: protein_id, length, exp_aa,
    first60, pred_hel, topology.
    """
    records = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            try:
                kv = {p.split("=")[0]: p.split("=")[1] for p in parts[1:] if "=" in p}
                records.append(
                    {
                        "protein_id": parts[0],
                        "length": int(kv.get("len", 0)),
                        "exp_aa": float(kv.get("ExpAA", 0)),
                        "first60": float(kv.get("First60", 0)),
                        "pred_hel": int(kv.get("PredHel", 0)),
                        "topology": kv.get("Topology", ""),
                    }
                )
            except (IndexError, ValueError):
                continue
    return pd.DataFrame(records)


def tmhmm_to_features(df: pd.DataFrame) -> pd.Series:
    """Aggregate per-protein TMHMM predictions to a genome-level feature vector."""
    if df.empty:
        return pd.Series(dtype=np.float32)

    n = len(df)
    features: dict[str, float] = {
        "tmhmm_frac_with_tm": float((df["pred_hel"] >= 1).sum()) / n,
        "tmhmm_mean_predhel": float(df["pred_hel"].mean()),
        "tmhmm_mean_expaa": float(df["exp_aa"].mean()),
    }
    for k in range(6):
        label = str(k) if k < 5 else "5plus"
        mask = (df["pred_hel"] == k) if k < 5 else (df["pred_hel"] >= 5)
        features[f"tmhmm_frac_hel_{label}"] = float(mask.sum()) / n

    return pd.Series(features, dtype=np.float32)


def build_tmhmm_matrix(
    annotation_paths: dict[str, Path],
) -> pd.DataFrame:
    """Build genome × TMHMM feature matrix."""
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="TMHMM features"):
        try:
            df = parse_tmhmm(path)
            rows[genome_id] = tmhmm_to_features(df)
        except Exception as e:
            logger.warning(f"Failed TMHMM parse for {genome_id}: {e}")
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"TMHMM matrix: {matrix.shape}")
    return matrix


# ── SignalP ───────────────────────────────────────────────────────────────────


def parse_signalp(path: Path) -> pd.DataFrame:
    """
    Parse SignalP-6 output.

    Format (tab-separated, # comment header):
        ID  Prediction  OTHER  SP(Sec/SPI)  CS Position

    The ID column contains two space-separated tokens; the first is used as
    protein_id. Returns DataFrame with columns: protein_id, prediction, sp_prob.
    """
    records = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            try:
                protein_id = parts[0].split()[0]
                prediction = parts[1].strip()
                sp_prob = float(parts[3])
                records.append(
                    {
                        "protein_id": protein_id,
                        "prediction": prediction,
                        "sp_prob": sp_prob,
                    }
                )
            except (IndexError, ValueError):
                continue
    return pd.DataFrame(records)


def signalp_to_features(df: pd.DataFrame) -> pd.Series:
    """Aggregate SignalP predictions to genome-level features."""
    if df.empty:
        return pd.Series(dtype=np.float32)
    n = len(df)
    return pd.Series(
        {
            "signalp_frac_sp": float((df["prediction"] == "SP").sum()) / n,
            "signalp_mean_sp_prob": float(df["sp_prob"].mean()),
        },
        dtype=np.float32,
    )


def build_signalp_matrix(annotation_paths: dict[str, Path]) -> pd.DataFrame:
    """Build genome × SignalP feature matrix."""
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="SignalP features"):
        try:
            df = parse_signalp(path)
            rows[genome_id] = signalp_to_features(df)
        except Exception as e:
            logger.warning(f"Failed SignalP parse for {genome_id}: {e}")
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"SignalP matrix: {matrix.shape}")
    return matrix


# ── WolfPSORT ─────────────────────────────────────────────────────────────────


def parse_wolfpsort(path: Path) -> pd.DataFrame:
    """
    Parse WolfPSORT output.

    Format (space-delimited, # comment header):
        protein_id  loc1 score1, loc2 score2, ...

    Returns DataFrame with columns: protein_id, top_loc, and one column per
    known localisation containing the kNN score (0 if absent).
    """
    records = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split at first space to separate protein_id from the rest
            tokens = line.split(" ", 1)
            if len(tokens) < 2:
                continue
            protein_id = tokens[0]
            loc_scores: dict[str, float] = {}
            for pair in tokens[1].split(", "):
                pair = pair.strip()
                parts = pair.split(" ")
                if len(parts) == 2:
                    try:
                        loc_scores[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
            top_loc = max(loc_scores, key=loc_scores.get) if loc_scores else "unknown"
            row: dict = {"protein_id": protein_id, "top_loc": top_loc}
            row.update(loc_scores)
            records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        return df
    for loc in WOLFPSORT_LOCS:
        if loc not in df.columns:
            df[loc] = 0.0
    df[WOLFPSORT_LOCS] = df[WOLFPSORT_LOCS].fillna(0.0)
    return df


def wolfpsort_to_features(df: pd.DataFrame) -> pd.Series:
    """Aggregate WolfPSORT predictions to genome-level features."""
    if df.empty:
        return pd.Series(dtype=np.float32)
    n = len(df)
    features: dict[str, float] = {}
    for loc in WOLFPSORT_LOCS:
        safe = loc.replace(".", "_").replace("/", "_")
        features[f"wolfpsort_frac_{safe}"] = float((df["top_loc"] == loc).sum()) / n
        if loc in df.columns:
            features[f"wolfpsort_mean_score_{safe}"] = float(df[loc].mean())
    return pd.Series(features, dtype=np.float32)


def build_wolfpsort_matrix(annotation_paths: dict[str, Path]) -> pd.DataFrame:
    """Build genome × WolfPSORT feature matrix."""
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="WolfPSORT features"):
        try:
            df = parse_wolfpsort(path)
            rows[genome_id] = wolfpsort_to_features(df)
        except Exception as e:
            logger.warning(f"Failed WolfPSORT parse for {genome_id}: {e}")
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"WolfPSORT matrix: {matrix.shape}")
    return matrix


# ── TargetP ───────────────────────────────────────────────────────────────────


def parse_targetp(path: Path) -> pd.DataFrame:
    """
    Parse TargetP-2 summary output.

    Format (tab-separated, # comment header):
        ID  Prediction  noTP  SP  mTP  CS Position

    Returns DataFrame with columns: protein_id, prediction, notp_prob,
    sp_prob, mtp_prob.
    """
    records = []
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                records.append(
                    {
                        "protein_id": parts[0].strip(),
                        "prediction": parts[1].strip(),
                        "notp_prob": float(parts[2]),
                        "sp_prob": float(parts[3]),
                        "mtp_prob": float(parts[4]),
                    }
                )
            except (IndexError, ValueError):
                continue
    return pd.DataFrame(records)


def targetp_to_features(df: pd.DataFrame) -> pd.Series:
    """Aggregate TargetP predictions to genome-level features."""
    if df.empty:
        return pd.Series(dtype=np.float32)
    n = len(df)
    return pd.Series(
        {
            "targetp_frac_noTP": float((df["prediction"] == "noTP").sum()) / n,
            "targetp_frac_SP": float((df["prediction"] == "SP").sum()) / n,
            "targetp_frac_mTP": float((df["prediction"] == "mTP").sum()) / n,
            "targetp_mean_sp_prob": float(df["sp_prob"].mean()),
            "targetp_mean_mtp_prob": float(df["mtp_prob"].mean()),
        },
        dtype=np.float32,
    )


def build_targetp_matrix(annotation_paths: dict[str, Path]) -> pd.DataFrame:
    """Build genome × TargetP feature matrix."""
    rows: dict[str, pd.Series] = {}
    for genome_id, path in tqdm(annotation_paths.items(), desc="TargetP features"):
        try:
            df = parse_targetp(path)
            rows[genome_id] = targetp_to_features(df)
        except Exception as e:
            logger.warning(f"Failed TargetP parse for {genome_id}: {e}")
    matrix = pd.DataFrame(rows).T.fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"TargetP matrix: {matrix.shape}")
    return matrix


# ── Combined subcellular matrix ───────────────────────────────────────────────


def build_subcellular_matrix(
    tmhmm_paths: dict[str, Path] | None = None,
    signalp_paths: dict[str, Path] | None = None,
    wolfpsort_paths: dict[str, Path] | None = None,
    targetp_paths: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """
    Concatenate all available subcellular feature blocks into one matrix.

    Any genome missing from a sub-block gets zeros for that block.
    """
    blocks = []
    if tmhmm_paths:
        blocks.append(build_tmhmm_matrix(tmhmm_paths))
    if signalp_paths:
        blocks.append(build_signalp_matrix(signalp_paths))
    if wolfpsort_paths:
        blocks.append(build_wolfpsort_matrix(wolfpsort_paths))
    if targetp_paths:
        blocks.append(build_targetp_matrix(targetp_paths))

    if not blocks:
        return pd.DataFrame()

    matrix = pd.concat(blocks, axis=1).fillna(0.0).astype(np.float32)
    matrix.index.name = "genome_id"
    logger.info(f"Combined subcellular matrix: {matrix.shape}")
    return matrix
