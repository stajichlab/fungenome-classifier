#!/usr/bin/env python3
"""
scripts/05_compare_targets.py

Compare classifier performance across multiple targets (taxonomy_order,
ecological_niche, lifestyle) and across feature blocks.

Produces:
  - Heatmap: block × target F1 scores
  - Bar chart: fusion model vs best single block per target
  - CSV: full comparison table
  - Phylogenetic signal vs accuracy scatter

Usage:
    python scripts/05_compare_targets.py \\
        --results-dir results/ \\
        --output-dir results/comparison
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Compare classifier performance across targets")
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    p.add_argument("--metric", default="f1_macro", help="Metric to compare")
    return p.parse_args()


def load_training_summary(results_dir: Path, target: str) -> dict | None:
    summary_path = results_dir / target / "training_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as fh:
        return json.load(fh)


def load_block_comparison(results_dir: Path, target: str) -> pd.DataFrame | None:
    path = results_dir / target / "block_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["target"] = target
    return df


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-discover trained targets
    targets = [
        d.name
        for d in args.results_dir.iterdir()
        if d.is_dir() and (d / "training_summary.json").exists()
    ]
    if not targets:
        logger.error(f"No training summaries found in {args.results_dir}")
        sys.exit(1)

    logger.info(f"Found results for targets: {targets}")

    # ── Load block comparison tables ──────────────────────────────────────────
    block_dfs = []
    summaries = {}
    for target in targets:
        df = load_block_comparison(args.results_dir, target)
        if df is not None:
            block_dfs.append(df)
        summary = load_training_summary(args.results_dir, target)
        if summary:
            summaries[target] = summary

    if not block_dfs:
        logger.error("No block_comparison.csv files found. Run 02_train.py first.")
        sys.exit(1)

    all_blocks = pd.concat(block_dfs, ignore_index=True)

    # ── Build block × target pivot ────────────────────────────────────────────
    mean_col = f"mean_{args.metric}"
    if mean_col not in all_blocks.columns:
        # Try to infer
        mean_col = [c for c in all_blocks.columns if "mean" in c and "f1" in c]
        if mean_col:
            mean_col = mean_col[0]
        else:
            logger.error(f"Column {mean_col} not found in block comparison table")
            sys.exit(1)

    pivot = all_blocks.pivot_table(index="block", columns="target", values=mean_col)
    pivot = pivot.sort_values(by=targets[0], ascending=False)

    # Save table
    pivot.round(3).to_csv(args.output_dir / "block_by_target_matrix.csv")
    logger.info(f"\nBlock × Target F1 matrix:\n{pivot.round(3).to_string()}")

    # ── Plot 1: Heatmap block × target ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, len(targets) * 1.8), max(5, len(pivot) * 0.7)))
    sns.heatmap(
        pivot,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": args.metric.replace("_", " ").title()},
    )
    ax.set_title(f"Feature Block Performance — {args.metric.replace('_', ' ').title()}")
    ax.set_xlabel("Target")
    ax.set_ylabel("Feature Block")
    plt.tight_layout()
    fig.savefig(args.output_dir / "block_by_target_heatmap.svg", dpi=150, bbox_inches="tight")
    logger.info("Saved block_by_target_heatmap.svg")
    plt.close()

    # ── Plot 2: Fusion vs best-block bar chart ────────────────────────────────
    # Extract fusion model scores from training summaries
    fusion_scores = {}
    best_block_scores = {}

    for target in targets:
        summary = summaries.get(target, {})
        block_cv = summary.get("block_cv_summary", {})
        best = 0.0
        for block_name, cv_data in block_cv.items():
            # cv_data is a dict of metric -> {mean, std, ...}
            f1_data = cv_data.get(args.metric, {})
            if isinstance(f1_data, dict):
                val = f1_data.get("mean", 0.0)
            else:
                val = float(f1_data) if f1_data else 0.0
            best = max(best, val)
        best_block_scores[target] = best

        # Fusion score from block comparison (last row often "fusion" or from evaluation)
        fusion_row = all_blocks[
            (all_blocks["target"] == target) & (all_blocks["block"] == "fusion")
        ]
        if not fusion_row.empty:
            fusion_scores[target] = float(fusion_row[mean_col].iloc[0])

    if fusion_scores:
        compare_df = pd.DataFrame(
            {
                "Best single block": best_block_scores,
                "Stacking fusion": fusion_scores,
            }
        ).T

        fig, ax = plt.subplots(figsize=(max(7, len(targets) * 1.5), 5))
        x = np.arange(len(targets))
        width = 0.35
        bars1 = ax.bar(
            x - width / 2,
            compare_df.loc["Best single block", targets],
            width,
            label="Best single block",
            color="#aec7e8",
            edgecolor="white",
        )
        bars2 = ax.bar(
            x + width / 2,
            compare_df.loc["Stacking fusion", targets],
            width,
            label="Stacking fusion",
            color="#1a6fa8",
            edgecolor="white",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=10)
        ax.set_ylabel(args.metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Fusion vs Best Single Block")
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with values
        for bar in bars1:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        for bar in bars2:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        fig.savefig(args.output_dir / "fusion_vs_best_block.svg", dpi=150, bbox_inches="tight")
        logger.info("Saved fusion_vs_best_block.svg")
        plt.close()

    # ── Plot 3: Block rank consistency across targets ─────────────────────────
    # Which block is most consistently informative?
    rank_pivot = pivot.rank(ascending=False)
    mean_rank = rank_pivot.mean(axis=1).sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(mean_rank)))
    mean_rank.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_xlabel("Mean rank across targets (lower = more consistently informative)")
    ax.set_title("Feature Block Consistency Across Targets")
    ax.axvline(mean_rank.mean(), color="grey", linestyle="--", alpha=0.7, label="Mean")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(args.output_dir / "block_rank_consistency.svg", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 4: Per-block performance profile ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(7, len(pivot.columns) * 1.5), 5))
    palette = sns.color_palette("tab10", n_colors=len(pivot.index))
    for (block, row), color in zip(pivot.iterrows(), palette):
        ax.plot(
            pivot.columns,
            row.values,
            marker="o",
            label=block,
            color=color,
            linewidth=2,
            markersize=7,
        )

    ax.set_ylabel(args.metric.replace("_", " ").title())
    ax.set_xlabel("Target")
    ax.set_title("Per-Block Performance Profile Across Targets")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(args.output_dir / "block_profile_across_targets.svg", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary report ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Multi-Target Comparison Summary")
    print(f"{'=' * 60}")
    print(f"\n  Metric: {args.metric}")
    print("\n  Block × Target F1 matrix:\n")
    print(pivot.round(3).to_string(float_format="{:.3f}".format))
    print("\n  Most consistently informative blocks:")
    for block, rank in mean_rank.items():
        print(f"    {block:20s}  mean rank = {rank:.1f}")
    print(f"\n  Outputs saved to: {args.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
