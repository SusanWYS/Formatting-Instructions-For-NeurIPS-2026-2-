#!/usr/bin/env python
"""Plot Gemini 3.6 Flash causal-rating distributions for the appendix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITIONS = ("control", "critical")
LABELS = ("Control", "Critical")
COLORS = ("#D17A22", "#2F6DAE")
SPLITS = (
    ("cross_sentence", "Cross-sentence", 480),
    ("within_sentence_ood", "Within-sentence", 150),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/gemini36_likert_histograms.pdf"),
    )
    parser.add_argument(
        "--derived",
        type=Path,
        default=Path("derived/gemini36_likert_rating_histogram.csv"),
    )
    return parser.parse_args()


def validate(frame: pd.DataFrame) -> None:
    required = {"id", "pair_id", "source_dataset", "condition", "rating"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Ratings file is missing columns: {sorted(missing)}")
    if len(frame) != 1260 or frame.id.nunique() != 1260:
        raise ValueError("Expected 1,260 unique rating rows")
    if set(frame.source_dataset) != {name for name, _, _ in SPLITS}:
        raise ValueError(f"Unexpected source datasets: {sorted(frame.source_dataset.unique())}")
    if set(frame.condition) != set(CONDITIONS):
        raise ValueError(f"Unexpected conditions: {sorted(frame.condition.unique())}")
    if not frame.rating.astype(int).between(1, 7).all():
        raise ValueError("Ratings must be integers from 1 through 7")
    for source, _, expected_pairs in SPLITS:
        local = frame[frame.source_dataset.eq(source)]
        if local.pair_id.nunique() != expected_pairs:
            raise ValueError(
                f"{source} has {local.pair_id.nunique()} pairs; expected {expected_pairs}"
            )
        counts = local.groupby(["pair_id", "condition"]).size().unstack(fill_value=0)
        if set(counts.columns) != set(CONDITIONS) or not (counts[list(CONDITIONS)] == 1).all().all():
            raise ValueError(f"{source} does not contain complete matched pairs")


def distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, title, expected_pairs in SPLITS:
        local = frame[frame.source_dataset.eq(source)]
        for condition in CONDITIONS:
            ratings = local.loc[local.condition.eq(condition), "rating"].astype(int)
            counts = ratings.value_counts().reindex(range(1, 8), fill_value=0)
            for rating, count in counts.items():
                rows.append(
                    {
                        "source_dataset": source,
                        "construction": title,
                        "condition": condition,
                        "rating": int(rating),
                        "count": int(count),
                        "total": int(len(ratings)),
                        "percentage": 100.0 * float(count) / len(ratings),
                        "pairs": expected_pairs,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.ratings, sep="\t", dtype={"id": str, "pair_id": str})
    validate(frame)
    derived = distribution_table(frame)
    args.derived.parent.mkdir(parents=True, exist_ok=True)
    derived.to_csv(args.derived, index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.25), sharex=True, sharey=True)
    x = np.arange(1, 8, dtype=float)
    width = 0.36
    offsets = (-width / 2, width / 2)
    for axis, (source, title, expected_pairs) in zip(axes, SPLITS, strict=True):
        for condition, label, color, offset in zip(
            CONDITIONS, LABELS, COLORS, offsets, strict=True
        ):
            values = (
                derived[
                    derived.source_dataset.eq(source) & derived.condition.eq(condition)
                ]
                .sort_values("rating")
                .percentage.to_numpy(dtype=float)
            )
            axis.bar(
                x + offset,
                values,
                width=width,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                label=label,
            )
        axis.set_title(f"{title} ({expected_pairs} pairs)", pad=3.0)
        axis.set_xticks(x)
        axis.set_xlim(0.48, 7.52)
        axis.set_ylim(0, 102)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Items (%)")
    figure.supxlabel("Gemini 3.6 Flash rating", y=0.055, fontsize=8.0)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.075, right=0.992, bottom=0.24, top=0.81, wspace=0.17)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


if __name__ == "__main__":
    main()
