#!/usr/bin/env python3
"""Plot the seven-model fixed-alpha intervention sweep for the ACL/ARR paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


MODELS = (
    "gemma2_2b",
    "gemma2_9b",
    "qwen3_4b",
    "qwen3_8b",
    "llama32_3b",
    "llama3_8b",
    "llama31_70b",
)
LABELS = {
    "gemma2_2b": "Gemma 2 2B",
    "gemma2_9b": "Gemma 2 9B",
    "qwen3_4b": "Qwen3 4B",
    "qwen3_8b": "Qwen3 8B",
    "llama32_3b": "Llama 3.2 3B",
    "llama3_8b": "Llama 3 8B",
    "llama31_70b": "Llama 3.1 70B",
}
ALL_ALPHAS = np.array((0.5, 1.0, 2.0, 3.0, 4.0, 5.0))
PLOT_ALPHAS = np.array((1.0, 2.0, 3.0, 4.0, 5.0))
CAUSAL_COLOR = "#315B7D"
RANDOM_COLOR = "#C25B3C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/data/cl/u/susanw26/eliciture_interp/results/"
            "caa_fixed_alpha_sweep_2026-10-07/fixed_alpha_sweep_summary.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/cl/u/susanw26/Formatting-Instructions-For-NeurIPS-2026-2-/"
            "figures/intervention_alpha_sweep.pdf"
        ),
    )
    return parser.parse_args()


def validate(frame: pd.DataFrame) -> None:
    expected = {(model, alpha) for model in MODELS for alpha in ALL_ALPHAS}
    observed = set(zip(frame.model, frame.alpha.astype(float), strict=True))
    if observed != expected:
        raise ValueError(
            f"Unexpected model/alpha cells: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    if not frame.n_heldout_pairs.eq(150).all():
        raise ValueError("Every estimate must use all 150 held-out item pairs")
    if not frame.n_random_vectors.eq(10).all():
        raise ValueError("Every random estimate must use ten directions")


def asymmetric_errors(
    frame: pd.DataFrame, mean: str, low: str, high: str
) -> tuple[np.ndarray, np.ndarray]:
    center = 100 * frame[mean].to_numpy(dtype=float)
    lower = 100 * frame[low].to_numpy(dtype=float)
    upper = 100 * frame[high].to_numpy(dtype=float)
    return center, np.vstack((center - lower, upper - center))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.summary)
    validate(frame)

    plt.rcParams.update(
        {
            # Match the sans-serif typography used in the paper's other plots.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # ACL/ARR's full text block is approximately 7.1 inches wide. A 2x8 grid
    # gives equal-sized panels: four Gemma/Qwen panels above three centered
    # Llama panels.
    figure = plt.figure(figsize=(7.1, 3.45))
    grid = figure.add_gridspec(
        2,
        8,
        left=0.075,
        right=0.993,
        bottom=0.145,
        top=0.875,
        wspace=1.05,
        hspace=0.46,
    )
    positions = (
        grid[0, 0:2],
        grid[0, 2:4],
        grid[0, 4:6],
        grid[0, 6:8],
        grid[1, 1:3],
        grid[1, 3:5],
        grid[1, 5:7],
    )
    axes = []
    shared = None
    for position in positions:
        axis = figure.add_subplot(position, sharex=shared, sharey=shared)
        axes.append(axis)
        if shared is None:
            shared = axis

    for index, (axis, model) in enumerate(zip(axes, MODELS, strict=True)):
        subset = frame[
            frame.model.eq(model) & frame.alpha.isin(PLOT_ALPHAS)
        ].sort_values("alpha")
        causal, causal_error = asymmetric_errors(
            subset, "causal_delta_accuracy", "causal_ci_low", "causal_ci_high"
        )
        random, random_error = asymmetric_errors(
            subset,
            "random_delta_accuracy_mean",
            "random_ci_low",
            "random_ci_high",
        )
        axis.axhline(0, color="#777777", linewidth=0.65, alpha=0.75, zorder=0)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65, zorder=0)
        axis.errorbar(
            PLOT_ALPHAS,
            random,
            yerr=random_error,
            color=RANDOM_COLOR,
            linestyle=(0, (3.2, 2.2)),
            marker="s",
            markersize=2.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.15,
            elinewidth=0.75,
            capsize=1.8,
            capthick=0.75,
            zorder=2,
        )
        axis.errorbar(
            PLOT_ALPHAS,
            causal,
            yerr=causal_error,
            color=CAUSAL_COLOR,
            linestyle="-",
            marker="o",
            markersize=3.0,
            markerfacecolor=CAUSAL_COLOR,
            markeredgewidth=0,
            linewidth=1.35,
            elinewidth=0.8,
            capsize=1.8,
            capthick=0.8,
            zorder=3,
        )
        axis.set_title(LABELS[model], fontweight="normal", pad=2.8)
        axis.set_xlim(0.75, 5.25)
        axis.set_ylim(-82, 18)
        axis.set_xticks(PLOT_ALPHAS, ("1", "2", "3", "4", "5"))
        axis.set_yticks((-80, -60, -40, -20, 0))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if index not in (0, 4):
            axis.tick_params(axis="y", labelleft=False)

    figure.supxlabel(r"Intervention magnitude $|\alpha|$", y=0.025)
    figure.supylabel(r"$\Delta$ paired accuracy (%)", x=0.012)
    handles = (
        Line2D(
            [0],
            [0],
            color=CAUSAL_COLOR,
            marker="o",
            markersize=3.2,
            linewidth=1.35,
            label="Causal vectors",
        ),
        Line2D(
            [0],
            [0],
            color=RANDOM_COLOR,
            linestyle=(0, (3.2, 2.2)),
            marker="s",
            markerfacecolor="white",
            markeredgewidth=0.8,
            markersize=3.1,
            linewidth=1.15,
            label="Norm-matched random vectors",
        ),
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.535, 0.992),
        frameon=False,
        ncol=2,
        columnspacing=1.8,
        handlelength=2.8,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


if __name__ == "__main__":
    main()
