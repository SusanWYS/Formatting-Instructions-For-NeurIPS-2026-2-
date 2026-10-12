#!/usr/bin/env python3
"""Plot the post hoc Llama 3.1 70B layer-36 intervention diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path("/data/cl/u/susanw26/eliciture_interp")
CAUSAL_ROOT = (
    ANALYSIS_ROOT / "results/llama31_70b_layer36_intervention_2026-10-11"
)
RANDOM_ROOT = (
    ANALYSIS_ROOT
    / "results/deeper_layer_random_intervention_2026-10-11/models/llama31_70b"
)
PAPER_ROOT = Path("/data/cl/u/susanw26/Formatting-Instructions-For-NeurIPS-2026-2-")
ALPHAS = np.arange(1, 6, dtype=float)
CAUSAL_COLOR = "#315B7D"
RANDOM_COLOR = "#C25B3C"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_ROOT / "figures/llama70b_layer36_intervention.pdf",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PAPER_ROOT / "derived/llama70b_layer36_intervention.csv",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=202610110)
    return parser.parse_args()


def crossed_bootstrap(
    matrix: np.ndarray, rng: np.random.Generator, repetitions: int
) -> tuple[float, float]:
    n_vectors, n_pairs = matrix.shape
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        vector_indices = rng.integers(0, n_vectors, size=n_vectors)
        pair_indices = rng.integers(0, n_pairs, size=n_pairs)
        estimates[repetition] = matrix[
            np.ix_(vector_indices, pair_indices)
        ].mean()
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def load_results(repetitions: int, seed: int) -> pd.DataFrame:
    rows = []
    reference_pairs: list[str] | None = None
    for alpha_index, alpha in enumerate(ALPHAS):
        tag = f"alpha_{alpha:g}"
        causal = json.loads(
            (CAUSAL_ROOT / tag / "summary.json").read_text(encoding="utf-8")
        )
        if causal["n_pairs"] != 150 or causal["layer"] != 36:
            raise ValueError(f"Unexpected causal result metadata for {tag}")

        random_values = []
        for vector_index in range(10):
            path = (
                RANDOM_ROOT
                / tag
                / "checkpoints"
                / f"random_{vector_index:02d}"
                / "heldout_pair_scores.csv"
            )
            frame = pd.read_csv(path, dtype={"pair_id": str}).sort_values("pair_id")
            pair_ids = frame.pair_id.tolist()
            if len(frame) != 150 or len(set(pair_ids)) != 150:
                raise ValueError(f"Incomplete random result: {path}")
            if reference_pairs is None:
                reference_pairs = pair_ids
            elif pair_ids != reference_pairs:
                raise ValueError(f"Held-out pairs changed: {path}")
            random_values.append(frame.delta_correct.to_numpy(dtype=float))

        random_matrix = np.stack(random_values)
        rng = np.random.default_rng(seed + alpha_index)
        random_low, random_high = crossed_bootstrap(
            random_matrix, rng, repetitions
        )
        rows.append(
            {
                "alpha": alpha,
                "causal_delta_accuracy": causal["delta_accuracy"],
                "causal_ci_low": causal["delta_ci_low"],
                "causal_ci_high": causal["delta_ci_high"],
                "random_delta_accuracy_mean": float(random_matrix.mean()),
                "random_ci_low": random_low,
                "random_ci_high": random_high,
                "n_pairs": 150,
                "n_random_vectors": 10,
            }
        )
    return pd.DataFrame(rows)


def errors(
    frame: pd.DataFrame, mean: str, low: str, high: str
) -> tuple[np.ndarray, np.ndarray]:
    center = 100 * frame[mean].to_numpy(dtype=float)
    lower = 100 * frame[low].to_numpy(dtype=float)
    upper = 100 * frame[high].to_numpy(dtype=float)
    return center, np.vstack((center - lower, upper - center))


def main() -> None:
    args = parse_args()
    frame = load_results(args.bootstrap_repetitions, args.bootstrap_seed)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.summary_output, index=False)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 7.2,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(3.45, 2.35))
    causal, causal_error = errors(
        frame, "causal_delta_accuracy", "causal_ci_low", "causal_ci_high"
    )
    random, random_error = errors(
        frame,
        "random_delta_accuracy_mean",
        "random_ci_low",
        "random_ci_high",
    )
    axis.axhline(0, color="#777777", linewidth=0.65, alpha=0.75, zorder=0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, alpha=0.65, zorder=0)
    axis.errorbar(
        ALPHAS,
        random,
        yerr=random_error,
        color=RANDOM_COLOR,
        linestyle=(0, (3.2, 2.2)),
        marker="s",
        markersize=3.0,
        markerfacecolor="white",
        markeredgewidth=0.8,
        linewidth=1.15,
        elinewidth=0.75,
        capsize=1.8,
        capthick=0.75,
        zorder=2,
    )
    axis.errorbar(
        ALPHAS,
        causal,
        yerr=causal_error,
        color=CAUSAL_COLOR,
        linestyle="-",
        marker="o",
        markersize=3.2,
        markerfacecolor=CAUSAL_COLOR,
        markeredgewidth=0,
        linewidth=1.35,
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        zorder=3,
    )
    axis.set_title("Llama 3.1 70B, layer 36", fontweight="normal", pad=3.0)
    axis.set_xlabel(r"Intervention magnitude $|\alpha|$")
    axis.set_ylabel(r"$\Delta$ paired accuracy (%)")
    axis.set_xlim(0.75, 5.25)
    axis.set_ylim(-90, 12)
    axis.set_xticks(ALPHAS, ("1", "2", "3", "4", "5"))
    axis.set_yticks((-80, -60, -40, -20, 0))
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    handles = (
        Line2D(
            [0], [0], color=CAUSAL_COLOR, marker="o", markersize=3.2,
            linewidth=1.35, label="Causal vector"
        ),
        Line2D(
            [0], [0], color=RANDOM_COLOR, linestyle=(0, (3.2, 2.2)),
            marker="s", markerfacecolor="white", markeredgewidth=0.8,
            markersize=3.1, linewidth=1.15, label="Random vectors"
        ),
    )
    axis.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        handlelength=2.6,
    )
    figure.subplots_adjust(left=0.19, right=0.985, bottom=0.20, top=0.88)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


if __name__ == "__main__":
    main()
