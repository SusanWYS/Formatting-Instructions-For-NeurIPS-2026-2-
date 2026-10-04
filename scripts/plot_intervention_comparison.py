#!/usr/bin/env python3
"""Plot learned-CAA and norm-matched-random intervention effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    "gemma2_2b": "Gemma 2\n2B",
    "gemma2_9b": "Gemma 2\n9B",
    "qwen3_4b": "Qwen3\n4B",
    "qwen3_8b": "Qwen3\n8B",
    "llama32_3b": "Llama 3.2\n3B",
    "llama3_8b": "Llama 3\n8B",
    "llama31_70b": "Llama 3.1\n70B",
}
RESULT_DIRS = {
    "control": "caa_one_sided_alpha_control_ood_2026-10-04",
    "critical": "caa_one_sided_alpha_critical_ood_2026-10-04",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/eliciture_interp"),
    )
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/Formatting-Instructions-For-NeurIPS-2026-2-"),
    )
    return parser.parse_args()


def load_learned(analysis_repo: Path) -> pd.DataFrame:
    frames = []
    for condition, result_dir in RESULT_DIRS.items():
        for model in MODELS:
            path = analysis_repo / "results" / result_dir / "models" / model / "summary.csv"
            frame = pd.read_csv(path)
            if len(frame) != 1:
                raise ValueError(f"Expected one summary row in {path}")
            if frame.iloc[0].evaluation_condition != condition:
                raise ValueError(f"Unexpected condition in {path}")
            frames.append(frame)
    learned = pd.concat(frames, ignore_index=True).rename(
        columns={
            "evaluation_condition": "condition",
            "selected_alpha": "learned_alpha",
            "mean_delta_surprisal_mean": "learned_mean",
            "mean_delta_surprisal_mean_ci_low": "learned_ci_low",
            "mean_delta_surprisal_mean_ci_high": "learned_ci_high",
        }
    )
    return learned[
        [
            "model",
            "condition",
            "learned_alpha",
            "learned_mean",
            "learned_ci_low",
            "learned_ci_high",
            "n_ood_items",
        ]
    ]


def load_comparison(args: argparse.Namespace) -> pd.DataFrame:
    learned = load_learned(args.analysis_repo)
    random = pd.read_csv(args.paper_repo / "derived/random_vector_intervention_summary.csv")
    comparison = learned.merge(
        random,
        on=["model", "condition", "n_ood_items"],
        how="left",
        validate="one_to_one",
    )
    if comparison.random_vector_mean_delta_surprisal.isna().any():
        raise ValueError("Missing random-vector intervention results")
    comparison["model"] = pd.Categorical(comparison.model, MODELS, ordered=True)
    return comparison.sort_values(["condition", "model"]).reset_index(drop=True)


def alpha_text(value: float) -> str:
    if np.isclose(value, round(value)):
        number = f"{value:.0f}"
    elif np.isclose(value * 2, round(value * 2)):
        number = f"{value:.1f}"
    else:
        number = f"{value:.2f}"
    return number


def plot_condition(frame: pd.DataFrame, condition: str, output_dir: Path) -> None:
    local = frame[frame.condition.eq(condition)].set_index("model").loc[list(MODELS)]
    x = np.arange(len(MODELS), dtype=float)
    offset = 0.13
    learned_x = x - offset
    random_x = x + offset

    learned_mean = local.learned_mean.to_numpy(dtype=float)
    learned_low = local.learned_ci_low.to_numpy(dtype=float)
    learned_high = local.learned_ci_high.to_numpy(dtype=float)
    learned_error = np.vstack((learned_mean - learned_low, learned_high - learned_mean))
    random_mean = local.random_vector_mean_delta_surprisal.to_numpy(dtype=float)
    random_sd = local.random_vector_sd_delta_surprisal.to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(3.25, 2.25))
    axis.errorbar(
        learned_x,
        learned_mean,
        yerr=learned_error,
        fmt="o",
        markersize=4.5,
        markerfacecolor="#3B6FB6",
        markeredgecolor="black",
        markeredgewidth=0.45,
        ecolor="#3B6FB6",
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        linestyle="none",
        label="Derived causal vector",
        zorder=4,
    )
    axis.errorbar(
        random_x,
        random_mean,
        yerr=random_sd,
        fmt="D",
        markersize=4.0,
        markerfacecolor="#D97A32",
        markeredgecolor="black",
        markeredgewidth=0.45,
        ecolor="#D97A32",
        elinewidth=0.8,
        capsize=1.8,
        capthick=0.8,
        linestyle="none",
        label="Random vectors",
        zorder=4,
    )

    lower = min(float(learned_low.min()), float((random_mean - random_sd).min()), 0.0)
    upper = max(float(learned_high.max()), float((random_mean + random_sd).max()), 0.0)
    data_range = max(upper - lower, 0.1)
    lower_space = (0.18 if condition == "critical" else 0.08) * data_range
    upper_space = (0.08 if condition == "critical" else 0.18) * data_range
    axis.set_ylim(lower - lower_space, upper + upper_space)

    learned_alpha = local.learned_alpha.to_numpy(dtype=float)
    random_alpha = local.random_vector_mean_alpha.to_numpy(dtype=float)
    learned_anchor = learned_high if condition == "control" else learned_low
    random_anchor = random_mean + random_sd if condition == "control" else random_mean - random_sd
    annotation_offset = 0.035 * data_range if condition == "control" else -0.035 * data_range
    vertical_alignment = "bottom" if condition == "control" else "top"
    for index in range(len(MODELS)):
        axis.text(
            learned_x[index],
            learned_anchor[index] + annotation_offset,
            alpha_text(learned_alpha[index]),
            ha="right",
            va=vertical_alignment,
            color="#2B568E",
            fontsize=6.3,
        )
        axis.text(
            random_x[index],
            random_anchor[index] + annotation_offset,
            alpha_text(random_alpha[index]),
            ha="left",
            va=vertical_alignment,
            color="#A95820",
            fontsize=6.3,
        )

    axis.axhline(0.0, color="#777777", linewidth=0.7, zorder=1)
    axis.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)
    axis.set_xticks(
        x,
        [LABELS[model].replace("\n", " ") for model in MODELS],
        fontsize=7.0,
        rotation=38,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_xlabel("Model")
    axis.set_ylabel(r"$\Delta$ surprisal")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.17, right=0.99, bottom=0.30, top=0.82)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / f"intervention_{condition}_learned_random"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    comparison = load_comparison(args)
    comparison.to_csv(args.paper_repo / "derived/intervention_learned_random_comparison.csv", index=False)
    for condition in ("control", "critical"):
        plot_condition(comparison, condition, args.paper_repo / "figures")


if __name__ == "__main__":
    main()
