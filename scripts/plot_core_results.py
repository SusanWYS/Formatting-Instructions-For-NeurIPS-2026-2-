#!/usr/bin/env python3
"""Regenerate the seven-model core paper figures from primary result files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/eliciture_interp"),
    )
    parser.add_argument(
        "--pipeline-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/eliciture_interp_paper"),
    )
    parser.add_argument(
        "--paper-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/Formatting-Instructions-For-NeurIPS-2026-2-"),
    )
    return parser.parse_args()


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def wilson_interval(accuracy: float, n: int, z: float = 1.96) -> tuple[float, float]:
    successes = int(round(accuracy * n))
    proportion = successes / n
    denominator = 1.0 + z**2 / n
    center = (proportion + z**2 / (2 * n)) / denominator
    radius = z * np.sqrt(
        proportion * (1.0 - proportion) / n + z**2 / (4 * n**2)
    ) / denominator
    return center - radius, center + radius


def load_rq1(args: argparse.Namespace) -> pd.DataFrame:
    probability = pd.concat(
        [
            pd.read_csv(
                args.analysis_repo
                / "results/probabilistic_behavior_eval_v2/cross_model_metrics.csv"
            ),
            pd.read_csv(
                args.analysis_repo
                / "results/extended_paper_models_2026-10-02/rq1_probability/cross_model_metrics.csv"
            ),
        ],
        ignore_index=True,
    )
    probability = probability[probability.model.isin(MODELS)]
    explicit = (
        probability[
            probability.measure.isin(("yesno_probability", "likert_expected"))
        ]
        .pivot(index="model", columns="measure", values="pairwise_accuracy")
        .rename(
            columns={
                "yesno_probability": "Yes/No (Explicit)",
                "likert_expected": "7-Point Likert (Explicit)",
            }
        )
    )

    idk = pd.read_csv(
        args.analysis_repo
        / "results/rq1_idk_exact_2026-10-07/cross_model_summary.csv"
    )
    idk = idk[
        idk.model.isin(MODELS)
        & idk.scope.eq("all")
        & idk.construction.eq("all")
        & idk.measure.eq("ignorance_surprisal_sum")
    ].set_index("model")
    prior = explicit.join(
        idk[["pairwise_accuracy"]].rename(
            columns={"pairwise_accuracy": "IDK Why (Surprisal)"}
        ),
        how="outer",
        validate="one_to_one",
    )

    revised = pd.read_csv(
        args.analysis_repo
        / "results/behavioral_connective_sentence_only_period_2026-10-04"
        / "cross_model_summary.csv"
    )
    revised = revised[
        revised.scope.eq("all")
        & revised.construction.eq("all")
        & revised.measure.eq("connective_lp_mean_diff")
    ].set_index("model")
    prior["Because-And (Surprisal)"] = revised["pairwise_accuracy"]

    columns = (
        "Yes/No (Explicit)",
        "7-Point Likert (Explicit)",
        "IDK Why (Surprisal)",
        "Because-And (Surprisal)",
    )
    missing = set(MODELS).difference(prior.index)
    if missing:
        raise ValueError(f"Missing RQ1 results for models: {sorted(missing)}")
    return prior.loc[list(MODELS), list(columns)].reset_index()


def plot_rq1(table: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.6, 1.05))
    x = np.arange(len(table))
    width = 0.18
    columns = (
        "Yes/No (Explicit)",
        "7-Point Likert (Explicit)",
        "IDK Why (Surprisal)",
        "Because-And (Surprisal)",
    )
    colors = ("#3B6FB6", "#D97A32", "#4F9D69", "#8A63A8")
    for offset, (column, color) in enumerate(zip(columns, colors, strict=True)):
        means = table[column].to_numpy(dtype=float)
        intervals = np.asarray([wilson_interval(value, 150) for value in means])
        errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        axis.bar(
            x + (offset - 1.5) * width,
            means,
            width,
            label=column,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            yerr=errors,
            capsize=1.8,
            error_kw={"linewidth": 0.7},
        )
    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8, label="Chance")
    axis.set_ylim(0.3, 1.03)
    axis.set_ylabel("Paired accuracy")
    axis.set_xticks(x, [LABELS[model] for model in table.model])
    axis.legend(
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.27),
        frameon=False,
        fontsize=7.3,
        columnspacing=0.9,
        handlelength=1.4,
        handletextpad=0.4,
    )
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.set_axisbelow(True)
    save_figure(figure, output / "rq1_behavior")


def result_dir(args: argparse.Namespace, model: str) -> Path:
    return (
        args.analysis_repo
        / "results/occasion_standardized_2026-10-07/fixed_caa_evaluation/models"
        / model
    )


def load_transfer(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        frame = pd.read_csv(result_dir(args, model) / "cross_transfer_metrics.csv")
        if set(frame.model.astype(str)) != {model}:
            raise ValueError(f"Model provenance mismatch for {model}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_heatmaps(transfer: pd.DataFrame, output: Path, derived: Path) -> None:
    comparisons = (
        (("causal", "occasion"), ("Caus.", "Occas."), "Occasion–Causal", "rq2a_causal_occasion_cells.csv"),
        (("causal", "plausibility"), ("Caus.", "Plaus."), "Plausibility–Causal", "rq2a_causal_plausibility_cells.csv"),
    )
    figure, axes = plt.subplots(2, len(MODELS), figsize=(10.7, 3.38))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    image = None
    for row_index, (domains, labels, title, filename) in enumerate(comparisons):
        selected = transfer[
            transfer.source.isin(domains) & transfer.target.isin(domains)
        ].copy()
        selected.to_csv(derived / filename, index=False)
        for model_index, model in enumerate(MODELS):
            axis = axes[row_index, model_index]
            local = selected[selected.model.eq(model)]
            matrix = np.asarray(
                [
                    [
                        float(
                            local[
                                local.source.eq(source) & local.target.eq(target)
                            ].pairwise_accuracy.iloc[0]
                        )
                        for target in domains
                    ]
                    for source in domains
                ]
            )
            image = axis.imshow(matrix, cmap="RdYlBu", norm=norm, aspect="equal")
            for row in range(2):
                for column in range(2):
                    color = "white" if matrix[row, column] < 0.25 or matrix[row, column] > 0.78 else "black"
                    axis.text(
                        column,
                        row,
                        f"{matrix[row, column]:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=11,
                    )
            if row_index == 0:
                axis.set_title(LABELS[model].replace("\n", " "), fontsize=10, pad=3)
            axis.set_xticks((0, 1), labels, fontsize=10)
            axis.get_xticklabels()[0].set_ha("right")
            axis.get_xticklabels()[1].set_ha("left")
            axis.set_yticks((0, 1))
            if model_index == 0:
                axis.set_yticklabels(labels, fontsize=10)
            else:
                axis.tick_params(axis="y", left=False, labelleft=False)
            axis.tick_params(length=2.5, pad=1)
    figure.supylabel("CAA training set", fontsize=13, x=0.012)
    figure.supxlabel("Held-Out In-Distribution Test Set", fontsize=13, y=0.015)
    figure.subplots_adjust(
        left=0.08, right=0.94, bottom=0.15, top=0.82, wspace=0.08, hspace=0.80
    )
    figure.text(0.51, 0.975, comparisons[0][2], ha="center", va="top", fontsize=13)
    row_gap_center = (axes[0, 0].get_position().y0 + axes[1, 0].get_position().y1) / 2
    figure.text(0.51, row_gap_center - 0.025, comparisons[1][2], ha="center", va="center", fontsize=13)
    if image is not None:
        colorbar_axis = figure.add_axes((0.95, 0.18, 0.012, 0.64))
        colorbar = figure.colorbar(image, cax=colorbar_axis)
        colorbar.set_label("Paired accuracy", fontsize=11, labelpad=5)
        colorbar.ax.tick_params(labelsize=9, length=2.5, pad=2)
    save_figure(figure, output / "rq2a_combined_cross_transfer_heatmaps")


def plot_syntax(args: argparse.Namespace, output: Path, derived: Path) -> None:
    rows = []
    for model in MODELS:
        metrics = pd.read_csv(result_dir(args, model) / "syntax_ood_metrics.csv")
        for source, label in (
            ("causal", "Causal CAA"),
            ("occasion", "Occasion CAA"),
            ("plausibility", "Plausibility CAA"),
        ):
            row = metrics[metrics.source.eq(source)].iloc[0]
            rows.append(
                {
                    "model": model,
                    "caa_condition": label,
                    "pairwise_accuracy": float(row.pairwise_accuracy),
                    "n_pairs": int(row.n_pairs),
                    "selected_layer": int(row.selected_layer),
                    "selected_on": row.selected_on,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(derived / "rq2b_causal_ood_caa_accuracy.csv", index=False)
    figure, axis = plt.subplots(figsize=(7.6, 1.05))
    x = np.arange(len(MODELS))
    width = 0.22
    conditions = ("Causal CAA", "Occasion CAA", "Plausibility CAA")
    colors = ("#C97A6A", "#8E7D9F", "#B5A46A")
    for index, (condition, color) in enumerate(zip(conditions, colors, strict=True)):
        local = table[table.caa_condition.eq(condition)].set_index("model").loc[list(MODELS)]
        means = local.pairwise_accuracy.to_numpy(dtype=float)
        intervals = np.asarray(
            [wilson_interval(value, int(n)) for value, n in zip(means, local.n_pairs, strict=True)]
        )
        errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
        axis.bar(
            x + (index - 1) * width,
            means,
            width,
            label=condition,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            yerr=errors,
            capsize=1.8,
            error_kw={"linewidth": 0.7},
        )
    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=0.8, label="Chance")
    axis.set_ylim(0.3, 1.03)
    axis.set_ylabel("Paired accuracy")
    axis.set_xticks(x, [LABELS[model] for model in MODELS])
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.27), frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.set_axisbelow(True)
    save_figure(figure, output / "rq2b_causal_ood_caa_accuracy")


def plot_projection_distributions(args: argparse.Namespace, output: Path) -> None:
    frames = []
    for model in MODELS[:6]:
        frame = pd.read_csv(
            args.pipeline_repo / "results/rq3_naturalistic/models" / model / "projection_scores.csv"
        )
        frame["model"] = model
        frames.append(frame)
    for model in MODELS[6:]:
        extended = pd.read_csv(
            args.analysis_repo
            / f"results/extended_paper_models_2026-10-02/naturalistic_demographic_projection/models/{model}/projection_scores.csv"
        )
        frames.append(extended[extended.outcome.eq("negative")])
    frame = pd.concat(frames, ignore_index=True)
    calibration = pd.read_csv(
        args.paper_repo / "derived/causal_train_projection_gaps.csv"
    )[["model", "critical_minus_control_train_gap"]]
    frame = frame.merge(calibration, on="model", how="left", validate="many_to_one")
    if frame.critical_minus_control_train_gap.isna().any():
        raise ValueError("Missing causal-training calibration gap")
    frame["calibrated_projection"] = (
        frame.unit_vector_projection / frame.critical_minus_control_train_gap
    )
    conditions = ("white_male", "white_female", "minority_male", "minority_female")
    labels = ("White male", "White female", "Minority male", "Minority female")
    colors = ("#555555", "#D8A03D", "#2A6F97", "#B23A48")
    linestyles = ("-", "--", "-.", ":")
    figure, axes = plt.subplots(2, 4, figsize=(9.4, 4.2))
    for axis, model in zip(axes.flat, MODELS, strict=False):
        local = frame[frame.model.eq(model)]
        values = local.calibrated_projection.to_numpy(dtype=float)
        bins = np.linspace(values.min(), values.max(), 12)
        for condition, label, color, linestyle in zip(
            conditions, labels, colors, linestyles, strict=True
        ):
            scores = local.loc[local.condition.eq(condition), "calibrated_projection"]
            axis.hist(
                scores,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        axis.set_title(LABELS[model].replace("\n", " "), fontsize=10)
        axis.tick_params(labelsize=8, length=2.5, pad=1.5)
    for axis in axes.flat[len(MODELS):]:
        axis.axis("off")
    figure.supylabel("Count", fontsize=9, x=0.015)
    figure.supxlabel("CAA projection (training-gap units)", fontsize=9, y=0.035)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        ncol=4,
        loc="upper center",
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.985),
    )
    figure.subplots_adjust(
        left=0.07, right=0.99, bottom=0.13, top=0.87, hspace=0.42, wspace=0.27
    )
    save_figure(figure, output / "rq3_self_blinding_unit_histograms")


def main() -> None:
    args = parse_args()
    figures = args.paper_repo / "figures"
    derived = args.paper_repo / "derived"
    figures.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rq1 = load_rq1(args)
    rq1.to_csv(derived / "rq1_behavior_pairwise_accuracy.csv", index=False)
    plot_rq1(rq1, figures)
    transfer = load_transfer(args)
    transfer.to_csv(derived / "all_caa_cross_transfer_metrics.csv", index=False)
    plot_heatmaps(transfer, figures, derived)
    plot_syntax(args, figures, derived)
    plot_projection_distributions(args, figures)


if __name__ == "__main__":
    main()
