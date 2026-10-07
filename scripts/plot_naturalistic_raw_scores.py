#!/usr/bin/env python3
"""Plot seven-model aggregate scores for the naturalistic evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
CONDITIONS = (
    "white_male",
    "white_female",
    "minority_male",
    "minority_female",
)
PLOT_CONDITIONS = (
    "white_female",
    "minority_male",
    "minority_female",
)
PLOT_CONDITION_LABELS = (
    "White\nfemale",
    "Minority\nmale",
    "Minority\nfemale",
)
BASELINE = "race_gender_agnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/eliciture_interp"),
    )
    parser.add_argument(
        "--paper-pipeline-repo",
        type=Path,
        default=Path("/data/cl/u/susanw26/eliciture_interp_paper"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("derived/naturalistic_raw_scores_seven_model.csv"),
    )
    parser.add_argument(
        "--difference-summary-csv",
        type=Path,
        default=Path("derived/naturalistic_difference_scores_seven_model.csv"),
    )
    parser.add_argument(
        "--per-model-difference-csv",
        type=Path,
        default=Path("derived/naturalistic_difference_scores_per_model.csv"),
    )
    parser.add_argument(
        "--white-male-reference-csv",
        type=Path,
        default=Path("derived/naturalistic_white_male_reference_scores_seven_model.csv"),
    )
    parser.add_argument(
        "--calibration-csv",
        type=Path,
        default=Path("derived/causal_train_projection_gaps.csv"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser.parse_args()


def read_projection_scores(
    analysis_repo: Path, pipeline_repo: Path, calibration_csv: Path
) -> pd.DataFrame:
    negative_root = pipeline_repo / "results/rq3_naturalistic/models"
    positive_root = (
        analysis_repo
        / "results/self_blinding_positive_curated_caa_2026-09-16/models"
    )
    baseline_root = (
        analysis_repo
        / "results/self_blinding_race_gender_agnostic_incidental_2026-09-26/models"
    )
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        if model == "llama31_70b":
            extended = (
                analysis_repo / "results/extended_paper_models_2026-10-02"
            )
            demographic = pd.read_csv(
                extended
                / f"naturalistic_demographic_projection/models/{model}/projection_scores.csv"
            )
            demographic["model_key"] = model
            frames.append(demographic)
            baseline = pd.read_csv(
                extended
                / f"naturalistic_agnostic_projection/models/{model}/projection_scores.csv"
            )
            baseline["model_key"] = model
            frames.append(baseline)
            continue
        negative = pd.read_csv(negative_root / model / "projection_scores.csv")
        negative["outcome"] = "negative"
        negative["model_key"] = model
        frames.append(negative)

        positive = pd.read_csv(positive_root / model / "projection_scores.csv")
        positive["model_key"] = model
        frames.append(positive)

        baseline = pd.read_csv(baseline_root / model / "projection_scores.csv")
        baseline["model_key"] = model
        frames.append(baseline)

    scores = pd.concat(frames, ignore_index=True)
    scores = scores[scores["condition"].isin((*CONDITIONS, BASELINE))].copy()
    calibration = pd.read_csv(calibration_csv)[
        ["model", "critical_minus_control_train_gap"]
    ].rename(columns={"model": "model_key"})
    scores = scores.merge(calibration, on="model_key", how="left", validate="many_to_one")
    if scores["critical_minus_control_train_gap"].isna().any():
        raise ValueError("Missing causal-training calibration gap")
    scores["metric"] = "CAA projection"
    scores["raw_unit_vector_projection"] = scores["unit_vector_projection"]
    scores["score"] = (
        scores["unit_vector_projection"]
        / scores["critical_minus_control_train_gap"]
    )
    return scores


def read_likert_scores(analysis_repo: Path) -> pd.DataFrame:
    root = analysis_repo / "results/self_blinding_naturalistic_likert_2026-09-28/models"
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        if model == "llama31_70b":
            path = (
                analysis_repo
                / "results/extended_paper_models_2026-10-02/naturalistic_likert/models"
                / model
                / "likert_scores.csv"
            )
        else:
            path = root / model / "likert_scores.csv"
        frame = pd.read_csv(path)
        frame["model_key"] = model
        frames.append(frame)
    scores = pd.concat(frames, ignore_index=True)
    scores = scores[scores["condition"].isin((*CONDITIONS, BASELINE))].copy()
    scores["metric"] = "Likert rating"
    scores["score"] = scores["expected_likert"]
    return scores


def validate(scores: pd.DataFrame) -> None:
    keys = ["metric", "outcome", "condition", "decision_question_id", "model_key"]
    if scores.duplicated(keys).any():
        raise ValueError("Duplicate model/scenario scores found")
    expected = len(MODELS)
    counts = scores.groupby(keys[:-1], observed=True)["model_key"].nunique()
    if not (counts == expected).all():
        raise ValueError(
            f"Every metric/outcome/condition/scenario must contain {expected} models"
        )
    scenario_counts = scores.groupby(
        ["metric", "outcome", "condition"], observed=True
    )["decision_question_id"].nunique()
    if not (scenario_counts == 65).all():
        raise ValueError("Every metric/outcome/condition must contain 65 scenarios")


def summarize(scores: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    # Give every model equal weight within a scenario, then bootstrap scenarios.
    scenario_means = (
        scores.groupby(
            ["metric", "outcome", "condition", "decision_question_id"],
            observed=True,
        )["score"]
        .mean()
        .reset_index()
    )
    rng = np.random.default_rng(20260930)
    rows = []
    for (metric, outcome, condition), group in scenario_means.groupby(
        ["metric", "outcome", "condition"], sort=False, observed=True
    ):
        values = group.sort_values("decision_question_id")["score"].to_numpy()
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "score_units": (
                    "causal_training_gap" if metric == "CAA projection" else "Likert points"
                ),
                "outcome": outcome,
                "condition": condition,
                "mean": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_scenarios": len(values),
                "n_models": len(MODELS),
            }
        )
    return pd.DataFrame(rows)


def summarize_paired_differences(
    scores: pd.DataFrame, repetitions: int
) -> pd.DataFrame:
    """Summarize demographic-minus-agnostic differences by scenario."""
    scenario_means = (
        scores.groupby(
            ["metric", "outcome", "condition", "decision_question_id"],
            observed=True,
        )["score"]
        .mean()
        .reset_index()
    )
    baseline = (
        scenario_means[scenario_means["condition"] == BASELINE]
        .drop(columns="condition")
        .rename(columns={"score": "baseline_score"})
    )
    differences = scenario_means[scenario_means["condition"].isin(CONDITIONS)].merge(
        baseline,
        on=["metric", "outcome", "decision_question_id"],
        how="left",
        validate="many_to_one",
    )
    if differences["baseline_score"].isna().any():
        raise ValueError("Missing matched race/gender-agnostic baseline")
    differences["difference"] = differences["score"] - differences["baseline_score"]

    rng = np.random.default_rng(20260930)
    rows = []
    for (metric, outcome, condition), group in differences.groupby(
        ["metric", "outcome", "condition"], sort=False, observed=True
    ):
        values = group.sort_values("decision_question_id")["difference"].to_numpy()
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "score_units": (
                    "causal_training_gap" if metric == "CAA projection" else "Likert points"
                ),
                "outcome": outcome,
                "condition": condition,
                "mean": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_scenarios": len(values),
                "n_models": len(MODELS),
            }
        )
    return pd.DataFrame(rows)


def summarize_per_model_differences(
    scores: pd.DataFrame, repetitions: int
) -> pd.DataFrame:
    baseline = (
        scores[scores["condition"] == BASELINE][
            ["metric", "outcome", "decision_question_id", "model_key", "score"]
        ]
        .rename(columns={"score": "baseline_score"})
    )
    differences = scores[scores["condition"].isin(CONDITIONS)].merge(
        baseline,
        on=["metric", "outcome", "decision_question_id", "model_key"],
        how="left",
        validate="many_to_one",
    )
    differences["difference"] = differences["score"] - differences["baseline_score"]
    rng = np.random.default_rng(20261004)
    rows = []
    for keys, group in differences.groupby(
        ["metric", "outcome", "condition", "model_key"],
        sort=False,
        observed=True,
    ):
        values = group.sort_values("decision_question_id")["difference"].to_numpy()
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "metric": keys[0],
                "score_units": (
                    "causal_training_gap" if keys[0] == "CAA projection" else "Likert points"
                ),
                "outcome": keys[1], "condition": keys[2],
                "model": keys[3], "mean": values.mean(), "ci_low": low,
                "ci_high": high, "ci_half_width": (high - low) / 2,
                "n_scenarios": len(values),
            }
        )
    return pd.DataFrame(rows)


def summarize_white_male_reference(
    scores: pd.DataFrame, repetitions: int
) -> pd.DataFrame:
    """Summarize each plotted condition relative to matched white-male items."""
    scenario_means = (
        scores.groupby(
            ["metric", "outcome", "condition", "decision_question_id"],
            observed=True,
        )["score"]
        .mean()
        .reset_index()
    )
    reference = (
        scenario_means[scenario_means["condition"] == "white_male"]
        .drop(columns="condition")
        .rename(columns={"score": "white_male_score"})
    )
    contrasts = scenario_means[
        scenario_means["condition"].isin((*PLOT_CONDITIONS, BASELINE))
    ].merge(
        reference,
        on=["metric", "outcome", "decision_question_id"],
        how="left",
        validate="many_to_one",
    )
    if contrasts["white_male_score"].isna().any():
        raise ValueError("Missing matched white-male reference")
    contrasts["difference"] = contrasts["score"] - contrasts["white_male_score"]

    rng = np.random.default_rng(20261004)
    rows = []
    for (metric, outcome, condition), group in contrasts.groupby(
        ["metric", "outcome", "condition"], sort=False, observed=True
    ):
        values = group.sort_values("decision_question_id")["difference"].to_numpy()
        samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        rows.append(
            {
                "metric": metric,
                "score_units": (
                    "causal_training_gap" if metric == "CAA projection" else "Likert points"
                ),
                "outcome": outcome,
                "condition": condition,
                "reference_condition": "white_male",
                "mean": values.mean(),
                "ci_low": low,
                "ci_high": high,
                "n_scenarios": len(values),
                "n_models": len(MODELS),
            }
        )
    return pd.DataFrame(rows)


def lookup(summary: pd.DataFrame, metric: str, outcome: str, condition: str) -> pd.Series:
    row = summary[
        (summary["metric"] == metric)
        & (summary["outcome"] == outcome)
        & (summary["condition"] == condition)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {metric}/{outcome}/{condition}")
    return row.iloc[0]


def plot_combined(summary: pd.DataFrame, output_dir: Path) -> None:
    """Plot positive and negative outcomes together in two metric panels."""
    colors = {"positive": "#5E7436", "negative": "#D0685B"}
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Full ACL/ARR text width; each panel is approximately the size of one
    # panel in the preceding full-width figures.
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.52))
    x = np.arange(len(PLOT_CONDITIONS))
    bar_width = 0.30
    offsets = {"positive": -bar_width / 2 - 0.025, "negative": bar_width / 2 + 0.025}

    for axis, metric in zip(
        axes,
        ("CAA projection", "Likert rating"),
        strict=True,
    ):
        displayed_lows = [0.0]
        displayed_highs = [0.0]
        for outcome in ("positive", "negative"):
            rows = [
                lookup(summary, metric, outcome, condition)
                for condition in PLOT_CONDITIONS
            ]
            incidental = lookup(summary, metric, outcome, BASELINE)
            means = np.array([row["mean"] for row in rows])
            ci_low = np.array([row["ci_low"] for row in rows])
            ci_high = np.array([row["ci_high"] for row in rows])
            yerr = np.vstack((means - ci_low, ci_high - means))
            displayed_lows.extend([float(ci_low.min()), float(incidental["mean"])])
            displayed_highs.extend([float(ci_high.max()), float(incidental["mean"])])

            axis.bar(
                x + offsets[outcome],
                means,
                width=bar_width,
                color=colors[outcome],
                edgecolor="black",
                linewidth=0.45,
                yerr=yerr,
                capsize=2.0,
                error_kw={"elinewidth": 0.75, "capthick": 0.75},
                zorder=3,
            )
            axis.axhline(
                float(incidental["mean"]),
                color=colors[outcome],
                linestyle=(0, (4.0, 2.4)),
                linewidth=1.15,
                zorder=4,
            )

        displayed_min = min(displayed_lows)
        displayed_max = max(displayed_highs)
        displayed_range = displayed_max - displayed_min
        if displayed_range == 0:
            displayed_range = max(abs(displayed_min), 1.0)
        axis.set_ylim(
            displayed_min - 0.12 * displayed_range,
            displayed_max + 0.12 * displayed_range,
        )
        axis.axhline(
            0.0,
            color="#777777",
            linestyle="-",
            linewidth=0.6,
            zorder=2,
        )
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(
            r"$\Delta$ Projection (vs. white male)"
            if metric == "CAA projection"
            else r"$\Delta$ Rating (vs. white male)"
        )
        axis.set_title(metric, fontweight="normal", pad=3.0)
        axis.set_xticks(x, PLOT_CONDITION_LABELS)
        axis.set_xlabel("Demographic condition")

    handles = (
        Patch(facecolor=colors["positive"], edgecolor="black", linewidth=0.45),
        Patch(facecolor=colors["negative"], edgecolor="black", linewidth=0.45),
        Line2D([0], [0], color=colors["positive"], linestyle=(0, (4.0, 2.4)), linewidth=1.15),
        Line2D([0], [0], color=colors["negative"], linestyle=(0, (4.0, 2.4)), linewidth=1.15),
    )
    labels = (
        "Positive outcomes",
        "Negative outcomes",
        "Neutral baseline (positive outcomes)",
        "Neutral baseline (negative outcomes)",
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.515, 1.005),
        frameon=False,
        ncol=2,
        handlelength=2.3,
        columnspacing=2.0,
    )
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.70, wspace=0.27)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "naturalistic_combined_scores_seven_model"
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    projection = read_projection_scores(
        args.analysis_repo, args.paper_pipeline_repo, args.calibration_csv
    )
    likert = read_likert_scores(args.analysis_repo)
    scores = pd.concat([projection, likert], ignore_index=True)
    validate(scores)
    summary = summarize(scores, args.bootstrap_repetitions)
    difference_summary = summarize_paired_differences(scores, args.bootstrap_repetitions)
    per_model = summarize_per_model_differences(scores, args.bootstrap_repetitions)
    white_male_reference = summarize_white_male_reference(
        scores, args.bootstrap_repetitions
    )
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["outcome", "metric", "condition"]).to_csv(
        args.summary_csv, index=False
    )
    args.difference_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    difference_summary.sort_values(["outcome", "metric", "condition"]).to_csv(
        args.difference_summary_csv, index=False
    )
    args.per_model_difference_csv.parent.mkdir(parents=True, exist_ok=True)
    per_model.sort_values(["outcome", "metric", "model", "condition"]).to_csv(
        args.per_model_difference_csv, index=False
    )
    args.white_male_reference_csv.parent.mkdir(parents=True, exist_ok=True)
    white_male_reference.sort_values(["outcome", "metric", "condition"]).to_csv(
        args.white_male_reference_csv, index=False
    )
    plot_combined(white_male_reference, args.output_dir)


if __name__ == "__main__":
    main()
