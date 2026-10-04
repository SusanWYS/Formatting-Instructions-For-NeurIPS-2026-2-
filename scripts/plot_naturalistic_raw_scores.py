#!/usr/bin/env python3
"""Plot seven-model aggregate scores for the naturalistic evaluation."""

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
CONDITIONS = (
    "white_male",
    "white_female",
    "minority_male",
    "minority_female",
)
CONDITION_LABELS = (
    "White\nmale",
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
        default=Path("derived/naturalistic_raw_scores_six_model.csv"),
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


def lookup(summary: pd.DataFrame, metric: str, outcome: str, condition: str) -> pd.Series:
    row = summary[
        (summary["metric"] == metric)
        & (summary["outcome"] == outcome)
        & (summary["condition"] == condition)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {metric}/{outcome}/{condition}")
    return row.iloc[0]


def plot_outcome(summary: pd.DataFrame, outcome: str, output_dir: Path) -> None:
    palette = {
        "positive": ("#556B2F", "#B5B87A"),
        "negative": ("#B94A48", "#F08080"),
    }
    projection_color, likert_color = palette[outcome]
    # Use the same Matplotlib typography as Figure 3.  The canvas and LaTeX
    # inclusion widths are both approximately half those of Figure 3.
    figure, axes = plt.subplots(2, 1, figsize=(3.24, 2.5), sharex=True)
    x = np.arange(len(CONDITIONS))

    for axis, metric, color in zip(
        axes,
        ("CAA projection", "Likert rating"),
        (projection_color, likert_color),
        strict=True,
    ):
        rows = [lookup(summary, metric, outcome, condition) for condition in CONDITIONS]
        means = np.array([row["mean"] for row in rows])
        ci_low = np.array([row["ci_low"] for row in rows])
        ci_high = np.array([row["ci_high"] for row in rows])
        yerr = np.vstack((means - ci_low, ci_high - means))
        displayed_min = min(ci_low.min(), 0.0)
        displayed_max = max(ci_high.max(), 0.0)
        displayed_range = displayed_max - displayed_min
        if displayed_range == 0:
            displayed_range = max(abs(displayed_min), 1.0)
        y_min = displayed_min - 0.10 * displayed_range
        y_max = displayed_max + 0.10 * displayed_range

        axis.bar(
            x,
            means,
            width=0.62,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            yerr=yerr,
            capsize=1.8,
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
            zorder=3,
        )
        axis.set_ylim(y_min, y_max)
        axis.axhline(
            0.0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            zorder=4,
        )
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_ylabel(r"$\Delta$ CAA" if metric == "CAA projection" else r"$\Delta$ Likert")

    axes[-1].set_xticks(x, CONDITION_LABELS)
    axes[-1].set_xlabel("Demographic condition")
    figure.subplots_adjust(left=0.19, right=0.985, bottom=0.19, top=0.98, hspace=0.22)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the established asset name so both manuscript formats continue
    # to resolve the same figure without duplicating binary files.
    stem = f"naturalistic_{outcome}_raw_scores_six_model"
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.015)
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
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
    for outcome in ("positive", "negative"):
        plot_outcome(difference_summary, outcome, args.output_dir)


if __name__ == "__main__":
    main()
