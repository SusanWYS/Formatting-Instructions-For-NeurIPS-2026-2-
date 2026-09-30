#!/usr/bin/env python3
"""Plot six-model aggregate scores for the naturalistic evaluation."""

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
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser.parse_args()


def read_projection_scores(analysis_repo: Path, pipeline_repo: Path) -> pd.DataFrame:
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
    scores["metric"] = "CAA projection"
    scores["score"] = scores["unit_vector_projection"]
    return scores


def read_likert_scores(analysis_repo: Path) -> pd.DataFrame:
    root = analysis_repo / "results/self_blinding_naturalistic_likert_2026-09-28/models"
    frames: list[pd.DataFrame] = []
    for model in MODELS:
        frame = pd.read_csv(root / model / "likert_scores.csv")
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
        raise ValueError("Every metric/outcome/condition/scenario must contain six models")
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
    # Figure 3 uses a 7.6 x 1.05 inch canvas for one panel.  This canvas
    # preserves the same wide, shallow panel footprint while stacking two.
    figure, axes = plt.subplots(2, 1, figsize=(7.6, 3.0), sharex=True)
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
        baseline = lookup(summary, metric, outcome, BASELINE)["mean"]

        displayed_min = min(ci_low.min(), baseline)
        displayed_max = max(ci_high.max(), baseline)
        displayed_range = displayed_max - displayed_min
        if displayed_range == 0:
            displayed_range = max(abs(displayed_min), 1.0)
        y_min = displayed_min - 0.10 * displayed_range
        y_max = displayed_max + 0.10 * displayed_range

        if metric == "Likert rating":
            axis.bar(
                x,
                means - y_min,
                width=0.62,
                bottom=y_min,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                yerr=yerr,
                capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
                zorder=3,
            )
            axis.set_ylabel("Expected Likert rating")
        else:
            axis.bar(
                x,
                means - y_min,
                width=0.62,
                bottom=y_min,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                yerr=yerr,
                capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
                zorder=3,
            )
            axis.set_ylabel("Unit CAA projection")

        axis.set_ylim(y_min, y_max)

        axis.axhline(
            baseline,
            color="black",
            linestyle=(0, (5, 3)),
            linewidth=1.3,
            zorder=4,
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9)

    axes[-1].set_xticks(x, CONDITION_LABELS)
    axes[-1].set_xlabel("Demographic condition")
    legend_line = Line2D(
        [0],
        [0],
        color="black",
        linestyle=(0, (5, 3)),
        linewidth=1.3,
        label="Incidental-detail baseline",
    )
    figure.legend(
        handles=[legend_line],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9,
    )
    figure.subplots_adjust(left=0.11, right=0.985, bottom=0.17, top=0.91, hspace=0.44)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"naturalistic_{outcome}_raw_scores_six_model"
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.015)
    figure.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.015)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    projection = read_projection_scores(args.analysis_repo, args.paper_pipeline_repo)
    likert = read_likert_scores(args.analysis_repo)
    scores = pd.concat([projection, likert], ignore_index=True)
    validate(scores)
    summary = summarize(scores, args.bootstrap_repetitions)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.sort_values(["outcome", "metric", "condition"]).to_csv(
        args.summary_csv, index=False
    )
    for outcome in ("positive", "negative"):
        plot_outcome(summary, outcome, args.output_dir)


if __name__ == "__main__":
    main()
