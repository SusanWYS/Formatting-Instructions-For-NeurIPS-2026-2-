#!/usr/bin/env python3
"""Build appendix tables for norm-matched random causal directions."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    "gemma2_2b": "Gemma~2 2B",
    "gemma2_9b": "Gemma~2 9B",
    "qwen3_4b": "Qwen3 4B",
    "qwen3_8b": "Qwen3 8B",
    "llama32_3b": "Llama~3.2 3B",
    "llama3_8b": "Llama~3 8B",
    "llama31_70b": "Llama~3.1 70B",
}
TARGETS = ("causal", "occasion", "plausibility", "syntax_ood")
CONDITIONS = ("white_male", "white_female", "minority_male", "minority_female")


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


def fixed_result_dir(args: argparse.Namespace, model: str) -> Path:
    if model == "llama31_70b":
        return (
            args.analysis_repo
            / "results/extended_paper_models_2026-10-02/fixed_caa_evaluation/models"
            / model
        )
    return args.pipeline_repo / "results/rq2_fixed_caa_evaluation/models" / model


def learned_accuracies(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        directory = fixed_result_dir(args, model)
        transfer = pd.read_csv(directory / "cross_transfer_metrics.csv")
        syntax = pd.read_csv(directory / "syntax_ood_metrics.csv")
        for target in TARGETS:
            if target == "syntax_ood":
                row = syntax[syntax.source.eq("causal")].iloc[0]
            else:
                row = transfer[
                    transfer.source.eq("causal") & transfer.target.eq(target)
                ].iloc[0]
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "learned_causal_accuracy": float(row.pairwise_accuracy),
                }
            )
    return pd.DataFrame(rows)


def load_random_results(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = (
        args.analysis_repo
        / "results/causal_random_vector_baselines_one_sided_alpha_2026-10-04/models"
    )
    accuracy_frames = []
    intervention_frames = []
    naturalistic_frames = []
    for model in MODELS:
        directory = root / model
        if not (directory / "completion.json").exists():
            raise FileNotFoundError(f"Random-vector run is incomplete for {model}")
        accuracy = pd.read_csv(directory / "accuracy_summary.csv")
        intervention = pd.read_csv(directory / "intervention_summary.csv")
        naturalistic = pd.read_csv(directory / "naturalistic_summary.csv")
        for frame, name in (
            (accuracy, "accuracy"),
            (intervention, "intervention"),
            (naturalistic, "naturalistic"),
        ):
            if set(frame.model.astype(str)) != {model}:
                raise ValueError(f"{name} provenance mismatch for {model}")
            if not frame.n_random_vectors.eq(10).all():
                raise ValueError(f"{name} does not contain ten vectors for {model}")
        accuracy_frames.append(accuracy)
        intervention_frames.append(intervention)
        naturalistic_frames.append(naturalistic)
    return (
        pd.concat(accuracy_frames, ignore_index=True),
        pd.concat(intervention_frames, ignore_index=True),
        pd.concat(naturalistic_frames, ignore_index=True),
    )


def normalize_naturalistic(args: argparse.Namespace, frame: pd.DataFrame) -> pd.DataFrame:
    calibration = pd.read_csv(
        args.paper_repo / "derived/causal_train_projection_gaps.csv"
    )[["model", "critical_minus_control_train_gap"]]
    result = frame.merge(calibration, on="model", how="left", validate="many_to_one")
    if result.critical_minus_control_train_gap.isna().any():
        raise ValueError("Missing learned-CAA calibration gap for random baseline")
    for column in (
        "random_vector_mean_delta_caa", "random_vector_sd_delta_caa",
        "random_vector_min_delta_caa", "random_vector_max_delta_caa",
    ):
        result[column] = result[column] / result.critical_minus_control_train_gap
    result["score_units"] = "learned_causal_training_gap"
    return result


def cell(mean: float, sd: float) -> str:
    return rf"${mean:.2f}\mathbin{{\pm}}{sd:.2f}$"


def accuracy_table(combined: pd.DataFrame) -> list[str]:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Paired accuracy of the learned causal CAA and ten norm-matched random directions evaluated at the same selected layer. Random entries report mean $\pm$ standard deviation across directions. \textbf{The learned causal direction is strongest on held-out causal items and within-sentence causal items; random-direction averages remain close to chance.}}",
        r"\label{tab:random-vector-accuracy}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Causal ID} & \multicolumn{2}{c}{Occasion ID} & \multicolumn{2}{c}{Plausibility ID} & \multicolumn{2}{c}{Within-sentence causal} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Model & Learned & Random & Learned & Random & Learned & Random & Learned & Random \\",
        r"\midrule",
    ]
    for model in MODELS:
        values = []
        for target in TARGETS:
            row = combined[combined.model.eq(model) & combined.target.eq(target)].iloc[0]
            values.extend(
                [
                    f"${float(row.learned_causal_accuracy):.2f}$",
                    cell(
                        float(row.random_vector_mean_accuracy),
                        float(row.random_vector_sd_accuracy),
                    ),
                ]
            )
        lines.append(f"{LABELS[model]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{table*}"])
    return lines


def intervention_table(frame: pd.DataFrame) -> list[str]:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Random-direction intervention controls. Each random direction is norm-matched to the learned causal CAA at the same layer and receives its own development-set-selected $\alpha$. Entries are means $\pm$ standard deviations across ten directions.}",
        r"\label{tab:random-vector-intervention}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"& \multicolumn{2}{c}{Critical} & \multicolumn{2}{c}{Control} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Model & $\alpha$ & $\Delta$ surprisal & $\alpha$ & $\Delta$ surprisal \\",
        r"\midrule",
    ]
    for model in MODELS:
        critical = frame[frame.model.eq(model) & frame.condition.eq("critical")].iloc[0]
        control = frame[frame.model.eq(model) & frame.condition.eq("control")].iloc[0]
        values = (
            cell(float(critical.random_vector_mean_alpha), float(critical.random_vector_sd_alpha)),
            cell(
                float(critical.random_vector_mean_delta_surprisal),
                float(critical.random_vector_sd_delta_surprisal),
            ),
            cell(float(control.random_vector_mean_alpha), float(control.random_vector_sd_alpha)),
            cell(
                float(control.random_vector_mean_delta_surprisal),
                float(control.random_vector_sd_delta_surprisal),
            ),
        )
        lines.append(f"{LABELS[model]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def naturalistic_table(frame: pd.DataFrame, outcome: str) -> list[str]:
    readable = "negative" if outcome == "negative" else "positive"
    label = f"tab:random-vector-naturalistic-{outcome}"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{Random-direction demographic-minus-agnostic CAA differences for {readable} outcomes. Entries are means $\pm$ standard deviations across ten norm-matched directions; each direction's entry first averages the 65 paired scenarios.}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & \shortstack{White\\male} & \shortstack{White\\female} & \shortstack{Minority\\male} & \shortstack{Minority\\female} \\",
        r"\midrule",
    ]
    for model in MODELS:
        values = []
        for condition in CONDITIONS:
            row = frame[
                frame.model.eq(model)
                & frame.outcome.eq(outcome)
                & frame.condition.eq(condition)
            ].iloc[0]
            values.append(
                cell(
                    float(row.random_vector_mean_delta_caa),
                    float(row.random_vector_sd_delta_caa),
                )
            )
        lines.append(f"{LABELS[model]} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def main() -> None:
    args = parse_args()
    accuracy, intervention, naturalistic = load_random_results(args)
    naturalistic = normalize_naturalistic(args, naturalistic)
    learned = learned_accuracies(args)
    combined = learned.merge(
        accuracy,
        on=["model", "target"],
        how="left",
        validate="one_to_one",
    )
    if combined.random_vector_mean_accuracy.isna().any():
        raise ValueError("Missing random-vector accuracy summary")
    derived = args.paper_repo / "derived"
    acl_derived = args.paper_repo / "acl/derived"
    derived.mkdir(parents=True, exist_ok=True)
    acl_derived.mkdir(parents=True, exist_ok=True)
    combined.to_csv(derived / "random_vector_accuracy_comparison.csv", index=False)
    intervention.to_csv(derived / "random_vector_intervention_summary.csv", index=False)
    naturalistic.to_csv(derived / "random_vector_naturalistic_summary.csv", index=False)
    lines = [
        r"\section{Norm-matched random-vector baselines}",
        r"\label{app:random-baselines}",
        (
            "For each model, we sample ten isotropic Gaussian directions at the causal CAA's "
            "selected layer and rescale each direction to exactly match the learned causal "
            "direction's norm. We reuse the same seeded directions for paired-accuracy, "
            "intervention, and naturalistic evaluations. For interventions, we tune $\\alpha$ "
            "separately for every random direction over the same one-sided grid and on the same "
            "development items used for the learned direction; no test result enters selection."
            " Naturalistic random-direction projections are divided by that model's learned-CAA "
            "critical-minus-control training gap, matching the learned-direction calibration."
        ),
        *accuracy_table(combined),
        *intervention_table(intervention),
        *naturalistic_table(naturalistic, "negative"),
        *naturalistic_table(naturalistic, "positive"),
    ]
    text = "\n".join(lines) + "\n"
    (derived / "random_vector_baselines.tex").write_text(text, encoding="utf-8")
    (acl_derived / "random_vector_baselines.tex").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
