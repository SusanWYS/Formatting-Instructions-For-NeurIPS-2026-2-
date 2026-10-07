#!/usr/bin/env python3
"""Build ACL-appendix tables for the fixed-magnitude intervention sweep."""

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
ALPHAS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)


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
        "--paper-repo",
        type=Path,
        default=Path(
            "/data/cl/u/susanw26/Formatting-Instructions-For-NeurIPS-2026-2-"
        ),
    )
    return parser.parse_args()


def signed(value: float) -> str:
    if abs(value) < 0.05:
        value = 0.0
    return f"{value:+.1f}"


def cell(mean: float, low: float, high: float) -> str:
    mean *= 100
    low *= 100
    high *= 100
    half_width = (high - low) / 2
    body = rf"{signed(mean)}\mathbin{{\pm}}{half_width:.1f}"
    if low > 0 or high < 0:
        body = rf"\boldsymbol{{{body}}}"
    return rf"${body}$"


def validate(frame: pd.DataFrame) -> None:
    expected = {(model, alpha) for model in MODELS for alpha in ALPHAS}
    observed = set(zip(frame.model, frame.alpha.astype(float), strict=True))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Unexpected model/alpha rows; missing={missing}, extra={extra}")
    if not frame.n_heldout_pairs.eq(150).all():
        raise ValueError("Every result must contain all 150 held-out pairs")
    if not frame.n_random_vectors.eq(10).all():
        raise ValueError("Every random baseline must contain ten directions")


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def table(frame: pd.DataFrame, alpha: float) -> list[str]:
    label = f"tab:fixed-alpha-sweep-{alpha_tag(alpha)}"
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        (
            rf"\caption{{Change in paired accuracy under fixed-magnitude interventions "
            rf"with $|\alpha|={alpha:g}$, in percentage points $\pm$ 95\% CI "
            "half-widths. The learned-direction CI bootstraps the 150 item pairs; "
            "the random-direction CI jointly bootstraps ten norm-matched directions "
            "and the shared item pairs. Bold CIs exclude zero; negative changes are "
            "predicted.}"
        ),
        rf"\label{{{label}}}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Model & Causal vectors & Random vectors \\",
        r"\midrule",
    ]
    for model in MODELS:
        row = frame[
            frame.model.eq(model) & frame.alpha.astype(float).eq(alpha)
        ].iloc[0]
        learned = cell(
            float(row.causal_delta_accuracy),
            float(row.causal_ci_low),
            float(row.causal_ci_high),
        )
        random = cell(
            float(row.random_delta_accuracy_mean),
            float(row.random_ci_low),
            float(row.random_ci_high),
        )
        lines.append(f"{LABELS[model]} & {learned} & {random}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return lines


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.summary)
    validate(frame)

    output = args.paper_repo / "acl/derived/fixed_alpha_sweep.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\section{Intervention magnitude sweep}",
        r"\label{app:fixed-alpha-sweep}",
        (
            "We repeat the paired-accuracy intervention from "
            "Section~\\ref{sec:intervention} at six fixed magnitudes. At every magnitude, "
            "we subtract the direction from critical contexts and add it to controls; "
            "all other settings are unchanged. Each table reports the learned causal "
            "direction and the average over ten norm-matched random directions. The "
            "$|\\alpha|=5$ table gives the largest-magnitude values plotted in "
            "Figure~\\ref{fig:intervention-alpha-sweep}."
        ),
    ]
    for alpha in ALPHAS:
        lines.extend(table(frame, alpha))
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
