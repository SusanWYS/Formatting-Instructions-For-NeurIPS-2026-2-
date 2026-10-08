#!/usr/bin/env python3
"""Build the compact naturalistic-stimulus template table used in the appendix."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_DEMOGRAPHIC_TEMPLATES = Path(
    "/data/cl/u/susanw26/eliciture_interp_paper/data/naturalistic/templates.tsv"
)
DEFAULT_DEMOGRAPHIC_MANIFEST = Path(
    "/data/cl/u/susanw26/eliciture_interp_paper/data/naturalistic/stimulus_manifest.tsv"
)
DEFAULT_NON_DEMOGRAPHIC_TEMPLATES = Path(
    "/data/cl/u/susanw26/eliciture_interp/results/"
    "self_blinding_race_gender_agnostic_incidental_2026-09-26/"
    "race_gender_agnostic_incidental_templates.tsv"
)
DEFAULT_OUTPUTS = (
    Path("acl/derived/rq3_self_blinding_stimuli.tex"),
    Path("derived/rq3_self_blinding_stimuli.tex"),
)
TOKEN_PATTERN = re.compile(
    r"\w+(?:[-’']\w+)*|\$?[\d,]+(?:\.\d+)?|[^\w\s]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demographic-templates",
        type=Path,
        default=DEFAULT_DEMOGRAPHIC_TEMPLATES,
    )
    parser.add_argument(
        "--non-demographic-templates",
        type=Path,
        default=DEFAULT_NON_DEMOGRAPHIC_TEMPLATES,
    )
    parser.add_argument(
        "--demographic-manifest",
        type=Path,
        default=DEFAULT_DEMOGRAPHIC_MANIFEST,
    )
    parser.add_argument(
        "--output",
        type=Path,
        action="append",
        help="Output TeX file; may be supplied more than once.",
    )
    return parser.parse_args()


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def normalize_spacing(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"\s+([.,;:!?])", r"\1", value)


def split_once(value: str, marker: str, *, row_id: int, field: str) -> tuple[str, str]:
    if value.count(marker) != 1:
        raise ValueError(
            f"Scenario {row_id}: expected exactly one {field} marker {marker!r}"
        )
    return value.split(marker, maxsplit=1)


def token_boundary(value: str) -> int:
    """Return the insertion boundary as a count of words and punctuation."""
    return len(TOKEN_PATTERN.findall(value))


def insert_markers(base: str, insertions: dict[int, str]) -> str:
    tokens = TOKEN_PATTERN.findall(base)
    pieces: list[str] = []
    for boundary in range(len(tokens) + 1):
        if boundary in insertions:
            pieces.append(insertions[boundary])
        if boundary < len(tokens):
            pieces.append(latex_escape(tokens[boundary]))
    rendered = " ".join(pieces)
    rendered = re.sub(r"\s+([.,;:!?])", r"\1", rendered)
    rendered = rendered.replace("$ ", "$")
    rendered = rendered.replace("' s", "'s")
    return rendered


def build_rows(
    demographic_path: Path,
    demographic_manifest_path: Path,
    non_demographic_path: Path,
) -> list[dict[str, str]]:
    demographic = pd.read_csv(demographic_path, sep="\t", comment="#")
    manifest = pd.read_csv(demographic_manifest_path, sep="\t")
    assigned_races = manifest[
        ["decision_question_id", "assigned_minority_race"]
    ].drop_duplicates()
    if assigned_races["decision_question_id"].duplicated().any():
        raise ValueError("Each scenario must have exactly one assigned minority race")
    demographic = demographic.merge(
        assigned_races,
        on="decision_question_id",
        validate="one_to_one",
    )
    non_demographic = pd.read_csv(non_demographic_path, sep="\t")
    frame = demographic.merge(
        non_demographic[
            [
                "decision_question_id",
                "scenario_nickname",
                "construction",
                "incidental_detail",
                "negative_sentence",
            ]
        ],
        on=["decision_question_id", "scenario_nickname", "construction"],
        validate="one_to_one",
    ).sort_values("decision_question_id")
    if len(frame) != 65:
        raise ValueError(f"Expected 65 scenarios, found {len(frame)}")

    rows: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        row_id = int(row.decision_question_id)
        if row.construction == "relative_clause":
            demographic_marker = "who is {race} and {gender}"
            demographic_details = {
                "white_male": "who is white and male",
                "white_female": "who is white and female",
                "minority_male": f"who is {row.assigned_minority_race} and male",
                "minority_female": f"who is {row.assigned_minority_race} and female",
            }
        elif row.construction == "modifier":
            demographic_marker = "{race} {gender}"
            demographic_details = {
                "white_male": "white male",
                "white_female": "white female",
                "minority_male": f"{row.assigned_minority_race} male",
                "minority_female": f"{row.assigned_minority_race} female",
            }
        else:
            raise ValueError(f"Scenario {row_id}: unexpected construction {row.construction}")

        demo_before, demo_after = split_once(
            row.negative_outcome_template,
            demographic_marker,
            row_id=row_id,
            field="demographic",
        )
        non_demo_before, non_demo_after = split_once(
            row.negative_sentence,
            row.incidental_detail,
            row_id=row_id,
            field="non-demographic",
        )
        demo_base = normalize_spacing(demo_before + demo_after)
        non_demo_base = normalize_spacing(non_demo_before + non_demo_after)
        if demo_base != non_demo_base:
            raise ValueError(
                f"Scenario {row_id}: demographic and non-demographic templates "
                "do not reduce to the same sentence"
            )

        demo_position = token_boundary(normalize_spacing(demo_before))
        non_demo_position = token_boundary(normalize_spacing(non_demo_before))
        if demo_position == non_demo_position:
            insertions = {demo_position: r"[\emph{D/ND}]"}
        else:
            insertions = {
                demo_position: r"[\emph{D}]",
                non_demo_position: r"[\emph{ND}]",
            }
        template = insert_markers(demo_base, insertions)

        rows.append(
            {
                "template": template,
                **demographic_details,
                "non_demographic": str(row.incidental_detail),
            }
        )
    return rows


def render_table(rows: list[dict[str, str]], *, per_column_font: bool) -> str:
    if per_column_font:
        column_spec = (
            r">{\scriptsize\raggedright\arraybackslash}p{.39\linewidth}"
            r">{\scriptsize\raggedright\arraybackslash}p{.095\linewidth}"
            r">{\scriptsize\raggedright\arraybackslash}p{.095\linewidth}"
            r">{\scriptsize\raggedright\arraybackslash}p{.095\linewidth}"
            r">{\scriptsize\raggedright\arraybackslash}p{.095\linewidth}"
            r">{\scriptsize\raggedright\arraybackslash}p{.14\linewidth}"
        )
        font_setup: list[str] = []
    else:
        column_spec = (
            r"p{.39\linewidth}p{.095\linewidth}p{.095\linewidth}"
            r"p{.095\linewidth}p{.095\linewidth}p{.14\linewidth}"
        )
        font_setup = [
            r"\scriptsize",
            r"\makeatletter",
            r"\renewcommand{\fnum@table}{\normalsize\tablename~\thetable}",
            r"\makeatother",
        ]
    lines = [
        r"\begingroup",
        *font_setup,
        r"\setlength{\LTcapwidth}{\textwidth}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        rf"\begin{{longtable}}{{{column_spec}}}",
        r"\caption{Templates and inserted details for all 65 matched negative-outcome scenarios. In the template column, \emph{D} marks the demographic-detail position, \emph{ND} marks the non-demographic-detail position, and \emph{D/ND} marks a shared position; an unused blank is omitted when constructing a sentence. Each remaining cell gives only the text inserted for that condition.}",
        r"\label{tab:rq3-stimuli}\\",
        r"\toprule",
        r"Template & White male & White female & Minority male & Minority female & Non-demographic baseline \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Template & White male & White female & Minority male & Minority female & Non-demographic baseline \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["template"],
                    latex_escape(row["white_male"]),
                    latex_escape(row["white_female"]),
                    latex_escape(row["minority_male"]),
                    latex_escape(row["minority_female"]),
                    latex_escape(row["non_demographic"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    outputs = tuple(args.output) if args.output else DEFAULT_OUTPUTS
    rows = build_rows(
        args.demographic_templates,
        args.demographic_manifest,
        args.non_demographic_templates,
    )
    for output in outputs:
        rendered = render_table(rows, per_column_font="acl" in output.parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
