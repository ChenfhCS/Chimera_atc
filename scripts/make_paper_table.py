#!/usr/bin/env python
"""Render results in the paper-table layout:

    | ARC-e        | ARC-c        | PIQA         | TruthfulQA   | CNN/DM         | PubMed         |
    | ACC↑  TPT↑   | ACC↑  TPT↑   | F1↑   TPT↑   | F1↑   TPT↑   | ROUGE↑ TPT↑    | ROUGE↑ TPT↑    |

Produces three files next to the requested output path: ``.csv``, ``.md``,
``.tex``. Numbers are rounded to 2 decimals and "error" rows are dropped.

Notes on metric labels:
  * For single-letter answer tasks (PIQA, TruthfulQA), F1 and ACC are
    numerically identical because the prediction is a single token after
    letter extraction. We label them "F1" in the header to match common
    paper convention. The underlying scores are computed by
    ``exact_choice_accuracy`` on the regex-extracted letter.
  * ROUGE in the table is ROUGE-L F1, our default for summarization.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_DATASETS = ["ARC-e", "ARC-c", "PIQA", "TruthfulQA", "CNN/DM", "PubMed"]
DEFAULT_METRIC_LABELS = {
    "ARC-e": "ACC",
    "ARC-c": "ACC",
    "PIQA": "F1",
    "TruthfulQA": "F1",
    "CNN/DM": "ROUGE",
    "PubMed": "ROUGE",
}


def _load_rows(paths):
    out = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _flat_table(rows, datasets, labels, decimals):
    """Return (models_list, cells) where cells[model][dataset] = (score, tpt)."""
    cells = {}
    for r in rows:
        if "error" in r:
            continue
        m = r.get("model")
        d = r.get("dataset")
        if m is None or d is None or d not in datasets:
            continue
        score = r.get("score")
        tpt = r.get("tpt")
        if score is None or tpt is None:
            continue
        cells.setdefault(m, {})[d] = (round(score, decimals), round(tpt, decimals))
    return sorted(cells.keys()), cells


def _format_csv(models, cells, datasets, labels):
    """Flat header: model, <ds> <metric>, <ds> TPT, ..."""
    header = ["model"]
    for d in datasets:
        header.append(f"{d} {labels[d]}")
        header.append(f"{d} TPT")
    lines = [",".join(header)]
    for m in models:
        row = [m]
        for d in datasets:
            v = cells.get(m, {}).get(d)
            if v is None:
                row += ["", ""]
            else:
                row += [f"{v[0]}", f"{v[1]}"]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def _format_markdown(models, cells, datasets, labels):
    # Two-row header: dataset names (spanning 2 cols each) then metric/TPT.
    top = "|       | " + " | ".join([f"**{d}** ||" for d in datasets]) + " |"
    sub_cells = []
    align_cells = []
    for d in datasets:
        sub_cells += [f"{labels[d]} ↑", "TPT ↑"]
        align_cells += [":---:", ":---:"]
    sub = "| model | " + " | ".join(sub_cells) + " |"
    align = "|:---|" + "|".join(align_cells) + "|"
    out = [top, sub, align]
    for m in models:
        row = [m]
        for d in datasets:
            v = cells.get(m, {}).get(d)
            if v is None:
                row += ["-", "-"]
            else:
                row += [f"{v[0]}", f"{v[1]}"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


def _format_latex(models, cells, datasets, labels):
    """LaTeX table mimicking the paper layout."""
    col_spec = "l|" + "|".join(["cc"] * len(datasets))
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    # Top row: dataset names spanning 2 cols each.
    top_cells = [r"\textbf{model}"]
    for d in datasets:
        ds_disp = d.replace("_", r"\_")
        top_cells.append(r"\multicolumn{2}{c|}{\textbf{" + ds_disp + "}}")
    # Drop final | for last group.
    top_line = " & ".join(top_cells).replace(
        "\\multicolumn{2}{c|}{\\textbf{" + datasets[-1].replace("_", r"\_") + "}}",
        "\\multicolumn{2}{c}{\\textbf{" + datasets[-1].replace("_", r"\_") + "}}",
    )
    lines.append(top_line + r" \\")
    # Sub-header line: metric↑ TPT↑
    sub_cells = [""]
    for d in datasets:
        sub_cells.append(f"{labels[d]} $\\uparrow$")
        sub_cells.append(r"TPT $\uparrow$")
    lines.append(" & ".join(sub_cells) + r" \\")
    lines.append(r"\midrule")
    for m in models:
        row = [m.replace("_", r"\_")]
        for d in datasets:
            v = cells.get(m, {}).get(d)
            if v is None:
                row += ["-", "-"]
            else:
                row += [f"{v[0]}", f"{v[1]}"]
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="One or more *.jsonl result files")
    ap.add_argument("--output", default="results/paper_table",
                    help="Stem; will produce <output>.csv / .md / .tex")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=f"Dataset order. Default: {DEFAULT_DATASETS}")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="One metric label per dataset (same length as --datasets)."
                         " Default: ACC/ACC/F1/F1/ROUGE/ROUGE")
    ap.add_argument("--decimals", type=int, default=2)
    args = ap.parse_args()

    datasets = args.datasets or list(DEFAULT_DATASETS)
    if args.labels:
        if len(args.labels) != len(datasets):
            print(f"--labels must have same length as --datasets ({len(datasets)})",
                  file=sys.stderr)
            sys.exit(1)
        labels = dict(zip(datasets, args.labels))
    else:
        labels = {d: DEFAULT_METRIC_LABELS.get(d, "Score") for d in datasets}

    rows = _load_rows(args.inputs)
    models, cells = _flat_table(rows, datasets, labels, args.decimals)
    if not models:
        print("no valid rows", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    csv_path = args.output + ".csv"
    md_path = args.output + ".md"
    tex_path = args.output + ".tex"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(_format_csv(models, cells, datasets, labels))
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_format_markdown(models, cells, datasets, labels))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(_format_latex(models, cells, datasets, labels))

    print(f"Wrote:")
    print(f"  {csv_path}")
    print(f"  {md_path}")
    print(f"  {tex_path}")
    print()
    print("=== Markdown preview ===")
    print(_format_markdown(models, cells, datasets, labels))


if __name__ == "__main__":
    main()
