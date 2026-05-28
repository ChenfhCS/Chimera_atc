#!/usr/bin/env python
"""Aggregate one or more *.jsonl result files into a flat CSV (+ markdown).

Output schema:
    model, score_<dataset>, ..., tpt_<dataset>, ..., flops_per_token_<dataset>, ...

* ``score`` and ``tpt`` are rounded to ``--decimals`` (default 2).
* ``flops_per_token`` is always rendered in 2-decimal scientific notation
  (e.g. ``6.00e+09``) since the raw integers are unwieldy.
Rows missing a (model, dataset) cell are written as empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


# Field-specific column ordering and formatter. Add to FLOAT_METRICS for
# decimal-rounded fields; add to SCI_METRICS for scientific-notation fields.
FLOAT_METRICS = ("score", "tpt")
SCI_METRICS = ("flops_per_token",)
ALL_METRICS = FLOAT_METRICS + SCI_METRICS


def _load_rows(paths):
    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _drop_errors(rows):
    keep = []
    for r in rows:
        if "error" in r:
            continue
        if r.get("score") is None or r.get("dataset") is None or r.get("model") is None:
            continue
        keep.append(r)
    return keep


def _format_sci(v, decimals: int = 2) -> str:
    """Render a float in NN.MMe+EE notation. Empty cell stays empty."""
    if v is None or pd.isna(v):
        return ""
    return f"{float(v):.{decimals}e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output_csv", default="results/summary.csv")
    ap.add_argument("--output_md", default=None,
                    help="Optional markdown twin of the CSV. Default: <csv>.md")
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--sci_decimals", type=int, default=2,
                    help="Decimal places inside the scientific-notation cells "
                         "(default 2 → e.g. 6.00e+09).")
    args = ap.parse_args()

    rows = _drop_errors(_load_rows(args.inputs))
    if not rows:
        print("no valid result rows found")
        sys.exit(1)
    df = pd.DataFrame(rows)

    # Decide which metrics actually appear in the data so we don't reference
    # missing columns (e.g. old jsonl that pre-dates flops_per_token).
    present_metrics = [m for m in ALL_METRICS if m in df.columns]
    if not present_metrics:
        print("input rows have no recognized metric columns (score/tpt/flops_per_token)")
        sys.exit(1)

    pivot = df.pivot_table(
        index="model", columns="dataset", values=present_metrics, aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{ds}" for metric, ds in pivot.columns]

    # Stable column order: score_* first, then tpt_*, then flops_per_token_*.
    datasets = sorted({c.split("_", 1)[1] for c in pivot.columns
                       if c.split("_", 1)[1] in df["dataset"].unique()})
    ordered = []
    for metric in present_metrics:
        for d in datasets:
            col = f"{metric}_{d}"
            if col in pivot.columns:
                ordered.append(col)
    pivot = pivot[ordered].sort_index()

    # Build a display copy: round score/tpt; format flops_per_token in sci.
    display = pivot.copy()
    for col in display.columns:
        metric = col.split("_", 1)[0]
        if metric == "flops_per_token":
            display[col] = display[col].apply(
                lambda v: _format_sci(v, args.sci_decimals)
            )
        else:
            display[col] = display[col].round(args.decimals)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    display.to_csv(args.output_csv)
    print(f"wrote {args.output_csv} ({display.shape[0]} models x {display.shape[1]} columns)")

    with pd.option_context("display.max_columns", None,
                           "display.width", 240,
                           "display.float_format", lambda v: f"{v:.{args.decimals}f}"):
        print(display)

    md_path = args.output_md or (args.output_csv.rsplit(".", 1)[0] + ".md")
    try:
        md = display.to_markdown()
    except Exception:
        md = display.reset_index().to_string(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
