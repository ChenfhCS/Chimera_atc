#!/usr/bin/env python
"""Aggregate one or more *.jsonl result files into a flat CSV (+ markdown).

Output schema:
    model, score_<dataset>, ..., tpt_<dataset>, ...

Numbers are rounded to 2 decimals. Rows missing a (model, dataset) cell are
written as empty.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output_csv", default="results/summary.csv")
    ap.add_argument("--output_md", default=None,
                    help="Optional markdown twin of the CSV. Default: <csv>.md")
    ap.add_argument("--decimals", type=int, default=2)
    args = ap.parse_args()

    rows = _drop_errors(_load_rows(args.inputs))
    if not rows:
        print("no valid result rows found")
        sys.exit(1)
    df = pd.DataFrame(rows)
    # Pivot to wide form, then flatten the MultiIndex to single-row headers.
    pivot = df.pivot_table(
        index="model", columns="dataset", values=["score", "tpt"], aggfunc="mean",
    )
    pivot.columns = [f"{metric}_{ds}" for metric, ds in pivot.columns]
    # Order columns so all score_* come first, then tpt_*, datasets alphabetical.
    datasets = sorted({c.split("_", 1)[1] for c in pivot.columns})
    ordered = [f"score_{d}" for d in datasets if f"score_{d}" in pivot.columns]
    ordered += [f"tpt_{d}" for d in datasets if f"tpt_{d}" in pivot.columns]
    pivot = pivot[ordered].sort_index()
    pivot = pivot.round(args.decimals)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    pivot.to_csv(args.output_csv)
    print(f"wrote {args.output_csv} ({pivot.shape[0]} models x {pivot.shape[1]} columns)")
    with pd.option_context("display.max_columns", None,
                           "display.width", 200,
                           "display.float_format", lambda v: f"{v:.{args.decimals}f}"):
        print(pivot)

    md_path = args.output_md or (args.output_csv.rsplit(".", 1)[0] + ".md")
    try:
        md = pivot.to_markdown(floatfmt=f".{args.decimals}f")
    except Exception:
        md = pivot.reset_index().to_string(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
