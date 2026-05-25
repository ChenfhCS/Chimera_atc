#!/usr/bin/env python
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--output_csv', default='results/summary.csv')
    args = ap.parse_args()
    rows=[]
    for path in args.inputs:
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    df=pd.DataFrame(rows)
    pivot=df.pivot_table(index='model', columns='dataset', values=['score','tpt'], aggfunc='mean')
    print(pivot)
    pivot.to_csv(args.output_csv)

if __name__=='__main__': main()
