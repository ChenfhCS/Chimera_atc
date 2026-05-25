from __future__ import annotations

import re
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_choice_accuracy(preds: List[str], labels: List[str]) -> float:
    ok = 0
    for p, y in zip(preds, labels):
        p = p.strip().upper()
        # Accept either first label character or literal answer text label.
        pred = p[0] if p else ""
        ok += int(pred == y.strip().upper()[0])
    return 100.0 * ok / max(1, len(labels))


def f1_score(pred: str, target: str) -> float:
    p = normalize_answer(pred).split()
    t = normalize_answer(target).split()
    common = Counter(p) & Counter(t)
    num_same = sum(common.values())
    if len(p) == 0 or len(t) == 0:
        return float(p == t)
    if num_same == 0:
        return 0.0
    precision = num_same / len(p)
    recall = num_same / len(t)
    return 2 * precision * recall / (precision + recall)


def mean_f1(preds: List[str], labels: List[str]) -> float:
    return 100.0 * sum(f1_score(p, y) for p, y in zip(preds, labels)) / max(1, len(labels))


def rouge_l(pred: str, target: str) -> float:
    x = normalize_answer(pred).split()
    y = normalize_answer(target).split()
    if not x or not y:
        return 0.0
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if x[i-1] == y[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[-1][-1]
    r = lcs / len(y)
    p = lcs / len(x)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def mean_rouge_l(preds: List[str], labels: List[str]) -> float:
    return 100.0 * sum(rouge_l(p, y) for p, y in zip(preds, labels)) / max(1, len(labels))
