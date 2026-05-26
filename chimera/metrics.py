from __future__ import annotations

import re
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


# Patterns used by extract_choice_letter, tried in order.
# Capture group 1 must be the letter.
_LETTER_PATTERNS = [
    r"\banswer\s*(?:is|:)\s*[\*\(\[\{]*\s*([A-Z])\b",   # "the answer is A", "Answer: B"
    r"\b(?:option|choice)\s*[\*\(\[\{]*\s*([A-Z])\b",   # "option A", "choice (B)"
    r"^\s*[\*\(\[\{]*\s*([A-Z])\s*[\)\]\}\.\:\,\*]",    # leading "A.", "A)", "**B**"
    r"^\s*[\*\(\[\{]*\s*([A-Z])\s*$",                    # whole output is just "A"
    r"\b([A-Z])\)",                                      # "A)" anywhere
    r"\(([A-Z])\)",                                      # "(A)" anywhere
    r"\b([A-Z])\b",                                      # first standalone uppercase letter
]


def extract_choice_letter(text: str, valid_letters: str = "ABCDEFGH") -> str:
    """Extract the model's chosen letter from a free-form prediction.

    Handles both bare outputs ("A") and instruct-style outputs
    ("The answer is **A)**", "Option A is correct", "(A) lava").
    Returns "" if no letter in ``valid_letters`` can be found.
    """
    if not text:
        return ""
    text_upper = text.strip().upper()
    valid = set(valid_letters)
    for pattern in _LETTER_PATTERNS:
        for m in re.finditer(pattern, text_upper):
            letter = m.group(1)
            if letter in valid:
                return letter
    return ""


def exact_choice_accuracy(preds: List[str], labels: List[str]) -> float:
    """Accuracy for letter-answer tasks (ARC, PIQA, TruthfulQA).

    Robust to instruct/chat outputs that surround the letter with preamble
    text ("The answer is **A**"), markdown, or parentheses. Falls back to
    "no letter found" -> wrong.
    """
    ok = 0
    for p, y in zip(preds, labels):
        gold = y.strip().upper()
        gold_letter = gold[0] if gold else ""
        pred_letter = extract_choice_letter(p)
        ok += int(pred_letter == gold_letter and pred_letter != "")
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
