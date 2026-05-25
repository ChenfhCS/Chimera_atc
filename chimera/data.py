from __future__ import annotations

import json
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Iterable, List, Optional

from datasets import load_dataset


@dataclass
class EvalExample:
    dataset: str
    prompt: str
    target: str
    choices: Optional[List[str]] = None
    answer_idx: Optional[int] = None


def _choice_prompt(question: str, choices: List[str]) -> str:
    labels = [chr(ord('A') + i) for i in range(len(choices))]
    opts = "\n".join([f"{l}. {c}" for l, c in zip(labels, choices)])
    return f"Question: {question}\n{opts}\nAnswer:"


def _load_with_fallback(
    repo_candidates: List[str],
    subset: Optional[str],
    split: str,
):
    """Try a list of HF repo IDs in order, returning the first that loads.

    The newer ``datasets`` library (>=3.0) refuses unqualified names like
    ``ai2_arc`` and requires ``namespace/name``. We therefore feed the
    canonical name first and keep the old short name as a backstop for older
    ``datasets`` versions.
    """
    last_err: Optional[Exception] = None
    for repo in repo_candidates:
        try:
            kwargs = dict(split=split)
            # ``trust_remote_code`` was added in datasets 2.16; older versions
            # raise TypeError if we pass it.
            try:
                if subset is not None:
                    return load_dataset(repo, subset, trust_remote_code=True, **kwargs)
                return load_dataset(repo, trust_remote_code=True, **kwargs)
            except TypeError:
                if subset is not None:
                    return load_dataset(repo, subset, **kwargs)
                return load_dataset(repo, **kwargs)
        except Exception as e:  # repo missing / script disabled / network
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load any of {repo_candidates} (subset={subset}, split={split}). "
        f"Last error: {last_err!r}"
    )


# ---------------- PIQA direct download (script-free fallback) ----------------

_PIQA_URLS = {
    "train": (
        "https://yonatanbisk.com/piqa/data/train.jsonl",
        "https://yonatanbisk.com/piqa/data/train-labels.lst",
    ),
    "validation": (
        "https://yonatanbisk.com/piqa/data/valid.jsonl",
        "https://yonatanbisk.com/piqa/data/valid-labels.lst",
    ),
    "valid": (
        "https://yonatanbisk.com/piqa/data/valid.jsonl",
        "https://yonatanbisk.com/piqa/data/valid-labels.lst",
    ),
}


def _piqa_from_official(split: str) -> List[dict]:
    """Download PIQA directly from the official URLs (parquet-free)."""
    if split not in _PIQA_URLS:
        raise ValueError(f"No PIQA URL for split={split!r}")
    jsonl_url, label_url = _PIQA_URLS[split]
    cache_dir = os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        "piqa_official",
    )
    os.makedirs(cache_dir, exist_ok=True)
    jsonl_path = os.path.join(cache_dir, os.path.basename(jsonl_url))
    label_path = os.path.join(cache_dir, os.path.basename(label_url))
    for url, path in [(jsonl_url, jsonl_path), (label_url, label_path)]:
        if not os.path.exists(path):
            tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir)
            os.close(tmp_fd)
            urllib.request.urlretrieve(url, tmp_path)
            os.replace(tmp_path, path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    assert len(rows) == len(labels), (len(rows), len(labels))
    for r, lab in zip(rows, labels):
        r["label"] = lab
    return rows


def _piqa_iter(split: str) -> Iterable[dict]:
    """PIQA loader: try parquet mirrors first, then direct official URLs."""
    candidates = ["ybisk/piqa", "piqa"]
    try:
        return _load_with_fallback(candidates, subset=None, split=split)
    except Exception as primary_err:
        try:
            return _piqa_from_official(split)
        except Exception as fallback_err:
            raise RuntimeError(
                "PIQA could not be loaded. Tried HuggingFace ("
                f"{primary_err!r}) and direct download ({fallback_err!r}). "
                "Either downgrade datasets to <4.0 (`pip install 'datasets<4.0'`) "
                "so script-based loaders work, or run from a host with outbound "
                "HTTPS to yonatanbisk.com."
            )


# ---------------- Public loader ----------------


def load_eval_dataset(name: str, split: str = "validation", max_samples: Optional[int] = None) -> List[EvalExample]:
    name_l = name.lower()
    rows: List[EvalExample] = []

    if name_l in {"arc-e", "arc_easy", "arce"}:
        ds = _load_with_fallback(["allenai/ai2_arc", "ai2_arc"], "ARC-Easy", split)
        for x in ds:
            choices = x["choices"]["text"]
            labels = x["choices"]["label"]
            answer_idx = labels.index(x["answerKey"]) if x["answerKey"] in labels \
                else ord(x["answerKey"].upper()) - ord('A')
            rows.append(EvalExample(
                "ARC-e", _choice_prompt(x["question"], choices),
                chr(ord('A') + answer_idx), choices, answer_idx,
            ))

    elif name_l in {"arc-c", "arc_challenge", "arcc"}:
        ds = _load_with_fallback(["allenai/ai2_arc", "ai2_arc"], "ARC-Challenge", split)
        for x in ds:
            choices = x["choices"]["text"]
            labels = x["choices"]["label"]
            answer_idx = labels.index(x["answerKey"]) if x["answerKey"] in labels \
                else ord(x["answerKey"].upper()) - ord('A')
            rows.append(EvalExample(
                "ARC-c", _choice_prompt(x["question"], choices),
                chr(ord('A') + answer_idx), choices, answer_idx,
            ))

    elif name_l == "piqa":
        ds = _piqa_iter(split)
        for x in ds:
            choices = [x["sol1"], x["sol2"]]
            label = int(x["label"])
            rows.append(EvalExample(
                "PIQA", _choice_prompt(x["goal"], choices),
                chr(ord('A') + label), choices, label,
            ))

    elif name_l == "truthfulqa":
        # truthful_qa "multiple_choice" only ships a "validation" split.
        tqa_split = "validation"
        ds = _load_with_fallback(
            ["truthfulqa/truthful_qa", "truthful_qa"], "multiple_choice", tqa_split,
        )
        for x in ds:
            choices = x["mc1_targets"]["choices"]
            labels = x["mc1_targets"]["labels"]
            answer_idx = labels.index(1) if 1 in labels else 0
            rows.append(EvalExample(
                "TruthfulQA", _choice_prompt(x["question"], choices),
                chr(ord('A') + answer_idx), choices, answer_idx,
            ))

    elif name_l in {"cnn/dm", "cnn_dm", "cnn_dailymail"}:
        ds = _load_with_fallback(
            ["abisee/cnn_dailymail", "ccdv/cnn_dailymail", "cnn_dailymail"],
            "3.0.0", split,
        )
        for x in ds:
            prompt = "Summarize the following article:\n" + x["article"] + "\nSummary:"
            rows.append(EvalExample("CNN/DM", prompt, x["highlights"]))

    elif name_l == "xsum":
        ds = _load_with_fallback(
            ["EdinburghNLP/xsum", "xsum"], None, split,
        )
        for x in ds:
            article = x.get("document") or x.get("article") or x.get("text")
            summary = x.get("summary")
            if article is None or summary is None:
                continue
            rows.append(EvalExample(
                "XSum",
                "Summarize the following article in a single sentence:\n"
                + article + "\nSummary:",
                summary,
            ))

    elif name_l in {"multinews", "multi_news", "multi-news"}:
        ds = _load_with_fallback(
            ["alexfabbri/multi_news", "multi_news"], None, split,
        )
        for x in ds:
            article = x.get("document") or x.get("article")
            summary = x.get("summary")
            if article is None or summary is None:
                continue
            rows.append(EvalExample(
                "MultiNews",
                "Summarize the following news cluster:\n" + article + "\nSummary:",
                summary,
            ))

    elif name_l == "govreport":
        ds = _load_with_fallback(
            ["ccdv/govreport-summarization", "launch/gov_report"], None, split,
        )
        for x in ds:
            article = x.get("report") or x.get("document") or x.get("text")
            summary = x.get("summary") or x.get("abstract")
            if article is None or summary is None:
                continue
            rows.append(EvalExample(
                "GovReport",
                "Summarize the following report:\n" + article + "\nSummary:",
                summary,
            ))

    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples is not None:
        rows = rows[:max_samples]
    return rows


def build_lm_training_texts(dataset_names: List[str], split: str = "train", max_samples_per_dataset: int = 1000) -> List[str]:
    texts = []
    for name in dataset_names:
        examples = load_eval_dataset(name, split=split, max_samples=max_samples_per_dataset)
        for ex in examples:
            texts.append(ex.prompt + " " + ex.target)
    return texts
