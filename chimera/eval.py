from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import EvalExample
from .metrics import exact_choice_accuracy, mean_f1, mean_rouge_l
from .utils import Timer


# Per-dataset recommended generation budgets. ARC/PIQA/TruthfulQA only emit a
# single letter so 4 tokens is plenty. Summarization needs ~reference length;
# anything shorter caps ROUGE-L recall (e.g. 32 tokens vs. a 700-token
# GovReport reference caps F1 at ~4-5).
DATASET_DEFAULT_MAX_NEW_TOKENS = {
    "ARC-e": 8,
    "ARC-c": 8,
    "PIQA": 8,
    "TruthfulQA": 8,
    "CNN/DM": 128,
    "GovReport": 512,
    "XSum": 64,
    "MultiNews": 256,
    "arXiv": 256,
    "PubMed": 256,
}


def load_hf_model_and_tokenizer(model_name_or_path: str, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}.get(torch_dtype, torch.bfloat16)
    kwargs = dict(device_map=device_map, trust_remote_code=trust_remote_code)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def resolve_max_new_tokens(dataset_name: str, override: Optional[int]) -> int:
    """Pick a generation budget. Explicit override wins; otherwise fall back
    to the per-dataset default so summarization isn't capped at 32 tokens."""
    if override is not None:
        return int(override)
    return DATASET_DEFAULT_MAX_NEW_TOKENS.get(dataset_name, 32)


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: List[str], max_new_tokens=32, batch_size=1, is_dynamic=False,
                   input_max_tokens: int = 4096):
    preds = []
    total_new = 0
    total_time = 0.0
    device = next(model.parameters()).device
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=input_max_tokens).to(device)
        input_len = enc.input_ids.shape[1]
        with Timer() as t:
            if is_dynamic and hasattr(model, "generate_greedy"):
                out = model.generate_greedy(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
            else:
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        total_time += t.elapsed
        total_new += int(out.shape[1] - input_len) * len(batch)
        gen = out[:, input_len:]
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    tps = total_new / max(total_time, 1e-9)
    return preds, tps


def evaluate_examples(model, tokenizer, examples: List[EvalExample], max_new_tokens=None, batch_size=1,
                      metric="auto", is_dynamic=False, input_max_tokens: int = 4096) -> Dict:
    ds_name = examples[0].dataset if examples else "unknown"
    effective_new_tokens = resolve_max_new_tokens(ds_name, max_new_tokens)
    preds, tps = generate_texts(model, tokenizer, [x.prompt for x in examples], effective_new_tokens,
                                batch_size, is_dynamic, input_max_tokens=input_max_tokens)
    labels = [x.target for x in examples]
    ds = ds_name
    if metric == "auto":
        # Prompts emit a single A/B/C/D letter for ARC, PIQA, and TruthfulQA,
        # so first-character accuracy is the only sensible default. F1 over a
        # one-token answer collapses to exact match anyway.
        if ds in {"ARC-e", "ARC-c", "PIQA", "TruthfulQA"}:
            metric = "acc"
        else:
            metric = "rouge"
    if metric == "acc":
        score = exact_choice_accuracy(preds, labels)
        key = "ACC"
    elif metric == "f1":
        score = mean_f1(preds, labels)
        key = "F1"
    else:
        score = mean_rouge_l(preds, labels)
        key = "ROUGE-L"
    return {
        "dataset": ds,
        "metric": key,
        "score": score,
        "tpt": tps,
        "num_samples": len(examples),
        "max_new_tokens": effective_new_tokens,
        "input_max_tokens": input_max_tokens,
    }
