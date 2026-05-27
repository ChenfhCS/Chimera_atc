#!/usr/bin/env python
"""Dump the actual prompt + generated output for a single sample.

Usage:
  python scripts/debug_generation.py --model_path nvidia/Nemotron-H-4B-Instruct-128K \
                                     --dataset pubmed \
                                     --max_new_tokens 128 \
                                     --input_max_tokens 2048
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from chimera.data import load_eval_dataset
from chimera.eval import (
    _is_nemotron_h,
    _manual_greedy_with_cache,
    _build_legacy_hybrid_cache,
    _maybe_apply_chat_template,
    load_hf_model_and_tokenizer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dataset", required=True,
                    help="e.g. pubmed | cnn/dm | arc-e")
    ap.add_argument("--sample_idx", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--input_max_tokens", type=int, default=2048)
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--torch_dtype", default="bfloat16")
    args = ap.parse_args()

    print(f"loading {args.model_path} ...")
    model, tok = load_hf_model_and_tokenizer(
        args.model_path, torch_dtype=args.torch_dtype, device_map="auto",
        trust_remote_code=True,
    )

    print(f"loading dataset {args.dataset} (sample idx={args.sample_idx}) ...")
    examples = load_eval_dataset(args.dataset, split="validation[:32]")
    ex = examples[args.sample_idx]

    raw_prompt = ex.prompt
    # Mirror the production eval path: multi-choice datasets emit a system
    # directive on chat models telling them to output only the letter.
    is_mc = ex.dataset in {"ARC-e", "ARC-c", "PIQA", "TruthfulQA"}
    formatted = _maybe_apply_chat_template(
        tok, raw_prompt, apply=not args.no_chat_template,
        multi_choice_hint=is_mc,
    )

    print("=" * 80)
    print(f"chat_template present: {bool(tok.chat_template)}")
    print(f"eos_token_id: {tok.eos_token_id}  (token: {tok.decode([tok.eos_token_id]) if tok.eos_token_id is not None else 'None'!r})")
    print(f"special tokens: {tok.all_special_tokens[:20]}")
    print("=" * 80)
    print("=== FORMATTED PROMPT (last 800 chars) ===")
    print(formatted[-800:])
    print("=" * 80)
    print("=== REFERENCE TARGET ===")
    print(ex.target)
    print("=" * 80)

    enc = tok(formatted, return_tensors="pt", truncation=True,
              max_length=args.input_max_tokens).to(next(model.parameters()).device)
    input_len = enc.input_ids.shape[1]
    print(f"input tokens: {input_len}")

    if _is_nemotron_h(model):
        # Match the production eval path: Nemotron-H's released custom
        # modeling does not implement an incremental decode (every forward
        # call re-prefills via mamba_chunk_scan_combined with cache_init=
        # True), so manual cache loops produce gibberish. The only path that
        # yields correct text is HF generate with use_cache=False.
        print("[hybrid-cache] use_cache=False fallback")
        out_ids = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=False,
        )
    else:
        out_ids = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, use_cache=True,
        )
    gen_ids = out_ids[0, input_len:].tolist()

    print("=" * 80)
    print("=== RAW GENERATED IDS ===")
    print(gen_ids)
    print()
    print("=== DECODED (skip_special_tokens=True) ===")
    print(tok.decode(gen_ids, skip_special_tokens=True))
    print()
    print("=== DECODED (skip_special_tokens=False) ===")
    print(tok.decode(gen_ids, skip_special_tokens=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
