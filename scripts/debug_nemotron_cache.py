#!/usr/bin/env python
"""Standalone smoke test for the Nemotron-H hybrid cache path.

Runs a single greedy decode of 4 new tokens via the manual loop, with full
traceback on failure. Use this to iterate on Nemotron-H specific bugs
without going through the multi-dataset baseline runner.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from chimera.eval import (  # noqa: E402
    _build_legacy_hybrid_cache,
    _manual_greedy_with_cache,
    load_hf_model_and_tokenizer,
)

REPO = "nvidia/Nemotron-H-4B-Instruct-128K"
PROMPT = (
    "Question: What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
)


def main():
    print(f"Loading {REPO} ...")
    model, tok = load_hf_model_and_tokenizer(REPO)
    device = next(model.parameters()).device
    print(f"device={device}, dtype={next(model.parameters()).dtype}")
    print(f"model class={model.__class__.__name__}, module={model.__class__.__module__}")

    enc = tok(PROMPT, return_tensors="pt").to(device)
    print(f"input_ids.shape={tuple(enc.input_ids.shape)}")

    cache = _build_legacy_hybrid_cache(
        model, batch_size=enc.input_ids.shape[0],
        max_len=enc.input_ids.shape[1] + 8,
    )
    print(f"cache type={type(cache).__name__}")
    print(f"cache attrs={[a for a in dir(cache) if not a.startswith('_')]}")

    try:
        out = _manual_greedy_with_cache(
            model,
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            max_new_tokens=4,
            eos_token_id=tok.eos_token_id,
            cache=cache,
        )
        print(f"out shape={tuple(out.shape)}")
        print("generated:", repr(tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)))
        print("OK")
    except Exception:
        print("\n--- FAILURE TRACEBACK ---")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
