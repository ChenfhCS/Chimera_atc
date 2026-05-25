#!/usr/bin/env python
"""Sanity-check every baseline before launching a long eval run.

For each model:
  * loads the tokenizer and config (no weights)
  * estimates VRAM at the configured dtype
  * flags missing trust_remote_code, gated access, etc.

Run this once after ``download_baselines.py`` to fail fast on broken repos.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoConfig, AutoTokenizer  # noqa: E402

from chimera.utils import read_yaml  # noqa: E402


_DTYPE_BYTES = {"float32": 4, "fp32": 4, "float16": 2, "fp16": 2, "bfloat16": 2, "bf16": 2}


def _estimate_params(cfg) -> int:
    """Coarse upper bound on parameter count from a HF config object."""
    for key in ("num_parameters", "n_params"):
        if hasattr(cfg, key):
            return int(getattr(cfg, key))
    h = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or 0
    L = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layers", None) or 0
    V = getattr(cfg, "vocab_size", 0) or 0
    # 12 * h^2 per transformer block is the standard Kaplan estimate.
    return int(12 * (h ** 2) * L + 2 * V * h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--models", nargs="+", default=None)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    wanted = set(args.models) if args.models else None
    top = cfg

    for m in cfg["models"]:
        if wanted is not None and m["name"] not in wanted:
            continue
        name, path = m["name"], m["path"]
        dtype = m.get("torch_dtype") or top.get("torch_dtype", "bfloat16")
        trust = m.get("trust_remote_code")
        if trust is None:
            trust = top.get("trust_remote_code", True)
        bytes_per = _DTYPE_BYTES.get(dtype.lower(), 2)
        print(f"\n--- {name} ({path}) ---")
        try:
            hf_cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust)
            params = _estimate_params(hf_cfg)
            gb = params * bytes_per / 1e9
            print(f"  arch        : {hf_cfg.__class__.__name__}")
            print(f"  params (est): {params/1e9:.2f}B")
            print(f"  VRAM @{dtype}: ~{gb:.1f} GB  (weights only)")
        except Exception as e:
            print(f"  CONFIG LOAD FAILED: {e}")
            continue
        try:
            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust)
            print(f"  tokenizer   : {tok.__class__.__name__} (vocab={tok.vocab_size})")
        except Exception as e:
            print(f"  TOKENIZER LOAD FAILED: {e}")


if __name__ == "__main__":
    main()
