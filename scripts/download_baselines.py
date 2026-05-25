#!/usr/bin/env python
"""Pre-download baseline checkpoints listed in ``configs/baselines.yaml``.

Downloads via ``huggingface_hub.snapshot_download`` to the default HF cache so
that ``evaluate_baselines.py`` doesn't have to fight slow connections mid-run.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download  # noqa: E402

from chimera.utils import read_yaml  # noqa: E402


# Files we do NOT need at eval time. Dropping them cuts download size by ~half
# for many repos that ship both .bin and .safetensors.
DEFAULT_IGNORE = [
    "*.bin",                # prefer safetensors when both exist
    "*.pt",
    "consolidated.*",
    "*.gguf",
    "original/*",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Optional whitelist of model names from the config.")
    ap.add_argument("--include_bin", action="store_true",
                    help="Also pull pytorch_model.bin shards (off by default).")
    ap.add_argument("--token_env", default="HF_TOKEN",
                    help="Env var holding the HuggingFace token (default: HF_TOKEN).")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    wanted = set(args.models) if args.models else None
    token = os.environ.get(args.token_env)
    ignore = [] if args.include_bin else list(DEFAULT_IGNORE)

    for m in cfg["models"]:
        if wanted is not None and m["name"] not in wanted:
            continue
        print(f"[download] {m['name']} <- {m['path']}")
        try:
            local = snapshot_download(
                repo_id=m["path"],
                token=token,
                ignore_patterns=ignore,
                resume_download=True,
            )
            print(f"           -> {local}")
        except Exception as e:
            print(f"           FAILED: {e}")


if __name__ == "__main__":
    main()
