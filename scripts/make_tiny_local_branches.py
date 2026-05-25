#!/usr/bin/env python
"""Materialize a tiny Llama and tiny Mamba checkpoint pair on disk.

This lets the build / train / eval scripts run end-to-end without hitting the
HuggingFace Hub. Hidden sizes are forced to match so ``DynamicHybridForCausalLM``
construction succeeds.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
)


def make_llama(out: str, hidden_size: int, num_layers: int, vocab_size: int, donor_tok: str):
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg)
    model.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(donor_tok, use_fast=True)
    tok.save_pretrained(out)
    return out


def make_mamba(out: str, hidden_size: int, num_layers: int, vocab_size: int, donor_tok: str):
    cfg = MambaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        state_size=4,
        conv_kernel=2,
        expand=2,
        intermediate_size=hidden_size * 2,
    )
    model = MambaForCausalLM(cfg)
    model.save_pretrained(out)
    tok = AutoTokenizer.from_pretrained(donor_tok, use_fast=True)
    tok.save_pretrained(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints/tiny", help="root dir for the two local checkpoints")
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--vocab_size", type=int, default=32000)
    ap.add_argument(
        "--donor_tokenizer",
        default="hf-internal-testing/llama-tokenizer",
        help="Any small HF tokenizer that fits vocab_size; defaults to the public llama test tokenizer.",
    )
    args = ap.parse_args()

    os.makedirs(args.root, exist_ok=True)
    attn_dir = os.path.join(args.root, "attn")
    ssm_dir = os.path.join(args.root, "ssm")
    torch.manual_seed(0)
    make_llama(attn_dir, args.hidden_size, args.num_layers, args.vocab_size, args.donor_tokenizer)
    make_mamba(ssm_dir, args.hidden_size, args.num_layers, args.vocab_size, args.donor_tokenizer)
    print("Wrote", attn_dir, "and", ssm_dir)


if __name__ == "__main__":
    main()
