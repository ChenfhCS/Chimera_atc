"""Test factories that build a tiny Dynamic Hybrid model without HF downloads."""
from __future__ import annotations

import torch
from transformers import LlamaConfig, LlamaForCausalLM, MambaConfig, MambaForCausalLM

from chimera.dynamic_hybrid_model import DynamicHybridConfig, DynamicHybridForCausalLM


def build_tiny_llama(hidden_size: int = 32, num_layers: int = 2, vocab_size: int = 64) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(cfg)


def build_tiny_mamba(hidden_size: int = 32, num_layers: int = 2, vocab_size: int = 64) -> MambaForCausalLM:
    cfg = MambaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        state_size=4,
        conv_kernel=2,
        expand=2,
        intermediate_size=hidden_size * 2,
    )
    return MambaForCausalLM(cfg)


def build_tiny_hybrid(
    hidden_size: int = 32,
    num_layers: int = 2,
    vocab_size: int = 64,
    seed: int = 0,
) -> DynamicHybridForCausalLM:
    torch.manual_seed(seed)
    attn = build_tiny_llama(hidden_size, num_layers, vocab_size)
    ssm = build_tiny_mamba(hidden_size, num_layers, vocab_size)
    cfg = DynamicHybridConfig(
        attn_model_name_or_path="<tiny-llama>",
        ssm_model_name_or_path="<tiny-mamba>",
        num_layers=num_layers,
        gate_hidden_size=16,
        routing_mode="soft",
        torch_dtype="float32",
        trust_remote_code=False,
    )
    return DynamicHybridForCausalLM(cfg, attn, ssm)
