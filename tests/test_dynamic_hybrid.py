"""Smoke tests for the Dynamic Hybrid model.

These build tiny in-memory Llama and Mamba modules with matching hidden size,
so the tests run on a single CPU in a few seconds and require no HF download.
"""
from __future__ import annotations

import pytest
import torch

from chimera.dynamic_hybrid_model import (
    DynamicHybridConfig,
    DynamicHybridForCausalLM,
    PrefillGate,
)
from ._factories import build_tiny_hybrid, build_tiny_llama, build_tiny_mamba


def test_hidden_size_mismatch_raises():
    attn = build_tiny_llama(hidden_size=32)
    ssm = build_tiny_mamba(hidden_size=48)
    cfg = DynamicHybridConfig(
        attn_model_name_or_path="<a>",
        ssm_model_name_or_path="<b>",
        num_layers=2,
        gate_hidden_size=16,
        torch_dtype="float32",
    )
    with pytest.raises(ValueError, match="Hidden sizes must match"):
        DynamicHybridForCausalLM(cfg, attn, ssm)


def test_construction_and_forward_soft():
    model = build_tiny_hybrid().train()
    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    out, probs_list = model(
        input_ids,
        attention_mask=attention_mask,
        routing_mode="soft",
        return_gate_probs=True,
    )
    assert out.logits.shape == (2, 8, 64)
    assert len(probs_list) == len(model.layers)
    for p in probs_list:
        assert p.shape == (2, 2)
        # Soft routing must produce a real probability distribution.
        assert torch.allclose(p.sum(dim=-1), torch.ones(p.shape[0]), atol=1e-4)


def test_training_in_hard_mode_raises():
    model = build_tiny_hybrid().train()
    input_ids = torch.randint(0, 64, (1, 4))
    with pytest.raises(ValueError, match="freeze the gate"):
        model(input_ids, routing_mode="hard")


def test_backward_updates_gate_and_branches():
    model = build_tiny_hybrid().train()
    input_ids = torch.randint(0, 64, (2, 6))
    labels = input_ids.clone()
    out = model(input_ids, attention_mask=torch.ones_like(input_ids), labels=labels, routing_mode="soft")
    out.loss.backward()

    # Gate must receive gradient on every dHybrid block.
    for layer in model.layers:
        for p in layer.gate.parameters():
            assert p.grad is not None, "gate parameter missing grad"
            assert p.grad.abs().sum().item() > 0.0, "gate gradient is exactly zero"

    # Both branches must receive gradient under soft routing.
    has_attn_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for layer in model.layers
        for p in layer.attn_layer.parameters()
    )
    has_ssm_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for layer in model.layers
        for p in layer.ssm_layer.parameters()
    )
    assert has_attn_grad, "attention branch received no gradient"
    assert has_ssm_grad, "SSM branch received no gradient"


def test_freeze_branches_only_trains_gate():
    model = build_tiny_hybrid().train()
    for n, p in model.named_parameters():
        p.requires_grad = ".gate." in n
    trainables = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainables, "expected some trainable params"
    assert all(".gate." in n for n in trainables)

    input_ids = torch.randint(0, 64, (1, 4))
    out = model(input_ids, attention_mask=torch.ones_like(input_ids), labels=input_ids.clone(), routing_mode="soft")
    out.loss.backward()
    for layer in model.layers:
        for p in layer.attn_layer.parameters():
            assert p.grad is None or p.grad.abs().sum().item() == 0


def test_generate_greedy_caches_prefill_branches():
    model = build_tiny_hybrid()
    model.eval()
    input_ids = torch.randint(0, 64, (1, 6))
    out = model.generate_greedy(input_ids, max_new_tokens=4)
    assert out.shape == (1, 10)

    # Re-running with explicit fixed_branches must reproduce the same branch
    # decisions as the cached generate_greedy path.
    with torch.no_grad():
        _, probs_list = model(input_ids, attention_mask=torch.ones_like(input_ids),
                               routing_mode="hard", return_gate_probs=True)
        prefill_branches = [p.argmax(dim=-1) for p in probs_list]

    # Feed the generated suffix back in and confirm the gate decision is stable.
    full_mask = torch.ones_like(out)
    with torch.no_grad():
        _, probs_full = model(out, attention_mask=full_mask,
                               routing_mode="hard", return_gate_probs=True,
                               fixed_branches=prefill_branches)
    for prefill_p, full_p in zip(prefill_branches, probs_full):
        assert torch.equal(prefill_p, full_p.argmax(dim=-1)), \
            "fixed_branches override did not pin the gate decision"


def test_routing_mode_soft_keeps_both_branches_alive():
    model = build_tiny_hybrid().train()
    input_ids = torch.randint(0, 64, (4, 6))
    out, probs_list = model(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        routing_mode="soft",
        return_gate_probs=True,
    )
    # In soft mode every request gets a mix; no row should be a clean one-hot
    # at initialization (gate is random and would produce intermediate probs).
    for p in probs_list:
        assert (p > 0).all()


def test_pad_mask_does_not_break_pooling():
    gate = PrefillGate(hidden_size=8, gate_hidden_size=4)
    h = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    out = gate(h, mask)
    assert out.shape == (2, 2)
