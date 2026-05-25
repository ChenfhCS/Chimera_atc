#!/usr/bin/env python
"""End-to-end smoke runner for the Dynamic Hybrid model.

Builds a tiny Llama + tiny Mamba pair in memory (no HF downloads), then runs
forward, backward, and greedy generation. Useful when ``pytest`` is not
installed.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from tests._factories import build_tiny_hybrid


def main() -> int:
    torch.manual_seed(0)
    model = build_tiny_hybrid(hidden_size=32, num_layers=2)
    print(model.trainable_parameter_summary())

    input_ids = torch.randint(0, 64, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    model.train()
    out = model(input_ids, attention_mask=attention_mask, labels=labels, routing_mode="soft")
    print("soft loss:", float(out.loss))
    out.loss.backward()
    print("backward OK")

    gate_grad = sum(
        float(p.grad.abs().sum())
        for layer in model.layers
        for p in layer.gate.parameters()
        if p.grad is not None
    )
    print("gate grad sum:", gate_grad)
    assert gate_grad > 0.0, "gate did not receive gradient"

    model.eval()
    gen = model.generate_greedy(input_ids[:1], max_new_tokens=4)
    print("generated shape:", tuple(gen.shape))
    assert gen.shape == (1, 12)
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
