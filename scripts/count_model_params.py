#!/usr/bin/env python
"""Summarize each baseline's architectural footprint:

    Param.   |   Act. Param.   |   N_attn   |   N_ssm

Loads each model on the ``meta`` device (no weights downloaded, no GPU
memory used — just shape/structure). Counts:

  * Total params: sum of p.numel() across all modules.
  * Active params: for dense models == total. For MoE models (Jamba),
    subtract inactive-expert weights based on
    ``num_experts`` / ``num_experts_per_tok`` from config.
  * N_attn / N_ssm: per architecture handler — Llama, Mamba/Mamba2, Jamba,
    Zamba2, Nemotron-H (uses ``hybrid_override_pattern`` like "MMAMM...").

Reads the same baselines.yaml the eval scripts use, or accepts an
explicit list with ``--models``.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from chimera.utils import read_yaml  # noqa: E402


def _fmt_b(n):
    if n is None or n < 0:
        return "?"
    return f"{n / 1e9:.2f}B"


def _count_layer_types(cfg, arch_name: str):
    """Return (n_attn, n_ssm). -1 means unknown for that architecture."""
    name = (arch_name or "").lower()

    # Pure attention families.
    if any(k in name for k in ("llama", "mistral", "qwen", "phi", "gpt", "falcon3")):
        return getattr(cfg, "num_hidden_layers", -1), 0

    # Pure SSM families.
    if "mamba" in name and "jamba" not in name and "nemotron" not in name and "zamba" not in name:
        return 0, getattr(cfg, "num_hidden_layers", -1)

    # Jamba: periodic attention + MoE mamba.
    if "jamba" in name:
        total = getattr(cfg, "num_hidden_layers", 0)
        period = getattr(cfg, "attn_layer_period", 0)
        offset = getattr(cfg, "attn_layer_offset", 0)
        if not period:
            return -1, -1
        n_attn = sum(1 for i in range(total) if (i - offset) % period == 0)
        return n_attn, total - n_attn

    # Zamba2: layers_block_type list of "mamba"/"hybrid"/etc.
    if "zamba" in name:
        types = getattr(cfg, "layers_block_type", None)
        if types:
            n_attn = sum(1 for t in types if "hybrid" in str(t).lower() or "attention" in str(t).lower())
            n_ssm = sum(1 for t in types if "mamba" in str(t).lower())
            return n_attn, n_ssm
        return -1, -1

    # Nemotron-H / Nemotron-Nano: hybrid_override_pattern of "M"/"A"/"*".
    if "nemotron" in name:
        pattern = getattr(cfg, "hybrid_override_pattern", "") or ""
        if pattern:
            return pattern.count("A"), pattern.count("M")
        # Some Nemotron-Nano configs use lists of strings instead.
        layer_types = getattr(cfg, "layer_types", None)
        if isinstance(layer_types, (list, tuple)):
            n_attn = sum(1 for t in layer_types if "attention" in str(t).lower())
            n_ssm = sum(1 for t in layer_types if "mamba" in str(t).lower() or "ssm" in str(t).lower())
            return n_attn, n_ssm
        return -1, -1

    return -1, -1


def _compute_params(model, cfg, arch_name: str):
    """Return (total_params, active_params)."""
    total = sum(p.numel() for p in model.parameters())
    num_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    num_active = (
        getattr(cfg, "num_experts_per_tok", None)
        or getattr(cfg, "num_experts_active", None)
    )
    if not num_experts or not num_active or num_active >= num_experts:
        return total, total

    expert_params = 0
    for name, p in model.named_parameters():
        nl = name.lower()
        if ("expert" in nl and "router" not in nl) or "moe.experts" in nl:
            expert_params += p.numel()
    if expert_params == 0:
        return total, total

    inactive_experts = num_experts - num_active
    inactive_params = expert_params * inactive_experts / num_experts
    return total, int(total - inactive_params)


def analyze(name: str, path: str):
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    arch = cfg.__class__.__name__
    n_attn, n_ssm = _count_layer_types(cfg, arch)

    total = active = None
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
        total, active = _compute_params(model, cfg, arch)
        # Some architectures (e.g. Jamba) report MoE via class introspection
        # even if config attrs are missing — fall back to introspection.
    except Exception as e:
        print(f"  [{name}] meta load failed: {type(e).__name__}: {e}", file=sys.stderr)

    return {
        "model": name,
        "path": path,
        "arch": arch,
        "total_params": total,
        "active_params": active,
        "n_attn": n_attn,
        "n_ssm": n_ssm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Optional whitelist of model 'name' fields.")
    ap.add_argument("--output", default="results/model_config",
                    help="Stem; writes <output>.csv and <output>.md")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    wanted = set(args.models) if args.models else None
    rows = []
    for m in cfg["models"]:
        if wanted and m["name"] not in wanted:
            continue
        print(f"analyzing {m['name']} ({m['path']}) ...")
        try:
            rows.append(analyze(m["name"], m["path"]))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            rows.append({
                "model": m["name"], "path": m["path"], "arch": "?",
                "total_params": None, "active_params": None,
                "n_attn": -1, "n_ssm": -1,
            })

    # Render
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    csv_path = args.output + ".csv"
    md_path = args.output + ".md"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,arch,Param.,Act. Param.,N_attn,N_ssm\n")
        for r in rows:
            total_s = _fmt_b(r["total_params"])
            active_s = _fmt_b(r["active_params"])
            na = "-" if r["n_attn"] in (0, -1) else str(r["n_attn"])
            ns = "-" if r["n_ssm"] in (0, -1) else str(r["n_ssm"])
            f.write(f'{r["model"]},{r["arch"]},{total_s},{active_s},{na},{ns}\n')

    # Markdown: mimic the paper layout.
    md = ["| model | arch | Param. | Act. Param. | $N_\\text{attn}$ | $N_\\text{ssm}$ |",
          "|:---|:---|:---:|:---:|:---:|:---:|"]
    for r in rows:
        total_s = _fmt_b(r["total_params"])
        active_s = _fmt_b(r["active_params"])
        na = "-" if r["n_attn"] in (0, -1) else str(r["n_attn"])
        ns = "-" if r["n_ssm"] in (0, -1) else str(r["n_ssm"])
        md.append(f'| {r["model"]} | {r["arch"]} | {total_s} | {active_s} | {na} | {ns} |')
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print()
    print("=== Markdown preview ===")
    print("\n".join(md))
    print()
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
