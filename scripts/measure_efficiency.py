#!/usr/bin/env python
"""Architecture-fair efficiency metrics for the baselines.

Throughput (tokens/sec) depends on implementation quirks (KV cache support,
fused kernels, scheduling) and is not directly comparable across
architectures. This script reports two architecture-neutral numbers:

  * decode_TFLOPs_per_token  : theoretical FLOPs for one decode step
                              (with KV / SSM cache) at a given context.
  * prefill_TFLOPs           : theoretical FLOPs to process the full prompt.

Combined with the eval scores it also computes:

  * Score per TFLOP/token    : accuracy normalized by per-token compute
                              budget. Higher = more compute-efficient.

Theoretical FLOPs follow the standard formula:

    transformer decode flops/token ≈ 2 * P_active + 4 * N_attn * d_model * L
    Mamba     decode flops/token   ≈ 2 * P_active    (SSM update is O(1))
    hybrid                          ≈ 2 * P_active + 4 * N_attn * d_model * L

    transformer prefill flops      ≈ 2 * P_active * L
                                    + 2 * N_attn * n_heads * head_dim * L^2
    Mamba prefill flops            ≈ 2 * P_active * L   (linear in L)

The 4 * N_attn * d * L term is the KV read cost for the attention layers
during decode. For pure SSM models N_attn = 0 so this term vanishes; for
hybrids it scales with the number of attention layers only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

from chimera.utils import read_yaml  # noqa: E402


# Hardware peak BF16 throughput. Adjust if running on a different GPU.
H800_PEAK_BF16 = 989e12  # 989 TFLOPS dense, NVIDIA H800 spec
A100_PEAK_BF16 = 312e12

# Per-dataset typical context = avg prompt tokens + half of max_new_tokens.
# These match the budgets we used in evaluate_baselines.
DATASET_CONTEXT = {
    "ARC-e": 96,
    "ARC-c": 96,
    "PIQA": 96,
    "TruthfulQA": 96,
    "CNN/DM": 800,
    "PubMed": 2200,
}


def _safe_get(cfg, *names, default=0):
    for name in names:
        v = getattr(cfg, name, None)
        if v is not None:
            return v
    return default


def _arch_info(cfg) -> Dict:
    arch = cfg.__class__.__name__
    hidden = _safe_get(cfg, "hidden_size", "d_model")
    n_layers = _safe_get(cfg, "num_hidden_layers", "n_layers")
    n_heads = _safe_get(cfg, "num_attention_heads", "n_heads")
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and n_heads and hidden:
        head_dim = hidden // n_heads
    vocab = _safe_get(cfg, "vocab_size")
    return dict(arch=arch, hidden=hidden, n_layers=n_layers,
                n_heads=n_heads, head_dim=head_dim or 0, vocab=vocab)


def _count_layer_types(cfg, arch_name: str):
    name = (arch_name or "").lower()
    n_layers = _safe_get(cfg, "num_hidden_layers", "n_layers")
    if any(k in name for k in ("llama", "mistral", "qwen", "phi", "gpt", "falcon3")):
        return n_layers, 0
    if "mamba" in name and "jamba" not in name and "nemotron" not in name and "zamba" not in name:
        return 0, n_layers
    if "jamba" in name:
        period = getattr(cfg, "attn_layer_period", 0)
        offset = getattr(cfg, "attn_layer_offset", 0)
        if period:
            n_attn = sum(1 for i in range(n_layers) if (i - offset) % period == 0)
            return n_attn, n_layers - n_attn
        return -1, -1
    if "zamba" in name:
        types = getattr(cfg, "layers_block_type", None)
        if types:
            n_attn = sum(1 for t in types if "hybrid" in str(t).lower() or "attention" in str(t).lower())
            n_ssm = sum(1 for t in types if "mamba" in str(t).lower())
            return n_attn, n_ssm
    if "nemotron" in name:
        pattern = getattr(cfg, "hybrid_override_pattern", "") or ""
        if pattern:
            return pattern.count("A"), pattern.count("M")
    return -1, -1


def _total_active_params(model, cfg):
    total = sum(p.numel() for p in model.parameters())
    num_e = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    num_act = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "num_experts_active", None)
    if not num_e or not num_act or num_act >= num_e:
        return total, total
    expert_p = 0
    for name, p in model.named_parameters():
        nl = name.lower()
        if ("expert" in nl and "router" not in nl) or "moe.experts" in nl:
            expert_p += p.numel()
    if expert_p == 0:
        return total, total
    inactive = num_e - num_act
    return total, int(total - expert_p * inactive / num_e)


def decode_flops_per_token(active_params, n_attn, hidden, context_len):
    """Theoretical FLOPs for one decode token with cache."""
    dense = 2 * active_params
    attn_kv = 4 * n_attn * hidden * context_len if n_attn > 0 else 0
    return dense + attn_kv


def prefill_flops(active_params, n_attn, n_heads, head_dim, prompt_len):
    """Theoretical FLOPs for prefilling a prompt of length L."""
    dense = 2 * active_params * prompt_len
    if n_attn > 0:
        attn_quad = 2 * n_attn * n_heads * head_dim * prompt_len * prompt_len
    else:
        attn_quad = 0
    return dense + attn_quad


def analyze(name: str, path: str) -> Dict:
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    arch = _arch_info(cfg)
    n_attn, n_ssm = _count_layer_types(cfg, arch["arch"])

    total_p = active_p = None
    try:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
                path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            )
        total_p, active_p = _total_active_params(model, cfg)
    except Exception as e:
        print(f"  [{name}] meta load failed: {type(e).__name__}: {e}", file=sys.stderr)

    return dict(model=name, path=path, **arch,
                n_attn=n_attn, n_ssm=n_ssm,
                total_params=total_p, active_params=active_p)


def _load_scores(jsonl_paths):
    scores = {}
    for path in jsonl_paths:
        if not os.path.exists(path):
            continue
        for line in open(path):
            try:
                r = json.loads(line)
                if "error" in r: continue
                k = (r["model"], r["dataset"])
                scores[k] = r
            except Exception:
                pass
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baselines.yaml")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--scores", nargs="+", default=["results/baselines.jsonl", "results/baseline_middle.jsonl"],
                    help="Optional jsonl files; used to compute score per TFLOP/token.")
    ap.add_argument("--peak_tflops", type=float, default=989.0,
                    help="GPU peak BF16 TFLOPS (default: H800 = 989).")
    ap.add_argument("--output", default="results/efficiency", help="Stem; writes .csv and .md")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    wanted = set(args.models) if args.models else None
    rows = []
    for m in cfg["models"]:
        if wanted and m["name"] not in wanted:
            continue
        print(f"analyzing {m['name']} ...")
        try:
            rows.append(analyze(m["name"], m["path"]))
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}", file=sys.stderr)

    scores = _load_scores(args.scores)

    # Build summary rows: per-dataset FLOPs + score
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    md_lines = []
    md_lines.append("## Theoretical FLOPs and Score per TFLOP")
    md_lines.append("")
    md_lines.append("Hardware peak: {:.0f} TFLOPS BF16".format(args.peak_tflops))
    md_lines.append("")

    # Wide table: per dataset, decode FLOPs/token + score / TFLOP
    datasets = ["ARC-e", "ARC-c", "PIQA", "TruthfulQA", "CNN/DM", "PubMed"]
    header = ["Model", "Active (B)", "N_attn", "N_ssm"]
    for d in datasets:
        header.append(f"{d} dec.TFLOPs/tok")
        header.append(f"{d} Score")
        header.append(f"{d} Score/TFLOP")
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join([":---"] + [":---:"] * (len(header) - 1)) + "|")

    csv_lines = [",".join(header)]
    for r in rows:
        active = r.get("active_params")
        if not active:
            cells = [r["model"], "?", "?", "?"] + ["?"] * (3 * len(datasets))
            md_lines.append("| " + " | ".join(cells) + " |")
            csv_lines.append(",".join(cells))
            continue
        n_attn = max(0, r.get("n_attn", 0))
        cells = [
            r["model"],
            f"{active/1e9:.2f}",
            str(n_attn) if r.get("n_attn", -1) >= 0 else "?",
            str(r.get("n_ssm", 0)) if r.get("n_ssm", -1) >= 0 else "?",
        ]
        for d in datasets:
            ctx = DATASET_CONTEXT.get(d, 512)
            d_flops = decode_flops_per_token(active, n_attn, r["hidden"], ctx)
            d_tflops = d_flops / 1e12
            score = scores.get((r["model"], d), {}).get("score")
            if score is not None:
                spt = score / d_tflops
                cells += [f"{d_tflops:.2e}", f"{score:.2f}", f"{spt:.2e}"]
            else:
                cells += [f"{d_tflops:.2e}", "-", "-"]
        md_lines.append("| " + " | ".join(cells) + " |")
        csv_lines.append(",".join(cells))

    md_path = args.output + ".md"
    csv_path = args.output + ".csv"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines) + "\n")

    print()
    print("\n".join(md_lines))
    print()
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
