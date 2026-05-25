# Chimera Artifact Code

This artifact provides a code scaffold for evaluating baseline LLMs and building a **Dynamic Hybrid LLM** from pretrained Attention and SSM/Mamba checkpoints.

## What changed in v2

The dynamic model is **not** a toy model anymore. `chimera/dynamic_hybrid_model.py` now builds each `DHybridBlock` by wrapping two pretrained HuggingFace decoder layers:

- an Attention/Transformer layer from `attn_model_name_or_path`;
- an SSM/Mamba layer from `ssm_model_name_or_path`;
- a new prefill gate that selects Attention or SSM per dynamic block.

During training, the default `routing_mode=soft` computes both branches and mixes them with gate probabilities, so gradients flow into the gate and both pretrained branches. During evaluation, hard routing selects one branch per request.

## Important compatibility requirement

The Attention and SSM checkpoints must have the same hidden dimension. This is required because the two branches share the same hidden-state interface inside a `dHybridBlock`.

If you see an error like:

```text
Hidden sizes must match for dHybrid blocks
```

replace the model pair in `configs/dynamic_hybrid.yaml` with two checkpoints that have matching hidden size, or add projection adapters in `dynamic_hybrid_model.py`.

## Directory structure

```text
chimera_artifact_v2/
├── chimera/
│   ├── dynamic_hybrid_model.py      # DynamicHybridForCausalLM built from pretrained branches
│   ├── adapters/hf_blocks.py        # HF module extraction and safe layer forward
│   ├── data.py                      # ARC-e, ARC-c, PIQA, TruthfulQA, CNN/DM, GovReport loaders
│   ├── eval.py                      # Generation + metric + throughput measurement
│   ├── metrics.py
│   └── utils.py
├── scripts/
│   ├── build_dynamic_hybrid.py      # Initialize Dynamic Hybrid from pretrained branches
│   ├── train_dynamic_hybrid.py      # Fine-tune gate and model branches
│   ├── evaluate_dynamic_hybrid.py
│   ├── evaluate_baselines.py
│   └── make_table.py
└── configs/
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Evaluate baselines

Edit `configs/baselines.yaml` to use the exact model IDs or local checkpoint paths used in your paper.

```bash
python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --output results/baselines.jsonl \
  --max_samples 256 \
  --batch_size 1 \
  --max_new_tokens 32
```

The script reports the dataset score and generation throughput in tokens/s.

## Build Dynamic Hybrid from pretrained branches

Edit `configs/dynamic_hybrid.yaml` so that `attn_model_name_or_path` and `ssm_model_name_or_path` point to compatible checkpoints.

```bash
python scripts/build_dynamic_hybrid.py \
  --config configs/dynamic_hybrid.yaml \
  --output_dir checkpoints/chimera_init
```

This initializes:

```text
DynamicHybridLM
├── embedding and lm_head from the Attention model
├── dHybrid block 1: pretrained Attention layer + pretrained SSM layer + new gate
├── dHybrid block 2: pretrained Attention layer + pretrained SSM layer + new gate
└── ...
```

## Fine-tune Dynamic Hybrid

By default, gate and branch parameters are all trainable.

```bash
python scripts/train_dynamic_hybrid.py \
  --config configs/dynamic_hybrid.yaml \
  --checkpoint checkpoints/chimera_init/dynamic_hybrid.pt \
  --output_dir checkpoints/chimera_ft \
  --datasets arc-e arc-c piqa truthfulqa cnn/dm govreport \
  --max_samples_per_dataset 2000 \
  --epochs 1 \
  --batch_size 1 \
  --routing_mode soft
```

For gate-only debugging:

```bash
python scripts/train_dynamic_hybrid.py \
  --config configs/dynamic_hybrid.yaml \
  --output_dir checkpoints/chimera_gate_only \
  --freeze_branches
```

## Evaluate Dynamic Hybrid

```bash
python scripts/evaluate_dynamic_hybrid.py \
  --model_or_config checkpoints/chimera_ft \
  --datasets arc-e arc-c piqa truthfulqa cnn/dm govreport \
  --output results/dynamic_hybrid.jsonl \
  --max_samples 256 \
  --batch_size 1 \
  --max_new_tokens 32
```

## Make a result table

```bash
python scripts/make_table.py \
  --inputs results/baselines.jsonl results/dynamic_hybrid.jsonl \
  --output_csv results/summary.csv
```

## Notes for paper-grade experiments

1. Replace placeholder model IDs with the exact checkpoints used in your experiments.
2. Verify hidden-size compatibility before building the dynamic model.
3. The current dynamic generation uses a simple greedy loop without KV cache. It is correct for functional evaluation but not optimized for throughput. The paper systems implementation should replace it with your hybrid batching engine and KV/SSM state cache.
4. The adapter code is intentionally generic. For a fixed model pair, writing explicit adapters is recommended for reproducibility.
