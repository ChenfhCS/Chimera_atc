# Baseline Evaluation Playbook

This document walks through downloading the 8 baseline models and evaluating
them on the 6 datasets (ARC-e, ARC-c, PIQA, TruthfulQA, CNN/DM, GovReport).

## 0. Resource budget

Rough estimates at bf16, weights only:

| Tier  | Model                | Params | VRAM (eval, bs=1) | Disk |
|-------|----------------------|--------|-------------------|------|
| Small | Llama-3.2-3B         |  3.2B  | ~7 GB             | ~6 GB |
| Small | Mamba-2 ~2.7B        |  2.7B  | ~6 GB             | ~6 GB |
| Small | Jamba-3.2B           |  3.2B  | ~7 GB             | ~7 GB |
| Small | Nemotron-H 4B Inst.  |  4.5B  | ~10 GB            | ~9 GB |
| Large | Llama-3 13B          | 13B    | ~28 GB            | ~26 GB |
| Large | Mamba-2 8B           |  8.2B  | ~17 GB            | ~16 GB |
| Large | Jamba 13B            | 13B    | ~28 GB            | ~26 GB |
| Large | Nemotron-H 12B       | 12B    | ~26 GB            | ~24 GB |

Total disk: **≈120 GB**. Run the small tier on a 24 GB GPU; the large tier
needs ≥40 GB (A100/H100) or `device_map="auto"` across multiple GPUs.

## 1. One-time environment setup

```bash
# 1) Python + deps (use a venv or conda env you control)
pip install -r requirements.txt
pip install -e .

# 2) HuggingFace login (Llama-3.2-3B and several NVIDIA models are gated)
export HF_TOKEN="hf_xxx_your_token"
huggingface-cli login --token "$HF_TOKEN"

# 3) Confirm transformers >= 4.46 (Mamba-2, Jamba, Nemotron-H require it)
python -c "import transformers; print(transformers.__version__)"

# 4) (optional) point HF cache to a large disk
export HF_HOME=/path/with/disk/space
```

### Known model substitutions

* **`state-spaces/mamba2-2.7b` is not loadable via `AutoModelForCausalLM`** —
  it ships as `mamba_ssm.MambaLMHeadModel`. Three options:
  1. **Substitute** (default in `configs/baselines.yaml`): use the HuggingFace
     `state-spaces/mamba2-hybrid-2.7b-300b`. This is the easiest path.
  2. **Same family, HF-native**: substitute with `state-spaces/mamba-2.8b-hf`
     (Mamba-1, hidden=2560).
  3. **Native loader**: install
     `pip install mamba_ssm causal-conv1d` and write a thin wrapper that
     exposes `.generate(...)` like an HF model; see "Custom mamba_ssm loader"
     at the bottom of this file.

* **`ai21labs/AI21-Jamba2-3B`, `OxxoCodes/jamba-small-v1`,
  `nvidia/Nemotron-H-4B-Instruct-128K`, `elinas/Llama-3-13B-Instruct`** —
  please verify the repo names against `huggingface.co` before downloading;
  community repos occasionally get renamed/withdrawn.

## 2. Pre-download everything

```bash
# Pull all 8 models to the HF cache. Add --models ... to grab a subset.
python scripts/download_baselines.py --config configs/baselines.yaml
```

Then sanity-check that each model's config + tokenizer is reachable and
estimate VRAM:

```bash
python scripts/preflight_baselines.py --config configs/baselines.yaml
```

Fix any "CONFIG LOAD FAILED" or "TOKENIZER LOAD FAILED" lines before moving on.

## 3. Run the 6-dataset evaluation

The eval script supports `--models` to run one checkpoint at a time. For 8B+
models you almost certainly want this — both because of VRAM and because a
mid-run failure shouldn't wipe out progress.

### 3.1 Tier-A (small models, single GPU)

```bash
python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Llama-3-3.2B \
  --output results/baselines.jsonl \
  --max_samples 256 \
  --batch_size 1 \
  --max_new_tokens 32

python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Mamba-2-2.7B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32

python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Jamba-3.2B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32

python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Nemotron-H-4.5B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32
```

You can also batch them in one process — only do this if a model OOM won't
crash you out:

```bash
python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Llama-3-3.2B Mamba-2-2.7B Jamba-3.2B Nemotron-H-4.5B \
  --output results/baselines.jsonl \
  --free_after \
  --max_samples 256 --batch_size 1 --max_new_tokens 32
```

`--free_after` deletes the model and empties the CUDA cache between models.

### 3.2 Tier-B (≥8B models, A100 40GB+ or multi-GPU)

```bash
python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Llama-3-13B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32

python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Mamba-2-8.2B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32

python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Jamba-13B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32

python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Nemotron-H-12B \
  --output results/baselines.jsonl \
  --max_samples 256 --batch_size 1 --max_new_tokens 32
```

Tip: if a 13B model is too big for one GPU, leave `device_map: auto` (already
the default) — HF will shard it across visible CUDA devices.

### 3.3 Dataset overrides

If you want to debug a single dataset, pass `--datasets`:

```bash
python scripts/evaluate_baselines.py \
  --config configs/baselines.yaml \
  --models Llama-3-3.2B \
  --datasets arc-e \
  --max_samples 32
```

Generation length policy:
* ARC-e/c, PIQA, TruthfulQA → letter answers, `--max_new_tokens 4` is enough.
* CNN/DM, GovReport → summaries, use `--max_new_tokens 128` (or higher for
  GovReport if you can spare the time).

For paper-grade throughput, **run the throughput-sensitive datasets with a
single `--max_new_tokens` value across baselines and dynamic hybrid**, e.g.
keep `--max_new_tokens 32` everywhere; otherwise tokens/s aren't comparable.

## 4. Aggregate into a table

```bash
python scripts/make_table.py \
  --inputs results/baselines.jsonl results/dynamic_hybrid.jsonl \
  --output_csv results/summary.csv
```

Open `results/summary.csv` — one row per model, columns are (score, tpt) per
dataset. The exact same metric code is used for baselines and the dynamic
model, so the score columns are directly comparable.

## 5. Custom `mamba_ssm` loader (only if you really need state-spaces/mamba2-2.7b)

The repo ships `mamba_ssm.MambaLMHeadModel`. Sketch of a wrapper:

```python
# chimera/loaders.py (drop in if needed)
import torch
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer

def load_state_spaces_mamba(repo_id: str, device: str = "cuda", dtype=torch.bfloat16):
    model = MambaLMHeadModel.from_pretrained(repo_id, device=device, dtype=dtype)
    tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")  # mamba's tokenizer
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    class _Adapter(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def parameters(self): return self.m.parameters()
        def to(self, *a, **k): self.m = self.m.to(*a, **k); return self
        @torch.no_grad()
        def generate(self, input_ids, attention_mask=None, max_new_tokens=32,
                     do_sample=False, pad_token_id=None, **kw):
            return self.m.generate(input_ids=input_ids, max_length=input_ids.shape[1]+max_new_tokens,
                                   temperature=0.0, top_p=1.0)
    return _Adapter(model), tok
```

Then add a tiny branch in `chimera/eval.py::load_hf_model_and_tokenizer` that
dispatches to this when `repo_id.startswith("state-spaces/mamba")` and the
HF AutoModel route raises. Keep this path opt-in — the substitution in
`configs/baselines.yaml` is fine for most experiments.
