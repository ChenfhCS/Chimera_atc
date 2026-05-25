from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import EvalExample
from .metrics import exact_choice_accuracy, mean_f1, mean_rouge_l
from .utils import Timer


# Per-dataset recommended generation budgets. ARC/PIQA/TruthfulQA only emit a
# single letter so 4 tokens is plenty. Summarization needs ~reference length;
# anything shorter caps ROUGE-L recall (e.g. 32 tokens vs. a 700-token
# GovReport reference caps F1 at ~4-5).
DATASET_DEFAULT_MAX_NEW_TOKENS = {
    "ARC-e": 8,
    "ARC-c": 8,
    "PIQA": 8,
    "TruthfulQA": 8,
    "CNN/DM": 128,
    "GovReport": 512,
    "XSum": 64,
    "MultiNews": 256,
    "arXiv": 256,
    "PubMed": 256,
    "SAMSum": 64,
    "BillSum": 256,
}


def load_hf_model_and_tokenizer(model_name_or_path: str, torch_dtype="bfloat16", device_map="auto", trust_remote_code=True):
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "auto": "auto"}.get(torch_dtype, torch.bfloat16)
    kwargs = dict(device_map=device_map, trust_remote_code=trust_remote_code)
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Force use_cache=True for every model. Some hybrid checkpoints (Nemotron-H,
    # Jamba, Mamba-2) ship with config.use_cache=False which causes .generate()
    # to re-prefill the whole prompt at every decode step -> O(n^2), 50-100x
    # slower than expected. The override is safe for pure-attention models too.
    try:
        model.config.use_cache = True
    except Exception:
        pass
    if hasattr(model, "generation_config") and model.generation_config is not None:
        try:
            model.generation_config.use_cache = True
        except Exception:
            pass
    return model, tok


def _hybrid_cache_kwargs(model) -> Dict:
    """Per-model kwargs to make .generate() actually keep cache across decode
    steps for hybrid architectures.

    Pure-attention models (Llama family) already get a DynamicCache for free,
    so we only emit extra kwargs when the architecture is Mamba/Jamba/
    Nemotron-H-flavored. Returns an empty dict when nothing to do.

    Note: Nemotron-H does NOT honor ``cache_implementation`` because its custom
    cache class is not a subclass of HF ``Cache``. For that path we build the
    cache manually via :func:`_build_legacy_hybrid_cache`. We still emit the
    kwarg for Jamba which does honor the modern API.
    """
    cls_name = (model.__class__.__name__ or "").lower()
    is_nemotron_h = "nemotronh" in cls_name or "nemotron_h" in cls_name
    is_jamba = "jamba" in cls_name
    is_mamba = "mamba" in cls_name and not is_nemotron_h
    if is_nemotron_h:
        return {}
    if is_jamba:
        return {"cache_implementation": "hybrid"}
    if is_mamba:
        return {"cache_implementation": "mamba"}
    return {}


def _find_custom_cache_class(model) -> Optional[type]:
    """Look up a hybrid cache class registered alongside the model's custom
    modeling code (loaded via trust_remote_code). Returns ``None`` if not
    found. Currently targets Nemotron-H's ``NemotronHHybridDynamicCache``.
    """
    mod_name = getattr(model.__class__, "__module__", None)
    if not mod_name:
        return None
    mod = sys.modules.get(mod_name)
    if mod is None:
        return None
    # Prefer the most specific name first.
    for candidate in dir(mod):
        if candidate.endswith("HybridDynamicCache"):
            return getattr(mod, candidate)
    for candidate in dir(mod):
        if "HybridCache" in candidate:
            return getattr(mod, candidate)
    return None


def _build_legacy_hybrid_cache(model, batch_size: int, max_len: int) -> Optional[Any]:
    """Try a few constructor signatures to materialize a per-batch hybrid
    cache. Returns ``None`` if no signature works — caller falls back to
    cache-less generation.
    """
    cls_name = (model.__class__.__name__ or "").lower()
    if "nemotronh" not in cls_name and "nemotron_h" not in cls_name:
        return None
    cache_cls = _find_custom_cache_class(model)
    if cache_cls is None:
        return None
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    # Nemotron-H's cache has gone through a few signatures across releases.
    attempts = [
        lambda: cache_cls(config=model.config, batch_size=batch_size,
                         max_cache_len=max_len, dtype=dtype, device=device),
        lambda: cache_cls(model.config, batch_size, max_len, dtype, device),
        lambda: cache_cls(config=model.config, batch_size=batch_size,
                         dtype=dtype, device=device),
        lambda: cache_cls(model.config, batch_size, dtype=dtype, device=device),
        lambda: cache_cls(model.config, batch_size, device=device),
        lambda: cache_cls(model.config, batch_size),
        lambda: cache_cls(model.config),
    ]
    last_err: Optional[Exception] = None
    for attempt in attempts:
        try:
            return attempt()
        except Exception as e:
            last_err = e
            continue
    print(f"[hybrid-cache] could not instantiate {cache_cls.__name__}: {last_err!r}")
    return None


def resolve_max_new_tokens(dataset_name: str, override: Optional[int]) -> int:
    """Pick a generation budget. Explicit override wins; otherwise fall back
    to the per-dataset default so summarization isn't capped at 32 tokens."""
    if override is not None:
        return int(override)
    return DATASET_DEFAULT_MAX_NEW_TOKENS.get(dataset_name, 32)


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: List[str], max_new_tokens=32, batch_size=1, is_dynamic=False,
                   input_max_tokens: int = 4096):
    preds = []
    total_new = 0
    total_time = 0.0
    device = next(model.parameters()).device
    hybrid_kwargs = _hybrid_cache_kwargs(model) if not is_dynamic else {}
    cache_announced = False
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=input_max_tokens).to(device)
        input_len = enc.input_ids.shape[1]
        with Timer() as t:
            if is_dynamic and hasattr(model, "generate_greedy"):
                out = model.generate_greedy(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
            else:
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                gen_kwargs.update(hybrid_kwargs)
                # Per-batch fresh cache for Nemotron-H-style models whose
                # custom modeling code refuses to construct one when
                # past_key_values is None.
                manual_cache = _build_legacy_hybrid_cache(
                    model, batch_size=enc.input_ids.shape[0],
                    max_len=int(input_len) + int(max_new_tokens),
                )
                if manual_cache is not None:
                    gen_kwargs["past_key_values"] = manual_cache
                    if not cache_announced:
                        print(f"[hybrid-cache] {model.__class__.__name__}: "
                              f"using {manual_cache.__class__.__name__}")
                        cache_announced = True
                try:
                    out = model.generate(**enc, **gen_kwargs)
                except (TypeError, ValueError) as e:
                    msg = str(e)
                    retried = False
                    if "cache_implementation" in msg and "cache_implementation" in gen_kwargs:
                        gen_kwargs.pop("cache_implementation", None)
                        retried = True
                    if "past_key_values" in msg and "past_key_values" in gen_kwargs:
                        gen_kwargs.pop("past_key_values", None)
                        retried = True
                    if retried:
                        out = model.generate(**enc, **gen_kwargs)
                    else:
                        raise
        total_time += t.elapsed
        total_new += int(out.shape[1] - input_len) * len(batch)
        gen = out[:, input_len:]
        preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    tps = total_new / max(total_time, 1e-9)
    return preds, tps


def evaluate_examples(model, tokenizer, examples: List[EvalExample], max_new_tokens=None, batch_size=1,
                      metric="auto", is_dynamic=False, input_max_tokens: int = 4096) -> Dict:
    ds_name = examples[0].dataset if examples else "unknown"
    effective_new_tokens = resolve_max_new_tokens(ds_name, max_new_tokens)
    preds, tps = generate_texts(model, tokenizer, [x.prompt for x in examples], effective_new_tokens,
                                batch_size, is_dynamic, input_max_tokens=input_max_tokens)
    labels = [x.target for x in examples]
    ds = ds_name
    if metric == "auto":
        # Prompts emit a single A/B/C/D letter for ARC, PIQA, and TruthfulQA,
        # so first-character accuracy is the only sensible default. F1 over a
        # one-token answer collapses to exact match anyway.
        if ds in {"ARC-e", "ARC-c", "PIQA", "TruthfulQA"}:
            metric = "acc"
        else:
            metric = "rouge"
    if metric == "acc":
        score = exact_choice_accuracy(preds, labels)
        key = "ACC"
    elif metric == "f1":
        score = mean_f1(preds, labels)
        key = "F1"
    else:
        score = mean_rouge_l(preds, labels)
        key = "ROUGE-L"
    return {
        "dataset": ds,
        "metric": key,
        "score": score,
        "tpt": tps,
        "num_samples": len(examples),
        "max_new_tokens": effective_new_tokens,
        "input_max_tokens": input_max_tokens,
    }
