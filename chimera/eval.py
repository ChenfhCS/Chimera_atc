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


def _ensure_dynamic_siblings_loaded(model) -> None:
    """Force-import every .py file that lives next to the model's modeling
    code. Cache classes are often defined in a sibling ``cache_utils.py`` or
    similar that the modeling file imports lazily, so ``dir(modeling_mod)``
    alone misses them until we explicitly load the package.
    """
    import importlib
    import os
    mod_name = getattr(model.__class__, "__module__", "") or ""
    if not mod_name.startswith("transformers_modules."):
        return
    parts = mod_name.split(".")
    if len(parts) < 2:
        return
    parent_pkg = ".".join(parts[:-1])
    parent_mod = sys.modules.get(parent_pkg)
    if parent_mod is None:
        try:
            parent_mod = importlib.import_module(parent_pkg)
        except Exception:
            return
    parent_path = getattr(parent_mod, "__path__", None)
    if not parent_path:
        return
    parent_dir = parent_path[0]
    if not os.path.isdir(parent_dir):
        return
    for fn in os.listdir(parent_dir):
        if fn.endswith(".py") and not fn.startswith("_"):
            sub_name = parent_pkg + "." + fn[:-3]
            if sub_name not in sys.modules:
                try:
                    importlib.import_module(sub_name)
                except Exception:
                    continue


def _find_custom_cache_class(model) -> Optional[type]:
    """Search every loaded module that belongs to the model's
    dynamic_modules package for a hybrid cache class. Tries the most
    specific name suffix first and falls back to fuzzy match.
    """
    mod_name = getattr(model.__class__, "__module__", "") or ""

    _ensure_dynamic_siblings_loaded(model)

    candidate_mods = []
    if mod_name:
        m = sys.modules.get(mod_name)
        if m is not None:
            candidate_mods.append(m)
    if mod_name.startswith("transformers_modules."):
        prefix = ".".join(mod_name.split(".")[:-1]) + "."
        for name, m in list(sys.modules.items()):
            if name.startswith(prefix) and m is not None and m not in candidate_mods:
                candidate_mods.append(m)

    seen_caches: List[Tuple[str, str]] = []
    # Priority 1: explicit Nemotron-H / hybrid-dynamic-cache match.
    for m in candidate_mods:
        for cname in dir(m):
            if cname.startswith("_"):
                continue
            cls = getattr(m, cname, None)
            if not isinstance(cls, type):
                continue
            if "Cache" in cname:
                seen_caches.append((m.__name__.rsplit(".", 1)[-1], cname))
            if cname.endswith("HybridDynamicCache"):
                return cls
    # Priority 2: any *Hybrid*Cache class.
    for m in candidate_mods:
        for cname in dir(m):
            if cname.startswith("_") or "Cache" not in cname or "Hybrid" not in cname:
                continue
            cls = getattr(m, cname, None)
            if isinstance(cls, type):
                return cls
    if seen_caches:
        print(f"[hybrid-cache] no *HybridDynamicCache match; saw cache classes {seen_caches}")
    else:
        print(f"[hybrid-cache] no Cache classes found across "
              f"{[m.__name__ for m in candidate_mods]}")
    return None


def _is_nemotron_h(model) -> bool:
    cls_name = (model.__class__.__name__ or "").lower()
    return "nemotronh" in cls_name or "nemotron_h" in cls_name


# Cache kwarg priority is Mamba-style first because the only models that hit
# our manual decode loop are hybrid Mamba/Attention checkpoints.
_CACHE_KWARG_CANDIDATES = ("cache_params", "past_key_values", "kv_cache", "cache")
_CACHE_OUTPUT_FIELDS = ("cache_params", "past_key_values", "kv_cache", "cache")


def _detect_cache_kwarg_name(model) -> Optional[str]:
    """Inspect ``model.forward`` to find the actual cache kwarg name.

    Trial-and-error fails for models whose signature ends in ``**kwargs`` (like
    Nemotron-H): wrong kwarg names get silently swallowed, the model runs with
    its real cache argument still defaulting to ``None``, and we see the
    "no cache will be returned" warning even though no TypeError was raised.
    """
    import inspect
    try:
        params = inspect.signature(model.forward).parameters
    except (ValueError, TypeError):
        return None
    for name in _CACHE_KWARG_CANDIDATES:
        if name in params:
            return name
    return None


def _call_model_with_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cache: Any,
    cache_kwarg_name: str,
    cache_position: Optional[torch.Tensor] = None,
) -> Any:
    """Call ``model.forward`` with an explicit cache kwarg + cache_position."""
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    kwargs[cache_kwarg_name] = cache
    if cache_position is not None:
        kwargs["cache_position"] = cache_position
    try:
        return model(**kwargs)
    except TypeError as e:
        # Some forwards don't accept cache_position; retry without it.
        if "cache_position" in str(e):
            kwargs.pop("cache_position", None)
            return model(**kwargs)
        raise


def _extract_returned_cache(out, fallback: Any) -> Any:
    """Pull the updated cache out of a model output object. Falls back to the
    cache we passed in (it may have been updated in-place).
    """
    for name in _CACHE_OUTPUT_FIELDS:
        v = getattr(out, name, None)
        if v is not None:
            return v
    return fallback


@torch.no_grad()
def _manual_greedy_with_cache(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    cache: Any,
) -> torch.Tensor:
    """Greedy decoding loop that threads a hybrid cache through every forward
    call by hand. Bypasses HF ``model.generate()`` so custom caches
    (Nemotron-H ``HybridMambaAttentionDynamicCache``) survive
    prepare_inputs_for_generation and ``_supports_cache_class`` checks.

    Looks up the actual cache kwarg name via ``inspect`` (Mamba-style models
    name it ``cache_params``, attention models name it ``past_key_values``),
    and threads ``cache_position`` so the Mamba branch knows we are decoding
    rather than re-prefilling.
    """
    device = input_ids.device
    kwarg_name = _detect_cache_kwarg_name(model) or "past_key_values"
    print(f"[hybrid-cache] forward kwarg='{kwarg_name}'")

    L = int(input_ids.shape[1])
    # Prefill on the full prompt.
    cache_position = torch.arange(L, device=device, dtype=torch.long)
    out = _call_model_with_cache(
        model, input_ids, attention_mask, cache,
        cache_kwarg_name=kwarg_name, cache_position=cache_position,
    )
    fields = [f for f in _CACHE_OUTPUT_FIELDS if getattr(out, f, None) is not None]
    print(f"[hybrid-cache] output cache fields={fields or 'NONE (in-place)'}")
    cache = _extract_returned_cache(out, fallback=cache)
    # Mark the cache as warm so subsequent decode calls hit the Mamba "use
    # stored state" branch instead of treating every step as a fresh prefill.
    # NemotronH stores this flag on the cache but does not always set it
    # automatically when prefill runs through a user-supplied cache.
    if hasattr(cache, "has_previous_state"):
        cache.has_previous_state = True
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_token]
    cur_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
    if eos_token_id is not None and (next_token == eos_token_id).all():
        return torch.cat([input_ids] + generated, dim=-1)

    for step in range(1, max_new_tokens):
        cache_position = torch.tensor([L + step - 1], device=device, dtype=torch.long)
        out = _call_model_with_cache(
            model, next_token, cur_mask, cache,
            cache_kwarg_name=kwarg_name, cache_position=cache_position,
        )
        cache = _extract_returned_cache(out, fallback=cache)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        cur_mask = torch.cat([cur_mask, torch.ones_like(next_token)], dim=-1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    return torch.cat([input_ids] + generated, dim=-1)


class _DeviceAwareList(list):
    """A list subclass that mirrors ``.device`` / ``.dtype`` of its first
    element. Patches over NemotronH's ``update_conv_state`` bug:

        self.conv_states[layer_idx] = new.to(self.conv_states.device)
                                                 ^^^^^^^^^^^^^^^^^^^^^
                                                 conv_states is a list

    The corrected call would have been ``self.conv_states[layer_idx].device``;
    we keep the source intact and just expose the missing attribute.
    """

    @property
    def device(self):
        for t in self:
            if hasattr(t, "device"):
                return t.device
        return None

    @property
    def dtype(self):
        for t in self:
            if hasattr(t, "dtype"):
                return t.dtype
        return None


def _augment_nemotron_cache(cache: Any, config) -> Any:
    """Make a freshly built HybridMambaAttentionDynamicCache compatible with
    NemotronH's modeling code. Two things need fixing:

      1. Some config-derived constants (``conv_kernel_size``,
         ``intermediate_size``, ``ssm_state_size``, ...) are stored only as
         ``__init__`` locals and not as instance attributes, yet the Mamba
         mixer reads them off the cache. We mirror them onto the instance.
      2. ``update_conv_state`` and ``update_ssm_state`` call
         ``self.conv_states.device`` directly on a Python ``list``. Wrap
         conv_states / ssm_states in a list subclass that exposes ``.device``
         delegating to the first stored tensor.
    """
    # 1) Mirror constants the modeling code reads from the cache.
    extras = {
        "conv_kernel_size": getattr(config, "conv_kernel", None),
        "intermediate_size": (
            getattr(config, "expand", 1) * getattr(config, "hidden_size", 0)
        ),
        "ssm_state_size": getattr(config, "ssm_state_size", None),
        "n_groups": getattr(config, "n_groups", None),
        "head_dim": getattr(config, "head_dim", None),
        "num_heads": getattr(config, "num_attention_heads", None),
        "batch_size": getattr(cache, "batch_size", None),
    }
    for name, val in extras.items():
        if val is None:
            continue
        if not hasattr(cache, name):
            try:
                setattr(cache, name, val)
            except Exception:
                pass

    # 2) Wrap the state lists so ``cache.conv_states.device`` works.
    for attr in ("conv_states", "ssm_states"):
        existing = getattr(cache, attr, None)
        if isinstance(existing, list) and not isinstance(existing, _DeviceAwareList):
            try:
                setattr(cache, attr, _DeviceAwareList(existing))
            except Exception:
                pass
    return cache


def _build_legacy_hybrid_cache(model, batch_size: int, max_len: int) -> Optional[Any]:
    """Try a few constructor signatures to materialize a per-batch hybrid
    cache. Returns ``None`` if no signature works — caller falls back to
    cache-less generation.
    """
    if not _is_nemotron_h(model):
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
            cache = attempt()
            return _augment_nemotron_cache(cache, model.config)
        except Exception as e:
            last_err = e
            continue
    print(f"[hybrid-cache] could not instantiate {cache_cls.__name__}: {last_err!r}")
    return None


def _maybe_apply_chat_template(tokenizer, prompt: str, apply: bool) -> str:
    """Wrap ``prompt`` in the tokenizer's chat template when ``apply`` is True
    and the tokenizer has one. Instruct/chat models (Nemotron-H-Instruct,
    Llama-3.2-Instruct, Jamba-Instruct, ...) ship a template; base models
    don't, so this is a no-op for the base baselines.

    Without this, instruct models score normally on letter-answer tasks
    (ARC/PIQA/TruthfulQA) but collapse on free-form generation (CNN/DM,
    PubMed) because the model expects ``<|assistant|>``-style turn tags
    around its output and emits noise when fed a raw "Summarize: ..." prompt.
    """
    if not apply:
        return prompt
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        return prompt
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return prompt


def resolve_max_new_tokens(dataset_name: str, override: Optional[int]) -> int:
    """Pick a generation budget. Explicit override wins; otherwise fall back
    to the per-dataset default so summarization isn't capped at 32 tokens."""
    if override is not None:
        return int(override)
    return DATASET_DEFAULT_MAX_NEW_TOKENS.get(dataset_name, 32)


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: List[str], max_new_tokens=32, batch_size=1, is_dynamic=False,
                   input_max_tokens: int = 4096, apply_chat_template: bool = True):
    preds = []
    total_new = 0
    total_time = 0.0
    device = next(model.parameters()).device
    hybrid_kwargs = _hybrid_cache_kwargs(model) if not is_dynamic else {}
    cache_announced = False
    use_manual_loop = (not is_dynamic) and _is_nemotron_h(model)
    # Pre-format prompts once. For base models tokenizer.chat_template is None,
    # so this is a no-op and we keep the raw "Summarize: ..." prompt.
    if apply_chat_template:
        prompts = [_maybe_apply_chat_template(tokenizer, p, True) for p in prompts]
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=input_max_tokens).to(device)
        input_len = enc.input_ids.shape[1]
        with Timer() as t:
            if is_dynamic and hasattr(model, "generate_greedy"):
                out = model.generate_greedy(enc.input_ids, attention_mask=enc.attention_mask, max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id)
            elif use_manual_loop:
                # Nemotron-H's _supports_cache_class is False, so HF generate()
                # silently drops the hybrid cache we pass in and re-prefills
                # the whole prompt at every decode step (~100x slowdown).
                # Run a hand-written greedy loop that keeps the cache alive.
                manual_cache = _build_legacy_hybrid_cache(
                    model, batch_size=enc.input_ids.shape[0],
                    max_len=int(input_len) + int(max_new_tokens),
                )
                if manual_cache is None:
                    # Fall back to vanilla generate if we cannot construct the
                    # cache class for this revision.
                    out = model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.eos_token_id, use_cache=True,
                    )
                else:
                    if not cache_announced:
                        print(f"[hybrid-cache] {model.__class__.__name__}: "
                              f"manual loop with {manual_cache.__class__.__name__}")
                        cache_announced = True
                    out = _manual_greedy_with_cache(
                        model,
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        cache=manual_cache,
                    )
            else:
                gen_kwargs = dict(
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                gen_kwargs.update(hybrid_kwargs)
                try:
                    out = model.generate(**enc, **gen_kwargs)
                except (TypeError, ValueError) as e:
                    msg = str(e)
                    if "cache_implementation" in msg and "cache_implementation" in gen_kwargs:
                        gen_kwargs.pop("cache_implementation", None)
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
                      metric="auto", is_dynamic=False, input_max_tokens: int = 4096,
                      apply_chat_template: bool = True) -> Dict:
    ds_name = examples[0].dataset if examples else "unknown"
    effective_new_tokens = resolve_max_new_tokens(ds_name, max_new_tokens)
    preds, tps = generate_texts(model, tokenizer, [x.prompt for x in examples], effective_new_tokens,
                                batch_size, is_dynamic, input_max_tokens=input_max_tokens,
                                apply_chat_template=apply_chat_template)
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
