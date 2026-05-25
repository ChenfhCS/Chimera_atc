from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class HFBackboneAdapter:
    """A light-weight adapter for extracting modules from HuggingFace causal LMs.

    The adapter intentionally avoids assuming a single class name. It supports
    common decoder-only layouts used by Llama-like and Mamba-like HF models.
    """

    model: nn.Module
    name: str = "model"

    def layers(self) -> nn.ModuleList:
        candidates = [
            "model.layers",
            "backbone.layers",
            "layers",
            "model.backbone.layers",
            "model.decoder.layers",
            "transformer.h",
        ]
        for path in candidates:
            mod = self._get(path)
            if mod is not None:
                return mod
        raise ValueError(f"Cannot find decoder layers in {self.name}. Tried {candidates}.")

    def embed_tokens(self) -> nn.Module:
        candidates = [
            "model.embed_tokens",
            "backbone.embeddings",
            "backbone.embed_tokens",
            "embed_tokens",
            "transformer.wte",
        ]
        for path in candidates:
            mod = self._get(path)
            if mod is not None:
                return mod
        raise ValueError(f"Cannot find token embedding in {self.name}.")

    def final_norm(self) -> Optional[nn.Module]:
        candidates = [
            "model.norm",
            "backbone.norm",
            "backbone.norm_f",
            "norm",
            "transformer.ln_f",
        ]
        for path in candidates:
            mod = self._get(path)
            if mod is not None:
                return mod
        return None

    def lm_head(self) -> nn.Module:
        candidates = ["lm_head", "model.lm_head"]
        for path in candidates:
            mod = self._get(path)
            if mod is not None:
                return mod
        raise ValueError(f"Cannot find lm_head in {self.name}.")

    def rotary_emb(self) -> Optional[nn.Module]:
        """Best-effort lookup of a model-level rotary embedding (Llama-family)."""
        candidates = ["model.rotary_emb", "rotary_emb"]
        for path in candidates:
            mod = self._get(path)
            if mod is not None:
                return mod
        # Older transformers keep rotary_emb on each attention module.
        try:
            first = self.layers()[0]
        except Exception:
            return None
        sa = getattr(first, "self_attn", None)
        if sa is not None and hasattr(sa, "rotary_emb"):
            return sa.rotary_emb
        return None

    def hidden_size(self) -> int:
        cfg = getattr(self.model, "config", None)
        for key in ["hidden_size", "d_model", "n_embd"]:
            if cfg is not None and hasattr(cfg, key):
                return int(getattr(cfg, key))
        return int(self.embed_tokens().weight.shape[-1])

    def vocab_size(self) -> int:
        return int(self.embed_tokens().weight.shape[0])

    def _get(self, path: str):
        cur = self.model
        for p in path.split('.'):
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            if isinstance(item, torch.Tensor):
                return item
    if hasattr(x, "last_hidden_state"):
        return x.last_hidden_state
    if hasattr(x, "hidden_states") and x.hidden_states is not None:
        return x.hidden_states[-1]
    raise TypeError(f"Cannot extract tensor from layer output of type {type(x)}")


def call_attn_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    causal_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Call an attention decoder layer (e.g. LlamaDecoderLayer).

    Modern HF Llama layers require ``position_embeddings`` (precomputed rotary
    cos/sin) and a 4D causal ``attention_mask``. We try the modern signature
    first and fall back gracefully on older versions.
    """
    patterns = [
        dict(hidden_states=hidden_states, attention_mask=causal_mask, position_ids=position_ids,
             position_embeddings=position_embeddings, past_key_value=None, use_cache=False),
        dict(hidden_states=hidden_states, attention_mask=causal_mask, position_ids=position_ids,
             past_key_value=None, use_cache=False),
        dict(hidden_states=hidden_states, attention_mask=causal_mask, position_ids=position_ids),
        dict(hidden_states=hidden_states, attention_mask=causal_mask),
        dict(hidden_states=hidden_states),
    ]
    last_err: Optional[Exception] = None
    for kwargs in patterns:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            out = layer(**kwargs) if kwargs else layer(hidden_states)
            return _first_tensor(out)
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to call attention layer {layer.__class__.__name__}: {last_err}")


def call_ssm_layer(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    cache_params=None,
) -> torch.Tensor:
    """Call an SSM/Mamba block. SSM blocks ignore position_ids/attention_mask
    and instead optionally take a recurrent ``cache_params`` state.
    """
    patterns = [
        dict(hidden_states=hidden_states, cache_params=cache_params, cache_position=None, attention_mask=None),
        dict(hidden_states=hidden_states, cache_params=cache_params),
        dict(hidden_states=hidden_states),
    ]
    last_err: Optional[Exception] = None
    for kwargs in patterns:
        try:
            out = layer(**kwargs)
            return _first_tensor(out)
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to call SSM layer {layer.__class__.__name__}: {last_err}")


# Backwards-compatible generic wrapper used by older code paths.
def safe_layer_forward(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    **extra,
) -> torch.Tensor:
    patterns = [
        dict(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids, use_cache=False),
        dict(hidden_states=hidden_states, attention_mask=attention_mask, position_ids=position_ids),
        dict(hidden_states=hidden_states, attention_mask=attention_mask),
        dict(hidden_states=hidden_states),
    ]
    last_err: Optional[Exception] = None
    for kwargs in patterns:
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            out = layer(**kwargs) if kwargs else layer(hidden_states)
            return _first_tensor(out)
        except TypeError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to call layer {layer.__class__.__name__}: {last_err}")
