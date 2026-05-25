from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutput

from .adapters import HFBackboneAdapter
from .adapters.hf_blocks import call_attn_layer, call_ssm_layer


@dataclass
class DynamicHybridConfig:
    attn_model_name_or_path: str
    ssm_model_name_or_path: str
    num_layers: Optional[int] = None
    gate_hidden_size: int = 1024
    gate_temperature: float = 1.0
    routing_mode: str = "soft"  # soft | hard | straight_through
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_layer_offset: int = 0
    ssm_layer_offset: int = 0
    tie_lm_head: bool = True


def _dtype(name: str):
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "auto": "auto",
    }
    return mapping.get(name.lower(), torch.bfloat16)


class PrefillGate(nn.Module):
    """Request-level gate for choosing Attention or SSM at one dHybrid block."""

    def __init__(self, hidden_size: int, gate_hidden_size: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, gate_hidden_size),
            nn.SiLU(),
            nn.Linear(gate_hidden_size, 2),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            mask = attention_mask.to(hidden_states.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden_states * mask).sum(dim=1) / denom
        else:
            pooled = hidden_states.mean(dim=1)
        return self.net(pooled)


class DHybridBlock(nn.Module):
    """A dynamic block built from pretrained Attention and SSM blocks."""

    def __init__(self, attn_layer: nn.Module, ssm_layer: nn.Module, hidden_size: int, gate_hidden_size: int):
        super().__init__()
        self.attn_layer = attn_layer
        self.ssm_layer = ssm_layer
        self.gate = PrefillGate(hidden_size, gate_hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        routing_mode: str = "soft",
        temperature: float = 1.0,
        fixed_branch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if fixed_branch is None:
            gate_logits = self.gate(hidden_states, attention_mask_2d)
            probs = F.softmax(gate_logits / max(temperature, 1e-6), dim=-1)
        else:
            probs = F.one_hot(fixed_branch.long(), num_classes=2).to(hidden_states.dtype)

        if self.training and routing_mode in {"soft", "straight_through"}:
            y_attn = call_attn_layer(self.attn_layer, hidden_states, causal_mask, position_ids, position_embeddings)
            y_ssm = call_ssm_layer(self.ssm_layer, hidden_states)
            if routing_mode == "straight_through":
                hard = F.one_hot(probs.argmax(dim=-1), num_classes=2).to(probs.dtype)
                weights = hard.detach() - probs.detach() + probs
            else:
                weights = probs
            out = weights[:, 0].view(-1, 1, 1) * y_attn + weights[:, 1].view(-1, 1, 1) * y_ssm
            return out, probs

        branch = probs.argmax(dim=-1)
        out = torch.empty_like(hidden_states)
        if (branch == 0).any():
            idx = torch.nonzero(branch == 0, as_tuple=False).view(-1)
            h_sub = hidden_states.index_select(0, idx)
            cm_sub = causal_mask.index_select(0, idx) if causal_mask is not None else None
            pi_sub = position_ids.index_select(0, idx) if position_ids is not None else None
            if position_embeddings is not None:
                cos, sin = position_embeddings
                pe_sub = (cos.index_select(0, idx), sin.index_select(0, idx)) if cos.dim() >= 2 else position_embeddings
            else:
                pe_sub = None
            out[idx] = call_attn_layer(self.attn_layer, h_sub, cm_sub, pi_sub, pe_sub)
        if (branch == 1).any():
            idx = torch.nonzero(branch == 1, as_tuple=False).view(-1)
            h_sub = hidden_states.index_select(0, idx)
            out[idx] = call_ssm_layer(self.ssm_layer, h_sub)
        return out, probs


class DynamicHybridForCausalLM(nn.Module):
    """Dynamic Hybrid Causal LM initialized from pretrained HF models."""

    def __init__(self, config: DynamicHybridConfig, attn_model: nn.Module, ssm_model: nn.Module):
        super().__init__()
        self.dhybrid_config = config

        self.attn_adapter = HFBackboneAdapter(attn_model, "attention_model")
        self.ssm_adapter = HFBackboneAdapter(ssm_model, "ssm_model")

        h_attn = self.attn_adapter.hidden_size()
        h_ssm = self.ssm_adapter.hidden_size()
        if h_attn != h_ssm:
            raise ValueError(
                f"Hidden sizes must match for dHybrid blocks, got attention={h_attn}, ssm={h_ssm}. "
                "Choose Llama/Mamba checkpoints with the same hidden dimension or add projection adapters."
            )
        self.hidden_size = h_attn
        self.embed_tokens = self.attn_adapter.embed_tokens()
        self.final_norm = self.attn_adapter.final_norm()
        self.lm_head = self.attn_adapter.lm_head()

        rotary = self.attn_adapter.rotary_emb()
        if rotary is not None:
            self.rotary_emb = rotary
        else:
            self.rotary_emb = None

        attn_layers = self.attn_adapter.layers()
        ssm_layers = self.ssm_adapter.layers()
        n = config.num_layers or min(
            len(attn_layers) - config.attn_layer_offset,
            len(ssm_layers) - config.ssm_layer_offset,
        )
        if n <= 0:
            raise ValueError("No layers available after offsets.")
        if config.attn_layer_offset + n > len(attn_layers):
            raise ValueError(
                f"Requested {n} attention layers from offset {config.attn_layer_offset}, "
                f"but attention model only has {len(attn_layers)} layers."
            )
        if config.ssm_layer_offset + n > len(ssm_layers):
            raise ValueError(
                f"Requested {n} SSM layers from offset {config.ssm_layer_offset}, "
                f"but SSM model only has {len(ssm_layers)} layers."
            )
        self.layers = nn.ModuleList([
            DHybridBlock(
                attn_layers[config.attn_layer_offset + i],
                ssm_layers[config.ssm_layer_offset + i],
                self.hidden_size,
                config.gate_hidden_size,
            )
            for i in range(n)
        ])

    @classmethod
    def from_pretrained_branches(cls, config: DynamicHybridConfig, **hf_kwargs):
        dtype = _dtype(config.torch_dtype)
        kwargs = dict(trust_remote_code=config.trust_remote_code)
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
        kwargs.update(hf_kwargs)
        attn_model = AutoModelForCausalLM.from_pretrained(config.attn_model_name_or_path, **kwargs)
        ssm_model = AutoModelForCausalLM.from_pretrained(config.ssm_model_name_or_path, **kwargs)
        return cls(config, attn_model, ssm_model)

    @staticmethod
    def _build_4d_causal_mask(
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        min_val = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
        causal = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
        causal = torch.triu(causal, diagonal=1)
        causal = causal[None, None].expand(batch_size, 1, seq_len, seq_len).contiguous()
        if attention_mask is not None:
            pad = attention_mask[:, None, None, :].to(dtype=dtype)
            causal = causal.masked_fill(pad == 0, min_val)
        return causal

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        routing_mode: Optional[str] = None,
        return_gate_probs: bool = False,
        fixed_branches: Optional[List[torch.Tensor]] = None,
    ):
        routing_mode = routing_mode or self.dhybrid_config.routing_mode
        if self.training and routing_mode == "hard":
            raise ValueError(
                "routing_mode='hard' would freeze the gate during training. "
                "Use 'soft' or 'straight_through' instead."
            )

        hidden_states = self.embed_tokens(input_ids)
        batch, seqlen = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(batch, -1)
        causal_mask = self._build_4d_causal_mask(attention_mask, seqlen, hidden_states.dtype, device, batch)
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if self.rotary_emb is not None:
            try:
                position_embeddings = self.rotary_emb(hidden_states, position_ids)
            except TypeError:
                position_embeddings = None

        all_gate_probs: List[torch.Tensor] = []
        for i, layer in enumerate(self.layers):
            fb = fixed_branches[i] if fixed_branches is not None else None
            hidden_states, probs = layer(
                hidden_states,
                attention_mask_2d=attention_mask,
                causal_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                routing_mode=routing_mode,
                temperature=self.dhybrid_config.gate_temperature,
                fixed_branch=fb,
            )
            all_gate_probs.append(probs)

        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_gate_probs:
            return CausalLMOutput(loss=loss, logits=logits), all_gate_probs
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_greedy(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
        reuse_prefill_branches: bool = True,
    ):
        """Greedy decoding that caches the prefill routing decision per layer.

        The first forward computes hard branch decisions from the prompt; all
        subsequent decode steps reuse those decisions, so the gate output is
        stable across the request as required by the dHybrid design.
        """
        self.eval()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        out, probs_list = self(
            input_ids,
            attention_mask=attention_mask,
            routing_mode="hard",
            return_gate_probs=True,
        )
        fixed_branches = [p.argmax(dim=-1) for p in probs_list] if reuse_prefill_branches else None

        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cur = torch.cat([input_ids, next_token], dim=-1)
        mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
        if eos_token_id is not None and (next_token == eos_token_id).all():
            return cur

        for _ in range(max_new_tokens - 1):
            out = self(
                cur,
                attention_mask=mask,
                routing_mode="hard",
                fixed_branches=fixed_branches,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_token], dim=-1)
            mask = torch.cat([mask, torch.ones_like(next_token)], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return cur

    def trainable_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def load_tokenizer_for_hybrid(attn_model_name_or_path: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(attn_model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
