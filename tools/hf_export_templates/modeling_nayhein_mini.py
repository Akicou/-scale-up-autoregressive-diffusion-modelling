from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_nayhein_mini import NayheinMiniConfig


class NayheinMiniRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rotary_dim = self.dim if self.dim % 2 == 0 else self.dim - 1
        self.base = float(base)
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        else:
            inv_freq = torch.empty(0)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(1, 0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1, 0), persistent=False)
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        self.max_seq_len_cached = int(seq_len)
        if self.rotary_dim == 0:
            self.cos_cached = torch.empty(seq_len, 0, device=self.inv_freq.device)
            self.sin_cached = torch.empty(seq_len, 0, device=self.inv_freq.device)
            return
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len].to(device=device, dtype=dtype),
            self.sin_cached[:seq_len].to(device=device, dtype=dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    if rotary_dim == 0:
        return q, k

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected rotary cache rank: {cos.dim()}")
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    if q_pass.shape[-1] > 0:
        q_embed = torch.cat((q_embed, q_pass), dim=-1)
    if k_pass.shape[-1] > 0:
        k_embed = torch.cat((k_embed, k_pass), dim=-1)
    return q_embed, k_embed


class NayheinMiniMLP(nn.Module):
    def __init__(self, config: NayheinMiniConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class NayheinMiniAttention(nn.Module):
    def __init__(self, config: NayheinMiniConfig) -> None:
        super().__init__()
        self.num_heads = int(config.num_heads)
        self.head_dim = int(config.head_dim)
        self.hidden_size = int(config.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = NayheinMiniRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0 if past_key_value is None else past_key_value[0].shape[2]
        total_len = past_len + seq_len
        cos, sin = self.rotary_emb(total_len, hidden_states.device, q.dtype)
        if position_ids is None:
            position_ids = torch.arange(past_len, total_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        cos = cos[position_ids]
        sin = sin[position_ids]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)

        present = (k, v) if use_cache else None

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(max(1, self.head_dim))
        kv_len = k.shape[2]

        causal_mask = torch.ones((seq_len, kv_len), device=hidden_states.device, dtype=torch.bool)
        causal_mask = torch.tril(causal_mask, diagonal=past_len)
        attn_scores = attn_scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(attn_scores.dtype).min)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                key_mask = attention_mask[:, None, None, :kv_len].to(dtype=torch.bool)
                attn_scores = attn_scores.masked_fill(~key_mask, torch.finfo(attn_scores.dtype).min)
            else:
                attn_scores = attn_scores + attention_mask[:, :, :, :kv_len].to(dtype=attn_scores.dtype)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output), present


class NayheinMiniDecoderLayer(nn.Module):
    def __init__(self, config: NayheinMiniConfig) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = NayheinMiniAttention(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = NayheinMiniMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_output, present = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class NayheinMiniModel(nn.Module):
    def __init__(self, config: NayheinMiniConfig) -> None:
        super().__init__()
        base_pos_len = min(config.max_position_embeddings, 8192)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(base_pos_len, config.hidden_size)
        self.embed_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([NayheinMiniDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> BaseModelOutputWithPast | Tuple[torch.Tensor, ...]:
        batch_size, seq_len = input_ids.shape
        past_key_values = past_key_values or tuple([None] * len(self.layers))
        past_len = 0 if not past_key_values or past_key_values[0] is None else past_key_values[0][0].shape[2]

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embed_ids = position_ids % self.position_embeddings.num_embeddings
        token_embeds = self.embed_tokens(input_ids)
        position_embeds = self.position_embeddings(pos_embed_ids)
        if position_embeds.size(-1) != token_embeds.size(-1):
            if position_embeds.size(-1) < token_embeds.size(-1):
                position_embeds = F.pad(position_embeds, (0, token_embeds.size(-1) - position_embeds.size(-1)))
            else:
                position_embeds = position_embeds[..., :token_embeds.size(-1)]
        hidden_states = token_embeds + position_embeds
        hidden_states = self.embed_layernorm(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        next_past = () if use_cache else None

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_past = past_key_values[layer_idx] if layer_idx < len(past_key_values) else None
            if self.gradient_checkpointing and self.training and not use_cache and layer_past is None:
                def custom_forward(*inputs):
                    return decoder_layer(inputs[0], attention_mask=inputs[1], position_ids=inputs[2], use_cache=False)[0]

                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
                if use_cache:
                    next_past = next_past + (present,)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            outputs = (hidden_states,)
            if use_cache:
                outputs = outputs + (next_past,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_past,
            hidden_states=all_hidden_states,
        )


class NayheinMiniPreTrainedModel(PreTrainedModel):
    config_class = NayheinMiniConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NayheinMiniDecoderLayer"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

class NayheinMiniForCausalLM(NayheinMiniPreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config: NayheinMiniConfig) -> None:
        super().__init__(config)
        self.model = NayheinMiniModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.model.gradient_checkpointing = True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        beam_idx: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        reordered = []
        for layer_past in past_key_values:
            reordered.append(tuple(state.index_select(0, beam_idx) for state in layer_past))
        return tuple(reordered)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | Tuple[torch.Tensor, ...]:
        if input_ids is None:
            raise ValueError("input_ids is required")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        use_cache = self.config.use_cache if use_cache is None else use_cache

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(model_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            outputs = (logits, model_outputs.past_key_values, model_outputs.hidden_states)
            return ((loss,) + outputs) if loss is not None else outputs

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
        )
