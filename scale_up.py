#!/usr/bin/env python3
"""
Model scaling script for expanding pretrained weights to larger models.

This script supports multiple scaling methods:
- Weight Replication: repeat weights across more layers
- Width Expansion: expand hidden sizes, attention dimensions
- Stack Layers: add additional transformer blocks

Usage:
    python scale_up.py \
        --input ./checkpoints/10m_pretrain \
        --output ./checkpoints/4b_scaled \
        --target-parameters 4000000000 \
        --method width+depth \
        --interpolate-pos-embeddings linear
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig, PreTrainedModel
from transformers.modeling_utils import load_state_dict

from pretrain import DualModeModel, ARHead, DiffusionHead
from utils.architecture import estimate_params, normalize_heads, calculate_target_config


# ============================================================================
# Helper Functions
# ============================================================================

def _estimate_params(config: Dict[str, Any]) -> int:
    """Estimate DualModeModel parameter count using architecture-aware formula."""
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    num_heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    mlp_ratio = float(config.get("mlp_ratio", 4.0))
    max_seq_len = int(config.get("max_seq_len", 2048))
    base_vocab_size = int(config["base_vocab_size"])
    mtp_enabled = bool(config.get("mtp_enabled", False))
    mtp_num_heads = int(config.get("mtp_num_heads", 0)) if mtp_enabled else 0

    # DualModeModel reserves one extra mask token internally.
    effective_vocab_size = base_vocab_size + 1
    base_pos_len = min(max_seq_len, 8192)

    attn_dim = num_heads * head_dim
    mlp_dim = int(hidden_size * mlp_ratio)

    # Embeddings + embed LN
    token_embed = effective_vocab_size * hidden_size
    pos_embed = base_pos_len * hidden_size
    embed_ln = 2 * hidden_size

    # Per transformer block
    # q/k/v/o (bias=False) + MLP gate/up/down (bias=False) + 2 layer norms
    per_layer = (4 * hidden_size * attn_dim) + (3 * hidden_size * mlp_dim) + (4 * hidden_size)

    # Final LN
    final_ln = 2 * hidden_size

    # Heads: AR lm_head (bias=False) + diffusion dense (bias=True) + diffusion LN + diffusion decoder (bias=False)
    ar_head = hidden_size * effective_vocab_size
    diffusion_head = (hidden_size * hidden_size + hidden_size) + (2 * hidden_size) + (hidden_size * effective_vocab_size)
    mtp_heads = mtp_num_heads * (hidden_size * effective_vocab_size)

    total = token_embed + pos_embed + embed_ln + (num_layers * per_layer) + final_ln + ar_head + diffusion_head + mtp_heads
    return int(total)


def _normalize_heads(hidden_size: int, preferred_heads: int) -> Tuple[int, int]:
    """Return (num_heads, head_dim) with hidden_size divisible by num_heads and even head_dim for RoPE."""
    num_heads = max(1, min(preferred_heads, hidden_size))
    while hidden_size % num_heads != 0 and num_heads > 1:
        num_heads -= 1

    head_dim = hidden_size // num_heads
    if head_dim % 2 != 0:
        # Prefer reducing heads so head_dim becomes even while preserving divisibility.
        found = False
        for nh in range(num_heads - 1, 0, -1):
            if hidden_size % nh == 0 and ((hidden_size // nh) % 2 == 0):
                num_heads = nh
                head_dim = hidden_size // nh
                found = True
                break
        if not found:
            # Fallback: bump hidden size minimally to make head_dim even
            hidden_size += num_heads
            head_dim = hidden_size // num_heads

    return num_heads, head_dim


def calculate_target_config(
    current_config: Dict[str, Any],
    target_params: int,
    method: str = "width+depth",
) -> Dict[str, Any]:
    """Calculate target configuration using architecture-aware parameter estimation."""
    base_vocab_size = int(current_config.get("base_vocab_size", current_config.get("vocab_size", 32000) - 1))
    hidden_size = int(current_config.get("hidden_size", 512))
    num_layers = int(current_config.get("num_layers", 10))
    num_heads = int(current_config.get("num_heads", 8))
    head_dim = int(current_config.get("head_dim", max(1, hidden_size // max(1, num_heads))))
    mlp_ratio = float(current_config.get("mlp_ratio", 4.0))
    max_seq_len = int(current_config.get("max_seq_len", 2048))

    def make_cfg(h: int, l: int, nh: Optional[int] = None) -> Dict[str, Any]:
        nh_val = nh if nh is not None else num_heads
        nh_val, hd_val = _normalize_heads(h, nh_val)
        return {
            "base_vocab_size": base_vocab_size,
            "vocab_size": base_vocab_size + 1,
            "hidden_size": int(h),
            "num_layers": int(max(1, l)),
            "num_heads": int(nh_val),
            "head_dim": int(hd_val),
            "mlp_ratio": mlp_ratio,
            "max_seq_len": max_seq_len,
            "mtp_enabled": bool(current_config.get("mtp_enabled", False)),
            "mtp_num_heads": int(current_config.get("mtp_num_heads", 0)),
        }

    current_exact = _estimate_params(make_cfg(hidden_size, num_layers, num_heads))
    target_params = int(target_params)

    if method == "depth":
        low, high = 1, max(2, int(num_layers * max(2.0, target_params / max(1, current_exact))))
        best_cfg = make_cfg(hidden_size, num_layers, num_heads)
        best_delta = abs(_estimate_params(best_cfg) - target_params)
        while low <= high:
            mid = (low + high) // 2
            cfg = make_cfg(hidden_size, mid, num_heads)
            p = _estimate_params(cfg)
            delta = abs(p - target_params)
            if delta < best_delta:
                best_cfg, best_delta = cfg, delta
            if p < target_params:
                low = mid + 1
            else:
                high = mid - 1
        return best_cfg

    if method == "width":
        low_h = max(num_heads, hidden_size // 2)
        high_h = max(hidden_size + num_heads, int(hidden_size * max(2.0, (target_params / max(1, current_exact)) ** 0.6)) + num_heads)
        best_cfg = make_cfg(hidden_size, num_layers, num_heads)
        best_delta = abs(_estimate_params(best_cfg) - target_params)

        lo = (low_h // num_heads) * num_heads
        hi = ((high_h + num_heads - 1) // num_heads) * num_heads
        if lo < num_heads:
            lo = num_heads

        while lo <= hi:
            mid = ((lo + hi) // (2 * num_heads)) * num_heads
            mid = max(num_heads, mid)
            cfg = make_cfg(mid, num_layers, num_heads)
            p = _estimate_params(cfg)
            delta = abs(p - target_params)
            if delta < best_delta:
                best_cfg, best_delta = cfg, delta
            if p < target_params:
                lo = mid + num_heads
            else:
                hi = mid - num_heads
        return best_cfg

    # width+depth: balanced start then local greedy refinement
    scale = max(1e-9, target_params / max(1, current_exact))
    start_hidden = max(num_heads, int(hidden_size * (scale ** 0.30)))
    start_hidden = ((start_hidden + num_heads - 1) // num_heads) * num_heads
    start_layers = max(1, int(round(num_layers * (scale ** 0.40))))

    best_cfg = make_cfg(start_hidden, start_layers, num_heads)
    best_params = _estimate_params(best_cfg)

    step_h = max(num_heads, (hidden_size // num_heads) * num_heads)
    step_l = max(1, num_layers // 4)

    for _ in range(120):
        improved = False
        candidates = []
        h = best_cfg["hidden_size"]
        l = best_cfg["num_layers"]

        for nh, nl in [
            (h + step_h, l),
            (max(num_heads, h - step_h), l),
            (h, l + step_l),
            (h, max(1, l - step_l)),
            (h + step_h, l + step_l),
            (max(num_heads, h - step_h), max(1, l - step_l)),
        ]:
            cfg = make_cfg(nh, nl, num_heads)
            p = _estimate_params(cfg)
            candidates.append((abs(p - target_params), cfg, p))

        candidates.sort(key=lambda x: x[0])
        cand_delta, cand_cfg, cand_params = candidates[0]
        if cand_delta < abs(best_params - target_params):
            best_cfg, best_params = cand_cfg, cand_params
            improved = True

        if not improved:
            if step_h > num_heads:
                step_h = max(num_heads, step_h // 2)
            if step_l > 1:
                step_l = max(1, step_l // 2)
            if step_h == num_heads and step_l == 1:
                break

    return best_cfg


def expand_embedding_dim(
    old_embed: nn.Embedding,
    new_embedding_dim: int,
) -> nn.Embedding:
    """
    Expand embedding layer's embedding dimension (hidden size).
    
    Args:
        old_embed: Old embedding layer
        new_embedding_dim: New embedding dimension
    
    Returns:
        New embedding layer with expanded dimension
    """
    old_num_embeddings, old_embedding_dim = old_embed.weight.shape
    new_embed = nn.Embedding(old_num_embeddings, new_embedding_dim)
    
    with torch.no_grad():
        # Copy existing embeddings
        new_embed.weight.data[:, :old_embedding_dim] = old_embed.weight.data
        # Initialize new dimensions with small random values
        if new_embedding_dim > old_embedding_dim:
            torch.nn.init.normal_(new_embed.weight.data[:, old_embedding_dim:], mean=0.0, std=0.02)
    
    return new_embed


def interpolate_position_embeddings(
    old_embed: nn.Embedding,
    new_seq_len: int,
    method: str = "linear",
) -> nn.Embedding:
    """
    Interpolate position embeddings to support longer sequences.
    
    Args:
        old_embed: Old position embedding layer
        new_seq_len: New maximum sequence length
        method: Interpolation method ("linear", "interleave")
    
    Returns:
        New position embedding layer
    """
    old_seq_len, hidden_dim = old_embed.weight.shape
    new_embed = nn.Embedding(new_seq_len, hidden_dim)
    
    if new_seq_len <= old_seq_len:
        # Just truncate if shorter
        new_embed.weight.data[:old_seq_len] = old_embed.weight.data[:old_seq_len]
    else:
        # Interpolate
        if method == "linear":
            # Linear interpolation
            indices = torch.linspace(0, old_seq_len - 1, new_seq_len)
            for i, idx in enumerate(indices):
                low = int(idx)
                high = min(low + 1, old_seq_len - 1)
                weight = idx - low
                new_embed.weight.data[i] = (1 - weight) * old_embed.weight.data[low] + weight * old_embed.weight.data[high]
        elif method == "interleave":
            # Interleave new positions
            for i in range(new_seq_len):
                old_idx = i * old_seq_len // new_seq_len
                new_embed.weight.data[i] = old_embed.weight.data[old_idx]
        else:
            # Copy and extend
            new_embed.weight.data[:old_seq_len] = old_embed.weight.data
            # Extrapolate
            if new_seq_len > old_seq_len:
                step = (old_embed.weight.data[-1] - old_embed.weight.data[0]) / (old_seq_len - 1)
                for i in range(old_seq_len, new_seq_len):
                    new_embed.weight.data[i] = old_embed.weight.data[-1] + step * (i - old_seq_len + 1)
    
    return new_embed


def expand_linear_layer(
    old_layer: nn.Linear,
    new_out_features: Optional[int] = None,
    new_in_features: Optional[int] = None,
) -> nn.Linear:
    """Expand a linear layer while preserving overlapping weights safely."""
    if new_out_features is None:
        new_out_features = old_layer.out_features
    if new_in_features is None:
        new_in_features = old_layer.in_features

    new_layer = nn.Linear(new_in_features, new_out_features, bias=old_layer.bias is not None)

    with torch.no_grad():
        copy_out = min(new_out_features, old_layer.out_features)
        copy_in = min(new_in_features, old_layer.in_features)
        new_layer.weight[:copy_out, :copy_in] = old_layer.weight[:copy_out, :copy_in]
        if old_layer.bias is not None and new_layer.bias is not None:
            new_layer.bias[:copy_out] = old_layer.bias[:copy_out]

    return new_layer


def _copy_param_overlap(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Copy overlapping region from src tensor into dst tensor without changing dst shape."""
    with torch.no_grad():
        if dst.ndim != src.ndim:
            raise ValueError(f"Rank mismatch: dst {dst.shape} vs src {src.shape}")
        slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
        dst[slices] = src[slices]


def scale_model_width(
    model: DualModeModel,
    new_config: Dict[str, Any],
) -> DualModeModel:
    """Scale model by expanding hidden dimensions."""
    
    old_config = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "head_dim": model.head_dim,
        "mlp_ratio": model.mlp_ratio if hasattr(model, 'mlp_ratio') else 4.0,
    }
    
    # Create new model
    new_model = DualModeModel(
        vocab_size=new_config.get("base_vocab_size", new_config["vocab_size"] - 1),
        hidden_size=new_config["hidden_size"],
        num_layers=new_config["num_layers"],
        num_heads=new_config["num_heads"],
        head_dim=new_config["head_dim"],
        mlp_ratio=new_config["mlp_ratio"],
        max_seq_len=new_config["max_seq_len"],
        mtp_enabled=getattr(model, "mtp_enabled", False),
        mtp_num_heads=getattr(model, "mtp_num_heads", 3),
        mtp_loss_weights=getattr(model, "mtp_loss_weights", [1.0, 0.7, 0.5]),
    )
    
    with torch.no_grad():
        # Scale token embeddings (expand embedding dimension for width scaling)
        new_model.token_embeddings = expand_embedding_dim(
            model.token_embeddings,
            new_embedding_dim=new_config["hidden_size"],
        )

        # Scale position embeddings (interpolate if needed)
        new_model.position_embeddings = interpolate_position_embeddings(
            model.position_embeddings,
            new_config["max_seq_len"],
        )

        # Scale layer norm
        _copy_param_overlap(new_model.embed_layernorm.weight, model.embed_layernorm.weight)
        _copy_param_overlap(new_model.embed_layernorm.bias, model.embed_layernorm.bias)

        # Scale each transformer layer
        for new_layer, old_layer in zip(new_model.layers, model.layers):
            # Attention projections (expand BOTH in/out dimensions as needed)
            _copy_param_overlap(new_layer.attention.q_proj.weight, old_layer.attention.q_proj.weight)
            if new_layer.attention.q_proj.bias is not None and old_layer.attention.q_proj.bias is not None:
                _copy_param_overlap(new_layer.attention.q_proj.bias, old_layer.attention.q_proj.bias)

            _copy_param_overlap(new_layer.attention.k_proj.weight, old_layer.attention.k_proj.weight)
            if new_layer.attention.k_proj.bias is not None and old_layer.attention.k_proj.bias is not None:
                _copy_param_overlap(new_layer.attention.k_proj.bias, old_layer.attention.k_proj.bias)

            _copy_param_overlap(new_layer.attention.v_proj.weight, old_layer.attention.v_proj.weight)
            if new_layer.attention.v_proj.bias is not None and old_layer.attention.v_proj.bias is not None:
                _copy_param_overlap(new_layer.attention.v_proj.bias, old_layer.attention.v_proj.bias)

            _copy_param_overlap(new_layer.attention.o_proj.weight, old_layer.attention.o_proj.weight)
            if new_layer.attention.o_proj.bias is not None and old_layer.attention.o_proj.bias is not None:
                _copy_param_overlap(new_layer.attention.o_proj.bias, old_layer.attention.o_proj.bias)

            # Update RoPE
            new_layer.attention.rope = new_model.layers[0].attention.rope

            # Scale MLP projections
            _copy_param_overlap(new_layer.mlp.gate_proj.weight, old_layer.mlp.gate_proj.weight)
            if new_layer.mlp.gate_proj.bias is not None and old_layer.mlp.gate_proj.bias is not None:
                _copy_param_overlap(new_layer.mlp.gate_proj.bias, old_layer.mlp.gate_proj.bias)

            _copy_param_overlap(new_layer.mlp.up_proj.weight, old_layer.mlp.up_proj.weight)
            if new_layer.mlp.up_proj.bias is not None and old_layer.mlp.up_proj.bias is not None:
                _copy_param_overlap(new_layer.mlp.up_proj.bias, old_layer.mlp.up_proj.bias)

            _copy_param_overlap(new_layer.mlp.down_proj.weight, old_layer.mlp.down_proj.weight)
            if new_layer.mlp.down_proj.bias is not None and old_layer.mlp.down_proj.bias is not None:
                _copy_param_overlap(new_layer.mlp.down_proj.bias, old_layer.mlp.down_proj.bias)

            # Scale layer norms
            _copy_param_overlap(new_layer.input_layernorm.weight, old_layer.input_layernorm.weight)
            _copy_param_overlap(new_layer.input_layernorm.bias, old_layer.input_layernorm.bias)
            _copy_param_overlap(new_layer.post_attention_layernorm.weight, old_layer.post_attention_layernorm.weight)
            _copy_param_overlap(new_layer.post_attention_layernorm.bias, old_layer.post_attention_layernorm.bias)

        # Scale final layer norm
        _copy_param_overlap(new_model.final_layernorm.weight, model.final_layernorm.weight)
        _copy_param_overlap(new_model.final_layernorm.bias, model.final_layernorm.bias)

        # Scale AR head
        _copy_param_overlap(new_model.ar_head.lm_head.weight, model.ar_head.lm_head.weight)
        if new_model.ar_head.lm_head.bias is not None and model.ar_head.lm_head.bias is not None:
            _copy_param_overlap(new_model.ar_head.lm_head.bias, model.ar_head.lm_head.bias)

        # Scale diffusion head
        _copy_param_overlap(new_model.diffusion_head.dense.weight, model.diffusion_head.dense.weight)
        if new_model.diffusion_head.dense.bias is not None and model.diffusion_head.dense.bias is not None:
            _copy_param_overlap(new_model.diffusion_head.dense.bias, model.diffusion_head.dense.bias)
        _copy_param_overlap(new_model.diffusion_head.layer_norm.weight, model.diffusion_head.layer_norm.weight)
        _copy_param_overlap(new_model.diffusion_head.layer_norm.bias, model.diffusion_head.layer_norm.bias)
        _copy_param_overlap(new_model.diffusion_head.decoder.weight, model.diffusion_head.decoder.weight)
        if new_model.diffusion_head.decoder.bias is not None and model.diffusion_head.decoder.bias is not None:
            _copy_param_overlap(new_model.diffusion_head.decoder.bias, model.diffusion_head.decoder.bias)

        # Scale MTP heads if enabled
        if getattr(model, "mtp_enabled", False) and len(getattr(model, "mtp_heads", [])) > 0:
            for idx, old_mtp_head in enumerate(model.mtp_heads):
                if idx < len(new_model.mtp_heads):
                    _copy_param_overlap(new_model.mtp_heads[idx].lm_head.weight, old_mtp_head.lm_head.weight)
                    if new_model.mtp_heads[idx].lm_head.bias is not None and old_mtp_head.lm_head.bias is not None:
                        _copy_param_overlap(new_model.mtp_heads[idx].lm_head.bias, old_mtp_head.lm_head.bias)
    
    return new_model


def scale_model_depth(
    model: DualModeModel,
    new_num_layers: int,
) -> DualModeModel:
    """Scale model by stacking more layers."""
    
    # Get mlp_ratio from model if available, otherwise compute from gate_proj
    if hasattr(model, 'mlp_ratio'):
        mlp_ratio = model.mlp_ratio
    elif len(model.layers) > 0 and hasattr(model.layers[0], 'mlp'):
        # Compute from gate_proj out_features / hidden_size
        mlp_ratio = model.layers[0].mlp.gate_proj.out_features / model.hidden_size
    else:
        mlp_ratio = 4.0
    
    # Create new model with more layers
    new_model = DualModeModel(
        vocab_size=getattr(model, "original_vocab_size", model.vocab_size - 1),
        hidden_size=model.hidden_size,
        num_layers=new_num_layers,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        mlp_ratio=mlp_ratio,
        max_seq_len=model.max_seq_len,
        mtp_enabled=getattr(model, "mtp_enabled", False),
        mtp_num_heads=getattr(model, "mtp_num_heads", 3),
        mtp_loss_weights=getattr(model, "mtp_loss_weights", [1.0, 0.7, 0.5]),
    )
    
    with torch.no_grad():
        # Copy token embeddings
        new_model.token_embeddings.weight.data = model.token_embeddings.weight.data.clone()
        new_model.position_embeddings.weight.data = model.position_embeddings.weight.data.clone()
        
        # Copy layer norms
        new_model.embed_layernorm.weight.data = model.embed_layernorm.weight.data.clone()
        new_model.embed_layernorm.bias.data = model.embed_layernorm.bias.data.clone()
        new_model.final_layernorm.weight.data = model.final_layernorm.weight.data.clone()
        new_model.final_layernorm.bias.data = model.final_layernorm.bias.data.clone()
        
        # Copy transformer layers - repeat in a round-robin fashion
        num_old_layers = len(model.layers)
        for i in range(new_num_layers):
            old_layer_idx = i % num_old_layers
            old_layer = model.layers[old_layer_idx]
            new_layer = new_model.layers[i]
            
            # Copy attention weights
            new_layer.attention.q_proj.weight.data = old_layer.attention.q_proj.weight.data.clone()
            if old_layer.attention.q_proj.bias is not None:
                new_layer.attention.q_proj.bias.data = old_layer.attention.q_proj.bias.data.clone()
            new_layer.attention.k_proj.weight.data = old_layer.attention.k_proj.weight.data.clone()
            if old_layer.attention.k_proj.bias is not None:
                new_layer.attention.k_proj.bias.data = old_layer.attention.k_proj.bias.data.clone()
            new_layer.attention.v_proj.weight.data = old_layer.attention.v_proj.weight.data.clone()
            if old_layer.attention.v_proj.bias is not None:
                new_layer.attention.v_proj.bias.data = old_layer.attention.v_proj.bias.data.clone()
            new_layer.attention.o_proj.weight.data = old_layer.attention.o_proj.weight.data.clone()
            if old_layer.attention.o_proj.bias is not None:
                new_layer.attention.o_proj.bias.data = old_layer.attention.o_proj.bias.data.clone()
            
            # Copy MLP weights
            new_layer.mlp.gate_proj.weight.data = old_layer.mlp.gate_proj.weight.data.clone()
            if old_layer.mlp.gate_proj.bias is not None:
                new_layer.mlp.gate_proj.bias.data = old_layer.mlp.gate_proj.bias.data.clone()
            new_layer.mlp.up_proj.weight.data = old_layer.mlp.up_proj.weight.data.clone()
            if old_layer.mlp.up_proj.bias is not None:
                new_layer.mlp.up_proj.bias.data = old_layer.mlp.up_proj.bias.data.clone()
            new_layer.mlp.down_proj.weight.data = old_layer.mlp.down_proj.weight.data.clone()
            if old_layer.mlp.down_proj.bias is not None:
                new_layer.mlp.down_proj.bias.data = old_layer.mlp.down_proj.bias.data.clone()
            
            # Copy layer norms
            new_layer.input_layernorm.weight.data = old_layer.input_layernorm.weight.data.clone()
            new_layer.input_layernorm.bias.data = old_layer.input_layernorm.bias.data.clone()
            new_layer.post_attention_layernorm.weight.data = old_layer.post_attention_layernorm.weight.data.clone()
            new_layer.post_attention_layernorm.bias.data = old_layer.post_attention_layernorm.bias.data.clone()
        
        # Copy heads
        new_model.ar_head.lm_head.weight.data = model.ar_head.lm_head.weight.data.clone()
        if model.ar_head.lm_head.bias is not None:
            new_model.ar_head.lm_head.bias.data = model.ar_head.lm_head.bias.data.clone()
        
        new_model.diffusion_head.dense.weight.data = model.diffusion_head.dense.weight.data.clone()
        if model.diffusion_head.dense.bias is not None:
            new_model.diffusion_head.dense.bias.data = model.diffusion_head.dense.bias.data.clone()
        new_model.diffusion_head.layer_norm.weight.data = model.diffusion_head.layer_norm.weight.data.clone()
        new_model.diffusion_head.layer_norm.bias.data = model.diffusion_head.layer_norm.bias.data.clone()
        new_model.diffusion_head.decoder.weight.data = model.diffusion_head.decoder.weight.data.clone()
        if model.diffusion_head.decoder.bias is not None:
            new_model.diffusion_head.decoder.bias.data = model.diffusion_head.decoder.bias.data.clone()

        # Copy MTP heads if enabled
        if getattr(model, "mtp_enabled", False) and len(getattr(model, "mtp_heads", [])) > 0:
            for idx, old_mtp_head in enumerate(model.mtp_heads):
                if idx < len(new_model.mtp_heads):
                    new_model.mtp_heads[idx].lm_head.weight.data = old_mtp_head.lm_head.weight.data.clone()
                    if old_mtp_head.lm_head.bias is not None and new_model.mtp_heads[idx].lm_head.bias is not None:
                        new_model.mtp_heads[idx].lm_head.bias.data = old_mtp_head.lm_head.bias.data.clone()
    
    return new_model


def scale_model(
    input_path: str,
    output_path: str,
    target_params: int,
    method: str = "width+depth",
    interpolate_pos: str = "linear",
) -> DualModeModel:
    """
    Scale model from input checkpoint to target size.
    
    Args:
        input_path: Path to input checkpoint
        output_path: Path to save scaled model
        target_params: Target parameter count
        method: Scaling method ("width", "depth", "width+depth")
        interpolate_pos: Position embedding interpolation method
    
    Returns:
        Scaled model
    """
    print(f"\n=== Loading Model from {input_path} ===")
    
    input_path = Path(input_path)
    
    # Load config
    config_path = input_path / "config.pt"
    model_path = input_path / "model.pt"
    
    if not config_path.exists() or not model_path.exists():
        raise ValueError(f"Cannot load model from {input_path}. Expected config.pt and model.pt files.")
    
    config = torch.load(config_path, map_location="cpu")
    state_dict = torch.load(model_path, map_location="cpu")
    
    # Create model from config
    model = DualModeModel(
        vocab_size=config.get("original_vocab_size", config.get("vocab_size", 16000) - 1),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        head_dim=config.get("head_dim", 64),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        max_seq_len=config.get("max_seq_len", 8192),
        mtp_enabled=config.get("mtp_enabled", False),
        mtp_num_heads=config.get("mtp_num_heads", 3),
        mtp_loss_weights=config.get("mtp_loss_weights", [1.0, 0.7, 0.5]),
    )
    model.load_state_dict(state_dict)
    
    print(f"Loaded model from {input_path}")
    
    # Get current config
    # DualModeModel internally adds one mask token; expose base vocab size for scaling math.
    base_vocab_size = getattr(model, "original_vocab_size", model.vocab_size - 1)
    inferred_mlp_ratio = 4.0
    if len(model.layers) > 0 and hasattr(model.layers[0], "mlp"):
        inferred_mlp_ratio = model.layers[0].mlp.gate_proj.out_features / model.hidden_size

    current_config = {
        "base_vocab_size": int(base_vocab_size),
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "head_dim": model.head_dim,
        "mlp_ratio": inferred_mlp_ratio,
        "max_seq_len": model.max_seq_len,
        "mtp_enabled": getattr(model, "mtp_enabled", False),
        "mtp_num_heads": getattr(model, "mtp_num_heads", 3),
        "mtp_loss_weights": getattr(model, "mtp_loss_weights", [1.0, 0.7, 0.5]),
    }
    
    print(f"Current config: {current_config}")
    print(f"Current parameters: {model.get_num_params():,}")
    
    # Calculate target config
    target_config = calculate_target_config(current_config, target_params, method)
    print(f"\nTarget config: {target_config}")
    estimated_target_params = _estimate_params(target_config)
    print(f"Estimated target parameters from config: {estimated_target_params:,}")
    
    # Scale model
    print(f"\n=== Scaling Model ({method}) ===")
    
    if method == "width":
        scaled_model = scale_model_width(model, target_config)
    elif method == "depth":
        scaled_model = scale_model_depth(model, target_config["num_layers"])
    else:  # width+depth
        # First scale width
        temp_config = target_config.copy()
        temp_config["num_layers"] = current_config["num_layers"]
        scaled_model = scale_model_width(model, temp_config)
        
        # Then scale depth
        scaled_model = scale_model_depth(scaled_model, target_config["num_layers"])
    
    print(f"Scaled parameters: {scaled_model.get_num_params():,}")
    
    # Save model
    print(f"\n=== Saving Model to {output_path} ===")
    os.makedirs(output_path, exist_ok=True)
    
    # Save model state dict
    torch.save(scaled_model.state_dict(), Path(output_path) / "model.pt")
    
    # Save config
    scaling_config = {
        "input_path": str(input_path),
        "output_path": output_path,
        "method": method,
        "interpolate_pos_embeddings": interpolate_pos,
        "target_parameters": target_params,
        "estimated_target_parameters": estimated_target_params,
        "actual_scaled_parameters": scaled_model.get_num_params(),
        "current_config": current_config,
        "target_config": target_config,
        "mtp_enabled": getattr(scaled_model, "mtp_enabled", False),
        "mtp_num_heads": getattr(scaled_model, "mtp_num_heads", 0),
        "mtp_loss_weights": getattr(scaled_model, "mtp_loss_weights", [1.0, 0.7, 0.5]),
    }
    torch.save(scaling_config, Path(output_path) / "config.pt")
    
    print(f"\nScaled model saved to {output_path}")
    print(f"Config saved to {Path(output_path) / 'config.pt'}")
    
    return scaled_model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Scale up a pretrained model")
    
    parser.add_argument("--input", type=str, required=True, help="Input model path")
    parser.add_argument("--output", type=str, required=True, help="Output model path")
    parser.add_argument("--target-parameters", type=int, required=True, help="Target parameter count")
    parser.add_argument("--method", type=str, default="width+depth", 
                       choices=["width", "depth", "width+depth"], help="Scaling method")
    parser.add_argument("--interpolate-pos-embeddings", type=str, default="linear",
                       choices=["linear", "interleave", "extrapolate"], 
                       help="Position embedding interpolation method")
    
    args = parser.parse_args()
    
    # Scale model
    model = scale_model(
        input_path=args.input,
        output_path=args.output,
        target_params=args.target_parameters,
        method=args.method,
        interpolate_pos=args.interpolate_pos_embeddings,
    )
    
    print(f"\n=== Complete ===")
    print(f"Final parameter count: {model.get_num_params():,}")


if __name__ == "__main__":
    main()
