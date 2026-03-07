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


# ============================================================================
# Scaling Methods
# ============================================================================

def calculate_target_config(
    current_config: Dict[str, Any],
    target_params: int,
    method: str = "width+depth",
) -> Dict[str, Any]:
    """
    Calculate target configuration based on desired parameter count.
    
    Args:
        current_config: Current model configuration
        target_params: Target parameter count
        method: Scaling method ("width", "depth", "width+depth")
    
    Returns:
        Target configuration dictionary
    """
    # Estimate current parameters
    vocab_size = current_config.get("vocab_size", 32000)
    hidden_size = current_config.get("hidden_size", 512)
    num_layers = current_config.get("num_layers", 10)
    num_heads = current_config.get("num_heads", 8)
    head_dim = current_config.get("head_dim", 64)
    
    # Rough parameter estimation
    # Embedding: vocab_size * hidden_size
    # Per layer: ~2 * hidden_size^2 + 4 * hidden_size * (hidden_size * 4)
    params_per_layer = 2 * hidden_size ** 2 + 4 * hidden_size * (hidden_size * 4)
    current_params = vocab_size * hidden_size + num_layers * params_per_layer + hidden_size * vocab_size
    
    # Calculate scaling factor
    scale_factor = (target_params / current_params) ** 0.5
    
    if method == "width":
        # Only scale width
        new_hidden_size = int(hidden_size * scale_factor)
        new_num_heads = max(1, int(num_heads * scale_factor))
        new_head_dim = new_hidden_size // new_num_heads
        new_num_layers = num_layers
        
    elif method == "depth":
        # Only scale depth
        new_hidden_size = hidden_size
        new_num_heads = num_heads
        new_head_dim = head_dim
        new_num_layers = int(num_layers * scale_factor)
        
    else:  # width+depth
        # Balance width and depth
        width_scale = scale_factor ** 0.7
        depth_scale = scale_factor ** 0.5
        
        new_hidden_size = int(hidden_size * width_scale)
        new_num_heads = max(1, int(num_heads * width_scale))
        new_head_dim = new_hidden_size // new_num_heads
        new_num_layers = int(num_layers * depth_scale)
    
    # Ensure head_dim divides hidden_size evenly
    while new_hidden_size % new_num_heads != 0:
        new_num_heads -= 1
    new_head_dim = new_hidden_size // new_num_heads
    
    return {
        "vocab_size": vocab_size,
        "hidden_size": new_hidden_size,
        "num_layers": new_num_layers,
        "num_heads": new_num_heads,
        "head_dim": new_head_dim,
        "mlp_ratio": current_config.get("mlp_ratio", 4.0),
        "max_seq_len": current_config.get("max_seq_len", 2048),
    }


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
    """
    Expand a linear layer to new dimensions.
    
    Args:
        old_layer: Old linear layer
        new_out_features: New output dimension
        new_in_features: New input dimension
    
    Returns:
        New expanded linear layer
    """
    if new_out_features is None:
        new_out_features = old_layer.out_features
    if new_in_features is None:
        new_in_features = old_layer.in_features
    
    new_layer = nn.Linear(new_in_features, new_out_features, bias=old_layer.bias is not None)
    
    # Copy weights
    with torch.no_grad():
        if new_out_features >= old_layer.out_features and new_in_features >= old_layer.in_features:
            # Expand
            new_layer.weight.data[:old_layer.out_features, :old_layer.in_features] = old_layer.weight.data
            if old_layer.bias is not None:
                new_layer.bias.data[:old_layer.out_features] = old_layer.bias.data
        elif new_out_features <= old_layer.out_features and new_in_features <= old_layer.in_features:
            # Truncate
            new_layer.weight.data = old_layer.weight.data[:new_out_features, :new_in_features]
            if old_layer.bias is not None:
                new_layer.bias.data = old_layer.bias.data[:new_out_features]
        else:
            # Copy what fits
            copy_out = min(new_out_features, old_layer.out_features)
            copy_in = min(new_in_features, old_layer.in_features)
            new_layer.weight.data[:copy_out, :copy_in] = old_layer.weight.data[:copy_out, :copy_in]
            if old_layer.bias is not None:
                new_layer.bias.data[:copy_out] = old_layer.bias.data[:copy_out]
    
    return new_layer


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
        vocab_size=new_config["vocab_size"],
        hidden_size=new_config["hidden_size"],
        num_layers=new_config["num_layers"],
        num_heads=new_config["num_heads"],
        head_dim=new_config["head_dim"],
        mlp_ratio=new_config["mlp_ratio"],
        max_seq_len=new_config["max_seq_len"],
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
        new_model.embed_layernorm.weight.data = model.embed_layernorm.weight.data.clone()
        new_model.embed_layernorm.bias.data = model.embed_layernorm.bias.data.clone()
        
        # Scale each transformer layer
        for new_layer, old_layer in zip(new_model.layers, model.layers):
            # Scale attention
            new_layer.attention.q_proj = expand_linear_layer(
                old_layer.attention.q_proj,
                new_out_features=new_config["num_heads"] * new_config["head_dim"],
            )
            new_layer.attention.k_proj = expand_linear_layer(
                old_layer.attention.k_proj,
                new_out_features=new_config["num_heads"] * new_config["head_dim"],
            )
            new_layer.attention.v_proj = expand_linear_layer(
                old_layer.attention.v_proj,
                new_out_features=new_config["num_heads"] * new_config["head_dim"],
            )
            new_layer.attention.o_proj = expand_linear_layer(
                old_layer.attention.o_proj,
                new_in_features=new_config["num_heads"] * new_config["head_dim"],
            )
            
            # Update RoPE
            new_layer.attention.rope = new_model.layers[0].attention.rope
            
            # Scale MLP
            mlp_expand = int(new_config["hidden_size"] * new_config["mlp_ratio"])
            old_mlp_expand = int(old_config["hidden_size"] * old_config["mlp_ratio"])
            
            new_layer.mlp.gate_proj = expand_linear_layer(
                old_layer.mlp.gate_proj,
                new_out_features=mlp_expand,
            )
            new_layer.mlp.up_proj = expand_linear_layer(
                old_layer.mlp.up_proj,
                new_out_features=mlp_expand,
            )
            new_layer.mlp.down_proj = expand_linear_layer(
                old_layer.mlp.down_proj,
                new_in_features=mlp_expand,
            )
            
            # Scale layer norms
            new_layer.input_layernorm.weight.data = old_layer.input_layernorm.weight.data.clone()
            new_layer.input_layernorm.bias.data = old_layer.input_layernorm.bias.data.clone()
            new_layer.post_attention_layernorm.weight.data = old_layer.post_attention_layernorm.weight.data.clone()
            new_layer.post_attention_layernorm.bias.data = old_layer.post_attention_layernorm.bias.data.clone()
        
        # Scale final layer norm
        new_model.final_layernorm.weight.data = model.final_layernorm.weight.data.clone()
        new_model.final_layernorm.bias.data = model.final_layernorm.bias.data.clone()
        
        # Scale AR head
        new_model.ar_head = expand_linear_layer(
            model.ar_head.lm_head,
            new_in_features=new_config["hidden_size"],
        )
        
        # Scale diffusion head
        new_model.diffusion_head.dense = expand_linear_layer(
            model.diffusion_head.dense,
            new_in_features=new_config["hidden_size"],
        )
        new_model.diffusion_head.decoder = expand_linear_layer(
            model.diffusion_head.decoder,
            new_out_features=new_config["vocab_size"],
        )
        new_model.diffusion_head.layer_norm.weight.data = model.diffusion_head.layer_norm.weight.data.clone()
        new_model.diffusion_head.layer_norm.bias.data = model.diffusion_head.layer_norm.bias.data.clone()
    
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
        vocab_size=model.vocab_size,
        hidden_size=model.hidden_size,
        num_layers=new_num_layers,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        mlp_ratio=mlp_ratio,
        max_seq_len=model.max_seq_len,
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
        new_model.ar_head.lm_head.bias.data = model.ar_head.lm_head.bias.data.clone()
        
        new_model.diffusion_head.dense.weight.data = model.diffusion_head.dense.weight.data.clone()
        new_model.diffusion_head.dense.bias.data = model.diffusion_head.dense.bias.data.clone()
        new_model.diffusion_head.layer_norm.weight.data = model.diffusion_head.layer_norm.weight.data.clone()
        new_model.diffusion_head.layer_norm.bias.data = model.diffusion_head.layer_norm.bias.data.clone()
        new_model.diffusion_head.decoder.weight.data = model.diffusion_head.decoder.weight.data.clone()
        new_model.diffusion_head.decoder.bias.data = model.diffusion_head.decoder.bias.data.clone()
    
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
        vocab_size=config.get("vocab_size", 16000),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        head_dim=config.get("head_dim", 64),
        max_seq_len=config.get("max_seq_len", 8192),
    )
    model.load_state_dict(state_dict)
    
    print(f"Loaded model from {input_path}")
    
    # Get current config
    current_config = {
        "vocab_size": model.vocab_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "head_dim": model.head_dim,
        "mlp_ratio": 4.0,
        "max_seq_len": model.max_seq_len,
    }
    
    print(f"Current config: {current_config}")
    print(f"Current parameters: {model.get_num_params():,}")
    
    # Calculate target config
    target_config = calculate_target_config(current_config, target_params, method)
    print(f"\nTarget config: {target_config}")
    
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
        "current_config": current_config,
        "target_config": target_config,
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
