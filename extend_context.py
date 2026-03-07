#!/usr/bin/env python3
"""
Extend context window using YaRN (Yet Another RoPE Extension).

YaRN improves RoPE extrapolation by:
1. Interpolating position embeddings
2. Adding attention scaling for better extrapolation

Reference: https://arxiv.org/abs/2309.00071

Usage:
    python extend_context.py --checkpoint model.pt --config config.pt --target-context 65536
"""

import argparse
import torch
import math


class YaRNPositionEmbedding:
    """YaRN Position Embedding for extended context windows."""
    
    def __init__(self, dim: int, base_seq_len: int, max_seq_len: int,
                 rope_scale: float = 1.0, rope_factor: float = 1.0):
        self.dim = dim
        self.base_seq_len = base_seq_len
        self.max_seq_len = max_seq_len
        self.rope_scale = rope_scale
        self.rope_factor = rope_factor
        
        self.original_inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.scaled_inv_freq = self.original_inv_freq * rope_scale
        
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.original_inv_freq.device)
        t_scaled = t * self.rope_factor
        freqs = torch.outer(t_scaled, self.scaled_inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
    
    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)
    
    def get_attention_scale(self, seq_len: int) -> float:
        if seq_len <= self.base_seq_len:
            return 1.0
        return (seq_len / self.base_seq_len) ** (-0.1 * math.log(self.rope_factor + 1e-5))


SCALING_FACTORS = {
    (8192, 16384): {"rope_scale": 0.5, "rope_factor": 1.0},
    (8192, 32768): {"rope_scale": 0.25, "rope_factor": 2.0},
    (8192, 65536): {"rope_scale": 0.125, "rope_factor": 4.0},
}


def get_scaling_params(base_ctx: int, target_ctx: int, method: str = "yarn") -> dict:
    key = (base_ctx, target_ctx)
    if method == "yarn" and key in SCALING_FACTORS:
        return SCALING_FACTORS[key]
    rope_scale = base_ctx / target_ctx
    if method == "yarn":
        rope_factor = target_ctx / base_ctx
        return {"rope_scale": rope_scale, "rope_factor": rope_factor}
    return {"rope_scale": rope_scale, "rope_factor": 1.0}


def extend_context(checkpoint_path: str, config_path: str, output_path: str,
                   target_context: int, base_context: int = 8192, method: str = "yarn"):
    scaling = get_scaling_params(base_context, target_context, method)
    rope_scale = scaling["rope_scale"]
    rope_factor = scaling["rope_factor"]
    
    print(f"Extending context: {base_context} -> {target_context}")
    print(f"Method: {method}, Scale: {rope_scale}, Factor: {rope_factor}")
    
    config = torch.load(config_path, map_location="cpu")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    config["max_seq_len"] = target_context
    config["context_extension"] = {
        "method": method, "base_context": base_context,
        "target_context": target_context, "rope_scale": rope_scale, "rope_factor": rope_factor,
    }
    
    config_path_out = output_path.replace(".pt", "_config.pt")
    torch.save(config, config_path_out)
    torch.save(state_dict, output_path)
    print(f"Saved to: {output_path}")


def apply_yarn_to_model(model, target_context: int, base_context: int = 8192, method: str = "yarn"):
    scaling = get_scaling_params(base_context, target_context, method)
    rope_scale = scaling["rope_scale"]
    rope_factor = scaling["rope_factor"]
    
    if hasattr(model, 'max_seq_len'):
        model.max_seq_len = target_context
    
    for name, module in model.named_modules():
        if "rope" in name.lower() and hasattr(module, "inv_freq"):
            module.inv_freq = module.inv_freq * rope_scale
            module.rope_scale = rope_scale
            module.rope_factor = rope_factor
    
    model.yarn_enabled = True
    model.yarn_scale = rope_scale
    model.yarn_factor = rope_factor
    model.yarn_base_ctx = base_context
    print(f"YaRN applied: {base_context} -> {target_context}")


def main():
    parser = argparse.ArgumentParser(description="Extend context window with YaRN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=str, required=True, help="Model config path")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--target-context", type=int, default=65536, help="Target context length")
    parser.add_argument("--base-context", type=int, default=8192, help="Base context length")
    parser.add_argument("--method", type=str, default="yarn", choices=["yarn", "linear"], help="Extension method")
    args = parser.parse_args()
    
    extend_context(args.checkpoint, args.config, args.output,
                   args.target_context, args.base_context, args.method)


if __name__ == "__main__":
    main()
