"""
Architecture utilities for model configuration and parameter estimation.

This module provides functions for:
- Estimating parameter counts for DualModeModel architectures
- Computing optimal architecture configurations from target parameter counts
- Normalizing head counts for RoPE compatibility
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


# Predefined base profiles for common model sizes
BASE_PROFILES = {
    "tiny": {
        "hidden_size": 128,
        "num_layers": 4,
        "num_heads": 2,
        "mlp_ratio": 4.0,
        "description": "~2M parameters",
    },
    "small": {
        "hidden_size": 256,
        "num_layers": 6,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "description": "~10M parameters",
    },
    "medium": {
        "hidden_size": 512,
        "num_layers": 10,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "description": "~50M parameters",
    },
    "large": {
        "hidden_size": 768,
        "num_layers": 14,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "description": "~150M parameters",
    },
    "xl": {
        "hidden_size": 1024,
        "num_layers": 18,
        "num_heads": 16,
        "mlp_ratio": 4.0,
        "description": "~400M parameters",
    },
}


def estimate_params(config: Dict[str, Any]) -> int:
    """
    Estimate DualModeModel parameter count using architecture-aware formula.
    
    Args:
        config: Model configuration dict with keys:
            - hidden_size
            - num_layers
            - num_heads
            - head_dim
            - mlp_ratio (optional, default 4.0)
            - max_seq_len (optional, default 2048)
            - base_vocab_size or vocab_size
            - mtp_enabled (optional, default False)
            - mtp_num_heads (optional, default 0)
    
    Returns:
        Estimated total parameter count
    """
    hidden_size = int(config["hidden_size"])
    num_layers = int(config["num_layers"])
    num_heads = int(config["num_heads"])
    head_dim = int(config["head_dim"])
    mlp_ratio = float(config.get("mlp_ratio", 4.0))
    max_seq_len = int(config.get("max_seq_len", 2048))
    base_vocab_size = int(config.get("base_vocab_size", config.get("vocab_size", 32000) - 1))
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


def normalize_heads(hidden_size: int, preferred_heads: int) -> Tuple[int, int]:
    """
    Return (num_heads, head_dim) with hidden_size divisible by num_heads and even head_dim for RoPE.
    
    RoPE requires even head dimensions because rotate_half splits the last dim in half.
    
    Args:
        hidden_size: Model hidden dimension
        preferred_heads: Preferred number of attention heads
    
    Returns:
        Tuple of (num_heads, head_dim) that satisfies constraints
    """
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
    """
    Calculate target configuration to reach a target parameter count.
    
    Args:
        current_config: Current model configuration
        target_params: Target parameter count
        method: Scaling method - "depth", "width", or "width+depth"
    
    Returns:
        New configuration dict with adjusted hidden_size and num_layers
    """
    base_vocab_size = int(current_config.get("base_vocab_size", current_config.get("vocab_size", 32000) - 1))
    hidden_size = int(current_config.get("hidden_size", 512))
    num_layers = int(current_config.get("num_layers", 10))
    num_heads = int(current_config.get("num_heads", 8))
    head_dim = int(current_config.get("head_dim", max(1, hidden_size // max(1, num_heads))))
    mlp_ratio = float(current_config.get("mlp_ratio", 4.0))
    max_seq_len = int(current_config.get("max_seq_len", 2048))

    def make_cfg(h: int, l: int, nh: Optional[int] = None) -> Dict[str, Any]:
        nh_val = nh if nh is not None else num_heads
        nh_val, hd_val = normalize_heads(h, nh_val)
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

    current_exact = estimate_params(make_cfg(hidden_size, num_layers, num_heads))
    target_params = int(target_params)

    if method == "depth":
        low, high = 1, max(2, int(num_layers * max(2.0, target_params / max(1, current_exact))))
        best_cfg = make_cfg(hidden_size, num_layers, num_heads)
        best_delta = abs(estimate_params(best_cfg) - target_params)
        while low <= high:
            mid = (low + high) // 2
            cfg = make_cfg(hidden_size, mid, num_heads)
            p = estimate_params(cfg)
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
        best_delta = abs(estimate_params(best_cfg) - target_params)

        lo = (low_h // num_heads) * num_heads
        hi = ((high_h + num_heads - 1) // num_heads) * num_heads
        if lo < num_heads:
            lo = num_heads

        while lo <= hi:
            mid = ((lo + hi) // (2 * num_heads)) * num_heads
            mid = max(num_heads, mid)
            cfg = make_cfg(mid, num_layers, num_heads)
            p = estimate_params(cfg)
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
    best_params = estimate_params(best_cfg)

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
            p = estimate_params(cfg)
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


def config_from_profile(profile_name: str) -> Dict[str, Any]:
    """
    Get a base configuration from a predefined profile.
    
    Args:
        profile_name: Profile name (tiny, small, medium, large, xl)
    
    Returns:
        Base configuration dict
    """
    if profile_name not in BASE_PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(BASE_PROFILES.keys())}")
    
    profile = BASE_PROFILES[profile_name]
    return {
        "hidden_size": profile["hidden_size"],
        "num_layers": profile["num_layers"],
        "num_heads": profile["num_heads"],
        "mlp_ratio": profile["mlp_ratio"],
    }


def config_from_params(
    target_params: int,
    base_profile: str = "small",
    method: str = "width+depth",
    vocab_size: int = 16000,
    max_seq_len: int = 2048,
    mtp_enabled: bool = False,
    mtp_num_heads: int = 0,
) -> Dict[str, Any]:
    """
    Calculate a model configuration to reach a target parameter count.
    
    Args:
        target_params: Target parameter count (e.g., 10_000_000 for 10M)
        base_profile: Starting profile (tiny, small, medium, large, xl)
        method: Scaling method (depth, width, width+depth)
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        mtp_enabled: Whether to enable MTP heads
        mtp_num_heads: Number of MTP heads
    
    Returns:
        Complete model configuration dict
    """
    base_config = config_from_profile(base_profile)
    base_config["vocab_size"] = vocab_size
    base_config["base_vocab_size"] = vocab_size - 1
    base_config["max_seq_len"] = max_seq_len
    base_config["mtp_enabled"] = mtp_enabled
    base_config["mtp_num_heads"] = mtp_num_heads
    
    target_config = calculate_target_config(base_config, target_params, method)
    return target_config
