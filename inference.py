#!/usr/bin/env python3
"""Inference for the dual-mode AR + Diffusion model."""

import argparse
import time
import torch
from typing import Dict, Any

from pretrain import DualModeModel, get_default_tokenizer

model = None
tokenizer = None
device = None


def load_model(checkpoint_path: str, config_path: str, device_str: str = None):
    global model, tokenizer, device
    config = torch.load(config_path, map_location="cpu")
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    model = DualModeModel(
        vocab_size=config.get("vocab_size", 16000),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        head_dim=config.get("head_dim", 64),
        max_seq_len=config.get("max_seq_len", 8192),
        use_flash_attn=False,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    tokenizer = get_default_tokenizer()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, tokenizer


def complete(prompt: str, max_tokens: int = 100, temperature: float = 1.0, top_p: float = 1.0, mode: str = "ar", verbose: bool = True) -> Dict[str, Any]:
    global model, tokenizer, device
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    start_time = time.time()
    first_token_time = None
    generated_ids = []
    max_length = min(input_length + max_tokens, model.max_seq_len)
    if verbose:
        print(f"[{mode}] Prompt: {prompt[:50]}...")
        print(f"[{mode}] Input tokens: {input_length}")
    with torch.no_grad():
        for i in range(max_length - input_length):
            outputs = model(input_ids, use_cache=False, mode=mode)
            # Get logits based on mode
            ar_logits = outputs.get("ar_logits")
            diffusion_logits = outputs.get("diffusion_logits")
            
            if mode == "combined" or mode == "reasoning":
                # Average logits from both modes
                if ar_logits is not None and diffusion_logits is not None:
                    logits = (ar_logits + diffusion_logits) / 2
                elif ar_logits is not None:
                    logits = ar_logits
                elif diffusion_logits is not None:
                    logits = diffusion_logits
                else:
                    raise RuntimeError(f"No logits for mode: {mode}")
            else:
                logits = ar_logits if ar_logits is not None else diffusion_logits
                if logits is None:
                    raise RuntimeError(f"No logits for mode: {mode}")
            next_token_logits = logits[0, -1, :] / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            if first_token_time is None:
                first_token_time = time.time() - start_time
                if verbose:
                    print(f"[{mode}] First token at {first_token_time:.3f}s")
            if next_token.item() == tokenizer.eos_token_id:
                break
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if verbose:
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"[{mode}] {len(generated_ids)}: {token_text!r}")
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    total_time = time.time() - start_time
    ttft = first_token_time if first_token_time else 0
    tps = len(generated_ids) / total_time if total_time > 0 else 0
    if verbose:
        print(f"[{mode}] Generated {len(generated_ids)} tokens in {total_time:.2f}s")
        print(f"[{mode}] TTFT: {ttft:.3f}s, TPS: {tps:.2f}")
        print(f"[{mode}] Output: {generated_text[:200]}...")
    return {"text": generated_text, "ttft": ttft, "tps": tps, "total_tokens": len(generated_ids), "finish_reason": "stop" if generated_ids else "length"}


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to config .pt file")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt for completion")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling")
    parser.add_argument("--mode", type=str, default="ar", choices=["ar", "diffusion", "combined", "reasoning"], help="Generation mode: ar=auto-regressive, diffusion=masked, combined=avg both, reasoning=AR then diffusion")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    load_model(args.checkpoint, args.config)
    result = complete(args.prompt, args.max_tokens, args.temperature, args.top_p, args.mode, verbose=not args.quiet)
    print(f"\n=== RESULT ===")
    print(f"Text: {result['text']}")
    print(f"TTFT: {result['ttft']:.3f}s, TPS: {result['tps']:.2f}, Tokens: {result['total_tokens']}")


if __name__ == "__main__":
    main()
