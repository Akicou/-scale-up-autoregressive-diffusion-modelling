#!/usr/bin/env python3
"""Inference for the dual-mode AR + Diffusion model."""

import argparse
import time
import math
import torch
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from pretrain import DualModeModel, get_default_tokenizer
from tools import ToolRegistry, ToolCallParser, detect_tool_calls, parse_tool_result
from tools.registry import get_default_registry

model = None
tokenizer = None
device = None
tool_registry = None
tool_parser = None
MASK_TOKEN_ID = 0


def init_tools():
    """Initialize tool registry and parser."""
    global tool_registry, tool_parser
    tool_registry = get_default_registry()
    tool_parser = ToolCallParser()
    print(f"Initialized {len(tool_registry.list_tools())} tools: {tool_registry.list_tools()}")


def load_model(checkpoint_path: str, config_path: str, device_str: str = None):
    global model, tokenizer, device
    
    checkpoint_path = Path(checkpoint_path)
    config_path = Path(config_path)
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
        diffusion_mask_prob=config.get("diffusion_mask_prob", 0.15),
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    tokenizer_dir = config_path.parent
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"Loaded tokenizer from {tokenizer_dir}")
    except Exception:
        tokenizer = get_default_tokenizer()
        print("Tokenizer files not found next to the checkpoint, using the default tokenizer fallback.")
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    return model, tokenizer


def apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus filtering to logits."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    filtered_logits = logits.clone()
    filtered_logits[sorted_indices[sorted_indices_to_remove]] = float("-inf")
    return filtered_logits


def sample_next_token(next_token_logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample a token id from logits."""
    scaled_logits = next_token_logits / max(temperature, 1e-5)
    filtered_logits = apply_top_p_filter(scaled_logits, top_p)
    probs = torch.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def complete_ar(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool) -> tuple:
    """Generate tokens using auto-regressive mode (one token at a time)."""
    generated_ids = []
    start_time = time.time()
    first_token_time = None
    input_length = input_ids.shape[1]
    max_length = min(input_length + max_tokens, model.max_seq_len)
    
    with torch.no_grad():
        for i in range(max_length - input_length):
            outputs = model(input_ids, use_cache=False, mode="ar")
            logits = outputs["ar_logits"]
            
            next_token = sample_next_token(logits[0, -1, :], temperature, top_p)
            
            if first_token_time is None:
                first_token_time = time.time() - start_time
                if verbose:
                    print(f"[ar] First token at {first_token_time:.3f}s")
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if verbose:
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"[ar] {len(generated_ids)}: {token_text!r}")
    
    return generated_ids, first_token_time if first_token_time else time.time() - start_time


def complete_combined(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool) -> tuple:
    """Generate tokens by blending AR next-token logits with diffusion logits on a masked next position."""
    generated_ids = []
    start_time = time.time()
    first_token_time = None
    input_length = input_ids.shape[1]
    max_length = min(input_length + max_tokens, model.max_seq_len)

    with torch.no_grad():
        for _ in range(max_length - input_length):
            ar_outputs = model(input_ids, use_cache=False, mode="ar")
            ar_logits = ar_outputs["ar_logits"][0, -1, :]

            masked_input = torch.cat(
                [input_ids, torch.full((input_ids.shape[0], 1), MASK_TOKEN_ID, dtype=input_ids.dtype, device=input_ids.device)],
                dim=1,
            )
            diffusion_outputs = model(masked_input, use_cache=False, mode="diffusion")
            diffusion_logits = diffusion_outputs["diffusion_logits"][0, -1, :]

            combined_logits = 0.5 * (ar_logits + diffusion_logits)
            next_token = sample_next_token(combined_logits, temperature, top_p)

            if first_token_time is None:
                first_token_time = time.time() - start_time
                if verbose:
                    print(f"[combined] First token at {first_token_time:.3f}s")

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            if verbose:
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"[combined] {len(generated_ids)}: {token_text!r}")

    return generated_ids, first_token_time if first_token_time else time.time() - start_time


def complete_diffusion(input_ids: torch.Tensor, max_tokens: int, num_steps: int = 10, temperature: float = 1.0, verbose: bool = True) -> tuple:
    """
    Generate tokens using diffusion mode (parallel iterative denoising).
    Much faster than AR since it generates all tokens in parallel and iteratively refines.
    
    Args:
        input_ids: Input token IDs [batch, seq_len]
        max_tokens: Maximum number of tokens to generate
        num_steps: Number of denoising steps (fewer = faster, more = better quality)
        temperature: Sampling temperature
        verbose: Print progress
    """
    batch_size = input_ids.shape[0]
    input_length = input_ids.shape[1]
    device = input_ids.device
    
    # Create fixed-size sequence with prompt + [MASK] for generation space
    mask_token_id = MASK_TOKEN_ID
    generation_seq = torch.full((batch_size, input_length + max_tokens), mask_token_id, dtype=torch.long, device=device)
    generation_seq[:, :input_length] = input_ids
    
    # Track which positions are masked (need to be predicted)
    masked_positions = torch.zeros((batch_size, input_length + max_tokens), dtype=torch.bool, device=device)
    masked_positions[:, input_length:] = True
    
    start_time = time.time()
    
    with torch.no_grad():
        # Iterative denoising: in each step, predict some masked tokens
        num_masked = max_tokens
        for step in range(num_steps):
            # Run diffusion head on the entire sequence
            outputs = model(generation_seq, use_cache=False, mode="diffusion")
            diffusion_logits = outputs["diffusion_logits"]  # [batch, seq_len, vocab_size]
            
            # Apply temperature
            diffusion_logits = diffusion_logits / temperature
            
            # Get predictions for all positions
            probs = torch.softmax(diffusion_logits, dim=-1)
            predictions = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(batch_size, -1)
            
            # Determine how many tokens to unmask this step
            # Use cosine schedule: unmask more in early steps, fewer in later steps
            progress = (step + 1) / num_steps
            remaining_ratio = 0.5 * (1 + math.cos(math.pi * progress))  # Cosine schedule
            target_masked = max(1, int(max_tokens * remaining_ratio))
            num_to_unmask = max(1, num_masked - target_masked)
            
            if verbose and step == 0:
                print(f"[diffusion] Starting denoising with {num_steps} steps, {max_tokens} tokens to generate")
            
            # Unmask tokens with highest confidence
            for b in range(batch_size):
                masked_indices = masked_positions[b].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    continue
                
                # Get logits for masked positions and compute confidence (max prob)
                masked_logits = diffusion_logits[b, masked_indices, :]
                masked_probs = torch.softmax(masked_logits, dim=-1)
                confidence = masked_probs.max(dim=-1).values
                
                # Select top-k confident predictions to unmask
                k = min(num_to_unmask, len(masked_indices))
                _, topk_indices = torch.topk(confidence, k)
                selected_positions = masked_indices[topk_indices]
                
                # Unmask selected positions with predicted tokens
                for pos in selected_positions:
                    generation_seq[b, pos] = predictions[b, pos]
                    masked_positions[b, pos] = False
            
            num_masked = masked_positions.sum().item()
            if verbose:
                print(f"[diffusion] Step {step + 1}/{num_steps}: {max_tokens - num_masked} tokens unmasked, {num_masked} remaining")
            
            # Early stopping if all tokens are unmasked
            if num_masked == 0:
                if verbose:
                    print(f"[diffusion] Early stopping at step {step + 1}")
                break
    
    total_time = time.time() - start_time
    first_token_time = total_time  # For diffusion, all tokens appear "at once"
    
    # Extract generated tokens (positions after input_length)
    generated_ids = generation_seq[0, input_length:].tolist()
    
    # Truncate at EOS if present
    if tokenizer.eos_token_id in generated_ids:
        eos_pos = generated_ids.index(tokenizer.eos_token_id)
        generated_ids = generated_ids[:eos_pos]
    
    return generated_ids, first_token_time


def complete_reasoning(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool) -> tuple:
    """Generate an explicit thinking phase with AR, then answer from the resulting context with diffusion."""
    thinking_open_ids = tokenizer.encode("<thinking>\n", add_special_tokens=False)
    thinking_close_ids = tokenizer.encode("</thinking>\n", add_special_tokens=False)
    output_open_ids = tokenizer.encode("<output>\n", add_special_tokens=False)

    context_ids = input_ids.clone()
    prompt_text = tokenizer.decode(context_ids[0], skip_special_tokens=False)
    if "<thinking>" not in prompt_text:
        context_ids = torch.cat(
            [context_ids, torch.tensor([thinking_open_ids], dtype=context_ids.dtype, device=context_ids.device)],
            dim=1,
        )

    thinking_budget = max(1, max_tokens // 2)
    thinking_ids, first_token_time = complete_ar(context_ids, thinking_budget, temperature, top_p, verbose)
    generated_thinking = thinking_ids[:]

    close_seq = thinking_close_ids
    if close_seq and generated_thinking[-len(close_seq):] != close_seq:
        generated_thinking.extend(close_seq)

    reasoning_context = torch.cat(
        [context_ids, torch.tensor([generated_thinking + output_open_ids], dtype=context_ids.dtype, device=context_ids.device)],
        dim=1,
    )
    remaining_budget = max(1, max_tokens - len(generated_thinking))
    answer_ids, _ = complete_diffusion(
        reasoning_context,
        remaining_budget,
        num_steps=min(10, max(4, remaining_budget)),
        temperature=temperature,
        verbose=verbose,
    )

    return generated_thinking + output_open_ids + answer_ids, first_token_time


def complete(prompt: str, max_tokens: int = 100, temperature: float = 1.0, top_p: float = 1.0, mode: str = "ar", verbose: bool = True) -> Dict[str, Any]:
    global model, tokenizer, device
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    start_time = time.time()
    
    if verbose:
        print(f"[{mode}] Prompt: {prompt[:50]}...")
        print(f"[{mode}] Input tokens: {input_length}")
    
    # Route to appropriate generation method
    if mode == "ar":
        generated_ids, first_token_time = complete_ar(
            input_ids, max_tokens, temperature, top_p, verbose
        )
    elif mode == "diffusion":
        generated_ids, first_token_time = complete_diffusion(
            input_ids, max_tokens, num_steps=10, temperature=temperature, verbose=verbose
        )
    elif mode == "combined":
        generated_ids, first_token_time = complete_combined(
            input_ids, max_tokens, temperature, top_p, verbose
        )
    elif mode == "reasoning":
        generated_ids, first_token_time = complete_reasoning(
            input_ids, max_tokens, temperature, top_p, verbose
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    total_time = time.time() - start_time
    ttft = first_token_time if first_token_time else total_time
    tps = len(generated_ids) / total_time if total_time > 0 else 0
    
    if verbose:
        print(f"[{mode}] Generated {len(generated_ids)} tokens in {total_time:.2f}s")
        print(f"[{mode}] TTFT: {ttft:.3f}s, TPS: {tps:.2f}")
        print(f"[{mode}] Output: {generated_text[:200]}...")
    
    return {"text": generated_text, "ttft": ttft, "tps": tps, "total_tokens": len(generated_ids), "finish_reason": "stop" if generated_ids else "length"}


def complete_with_tools(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    mode: str = "ar",
    max_tool_cycles: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Generate completion with tool calling support.
    
    Args:
        prompt: Input prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        mode: Generation mode (ar, diffusion, combined, reasoning)
        max_tool_cycles: Maximum number of tool call cycles
        verbose: Print verbose output
    
    Returns:
        Dict with text, tool_calls, and metadata
    """
    global model, tokenizer, device, tool_registry, tool_parser
    
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    
    if tool_registry is None or tool_parser is None:
        init_tools()
    
    # Add tool schemas to prompt if tools are available
    tool_schemas = tool_registry.get_schemas_text() if tool_registry else ""
    
    # Build the full prompt with tool instructions
    full_prompt = prompt
    if tool_schemas:
        full_prompt = f"{prompt}\n\nAvailable tools:\n{tool_schemas}\n\nWhen you need to use a tool, output the tool call in JSON format: {{\"tool\": \"tool_name\", \"args\": {{...}}}}"
    
    tool_calls_executed = []
    current_prompt = full_prompt
    
    for cycle in range(max_tool_cycles):
        if verbose:
            print(f"\n=== Tool Cycle {cycle + 1} ===")
        
        # Generate completion
        result = complete(
            current_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            mode=mode,
            verbose=verbose
        )
        
        generated_text = result["text"]
        
        # Check for tool calls in the output
        tool_calls = tool_parser.parse(generated_text)
        
        if not tool_calls:
            if verbose:
                print(f"[complete_with_tools] No tool calls detected, returning result")
            return {
                "text": generated_text,
                "tool_calls": tool_calls_executed,
                "cycles": cycle + 1,
                "final": True
            }
        
        # Execute tool calls
        if verbose:
            print(f"[complete_with_tools] Executing {len(tool_calls)} tool call(s)")
        
        for tc in tool_calls:
            if verbose:
                print(f"[complete_with_tools] Tool: {tc.tool_name}, args: {tc.arguments}")
            
            try:
                tool_result = tool_registry.execute(tc.tool_name, tc.arguments)
                tool_calls_executed.append({
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result": tool_result,
                    "success": True
                })
                
                if verbose:
                    print(f"[complete_with_tools] Result: {tool_result}")
                    
            except Exception as e:
                error_msg = str(e)
                tool_calls_executed.append({
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "error": error_msg,
                    "success": False
                })
                
                if verbose:
                    print(f"[complete_with_tools] Error: {error_msg}")
        
        # Format tool results for the model
        tool_results_text = "\n\n".join([
            f"Tool '{tc['tool']}' result: {json.dumps(tc.get('result', tc.get('error')))}"
            for tc in tool_calls_executed[-len(tool_calls):]
        ])
        
        # Feed results back to model for final answer
        current_prompt = f"{current_prompt}\n\n{generated_text}\n\n{tool_results_text}\n\nBased on the tool results, provide your final answer:"
        max_tokens = max_tokens // 2  # Reduce tokens for subsequent cycles
    
    # Return after max cycles
    return {
        "text": generated_text,
        "tool_calls": tool_calls_executed,
        "cycles": max_tool_cycles,
        "final": False,
        "reason": "max_cycles"
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--config", type=str, required=True, help="Path to config .pt file")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt for completion")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling")
    parser.add_argument("--mode", type=str, default="ar", choices=["ar", "diffusion", "combined", "reasoning"], help="Generation mode: ar=auto-regressive, diffusion=masked, combined=avg both, reasoning=AR then diffusion")
    parser.add_argument("--use-tools", action="store_true", help="Enable tool calling mode")
    parser.add_argument("--max-tool-cycles", type=int, default=3, help="Max tool call cycles in tool mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()
    
    load_model(args.checkpoint, args.config)
    
    if args.use_tools:
        result = complete_with_tools(
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.mode,
            args.max_tool_cycles,
            verbose=not args.quiet
        )
        print(f"\n=== RESULT (with tools) ===")
        print(f"Text: {result['text']}")
        print(f"Tool calls executed: {len(result.get('tool_calls', []))}")
        print(f"Cycles: {result.get('cycles', 1)}")
        if result.get('tool_calls'):
            print("Tool call details:")
            for tc in result['tool_calls']:
                status = "success" if tc.get('success') else f"error: {tc.get('error')}"
                print(f"  - {tc['tool']}({tc['args']}): {status}")
    else:
        result = complete(args.prompt, args.max_tokens, args.temperature, args.top_p, args.mode, verbose=not args.quiet)
        print(f"\n=== RESULT ===")
        print(f"Text: {result['text']}")
        print(f"TTFT: {result['ttft']:.3f}s, TPS: {result['tps']:.2f}, Tokens: {result['total_tokens']}")


if __name__ == "__main__":
    main()
