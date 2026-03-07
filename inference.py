#!/usr/bin/env python3
"""Inference for the dual-mode AR + Diffusion model."""

import argparse
import time
import math
import torch
from typing import Dict, Any, List, Optional
import json

from pretrain import DualModeModel, get_default_tokenizer
from tools import ToolRegistry, ToolCallParser, detect_tool_calls, parse_tool_result
from tools.registry import get_default_registry

model = None
tokenizer = None
device = None
tool_registry = None
tool_parser = None


def init_tools():
    """Initialize tool registry and parser."""
    global tool_registry, tool_parser
    tool_registry = get_default_registry()
    tool_parser = ToolCallParser()
    print(f"Initialized {len(tool_registry.list_tools())} tools: {tool_registry.list_tools()}")


def load_model(checkpoint_path: str, config_path: str, device_str: str = None):
    global model, tokenizer, device
    
    config = torch.load(config_path, map_location="cpu")
    
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")
    
    model = DualModeModel(
        vocab_size=config.get("original_vocab_size", config.get("vocab_size", 16000) - 1),
        hidden_size=config.get("hidden_size", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        head_dim=config.get("head_dim", 64),
        max_seq_len=config.get("max_seq_len", 8192),
        use_flash_attn=False,
        mtp_enabled=config.get("mtp_enabled", False),
        mtp_num_heads=config.get("mtp_num_heads", 3),
        mtp_loss_weights=config.get("mtp_loss_weights", [1.0, 0.7, 0.5]),
    )
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    tokenizer = get_default_tokenizer()
    
    # Add special tokens for reasoning
    special_tokens = ["<thinking>", "</thinking>", "<reasoning>", "</reasoning>", "<output>", "</output>"]
    num_added = tokenizer.add_tokens(special_tokens)
    if num_added > 0:
        print(f"Added {num_added} special tokens: {special_tokens}")
        # Resize model embeddings to accommodate new tokens
        model.token_embeddings = torch.nn.Embedding(len(tokenizer), model.hidden_size).to(device)
        # Initialize new embeddings
        with torch.no_grad():
            torch.nn.init.normal_(model.token_embeddings.weight, mean=0.0, std=0.02)
        print(f"Resized token embeddings to {len(tokenizer)} tokens")
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    
    return model, tokenizer


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
                    print(f"[ar] First token at {first_token_time:.3f}s")
            
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if verbose:
                token_text = tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"[ar] {len(generated_ids)}: {token_text!r}")
    
    return generated_ids, first_token_time if first_token_time else time.time() - start_time


def complete_diffusion(input_ids: torch.Tensor, max_tokens: int, num_steps: int = 12, temperature: float = 0.9, verbose: bool = True) -> tuple:
    """
    Generate tokens using diffusion mode with confidence-guided iterative denoising.
    Strategy:
    - Start from all MASK tokens in generation region
    - Iteratively predict masked positions
    - Freeze only high-confidence positions each step (progressive threshold)
    - Keep uncertain positions masked for later refinement
    """
    batch_size = input_ids.shape[0]
    input_length = input_ids.shape[1]
    device = input_ids.device

    mask_token_id = model.mask_token_id if hasattr(model, "mask_token_id") else (tokenizer.pad_token_id if tokenizer.pad_token_id else 0)
    generation_seq = torch.full((batch_size, input_length + max_tokens), mask_token_id, dtype=torch.long, device=device)
    generation_seq[:, :input_length] = input_ids

    masked_positions = torch.zeros((batch_size, input_length + max_tokens), dtype=torch.bool, device=device)
    masked_positions[:, input_length:] = True

    start_time = time.time()

    with torch.no_grad():
        if verbose:
            print(f"[diffusion] Starting denoising with {num_steps} steps, {max_tokens} tokens to generate")

        for step in range(num_steps):
            outputs = model(generation_seq, use_cache=False, mode="diffusion")
            diffusion_logits = outputs["diffusion_logits"] / max(1e-6, temperature)
            probs = torch.softmax(diffusion_logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)
            pred_conf = probs.max(dim=-1).values

            # Progressive confidence threshold: strict early, relaxed later
            progress = (step + 1) / num_steps
            conf_threshold = 0.92 - (0.42 * progress)  # 0.92 -> 0.50

            total_newly_fixed = 0
            for b in range(batch_size):
                masked_idx = masked_positions[b].nonzero(as_tuple=True)[0]
                if len(masked_idx) == 0:
                    continue

                # Candidate masked positions and confidences
                m_conf = pred_conf[b, masked_idx]
                m_tok = pred_tokens[b, masked_idx]

                # Ensure minimum progress each step with top-k fallback
                remaining = len(masked_idx)
                min_fix = max(1, remaining // max(1, (num_steps - step)))
                threshold_sel = m_conf >= conf_threshold
                if threshold_sel.sum().item() < min_fix:
                    _, topk_idx = torch.topk(m_conf, k=min_fix)
                    sel = torch.zeros_like(threshold_sel)
                    sel[topk_idx] = True
                else:
                    sel = threshold_sel

                selected_positions = masked_idx[sel]
                selected_tokens = m_tok[sel]

                if len(selected_positions) > 0:
                    generation_seq[b, selected_positions] = selected_tokens
                    masked_positions[b, selected_positions] = False
                    total_newly_fixed += len(selected_positions)

            remaining_masked = int(masked_positions.sum().item())
            if verbose:
                print(f"[diffusion] Step {step + 1}/{num_steps}: fixed {total_newly_fixed}, remaining {remaining_masked}, thr={conf_threshold:.2f}")

            if remaining_masked == 0:
                if verbose:
                    print(f"[diffusion] Early stopping at step {step + 1}")
                break

    total_time = time.time() - start_time
    first_token_time = total_time

    generated_ids = generation_seq[0, input_length:].tolist()
    if tokenizer.eos_token_id in generated_ids:
        eos_pos = generated_ids.index(tokenizer.eos_token_id)
        generated_ids = generated_ids[:eos_pos]

    return generated_ids, first_token_time


def complete_medusa(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool) -> tuple:
    """Medusa-like decoding using base AR token plus MTP speculative tokens."""
    generated_ids = []
    start_time = time.time()
    first_token_time = None
    input_length = input_ids.shape[1]
    max_length = min(input_length + max_tokens, model.max_seq_len)

    if not getattr(model, "mtp_enabled", False):
        return complete_ar(input_ids, max_tokens, temperature, top_p, verbose)

    with torch.no_grad():
        while input_ids.shape[1] < max_length and len(generated_ids) < max_tokens:
            outputs = model(input_ids, use_cache=False, mode="ar")
            ar_logits = outputs["ar_logits"]
            mtp_logits = outputs.get("mtp_logits", [])

            # Base AR next token
            next_token_logits = ar_logits[0, -1, :] / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
            probs = torch.softmax(next_token_logits, dim=-1)
            candidate_tokens = [torch.multinomial(probs, num_samples=1).item()]

            # Add speculative tokens from MTP heads
            for h_logits in mtp_logits[:3]:
                h_token_logits = h_logits[0, -1, :] / temperature
                h_probs = torch.softmax(h_token_logits, dim=-1)
                candidate_tokens.append(torch.multinomial(h_probs, num_samples=1).item())

            accepted = 0
            for tok in candidate_tokens:
                if input_ids.shape[1] >= max_length or len(generated_ids) >= max_tokens:
                    break
                tok_tensor = torch.tensor([[tok]], device=input_ids.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, tok_tensor], dim=1)
                generated_ids.append(tok)
                accepted += 1

                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    if verbose:
                        print(f"[medusa] First token at {first_token_time:.3f}s")

                if tok == tokenizer.eos_token_id:
                    break

            if verbose:
                print(f"[medusa] Accepted {accepted} token(s), total={len(generated_ids)}")

            if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
                break

    return generated_ids[:max_tokens], first_token_time if first_token_time else time.time() - start_time


def complete_medusa_diffusion(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool = True) -> tuple:
    """Hybrid mode: medusa draft first, then diffusion refinement conditioned on the original prompt."""
    # Phase 1: draft with medusa
    drafted_ids, medusa_ttft = complete_medusa(input_ids.clone(), max_tokens, temperature, top_p, verbose)
    if len(drafted_ids) == 0:
        return drafted_ids, medusa_ttft

    # Phase 2: refine drafted span while preserving prompt conditioning
    refined_ids, _ = complete_diffusion(input_ids.clone(), max_tokens=len(drafted_ids), num_steps=12, temperature=max(0.8, temperature), verbose=verbose)

    # Keep length consistent with draft for stable behavior
    final_ids = refined_ids[:len(drafted_ids)] if len(refined_ids) > 0 else drafted_ids
    if verbose:
        print(f"[medusa+diffusion] Drafted={len(drafted_ids)} Refined={len(final_ids)}")

    return final_ids, medusa_ttft


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
        # Use AR for now - combined mode would need special implementation
        generated_ids, first_token_time = complete_ar(
            input_ids, max_tokens, temperature, top_p, verbose
        )
    elif mode == "reasoning":
        # Use AR for thinking phase
        generated_ids, first_token_time = complete_ar(
            input_ids, max_tokens, temperature, top_p, verbose
        )
    elif mode == "medusa":
        generated_ids, first_token_time = complete_medusa(
            input_ids, max_tokens, temperature, top_p, verbose
        )
    elif mode == "medusa+diffusion":
        generated_ids, first_token_time = complete_medusa_diffusion(
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
    parser.add_argument("--mode", type=str, default="ar", choices=["ar", "diffusion", "combined", "reasoning", "medusa", "medusa+diffusion"], help="Generation mode: ar=auto-regressive, diffusion=masked, combined=avg both, reasoning=AR then diffusion, medusa=mtp speculative decoding, medusa+diffusion=medusa draft then diffusion refine")
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
