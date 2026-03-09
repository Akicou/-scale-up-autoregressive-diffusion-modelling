#!/usr/bin/env python3
"""Inference for custom DualMode checkpoints, HF models, and PEFT adapters."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain import DualModeModel, get_default_tokenizer
from tools import ToolCallParser
from tools.registry import get_default_registry

model = None
tokenizer = None
device = None
tool_registry = None
tool_parser = None
load_context = None
_temp_dirs: List[tempfile.TemporaryDirectory] = []


@dataclass
class ModelCapabilities:
    supports_ar: bool
    supports_diffusion: bool
    supports_mtp: bool
    supports_tool_mode: bool = True


@dataclass
class LoadContext:
    backend_type: str
    capabilities: ModelCapabilities
    tokenizer_source: str
    source_paths: Dict[str, str]


def init_tools():
    """Initialize tool registry and parser."""
    global tool_registry, tool_parser
    tool_registry = get_default_registry()
    tool_parser = ToolCallParser()
    print(f"Initialized {len(tool_registry.list_tools())} tools: {tool_registry.list_tools()}")


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device_str)
    print(f"Using device: {resolved}")
    return resolved


def _resolve_torch_dtype(dtype_name: str, resolved_device: torch.device) -> Optional[torch.dtype]:
    dtype_name = (dtype_name or "auto").lower()
    if dtype_name == "auto":
        if resolved_device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if resolved_device.type == "cuda":
            return torch.float16
        return None
    mapping = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return mapping[dtype_name]


def _is_custom_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "model.pt").exists() and (path / "config.pt").exists()


def _is_adapter_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "adapter_config.json").exists():
        return False
    return (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists()


def _is_hf_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    if (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists():
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    return any(path.glob("*.safetensors"))


def _classify_model_path(model_path: str, config_path: Optional[str] = None) -> tuple[str, Path]:
    path = Path(model_path)
    if _is_adapter_dir(path):
        return "adapter_dir", path
    if _is_custom_checkpoint_dir(path):
        return "custom_dir", path
    if _is_hf_dir(path):
        return "hf_dir", path
    if path.is_file() and path.name == "model.pt":
        return "custom_file", path
    if path.is_file() and path.suffix == ".safetensors":
        sibling_config = path.parent / "config.json"
        if sibling_config.exists() or (config_path and Path(config_path).suffix == ".json"):
            return "hf_file", path
        raise FileNotFoundError(
            f"Single safetensors file {path} requires a sibling config.json or --config-path pointing to one."
        )
    raise FileNotFoundError(f"Could not classify model path: {model_path}")


def _load_tokenizer_from_path(path: Path, trust_remote_code: bool) -> Optional[Any]:
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    except Exception:
        return None


def _resolve_custom_tokenizer(
    checkpoint_dir: Path,
    tokenizer_path: Optional[str],
    trust_remote_code: bool,
) -> tuple[Any, str]:
    if tokenizer_path:
        tokenizer_obj = _load_tokenizer_from_path(Path(tokenizer_path), trust_remote_code)
        if tokenizer_obj is None:
            raise FileNotFoundError(f"Failed to load tokenizer from {tokenizer_path}")
        return tokenizer_obj, str(tokenizer_path)

    tokenizer_obj = _load_tokenizer_from_path(checkpoint_dir, trust_remote_code)
    if tokenizer_obj is not None:
        return tokenizer_obj, str(checkpoint_dir)

    tokenizer_obj = get_default_tokenizer()
    return tokenizer_obj, "default_tokenizer"


def _resolve_hf_tokenizer(
    primary_path: str | Path,
    tokenizer_path: Optional[str],
    trust_remote_code: bool,
    secondary_path: Optional[str | Path] = None,
) -> tuple[Any, str]:
    search_paths: List[Path] = []
    if tokenizer_path:
        search_paths.append(Path(tokenizer_path))
    search_paths.append(Path(primary_path))
    if secondary_path is not None:
        search_paths.append(Path(secondary_path))

    for candidate in search_paths:
        tokenizer_obj = _load_tokenizer_from_path(candidate, trust_remote_code)
        if tokenizer_obj is not None:
            return tokenizer_obj, str(candidate)

    raise FileNotFoundError(
        f"Tokenizer not found. Checked: {', '.join(str(candidate) for candidate in search_paths)}"
    )


def _patch_custom_state_dict(model_obj: DualModeModel, state_dict: Dict[str, torch.Tensor]) -> None:
    model_state = model_obj.state_dict()
    exact_state = {}
    mismatched = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape == value.shape:
            exact_state[key] = value
        else:
            mismatched.append((key, value, model_state[key]))

    model_obj.load_state_dict(exact_state, strict=False)
    for key, source, destination in mismatched:
        patched = destination.clone()
        slices = tuple(slice(0, min(dst, src)) for dst, src in zip(destination.shape, source.shape))
        patched[slices] = source[slices]
        model_obj.load_state_dict({key: patched}, strict=False)


def _infer_capabilities(model_obj: Any, backend_type: str) -> ModelCapabilities:
    if backend_type == "custom_native":
        return ModelCapabilities(
            supports_ar=True,
            supports_diffusion=True,
            supports_mtp=bool(getattr(model_obj, "mtp_enabled", False)),
        )

    config = getattr(model_obj, "config", None)
    supports_mtp = bool(getattr(config, "mtp_enabled", False)) if config is not None else False
    return ModelCapabilities(
        supports_ar=True,
        supports_diffusion=False,
        supports_mtp=supports_mtp,
    )


def _finalize_loaded_model(
    model_obj: Any,
    tokenizer_obj: Any,
    resolved_device: torch.device,
    backend_type: str,
    tokenizer_source: str,
    source_paths: Dict[str, str],
) -> tuple[Any, Any]:
    global model, tokenizer, device, load_context

    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token

    model = model_obj
    tokenizer = tokenizer_obj
    device = resolved_device
    load_context = LoadContext(
        backend_type=backend_type,
        capabilities=_infer_capabilities(model_obj, backend_type),
        tokenizer_source=tokenizer_source,
        source_paths=source_paths,
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded backend: {backend_type}")
    print(f"Tokenizer source: {tokenizer_source}")
    print(
        "Capabilities: "
        f"AR={load_context.capabilities.supports_ar}, "
        f"diffusion={load_context.capabilities.supports_diffusion}, "
        f"mtp={load_context.capabilities.supports_mtp}"
    )
    print(f"Model loaded: {param_count / 1e6:.1f}M params")

    return model, tokenizer


def _load_custom_dir(
    checkpoint_dir: Path,
    tokenizer_path: Optional[str],
    resolved_device: torch.device,
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    checkpoint_file: Optional[Path] = None,
    config_file: Optional[Path] = None,
) -> tuple[Any, Any]:
    resolved_checkpoint = checkpoint_file or (checkpoint_dir / "model.pt")
    resolved_config = config_file or (checkpoint_dir / "config.pt")

    config = torch.load(resolved_config, map_location="cpu")
    model_obj = DualModeModel(
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
    state_dict = torch.load(resolved_checkpoint, map_location="cpu")
    _patch_custom_state_dict(model_obj, state_dict)
    if dtype is not None and resolved_device.type != "cpu":
        model_obj = model_obj.to(device=resolved_device, dtype=dtype)
    else:
        model_obj = model_obj.to(resolved_device)
    model_obj.eval()

    tokenizer_obj, tokenizer_source = _resolve_custom_tokenizer(checkpoint_dir, tokenizer_path, trust_remote_code)
    return _finalize_loaded_model(
        model_obj,
        tokenizer_obj,
        resolved_device,
        "custom_native",
        tokenizer_source,
        {"model_path": str(resolved_checkpoint), "config_path": str(resolved_config)},
    )


def _load_custom_file(
    checkpoint_path: Path,
    config_path: Optional[str],
    tokenizer_path: Optional[str],
    resolved_device: torch.device,
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    resolved_config = Path(config_path) if config_path else checkpoint_path.with_name("config.pt")
    if not resolved_config.exists():
        raise FileNotFoundError(
            f"Config not found for {checkpoint_path}. Pass --config-path or place config.pt beside model.pt."
        )
    checkpoint_dir = checkpoint_path.parent
    return _load_custom_dir(
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        resolved_device=resolved_device,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        checkpoint_file=checkpoint_path,
        config_file=resolved_config,
    )


def _assemble_single_file_hf_dir(model_file: Path, config_path: Optional[str], tokenizer_path: Optional[str]) -> Path:
    resolved_config = Path(config_path) if config_path else model_file.parent / "config.json"
    if not resolved_config.exists():
        raise FileNotFoundError(
            f"config.json not found for {model_file}. Pass --config-path or place config.json beside the safetensors file."
        )

    temp_dir = tempfile.TemporaryDirectory(prefix="inference_hf_single_")
    _temp_dirs.append(temp_dir)
    target_dir = Path(temp_dir.name)
    (target_dir / model_file.name).write_bytes(model_file.read_bytes())
    (target_dir / "config.json").write_bytes(resolved_config.read_bytes())

    tokenizer_source = Path(tokenizer_path) if tokenizer_path else resolved_config.parent
    for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json"]:
        source = tokenizer_source / name
        if source.exists():
            (target_dir / name).write_bytes(source.read_bytes())

    return target_dir


def _load_hf_dir(
    model_dir: Path,
    tokenizer_path: Optional[str],
    resolved_device: torch.device,
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
    backend_type: str = "hf_causallm",
    source_paths: Optional[Dict[str, str]] = None,
) -> tuple[Any, Any]:
    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    model_obj = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model_obj = model_obj.to(resolved_device)
    model_obj.eval()

    tokenizer_obj, tokenizer_source = _resolve_hf_tokenizer(model_dir, tokenizer_path, trust_remote_code)
    return _finalize_loaded_model(
        model_obj,
        tokenizer_obj,
        resolved_device,
        backend_type,
        tokenizer_source,
        source_paths or {"model_path": str(model_dir)},
    )


def _resolve_adapter_base_path(adapter_dir: Path, base_model_path: Optional[str]) -> str:
    if base_model_path:
        return base_model_path

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    recorded = adapter_config.get("base_model_name_or_path")
    if not recorded:
        raise FileNotFoundError(
            f"Adapter directory {adapter_dir} does not define base_model_name_or_path; pass --base-model-path."
        )

    recorded_path = Path(recorded)
    if recorded_path.exists():
        return str(recorded_path)

    relative_candidate = (adapter_dir / recorded).resolve()
    if relative_candidate.exists():
        return str(relative_candidate)

    return recorded


def _load_adapter_dir(
    adapter_dir: Path,
    base_model_path: Optional[str],
    tokenizer_path: Optional[str],
    resolved_device: torch.device,
    dtype: Optional[torch.dtype],
    trust_remote_code: bool,
) -> tuple[Any, Any]:
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("PEFT is required to load adapter checkpoints.") from exc

    resolved_base = _resolve_adapter_base_path(adapter_dir, base_model_path)
    base_path = Path(resolved_base)
    if base_path.exists() and _is_custom_checkpoint_dir(base_path):
        from finetune.custom_checkpoint import ensure_hf_export

        resolved_base = str(ensure_hf_export(base_path))

    load_kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype

    base_model = AutoModelForCausalLM.from_pretrained(resolved_base, **load_kwargs)
    model_obj = PeftModel.from_pretrained(base_model, adapter_dir)
    model_obj = model_obj.to(resolved_device)
    model_obj.eval()

    tokenizer_obj, tokenizer_source = _resolve_hf_tokenizer(
        adapter_dir,
        tokenizer_path,
        trust_remote_code,
        secondary_path=resolved_base,
    )
    return _finalize_loaded_model(
        model_obj,
        tokenizer_obj,
        resolved_device,
        "peft_adapter",
        tokenizer_source,
        {"model_path": str(adapter_dir), "base_model_path": str(resolved_base)},
    )


def load_model(
    model_path: str,
    config_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    device_str: Optional[str] = None,
    trust_remote_code: bool = True,
    dtype: str = "auto",
):
    resolved_device = _resolve_device(device_str)
    resolved_dtype = _resolve_torch_dtype(dtype, resolved_device)
    model_type, resolved_path = _classify_model_path(model_path, config_path)

    if model_type == "custom_dir":
        return _load_custom_dir(resolved_path, tokenizer_path, resolved_device, resolved_dtype, trust_remote_code)
    if model_type == "custom_file":
        return _load_custom_file(resolved_path, config_path, tokenizer_path, resolved_device, resolved_dtype, trust_remote_code)
    if model_type == "hf_dir":
        return _load_hf_dir(resolved_path, tokenizer_path, resolved_device, resolved_dtype, trust_remote_code)
    if model_type == "hf_file":
        assembled_dir = _assemble_single_file_hf_dir(resolved_path, config_path, tokenizer_path)
        return _load_hf_dir(
            assembled_dir,
            tokenizer_path,
            resolved_device,
            resolved_dtype,
            trust_remote_code,
            source_paths={"model_path": str(resolved_path), "config_path": str(config_path or (resolved_path.parent / "config.json"))},
        )
    if model_type == "adapter_dir":
        return _load_adapter_dir(
            resolved_path,
            base_model_path,
            tokenizer_path,
            resolved_device,
            resolved_dtype,
            trust_remote_code,
        )
    raise RuntimeError(f"Unhandled model type: {model_type}")


def _ensure_loaded() -> None:
    if model is None or tokenizer is None or load_context is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")


def _warn_mode_fallback(requested_mode: str, fallback_mode: str, verbose: bool) -> None:
    if verbose:
        print(f"[{requested_mode}] Capability unavailable for backend {load_context.backend_type}; falling back to {fallback_mode}.")


def _forward_ar_outputs(input_ids: torch.Tensor) -> tuple[torch.Tensor, Any]:
    if load_context.backend_type == "custom_native":
        outputs = model(input_ids, use_cache=False, mode="ar")
        return outputs["ar_logits"], outputs

    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
    return outputs.logits, outputs


def _forward_diffusion_logits(input_ids: torch.Tensor) -> torch.Tensor:
    if not load_context.capabilities.supports_diffusion:
        raise RuntimeError(
            f"`diffusion` mode requires a native DualModeModel checkpoint; backend {load_context.backend_type} only supports causal LM generation."
        )
    outputs = model(input_ids, use_cache=False, mode="diffusion")
    return outputs["diffusion_logits"]


def _forward_mtp_logits(outputs: Any) -> List[torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs.get("mtp_logits", []) or []
    mtp_logits = getattr(outputs, "mtp_logits", None)
    return mtp_logits or []


def complete_ar(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool) -> tuple:
    """Generate tokens using auto-regressive mode (one token at a time)."""
    generated_ids = []
    start_time = time.time()
    first_token_time = None
    input_length = input_ids.shape[1]
    model_max_len = getattr(model, "max_seq_len", getattr(getattr(model, "config", None), "max_position_embeddings", input_length + max_tokens))
    max_length = min(input_length + max_tokens, model_max_len)

    with torch.no_grad():
        for _ in range(max_length - input_length):
            logits, _ = _forward_ar_outputs(input_ids)
            next_token_logits = logits[0, -1, :] / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")

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


def complete_diffusion(
    input_ids: torch.Tensor,
    max_tokens: int,
    num_steps: int = 12,
    temperature: float = 0.9,
    top_p: float = 0.95,
    top_k: int = 64,
    block_size: int = 256,
    repetition_penalty: float = 1.1,
    verbose: bool = True,
) -> tuple:
    """Generate tokens using block-wise confidence-guided diffusion decoding."""
    if not load_context.capabilities.supports_diffusion:
        raise RuntimeError(
            f"`diffusion` mode requires a native DualModeModel checkpoint; backend {load_context.backend_type} only supports causal LM generation."
        )
    batch_size = input_ids.shape[0]
    current_device = input_ids.device
    if batch_size != 1:
        raise ValueError("Diffusion decoding currently supports batch_size=1")

    mask_token_id = model.mask_token_id if hasattr(model, "mask_token_id") else (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    vocab_size = model.vocab_size

    banned_ids = {mask_token_id}
    if pad_token_id is not None:
        banned_ids.add(pad_token_id)

    start_time = time.time()
    generated_ids: List[int] = []
    first_token_time = None
    context_ids = input_ids.clone()
    model_max_len = getattr(model, "max_seq_len", input_ids.shape[1] + max_tokens)

    with torch.no_grad():
        if verbose:
            print(f"[diffusion] Starting denoising with {num_steps} steps, {max_tokens} tokens to generate")

        while len(generated_ids) < max_tokens:
            remaining_budget = max_tokens - len(generated_ids)
            max_block = min(block_size, remaining_budget)
            if context_ids.shape[1] + max_block > model_max_len:
                max_block = model_max_len - context_ids.shape[1]
            if max_block <= 0:
                break

            input_length = context_ids.shape[1]
            generation_seq = torch.full(
                (batch_size, input_length + max_block),
                mask_token_id,
                dtype=torch.long,
                device=current_device,
            )
            generation_seq[:, :input_length] = context_ids

            masked_positions = torch.zeros(
                (batch_size, input_length + max_block),
                dtype=torch.bool,
                device=current_device,
            )
            masked_positions[:, input_length:] = True

            for step in range(num_steps):
                diffusion_logits = _forward_diffusion_logits(generation_seq) / max(1e-6, temperature)

                for bad_id in banned_ids:
                    if 0 <= bad_id < vocab_size:
                        diffusion_logits[:, input_length:, bad_id] = float("-inf")
                if eos_token_id is not None and step < num_steps - 1 and 0 <= eos_token_id < vocab_size:
                    diffusion_logits[:, input_length:, eos_token_id] = float("-inf")

                probs = torch.softmax(diffusion_logits, dim=-1)
                pred_conf = probs.max(dim=-1).values

                progress = (step + 1) / num_steps
                conf_threshold = 0.92 - (0.42 * progress)

                total_newly_fixed = 0
                for batch_index in range(batch_size):
                    masked_idx = masked_positions[batch_index].nonzero(as_tuple=True)[0]
                    if len(masked_idx) == 0:
                        continue

                    masked_conf = pred_conf[batch_index, masked_idx]
                    masked_logits = diffusion_logits[batch_index, masked_idx, :].clone()

                    if repetition_penalty > 1.0:
                        recent_ids = generation_seq[batch_index, :].tolist()[-512:]
                        if recent_ids:
                            recent_unique = torch.tensor(sorted(set(recent_ids)), device=masked_logits.device, dtype=torch.long)
                            recent_unique = recent_unique[(recent_unique >= 0) & (recent_unique < masked_logits.shape[-1])]
                            if recent_unique.numel() > 0:
                                masked_logits[:, recent_unique] = masked_logits[:, recent_unique] / repetition_penalty

                    if top_k > 0 and top_k < masked_logits.shape[-1]:
                        topk_vals, _ = torch.topk(masked_logits, k=top_k, dim=-1)
                        kth_vals = topk_vals[:, -1].unsqueeze(-1)
                        masked_logits = torch.where(
                            masked_logits < kth_vals,
                            torch.full_like(masked_logits, float("-inf")),
                            masked_logits,
                        )

                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(masked_logits, descending=True, dim=-1)
                        sorted_probs = torch.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = False
                        to_remove = torch.zeros_like(masked_logits, dtype=torch.bool)
                        to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                        masked_logits = masked_logits.masked_fill(to_remove, float("-inf"))

                    masked_probs = torch.softmax(masked_logits, dim=-1)
                    sampled = torch.multinomial(masked_probs, num_samples=1).squeeze(-1)

                    remaining = len(masked_idx)
                    min_fix = max(1, remaining // max(1, (num_steps - step)))
                    threshold_sel = masked_conf >= conf_threshold
                    if threshold_sel.sum().item() < min_fix:
                        _, topk_idx = torch.topk(masked_conf, k=min_fix)
                        selected_mask = torch.zeros_like(threshold_sel)
                        selected_mask[topk_idx] = True
                    else:
                        selected_mask = threshold_sel

                    selected_positions = masked_idx[selected_mask]
                    selected_tokens = sampled[selected_mask]

                    if len(selected_positions) > 0:
                        generation_seq[batch_index, selected_positions] = selected_tokens
                        masked_positions[batch_index, selected_positions] = False
                        total_newly_fixed += len(selected_positions)

                remaining_masked = int(masked_positions.sum().item())
                if verbose:
                    print(
                        f"[diffusion] Step {step + 1}/{num_steps}: "
                        f"fixed {total_newly_fixed}, remaining {remaining_masked}, thr={conf_threshold:.2f}"
                    )
                if remaining_masked == 0:
                    if verbose:
                        print(f"[diffusion] Early stopping at step {step + 1}")
                    break

            block_ids = generation_seq[0, input_length:].tolist()
            if first_token_time is None and block_ids:
                first_token_time = time.time() - start_time

            if eos_token_id is not None and eos_token_id in block_ids:
                eos_pos = block_ids.index(eos_token_id)
                block_ids = block_ids[:eos_pos]
                generated_ids.extend(block_ids)
                break

            if not block_ids:
                break

            generated_ids.extend(block_ids)
            context_append = torch.tensor([block_ids], device=current_device, dtype=torch.long)
            context_ids = torch.cat([context_ids, context_append], dim=1)
            if context_ids.shape[1] > model_max_len:
                context_ids = context_ids[:, -model_max_len:]

    elapsed = first_token_time if first_token_time is not None else (time.time() - start_time)
    return generated_ids[:max_tokens], elapsed


def complete_medusa(input_ids: torch.Tensor, max_tokens: int, temperature: float, top_p: float, verbose: bool) -> tuple:
    """Medusa-like decoding using base AR token plus MTP speculative tokens."""
    if not load_context.capabilities.supports_mtp:
        _warn_mode_fallback("medusa", "ar", verbose)
        return complete_ar(input_ids, max_tokens, temperature, top_p, verbose)

    generated_ids = []
    start_time = time.time()
    first_token_time = None
    input_length = input_ids.shape[1]
    model_max_len = getattr(model, "max_seq_len", getattr(getattr(model, "config", None), "max_position_embeddings", input_length + max_tokens))
    max_length = min(input_length + max_tokens, model_max_len)

    with torch.no_grad():
        while input_ids.shape[1] < max_length and len(generated_ids) < max_tokens:
            ar_logits, outputs = _forward_ar_outputs(input_ids)
            mtp_logits = _forward_mtp_logits(outputs)

            next_token_logits = ar_logits[0, -1, :] / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")
            probs = torch.softmax(next_token_logits, dim=-1)
            candidate_tokens = [torch.multinomial(probs, num_samples=1).item()]

            for head_logits in mtp_logits[:3]:
                head_token_logits = head_logits[0, -1, :] / temperature
                head_probs = torch.softmax(head_token_logits, dim=-1)
                candidate_tokens.append(torch.multinomial(head_probs, num_samples=1).item())

            accepted = 0
            for token_id in candidate_tokens:
                if input_ids.shape[1] >= max_length or len(generated_ids) >= max_tokens:
                    break
                token_tensor = torch.tensor([[token_id]], device=input_ids.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, token_tensor], dim=1)
                generated_ids.append(token_id)
                accepted += 1

                if first_token_time is None:
                    first_token_time = time.time() - start_time
                    if verbose:
                        print(f"[medusa] First token at {first_token_time:.3f}s")

                if token_id == tokenizer.eos_token_id:
                    break

            if verbose:
                print(f"[medusa] Accepted {accepted} token(s), total={len(generated_ids)}")

            if generated_ids and generated_ids[-1] == tokenizer.eos_token_id:
                break

    elapsed = first_token_time if first_token_time else time.time() - start_time
    return generated_ids[:max_tokens], elapsed


def complete_medusa_diffusion(
    input_ids: torch.Tensor,
    max_tokens: int,
    temperature: float,
    top_p: float,
    verbose: bool = True,
) -> tuple:
    """Hybrid mode: medusa draft first, then diffusion refinement conditioned on the original prompt."""
    if not (load_context.capabilities.supports_mtp and load_context.capabilities.supports_diffusion):
        _warn_mode_fallback("medusa+diffusion", "ar", verbose)
        return complete_ar(input_ids, max_tokens, temperature, top_p, verbose)

    drafted_ids, medusa_ttft = complete_medusa(input_ids.clone(), max_tokens, temperature, top_p, verbose)
    if len(drafted_ids) == 0:
        return drafted_ids, medusa_ttft

    refined_ids, _ = complete_diffusion(
        input_ids.clone(),
        max_tokens=len(drafted_ids),
        num_steps=12,
        temperature=max(0.8, temperature),
        verbose=verbose,
    )

    final_ids = refined_ids[: len(drafted_ids)] if refined_ids else drafted_ids
    if verbose:
        print(f"[medusa+diffusion] Drafted={len(drafted_ids)} Refined={len(final_ids)}")

    return final_ids, medusa_ttft


def complete(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    mode: str = "ar",
    verbose: bool = True,
) -> Dict[str, Any]:
    _ensure_loaded()

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    input_length = input_ids.shape[1]
    start_time = time.time()

    if verbose:
        print(f"[{mode}] Prompt: {prompt[:50]}...")
        print(f"[{mode}] Input tokens: {input_length}")

    if mode == "ar":
        generated_ids, first_token_time = complete_ar(input_ids, max_tokens, temperature, top_p, verbose)
    elif mode == "diffusion":
        generated_ids, first_token_time = complete_diffusion(
            input_ids,
            max_tokens,
            num_steps=12,
            temperature=temperature,
            top_p=top_p,
            top_k=64,
            block_size=256,
            repetition_penalty=1.1,
            verbose=verbose,
        )
    elif mode == "combined":
        generated_ids, first_token_time = complete_ar(input_ids, max_tokens, temperature, top_p, verbose)
    elif mode == "reasoning":
        generated_ids, first_token_time = complete_ar(input_ids, max_tokens, temperature, top_p, verbose)
    elif mode == "medusa":
        generated_ids, first_token_time = complete_medusa(input_ids, max_tokens, temperature, top_p, verbose)
    elif mode == "medusa+diffusion":
        generated_ids, first_token_time = complete_medusa_diffusion(
            input_ids,
            max_tokens,
            temperature,
            top_p,
            verbose,
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

    return {
        "text": generated_text,
        "ttft": ttft,
        "tps": tps,
        "total_tokens": len(generated_ids),
        "finish_reason": "stop" if generated_ids else "length",
    }


def complete_with_tools(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    mode: str = "ar",
    max_tool_cycles: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Generate completion with tool calling support."""
    _ensure_loaded()

    if tool_registry is None or tool_parser is None:
        init_tools()

    tool_schemas = tool_registry.get_schemas_text() if tool_registry else ""
    full_prompt = prompt
    if tool_schemas:
        full_prompt = (
            f"{prompt}\n\nAvailable tools:\n{tool_schemas}\n\n"
            'When you need to use a tool, output the tool call in JSON format: {"tool": "tool_name", "args": {...}}'
        )

    tool_calls_executed = []
    current_prompt = full_prompt

    for cycle in range(max_tool_cycles):
        if verbose:
            print(f"\n=== Tool Cycle {cycle + 1} ===")

        result = complete(
            current_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            mode=mode,
            verbose=verbose,
        )

        generated_text = result["text"]
        tool_calls = tool_parser.parse(generated_text)

        if not tool_calls:
            if verbose:
                print("[complete_with_tools] No tool calls detected, returning result")
            return {
                "text": generated_text,
                "tool_calls": tool_calls_executed,
                "cycles": cycle + 1,
                "final": True,
            }

        if verbose:
            print(f"[complete_with_tools] Executing {len(tool_calls)} tool call(s)")

        for tool_call in tool_calls:
            if verbose:
                print(f"[complete_with_tools] Tool: {tool_call.tool_name}, args: {tool_call.arguments}")

            try:
                tool_result = tool_registry.execute(tool_call.tool_name, tool_call.arguments)
                tool_calls_executed.append(
                    {
                        "tool": tool_call.tool_name,
                        "args": tool_call.arguments,
                        "result": tool_result,
                        "success": True,
                    }
                )
                if verbose:
                    print(f"[complete_with_tools] Result: {tool_result}")
            except Exception as exc:
                error_msg = str(exc)
                tool_calls_executed.append(
                    {
                        "tool": tool_call.tool_name,
                        "args": tool_call.arguments,
                        "error": error_msg,
                        "success": False,
                    }
                )
                if verbose:
                    print(f"[complete_with_tools] Error: {error_msg}")

        tool_results_text = "\n\n".join(
            f"Tool '{tool_call['tool']}' result: {json.dumps(tool_call.get('result', tool_call.get('error')))}"
            for tool_call in tool_calls_executed[-len(tool_calls):]
        )

        current_prompt = (
            f"{current_prompt}\n\n{generated_text}\n\n{tool_results_text}\n\n"
            "Based on the tool results, provide your final answer:"
        )
        max_tokens = max_tokens // 2

    return {
        "text": generated_text,
        "tool_calls": tool_calls_executed,
        "cycles": max_tool_cycles,
        "final": False,
        "reason": "max_cycles",
    }


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model-path", type=str, help="Path to custom checkpoint dir/file, HF model dir/file, or adapter dir")
    parser.add_argument("--config-path", type=str, help="Optional config path for model.pt or single safetensors inputs")
    parser.add_argument("--tokenizer-path", type=str, help="Optional tokenizer override path")
    parser.add_argument("--base-model-path", type=str, help="Optional base model override when loading an adapter directory")
    parser.add_argument("--checkpoint", type=str, help="Deprecated alias for --model-path when loading model.pt")
    parser.add_argument("--config", type=str, help="Deprecated alias for --config-path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", help="Prompt for completion")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling")
    parser.add_argument(
        "--mode",
        type=str,
        default="ar",
        choices=["ar", "diffusion", "combined", "reasoning", "medusa", "medusa+diffusion"],
        help="Generation mode",
    )
    parser.add_argument("--device", type=str, help="Device override, e.g. cpu or cuda")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp32", "fp16", "bf16"], help="Model dtype")
    parser.add_argument("--use-tools", action="store_true", help="Enable tool calling mode")
    parser.add_argument("--max-tool-cycles", type=int, default=3, help="Max tool call cycles in tool mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", help="Allow remote code when loading HF models")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="Disable remote code when loading HF models")
    parser.set_defaults(trust_remote_code=True)
    args = parser.parse_args()

    if args.checkpoint or args.config:
        print("Warning: --checkpoint/--config are deprecated; use --model-path/--config-path instead.")

    model_path = args.model_path or args.checkpoint
    config_path = args.config_path or args.config
    if not model_path:
        parser.error("Provide --model-path, or use deprecated --checkpoint.")

    load_model(
        model_path=model_path,
        config_path=config_path,
        tokenizer_path=args.tokenizer_path,
        base_model_path=args.base_model_path,
        device_str=args.device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    if args.use_tools:
        result = complete_with_tools(
            args.prompt,
            args.max_tokens,
            args.temperature,
            args.top_p,
            args.mode,
            args.max_tool_cycles,
            verbose=not args.quiet,
        )
        print("\n=== RESULT (with tools) ===")
        print(f"Text: {result['text']}")
        print(f"Tool calls executed: {len(result.get('tool_calls', []))}")
        print(f"Cycles: {result.get('cycles', 1)}")
        if result.get("tool_calls"):
            print("Tool call details:")
            for tool_call in result["tool_calls"]:
                status = "success" if tool_call.get("success") else f"error: {tool_call.get('error')}"
                print(f"  - {tool_call['tool']}({tool_call['args']}): {status}")
        return

    result = complete(
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.mode,
        verbose=not args.quiet,
    )
    print("\n=== RESULT ===")
    print(f"Text: {result['text']}")
    print(f"TTFT: {result['ttft']:.3f}s, TPS: {result['tps']:.2f}, Tokens: {result['total_tokens']}")


if __name__ == "__main__":
    main()
