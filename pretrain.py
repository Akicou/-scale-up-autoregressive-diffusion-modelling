#!/usr/bin/env python3
"""
Pretraining script for compact language model with dual-mode AR + Diffusion architecture.

This script trains a ~10M parameter transformer from scratch with:
- Auto-Regressive (AR) head for next-token prediction
- Diffusion head for masked token prediction
- Shared backbone transformer

Usage:
    python pretrain.py \
        --data-path ./data/train.bin \
        --val-data-path ./data/val.bin \
        --output-dir ./checkpoints/10m_pretrain \
        --epochs 3 \
        --batch-size 64 \
        --lr 1e-4 \
        --warmup-steps 100 \
        --save-interval 1000 \
        --wandb-project my-pretrain
"""

import argparse
import itertools
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm
import numpy as np
import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# Try to import tokenizer
try:
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
except ImportError:
    Tokenizer = None
    PreTrainedTokenizerFast = None

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

# Try to import Flash Attention 2
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    flash_attn_func = None
    HAS_FLASH_ATTN = False


# ============================================================================
# Model Components
# ============================================================================

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for efficient positional encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Create inverse frequency table
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute sin/cos cache
        self._set_cos_sin_cache(max_seq_len)
    
    def _set_cos_sin_cache(self, seq_len: int):
        """Precompute sin and cos values for efficiency."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half of the hidden dims."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    # q, k: [batch, heads, seq, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SwiGLU(nn.Module):
    """SwiGLU activation function for MLP."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE, KV caching, and Flash Attention 2 support."""
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, max_seq_len: int = 2048, use_flash_attn: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rope = RotaryPositionEmbedding(head_dim, max_seq_len)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(seq_len, x.device)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache
        if past_key_values is not None:
            past_k, past_v = past_key_values
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        
        if use_cache:
            present = (k, v)
        else:
            present = None
        
        # Use Flash Attention 2 if enabled and available
        if self.use_flash_attn and not use_cache and not past_key_values:
            # Flash Attention 2: [batch, seq, num_heads, head_dim]
            # Input needs to be [batch, num_heads, seq, head_dim]
            q = q.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Flash Attention requires contiguous tensor
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            
            # Call flash_attn_func
            # Output: [batch, num_heads, seq, head_dim] -> reshape
            attn_output = flash_attn_func(
                q, k, v,
                softmax_scale=None,  # Will be computed internally
                causal=True,  # Causal masking for AR model
            )
            
            # Reshape: [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads * head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        else:
            # Standard attention (slower but works with caching)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        output = self.o_proj(attn_output)
        
        return output, present


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""
    
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0, max_seq_len: int = 2048, use_flash_attn: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.gradient_checkpointing = False
        
        # Pre-norm architecture
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        self.attention = MultiHeadAttention(hidden_size, num_heads, head_dim, max_seq_len, use_flash_attn)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm attention
        x_norm = self.input_layernorm(x)
        attn_output, present = self.attention(x_norm, attention_mask, use_cache=use_cache, past_key_values=past_key_values)
        x = x + attn_output
        
        # Pre-norm MLP
        x = x + self.mlp(self.post_attention_layernorm(x))
        
        return x, present


class ARHead(nn.Module):
    """Auto-Regressive head for next-token prediction."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class DiffusionHead(nn.Module):
    """Diffusion head for masked token prediction (BERT/T5-style)."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return self.decoder(hidden_states)


class DualModeModel(nn.Module):
    """
    Dual-mode transformer with shared backbone and two heads:
    - AR (Auto-Regressive) head for next-token prediction
    - Diffusion head for masked token prediction
    """
    
    def __init__(
        self,
        vocab_size: int = 16000,
        hidden_size: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        use_cache: bool = False,
        use_flash_attn: bool = False,
        gradient_checkpointing: bool = False,
        diffusion_mask_prob: float = 0.15,
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        self.mlp_ratio = mlp_ratio
        self.diffusion_mask_prob = diffusion_mask_prob
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        # Position embeddings - use smaller base since RoPE handles extrapolation
        # For long context (256K), use 8192 as base and interpolate
        base_pos_len = min(max_seq_len, 8192)
        self.position_embeddings = nn.Embedding(base_pos_len, hidden_size)
        self.max_seq_len = max_seq_len  # RoPE will extrapolate beyond base_pos_len
        self.embed_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, head_dim, mlp_ratio, max_seq_len, use_flash_attn)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_layernorm = nn.LayerNorm(hidden_size, eps=1e-6)
        
        # Two heads sharing the backbone
        self.ar_head = ARHead(hidden_size, vocab_size)
        self.diffusion_head = DiffusionHead(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def get_num_params(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> Dict[str, Any]:
        """Return a serializable model config."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "mlp_ratio": self.mlp_ratio,
            "max_seq_len": self.max_seq_len,
            "diffusion_mask_prob": self.diffusion_mask_prob,
        }

    def save_pretrained(self, save_directory: str):
        """Save the checkpoint in the repo's native format."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), save_path / "model.pt")
        torch.save(self.get_config(), save_path / "config.pt")

    @classmethod
    def from_pretrained(cls, load_directory: str, map_location: str | torch.device = "cpu") -> "DualModeModel":
        """Load a checkpoint saved by save_pretrained."""
        load_path = Path(load_directory)
        config = torch.load(load_path / "config.pt", map_location="cpu")
        model = cls(**config)
        state_dict = torch.load(load_path / "model.pt", map_location=map_location)
        model.load_state_dict(state_dict)
        return model
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all layers."""
        self.gradient_checkpointing = True
        for layer in self.layers:
            layer.gradient_checkpointing = True

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run the shared transformer backbone."""
        batch_size, seq_len = input_ids.shape

        input_ids = input_ids.clamp(0, self.vocab_size - 1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_ids = position_ids % self.position_embeddings.num_embeddings

        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.embed_layernorm(hidden_states)

        if attention_mask is None:
            attention_mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            attention_mask = torch.triu(attention_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        present = None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and layer.gradient_checkpointing:
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, use_cache, past, use_reentrant=False
                )
            else:
                hidden_states, present = layer(hidden_states, attention_mask, use_cache=use_cache, past_key_values=past)

        hidden_states = self.final_layernorm(hidden_states)
        return hidden_states, present
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        mode: str = "both",
    ) -> dict:
        """
        Forward pass supporting both AR and diffusion modes.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Labels for training [batch, seq_len]
            use_cache: Whether to use KV caching
            past_key_values: Past KV cache
            mode: "ar", "diffusion", or "both"
        
        Returns:
            Dictionary with losses and logits
        """
        outputs = {}
        present = None
        ar_hidden_states = None
        diffusion_hidden_states = None
        diffusion_labels = labels
        
        if mode in ("ar", "both"):
            ar_hidden_states, present = self._encode(
                input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_values=past_key_values,
            )
        
        # AR head - for next-token prediction
        if mode in ("ar", "both"):
            ar_logits = self.ar_head(ar_hidden_states)
            outputs["ar_logits"] = ar_logits
            
            if labels is not None:
                # Shift for causal language modeling
                shift_logits = ar_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Clamp labels to vocab_size range
                shift_labels = shift_labels.clamp(0, self.vocab_size - 1)
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ar_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                outputs["ar_loss"] = ar_loss
        
        # Diffusion head - for masked token prediction
        if mode in ("diffusion", "both"):
            diffusion_input_ids = input_ids
            if labels is not None:
                mask = torch.rand(input_ids.shape, device=input_ids.device) < self.diffusion_mask_prob
                diffusion_labels = labels.clone()
                diffusion_labels[~mask] = -100
                diffusion_input_ids = input_ids.clone()
                diffusion_input_ids[mask] = 0
            diffusion_hidden_states, _ = self._encode(
                diffusion_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                past_key_values=None,
            )
            diffusion_logits = self.diffusion_head(diffusion_hidden_states)
            outputs["diffusion_logits"] = diffusion_logits
            
            if diffusion_labels is not None:
                # Clamp labels to vocab_size range
                clamped_labels = diffusion_labels.clamp(-100, self.vocab_size - 1)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                diffusion_loss = loss_fct(diffusion_logits.view(-1, self.vocab_size), clamped_labels.view(-1))
                outputs["diffusion_loss"] = diffusion_loss
        
        if use_cache:
            outputs["present"] = present
        
        return outputs
    
    def generate_ar(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens using AR mode."""
        self.eval()
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get last token
                logits = self.forward(generated, mode="ar")["ar_logits"]
                logits = logits[:, -1, :] / temperature
                
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, -1:]] = float("-inf")
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def generate_diffusion(
        self,
        input_ids: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Generate tokens using Diffusion mode (masked token prediction).
        Simplified iterative unmasking process.
        """
        self.eval()
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Start with all tokens masked
        masked_input = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        
        # Gradually unmask tokens
        num_masked = seq_len
        for step in range(num_steps):
            # Predict unmasked tokens
            outputs = self.forward(masked_input, mode="diffusion")
            logits = outputs["diffusion_logits"]
            
            # Unmask a portion of tokens each step
            num_to_unmask = max(1, num_masked // (num_steps - step))
            
            # Get predictions for masked positions
            predictions = logits.argmax(dim=-1)
            
            # Update masked positions
            for b in range(batch_size):
                for i in range(seq_len):
                    if masked_input[b, i] == 0:
                        if num_to_unmask > 0:
                            masked_input[b, i] = predictions[b, i]
                            num_to_unmask -= 1
        
        return masked_input


# ============================================================================
# Dataset
# ============================================================================

class BinaryDataset(Dataset):
    """Dataset for binary token files."""
    
    def __init__(self, file_path: str, seq_len: int = 512):
        self.file_path = file_path
        self.seq_len = seq_len
        
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Binary dataset not found: {file_path}")

        self.data = np.fromfile(file_path, dtype=np.uint16)
        if len(self.data) < seq_len:
            raise ValueError(f"Binary dataset '{file_path}' does not contain a full sequence of length {seq_len}.")
        
        self.num_samples = len(self.data) // seq_len
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len
        tokens = self.data[start:end]
        return torch.tensor(tokens, dtype=torch.long)


class TextDataset(Dataset):
    """Dataset for text files with on-the-fly tokenization."""
    
    def __init__(self, file_path: str, tokenizer, seq_len: int = 512):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Text dataset not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            self.texts = [line for line in f.readlines() if line.strip()]

        if not self.texts:
            raise ValueError(f"Text dataset '{file_path}' is empty.")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self.texts[idx][: self.seq_len * 4]  # Approximate truncation
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Pad or truncate
        if len(tokens) < self.seq_len:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.seq_len - len(tokens))
        else:
            tokens = tokens[: self.seq_len]
        
        return torch.tensor(tokens, dtype=torch.long)


class HFDataset(Dataset):
    """Dataset loaded from Hugging Face."""
    
    def __init__(
        self,
        dataset_name: str,
        dataset_column: str,
        split: str = "train",
        seq_len: int = 512,
        tokenizer=None,
        max_samples: Optional[int] = None,
        dataset_config: Optional[str] = None,
        streaming: bool = True,
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_column = dataset_column
        self.dataset_config = dataset_config
        self.split = split
        self.streaming = streaming
        self._texts: List[str] = []
        self.total_tokens = 0
        
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading Hugging Face dataset '{dataset_name}' split '{split}'...")
        dataset = load_dataset(
            dataset_name,
            name=dataset_config,
            split=split,
            streaming=streaming,
            trust_remote_code=False,
        )

        iterator = dataset if max_samples is None else itertools.islice(dataset, max_samples)
        for item in iterator:
            text = item.get(dataset_column)
            if isinstance(text, str) and text.strip():
                self._texts.append(text)

        if not self._texts:
            raise ValueError(
                f"Dataset '{dataset_name}' split '{split}' yielded no usable rows from column '{dataset_column}'."
            )

        if tokenizer is not None:
            sample_size = min(len(self._texts), 128)
            token_counts = [
                min(seq_len, len(tokenizer.encode(text, add_special_tokens=True)))
                for text in self._texts[:sample_size]
            ]
            avg_tokens = sum(token_counts) / max(1, len(token_counts))
            self.total_tokens = int(avg_tokens * len(self._texts))
        else:
            self.total_tokens = len(self._texts) * seq_len

        print(f"Loaded {len(self._texts)} samples from {dataset_name}")
    
    def __len__(self) -> int:
        return len(self._texts)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        text = self._texts[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)


def get_default_tokenizer():
    """Get a default tokenizer for the dataset."""
    from transformers import GPT2Tokenizer
    # Use GPT-2 tokenizer (public, no auth required)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens for reasoning/thinking
    special_tokens = [
        "<thinking>",
        "</thinking>",
        "<reasoning>",
        "</reasoning>",
        "<output>",
        "</output>",
    ]
    num_added = tokenizer.add_tokens(special_tokens)
    if num_added > 0:
        print(f"Added {num_added} special tokens for reasoning: {special_tokens}")
    
    return tokenizer


# ============================================================================
# Training
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    hf_dataset: str = "openbmb/Ultra-FineWeb"
    hf_dataset_column: str = "content"
    hf_dataset_split: str = "en"
    hf_dataset_config: Optional[str] = None
    hf_streaming: bool = True
    hf_max_samples: Optional[int] = None
    output_dir: str = "./checkpoints/10m_pretrain"
    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-4
    warmup_steps: int = 100
    save_interval: int = 1000
    log_interval: int = 10
    eval_interval: int = 1000
    max_seq_len: int = 512
    
    # Model config
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 10
    num_heads: int = 8
    head_dim: int = 64
    mlp_ratio: float = 4.0
    
    # Training options
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    accumulation_steps: int = 1
    use_flash_attention: bool = False  # Flash Attention 2 for long context
    diffusion_mask_prob: float = 0.15
    max_grad_norm: float = 1.0
    num_workers: int = 0
    seed: int = 42
    
    # Loss options
    loss_mode: str = "both"  # "ar", "diffusion", or "both"
    ar_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    
    # Logging
    wandb_project: Optional[str] = None
    min_tokens_per_param: float = 20.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Create a learning rate scheduler with linear warmup."""
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


def get_accelerate_mixed_precision(mixed_precision: str) -> str:
    """Map CLI mixed precision mode to Accelerate values."""
    return "no" if mixed_precision == "fp32" else mixed_precision


def compute_training_loss(outputs: Dict[str, torch.Tensor], config: TrainingConfig) -> torch.Tensor:
    """Compute the configured training loss from model outputs."""
    if config.loss_mode == "both":
        return config.ar_loss_weight * outputs.get("ar_loss", 0) + \
               config.diffusion_loss_weight * outputs.get("diffusion_loss", 0)
    if config.loss_mode == "ar":
        return outputs.get("ar_loss", 0)
    if config.loss_mode == "diffusion":
        return outputs.get("diffusion_loss", 0)
    return outputs.get("ar_loss", 0)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    accelerator: Accelerator,
    config: TrainingConfig,
    epoch: int,
    global_step: int,
    val_dataloader: Optional[DataLoader] = None,
    tokenizer=None,
) -> Tuple[dict, int]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_ar_loss = 0.0
    total_diffusion_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        with accelerator.accumulate(model):
            outputs = model(batch, labels=batch, mode=config.loss_mode)
            loss = compute_training_loss(outputs, config)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        reduced_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
        total_loss += reduced_loss
        if outputs.get("ar_loss") is not None:
            reduced_ar_loss = accelerator.gather(outputs["ar_loss"].detach().float().reshape(1)).mean().item()
            total_ar_loss += reduced_ar_loss
        if outputs.get("diffusion_loss") is not None:
            reduced_diffusion_loss = accelerator.gather(outputs["diffusion_loss"].detach().float().reshape(1)).mean().item()
            total_diffusion_loss += reduced_diffusion_loss
        num_batches += 1

        if accelerator.is_local_main_process:
            pbar.set_postfix({
                "loss": f"{total_loss / num_batches:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

        if accelerator.sync_gradients:
            global_step += 1

            if config.wandb_project and HAS_WANDB and accelerator.is_main_process and global_step % config.log_interval == 0:
                wandb.log({
                    "train/loss": total_loss / num_batches,
                    "train/ar_loss": total_ar_loss / max(1, num_batches),
                    "train/diffusion_loss": total_diffusion_loss / max(1, num_batches),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/step": global_step,
                })

            if config.save_interval > 0 and global_step % config.save_interval == 0:
                save_checkpoint(model, optimizer, scheduler, accelerator, epoch, global_step, config, tokenizer=tokenizer)

            if val_dataloader is not None and config.eval_interval > 0 and global_step % config.eval_interval == 0:
                eval_metrics = evaluate(model, val_dataloader, accelerator, config)
                accelerator.print(f"Evaluation @ step {global_step}: {eval_metrics}")
                model.train()
                if config.wandb_project and HAS_WANDB and accelerator.is_main_process:
                    wandb.log({"eval/loss": eval_metrics["loss"], "eval/step": global_step})
    
    metrics = {
        "loss": total_loss / num_batches,
        "ar_loss": total_ar_loss / num_batches if total_ar_loss > 0 else None,
        "diffusion_loss": total_diffusion_loss / num_batches if total_diffusion_loss > 0 else None,
    }
    return metrics, global_step


def evaluate(model: nn.Module, dataloader: DataLoader, accelerator: Accelerator, config: TrainingConfig) -> dict:
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            outputs = model(batch, labels=batch, mode=config.loss_mode)
            loss = compute_training_loss(outputs, config)
            reduced_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            total_loss += reduced_loss
            num_batches += 1
    
    return {"loss": total_loss / num_batches}


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
    epoch: int,
    step: int,
    config: TrainingConfig,
    tokenizer=None,
    checkpoint_name: Optional[str] = None,
):
    """Save model checkpoint."""
    save_path = Path(config.output_dir) / (checkpoint_name or f"checkpoint-epoch{epoch}-step{step}")
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        save_path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(accelerator.get_state_dict(model), save_path / "model.pt")
    accelerator.save(unwrapped_model.get_config(), save_path / "config.pt")
    accelerator.save({
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "world_size": accelerator.num_processes,
    }, save_path / "trainer_state.pt")

    if accelerator.is_main_process and tokenizer is not None:
        tokenizer.save_pretrained(save_path)
        print(f"Saved checkpoint to {save_path}")

    accelerator.wait_for_everyone()


def estimate_available_tokens(dataset: Dataset, tokenizer=None) -> int:
    """Estimate how many non-padding training tokens are available."""
    if hasattr(dataset, "total_tokens") and getattr(dataset, "total_tokens") is not None:
        return int(dataset.total_tokens)
    if isinstance(dataset, BinaryDataset):
        return int(len(dataset.data))
    if isinstance(dataset, TextDataset):
        total_tokens = 0
        for text in dataset.texts:
            total_tokens += min(dataset.seq_len, len(tokenizer.encode(text[: dataset.seq_len * 4], add_special_tokens=True)))
        return total_tokens
    return len(dataset) * getattr(dataset, "seq_len", 0)


def required_pretraining_tokens(model: DualModeModel, min_tokens_per_param: float) -> int:
    """Compute the minimum recommended token budget using a Chinchilla-style heuristic."""
    return math.ceil(model.get_num_params() * min_tokens_per_param)


def validate_pretraining_data(dataset: Dataset, model: DualModeModel, config: TrainingConfig, tokenizer=None, print_fn=print):
    """Exit early if the available token budget is too small for the model size."""
    available_tokens = estimate_available_tokens(dataset, tokenizer=tokenizer)
    required_tokens = required_pretraining_tokens(model, config.min_tokens_per_param)
    print_fn(f"Estimated training tokens: {available_tokens:,}")
    print_fn(f"Required minimum tokens: {required_tokens:,} ({config.min_tokens_per_param:g} tokens/parameter)")
    if available_tokens < required_tokens:
        raise ValueError(
            "Insufficient data for pretraining. "
            f"Need at least {required_tokens:,} tokens for a {model.get_num_params():,}-parameter model, "
            f"but only found about {available_tokens:,}."
        )


def calculate_target_model_config(
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    mlp_ratio: float,
    target_params: int,
    method: str = "width+depth",
) -> Dict[str, Any]:
    """Estimate a model configuration for a requested target parameter count."""
    if target_params <= 0:
        raise ValueError(f"target_params must be positive, received {target_params}")

    params_per_layer = 2 * hidden_size ** 2 + 4 * hidden_size * int(hidden_size * mlp_ratio)
    current_params = vocab_size * hidden_size + num_layers * params_per_layer + hidden_size * vocab_size
    scale_factor = (target_params / max(1, current_params)) ** 0.5

    if method == "width":
        new_hidden_size = int(hidden_size * scale_factor)
        new_num_heads = max(1, int(num_heads * scale_factor))
        new_num_layers = num_layers
    elif method == "depth":
        new_hidden_size = hidden_size
        new_num_heads = num_heads
        new_num_layers = max(1, int(num_layers * scale_factor))
    elif method == "width+depth":
        width_scale = scale_factor ** 0.7
        depth_scale = scale_factor ** 0.5
        new_hidden_size = int(hidden_size * width_scale)
        new_num_heads = max(1, int(num_heads * width_scale))
        new_num_layers = max(1, int(num_layers * depth_scale))
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    new_hidden_size = max(new_num_heads, new_hidden_size)
    while new_num_heads > 1 and new_hidden_size % new_num_heads != 0:
        new_num_heads -= 1
    new_head_dim = max(1, new_hidden_size // max(1, new_num_heads))

    return {
        "vocab_size": vocab_size,
        "hidden_size": new_hidden_size,
        "num_layers": new_num_layers,
        "num_heads": new_num_heads,
        "head_dim": new_head_dim,
        "mlp_ratio": mlp_ratio,
        "max_seq_len": max_seq_len,
        "target_parameters": target_params,
        "estimated_parameters": (
            vocab_size * new_hidden_size
            + new_num_layers * (2 * new_hidden_size ** 2 + 4 * new_hidden_size * int(new_hidden_size * mlp_ratio))
            + new_hidden_size * vocab_size
        ),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pretrain a dual-mode AR + Diffusion model")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, default=None, help="Path to training data (.bin or .txt). Leave unset to use the default Hugging Face dataset.")
    parser.add_argument("--val-data-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--hf-dataset", type=str, default="openbmb/Ultra-FineWeb", help="Default Hugging Face dataset to use when --data-path is not provided")
    parser.add_argument("--hf-dataset-column", type=str, default="content", help="Text column to read from the Hugging Face dataset")
    parser.add_argument("--hf-dataset-split", type=str, default="en", help="Split to load from the Hugging Face dataset")
    parser.add_argument("--hf-dataset-config", type=str, default=None, help="Optional dataset config name")
    parser.add_argument("--hf-max-samples", type=int, default=None, help="Optional cap on loaded Hugging Face rows")
    parser.add_argument("--no-hf-streaming", action="store_true", help="Disable Hugging Face streaming mode")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/10m_pretrain", help="Output directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (smaller for long sequences)")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count per process")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments - defaults for ~10M model
    parser.add_argument("--vocab-size", type=int, default=16000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP expansion ratio")
    parser.add_argument("--target-parameters", type=int, default=None, help="Optional target parameter count for auto-sizing the model")
    parser.add_argument("--scaling-method", type=str, default="width+depth", choices=["width", "depth", "width+depth"], help="How to auto-scale the model when --target-parameters is set")
    
    # Training options
    parser.add_argument("--use-flash-attention", action="store_true", help="Enable Flash Attention 2 for long context (requires flash-attn package)")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--loss-mode", type=str, default="both", choices=["ar", "diffusion", "both"], help="Loss mode")
    parser.add_argument("--ar-loss-weight", type=float, default=1.0, help="AR loss weight")
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0, help="Diffusion loss weight")
    parser.add_argument("--diffusion-mask-prob", type=float, default=0.15, help="Mask probability used for diffusion pretraining")
    parser.add_argument("--min-tokens-per-param", type=float, default=20.0, help="Minimum data budget required before pretraining starts")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        hf_dataset=args.hf_dataset,
        hf_dataset_column=args.hf_dataset_column,
        hf_dataset_split=args.hf_dataset_split,
        hf_dataset_config=args.hf_dataset_config,
        hf_streaming=not args.no_hf_streaming,
        hf_max_samples=args.hf_max_samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        max_seq_len=args.seq_len,
        num_workers=args.num_workers,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        mlp_ratio=args.mlp_ratio,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_flash_attention=args.use_flash_attention,
        diffusion_mask_prob=args.diffusion_mask_prob,
        mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
        loss_mode=args.loss_mode,
        ar_loss_weight=args.ar_loss_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
        wandb_project=args.wandb_project,
        min_tokens_per_param=args.min_tokens_per_param,
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.loss_mode != "both")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.accumulation_steps,
        mixed_precision=get_accelerate_mixed_precision(config.mixed_precision),
        kwargs_handlers=[ddp_kwargs],
    )
    set_seed(config.seed)
    config.device = str(accelerator.device)
    accelerator.print(f"Using device: {accelerator.device}")
    accelerator.print(f"Distributed world size: {accelerator.num_processes}")
    
    tokenizer = None
    using_hf_dataset = config.data_path is None
    if using_hf_dataset or (config.data_path and config.data_path.endswith(".txt")):
        tokenizer = get_default_tokenizer()
        if len(tokenizer) > config.vocab_size:
            config.vocab_size = len(tokenizer)

    if args.target_parameters is not None:
        resolved_model_config = calculate_target_model_config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            mlp_ratio=config.mlp_ratio,
            target_params=args.target_parameters,
            method=args.scaling_method,
        )
        config.hidden_size = resolved_model_config["hidden_size"]
        config.num_layers = resolved_model_config["num_layers"]
        config.num_heads = resolved_model_config["num_heads"]
        config.head_dim = resolved_model_config["head_dim"]
    else:
        resolved_model_config = None

    # Create model
    accelerator.print("\n=== Model Configuration ===")
    accelerator.print(f"Vocab size: {config.vocab_size}")
    accelerator.print(f"Hidden size: {config.hidden_size}")
    accelerator.print(f"Num layers: {config.num_layers}")
    accelerator.print(f"Num heads: {config.num_heads}")
    accelerator.print(f"Head dim: {config.head_dim}")
    accelerator.print(f"Flash Attention: {config.use_flash_attention}")
    if resolved_model_config is not None:
        accelerator.print(f"Target parameters: {resolved_model_config['target_parameters']:,}")
        accelerator.print(f"Estimated parameters: {resolved_model_config['estimated_parameters']:,}")
        accelerator.print(f"Scaling method: {args.scaling_method}")
    
    model = DualModeModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        mlp_ratio=config.mlp_ratio,
        max_seq_len=config.max_seq_len,
        use_flash_attn=config.use_flash_attention,
        gradient_checkpointing=config.use_gradient_checkpointing,
        diffusion_mask_prob=config.diffusion_mask_prob,
    )
    
    num_params = model.get_num_params()
    accelerator.print(f"\nTotal parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # Create datasets
    accelerator.print("\n=== Loading Dataset ===")
    
    if using_hf_dataset:
        accelerator.print(f"Using Hugging Face dataset '{config.hf_dataset}' (default)")
        required_tokens = required_pretraining_tokens(model, config.min_tokens_per_param)
        default_hf_samples = math.ceil(required_tokens / max(1, config.max_seq_len))
        train_dataset = HFDataset(
            dataset_name=config.hf_dataset,
            dataset_column=config.hf_dataset_column,
            split=config.hf_dataset_split,
            seq_len=config.max_seq_len,
            tokenizer=tokenizer,
            dataset_config=config.hf_dataset_config,
            streaming=config.hf_streaming,
            max_samples=config.hf_max_samples or default_hf_samples,
        )
    elif config.data_path.endswith(".txt"):
        train_dataset = TextDataset(config.data_path, tokenizer, config.max_seq_len)
    else:
        train_dataset = BinaryDataset(config.data_path, config.max_seq_len)

    validate_pretraining_data(train_dataset, model, config, tokenizer=tokenizer, print_fn=accelerator.print)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )
    
    val_loader = None
    if config.val_data_path:
        if config.val_data_path == "hf":
            val_dataset = HFDataset(
                dataset_name=config.hf_dataset,
                dataset_column=config.hf_dataset_column,
                split=config.hf_dataset_split,
                seq_len=config.max_seq_len,
                tokenizer=tokenizer,
                dataset_config=config.hf_dataset_config,
                streaming=config.hf_streaming,
                max_samples=max(128, config.batch_size * 8),
            )
        elif config.val_data_path.endswith(".txt"):
            val_dataset = TextDataset(config.val_data_path, tokenizer, config.max_seq_len)
        else:
            val_dataset = BinaryDataset(config.val_data_path, config.max_seq_len)
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=config.num_workers > 0,
        )
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    if config.wandb_project and HAS_WANDB and accelerator.is_main_process:
        wandb.init(project=config.wandb_project, config=vars(config))

    if val_loader is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    num_training_steps = math.ceil(len(train_loader) / max(1, config.accumulation_steps)) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, num_training_steps)
    scheduler = accelerator.prepare(scheduler)
    
    accelerator.print("\n=== Starting Training ===")
    global_step = 0
    
    for epoch in range(config.epochs):
        train_metrics, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            accelerator,
            config,
            epoch,
            global_step,
            val_dataloader=val_loader,
            tokenizer=tokenizer,
        )
        accelerator.print(f"Epoch {epoch}: {train_metrics}")
        save_checkpoint(model, optimizer, scheduler, accelerator, epoch, global_step, config, tokenizer=tokenizer)
    
    # Final save
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        accelerator,
        config.epochs - 1,
        global_step,
        config,
        tokenizer=tokenizer,
        checkpoint_name="final",
    )
    accelerator.print(f"\nFinal model saved to {Path(config.output_dir) / 'final'}")
    
    # Log model size
    final_model = accelerator.unwrap_model(model)
    accelerator.print(f"\n=== Final Model Statistics ===")
    accelerator.print(f"Total parameters: {final_model.get_num_params():,}")
    accelerator.print(f"AR head parameters: {sum(p.numel() for p in final_model.ar_head.parameters()):,}")
    accelerator.print(f"Diffusion head parameters: {sum(p.numel() for p in final_model.diffusion_head.parameters()):,}")

    if config.wandb_project and HAS_WANDB and accelerator.is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
