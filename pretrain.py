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
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR
from tqdm import tqdm
import numpy as np

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
        vocab_size: int = 32000,
        hidden_size: int = 512,
        num_layers: int = 10,
        num_heads: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
        use_cache: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_seq_len, hidden_size)
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
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden_states = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        hidden_states = self.embed_layernorm(hidden_states)
        
        # Create attention mask for causal attention
        if attention_mask is None:
            # Create causal mask
            attention_mask = torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device)
            attention_mask = torch.triu(attention_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        
        # Transformer layers
        present = None
        for i, layer in enumerate(self.layers):
            past = past_key_values[i] if past_key_values is not None else None
            hidden_states, present = layer(hidden_states, attention_mask, use_cache=use_cache, past_key_values=past)
        
        hidden_states = self.final_layernorm(hidden_states)
        
        outputs = {}
        
        # AR head - for next-token prediction
        if mode in ("ar", "both"):
            ar_logits = self.ar_head(hidden_states)
            outputs["ar_logits"] = ar_logits
            
            if labels is not None:
                # Shift for causal language modeling
                shift_logits = ar_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ar_loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))
                outputs["ar_loss"] = ar_loss
        
        # Diffusion head - for masked token prediction
        if mode in ("diffusion", "both"):
            diffusion_logits = self.diffusion_head(hidden_states)
            outputs["diffusion_logits"] = diffusion_logits
            
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                diffusion_loss = loss_fct(diffusion_logits.view(-1, self.vocab_size), labels.view(-1))
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
        masked_input = torch.full((batch_size, seq_len), self.vocab_size - 1, dtype=torch.long, device=device)
        
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
                    if masked_input[b, i] == self.vocab_size - 1:
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
        
        # Load data
        if os.path.exists(file_path):
            self.data = np.fromfile(file_path, dtype=np.uint16)
        else:
            # Generate dummy data for testing
            print(f"Warning: {file_path} not found. Using dummy data.")
            self.data = np.random.randint(0, 32000, size=(100000,), dtype=np.uint16)
        
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
        
        # Load text data
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                self.texts = f.readlines()
        else:
            print(f"Warning: {file_path} not found. Using dummy data.")
            self.texts = ["This is a sample text for training." * 10] * 1000
    
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


class UltraFineWebDataset(Dataset):
    """Dataset for Ultra-FineWeb from HuggingFace."""
    
    def __init__(
        self,
        split: str = "train",
        seq_len: int = 512,
        tokenizer=None,
        max_samples: Optional[int] = None,
    ):
        """
        Load Ultra-FineWeb dataset.
        
        Args:
            split: "train" or "test"
            seq_len: Sequence length
            tokenizer: Tokenizer to use
            max_samples: Maximum number of samples to load (None = all)
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        
        # Try to import datasets
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading Ultra-FineWeb dataset ({split} split)...")
        
        # Load dataset - use streaming to handle large dataset
        try:
            self.dataset = load_dataset(
                "openbmb/Ultra-FineWeb",
                split=split,
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Failed to load Ultra-FineWeb: {e}")
            print("Falling back to dummy data.")
            self.dataset = None
            self._dummy_mode = True
            return
        
        self._dummy_mode = False
        self._iterator = iter(self.dataset)
        
        # Cache for efficiency
        self._cache = []
        self._cache_size = 10000
        self._cache_idx = 0
        
        # Count samples if max_samples is specified
        if max_samples:
            self._max_samples = max_samples
        else:
            self._max_samples = float('inf')
        
        self._num_loaded = 0
    
    def __len__(self) -> int:
        if self._dummy_mode:
            return 1000
        # Ultra-FineWeb has ~100M+ samples, limit for practical use
        return min(self._max_samples, 1000000)  # Cap at 1M samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._dummy_mode:
            return torch.randint(0, 32000, (self.seq_len,), dtype=torch.long)
        
        # Simple approach: load and tokenize on demand
        if self._num_loaded < self._max_samples:
            try:
                item = next(self._iterator)
                # Ultra-FineWeb has "content" column
                text = item.get("content", item.get("text", ""))
                
                if not text:
                    # Skip empty texts
                    return self.__getitem__(idx)
                
                # Tokenize
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                
                self._num_loaded += 1
                return tokens.squeeze(0)
                
            except StopIteration:
                # Restart iterator
                self._iterator = iter(self.dataset)
                return self.__getitem__(idx)
            except Exception as e:
                # Skip problematic items
                return self.__getitem__(idx)
        else:
            # Return cached or generate
            return torch.randint(0, 32000, (self.seq_len,), dtype=torch.long)


def get_default_tokenizer():
    """Get a default tokenizer for the dataset."""
    try:
        from transformers import AutoTokenizer
        # Use a small, efficient tokenizer
        return AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B",
            trust_remote_code=True,
            use_fast=False,
        )
    except Exception as e:
        print(f"Warning: Could not load Llama tokenizer: {e}")
        print("Using GPT-2 tokenizer as fallback...")
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained("gpt2")


# ============================================================================
# Training
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for training."""
    data_path: str = "./data/train.bin"
    val_data_path: Optional[str] = None
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
    
    # Loss options
    loss_mode: str = "both"  # "ar", "diffusion", or "both"
    ar_loss_weight: float = 1.0
    diffusion_loss_weight: float = 1.0
    
    # Logging
    wandb_project: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Create a learning rate scheduler with linear warmup."""
    
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    config: TrainingConfig,
    epoch: int,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_ar_loss = 0.0
    total_diffusion_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(config.device)
        
        # Forward pass
        outputs = model(batch, labels=batch, mode=config.loss_mode)
        
        # Compute loss
        if config.loss_mode == "both":
            loss = config.ar_loss_weight * outputs.get("ar_loss", 0) + \
                   config.diffusion_loss_weight * outputs.get("diffusion_loss", 0)
        elif config.loss_mode == "ar":
            loss = outputs.get("ar_loss", 0)
        elif config.loss_mode == "diffusion":
            loss = outputs.get("diffusion_loss", 0)
        else:
            loss = outputs.get("ar_loss", 0)
        
        # Scale loss for gradient accumulation
        loss = loss / config.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights
        if (batch_idx + 1) % config.accumulation_steps == 0:
            if config.use_gradient_checkpointing:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
        
        # Logging
        total_loss += loss.item() * config.accumulation_steps
        if outputs.get("ar_loss") is not None:
            total_ar_loss += outputs["ar_loss"].item()
        if outputs.get("diffusion_loss") is not None:
            total_diffusion_loss += outputs["diffusion_loss"].item()
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{total_loss / num_batches:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })
        
        # Log to wandb
        if config.wandb_project and HAS_WANDB and batch_idx % config.log_interval == 0:
            wandb.log({
                "train/loss": total_loss / num_batches,
                "train/ar_loss": total_ar_loss / max(1, num_batches),
                "train/diffusion_loss": total_diffusion_loss / max(1, num_batches),
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/step": epoch * len(dataloader) + batch_idx,
            })
    
    return {
        "loss": total_loss / num_batches,
        "ar_loss": total_ar_loss / num_batches if total_ar_loss > 0 else None,
        "diffusion_loss": total_diffusion_loss / num_batches if total_diffusion_loss > 0 else None,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, config: TrainingConfig) -> dict:
    """Evaluate the model."""
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(config.device)
            outputs = model(batch, labels=batch, mode=config.loss_mode)
            
            if config.loss_mode == "both":
                loss = config.ar_loss_weight * outputs.get("ar_loss", 0) + \
                       config.diffusion_loss_weight * outputs.get("diffusion_loss", 0)
            elif config.loss_mode == "ar":
                loss = outputs.get("ar_loss", 0)
            else:
                loss = outputs.get("ar_loss", 0)
            
            total_loss += loss.item()
            num_batches += 1
    
    return {"loss": total_loss / num_batches}


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, epoch: int, step: int, config: TrainingConfig):
    """Save model checkpoint."""
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save model in HuggingFace format
    save_path = Path(config.output_dir) / f"checkpoint-epoch{epoch}-step{step}"
    save_path.mkdir(exist_ok=True)
    
    model.save_pretrained(save_path)
    
    # Save optimizer state
    torch.save({
        "epoch": epoch,
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
    }, save_path / "trainer_state.pt")
    
    print(f"Saved checkpoint to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pretrain a dual-mode AR + Diffusion model")
    
    # Data arguments
    parser.add_argument("--data-path", type=str, default="./data/train.bin", help="Path to training data")
    parser.add_argument("--val-data-path", type=str, default=None, help="Path to validation data")
    parser.add_argument("--output-dir", type=str, default="./checkpoints/10m_pretrain", help="Output directory")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save interval")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    parser.add_argument("--eval-interval", type=int, default=1000, help="Evaluation interval")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    
    # Model arguments
    parser.add_argument("--vocab-size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--hidden-size", type=int, default=512, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=10, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="MLP expansion ratio")
    
    # Training options
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--use-flash-attention", action="store_true", help="Enable Flash Attention 2 for long context (requires flash-attn package)")
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Mixed precision")
    parser.add_argument("--accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--loss-mode", type=str, default="both", choices=["ar", "diffusion", "both"], help="Loss mode")
    parser.add_argument("--ar-loss-weight", type=float, default=1.0, help="AR loss weight")
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0, help="Diffusion loss weight")
    
    # Logging
    parser.add_argument("--wandb-project", type=str, default=None, help="WandB project name")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_path=args.data_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        max_seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        mlp_ratio=args.mlp_ratio,
        use_gradient_checkpointing=args.gradient_checkpointing,
        use_flash_attention=args.use_flash_attention,
        mixed_precision=args.mixed_precision,
        accumulation_steps=args.accumulation_steps,
        loss_mode=args.loss_mode,
        ar_loss_weight=args.ar_loss_weight,
        diffusion_loss_weight=args.diffusion_loss_weight,
        wandb_project=args.wandb_project,
    )
    
    # Initialize wandb
    if config.wandb_project and HAS_WANDB:
        wandb.init(project=config.wandb_project, config=vars(config))
    
    # Set device
    device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Create model
    print("\n=== Model Configuration ===")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num layers: {config.num_layers}")
    print(f"Num heads: {config.num_heads}")
    print(f"Head dim: {config.head_dim}")
    print(f"Flash Attention: {config.use_flash_attention}")
    
    model = DualModeModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        mlp_ratio=config.mlp_ratio,
        max_seq_len=config.max_seq_len,
        use_flash_attn=config.use_flash_attention,
    )
    
    num_params = model.get_num_params()
    print(f"\nTotal parameters: {num_params:,} ({num_params / 1e6:.1f}M)")
    
    model = model.to(device)
    
    # Enable gradient checkpointing
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Create datasets
    print("\n=== Loading Dataset ===")
    
    # Default to Ultra-FineWeb if no data path provided
    if config.data_path is None or config.data_path == "./data/train.bin":
        print("Using Ultra-FineWeb dataset (default)")
        # Get tokenizer for dataset
        tokenizer = get_default_tokenizer()
        
        train_dataset = UltraFineWebDataset(
            split="train",
            seq_len=config.max_seq_len,
            tokenizer=tokenizer,
        )
    elif config.data_path.endswith(".txt"):
        # Text file
        tokenizer = get_default_tokenizer()
        train_dataset = TextDataset(config.data_path, tokenizer, config.max_seq_len)
    else:
        # Binary token file
        train_dataset = BinaryDataset(config.data_path, config.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = None
    if config.val_data_path:
        if config.val_data_path == "ultra-fine-web":
            # Use validation split from Ultra-FineWeb
            tokenizer = get_default_tokenizer()
            val_dataset = UltraFineWebDataset(
                split="test",
                seq_len=config.max_seq_len,
                tokenizer=tokenizer,
            )
        elif config.val_data_path.endswith(".txt"):
            tokenizer = get_default_tokenizer()
            val_dataset = TextDataset(config.val_data_path, tokenizer, config.max_seq_len)
        else:
            val_dataset = BinaryDataset(config.val_data_path, config.max_seq_len)
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True,
        )
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    
    # Create scheduler
    num_training_steps = len(train_loader) * config.epochs // config.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, config.warmup_steps, num_training_steps)
    
    # Training loop
    print("\n=== Starting Training ===")
    global_step = 0
    
    for epoch in range(config.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        print(f"Epoch {epoch}: {train_metrics}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, config)
        
        # Evaluate
        if val_loader is not None and (epoch + 1) % config.eval_interval == 0:
            eval_metrics = evaluate(model, val_loader, config)
            print(f"Evaluation: {eval_metrics}")
            
            if config.wandb_project and HAS_WANDB:
                wandb.log({"eval/loss": eval_metrics["loss"]})
        
        global_step += len(train_loader)
    
    # Final save
    save_path = Path(config.output_dir) / "final"
    save_path.mkdir(exist_ok=True)
    model.save_pretrained(save_path)
    print(f"\nFinal model saved to {save_path}")
    
    # Log model size
    print(f"\n=== Final Model Statistics ===")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"AR head parameters: {sum(p.numel() for p in model.ar_head.parameters()):,}")
    print(f"Diffusion head parameters: {sum(p.numel() for p in model.diffusion_head.parameters()):,}")


if __name__ == "__main__":
    main()
