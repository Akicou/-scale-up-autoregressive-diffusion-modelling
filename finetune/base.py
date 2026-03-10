#!/usr/bin/env python3
"""
Base finetuner class with common training functionality.
"""

import argparse
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)
import accelerate
from pretrain import get_default_tokenizer

# Try to import wandb
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


@dataclass
class TrainerConfig:
    """Configuration for finetuning."""
    
    # Model
    model_path: str = "./checkpoints/10m_pretrain"
    tokenizer_path: Optional[str] = None
    
    # Training
    output_dir: str = "./checkpoints/finetuned"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    max_grad_norm: float = 1.0
    
    # LoRA/QLoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # QLoRA
    use_qlora: bool = False
    quantization_bits: int = 4
    quantization_compute_dtype: str = "bfloat16"
    
    # Optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    wandb_project: Optional[str] = None
    
    # Device
    device: str = "auto"
    local_rank: int = 0
    
    # Data
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    tool_calling: bool = False
    template_format: str = "chat"


class BaseFinetuner(ABC):
    """
    Base finetuner class with common training functionality.
    
    Supports:
    - Single GPU, multi-GPU, and DeepSpeed training
    - LoRA and QLoRA finetuning
    - Full-parameter finetuning
    - Mixed precision training
    - Gradient checkpointing
    """
    
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
        self.train_loader: Optional[DataLoader] = None
        self.eval_loader: Optional[DataLoader] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        
        self.global_step = 0
        self.current_epoch = 0
        
        # Initialize accelerator
        self.accelerator = None
    
    @abstractmethod
    def setup_model(self) -> PreTrainedModel:
        """Setup and return the model to finetune."""
        pass
    
    @abstractmethod
    def setup_data(self) -> tuple:
        """Setup training and evaluation data."""
        pass
    
    @abstractmethod
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        pass
    
    @abstractmethod
    def evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single evaluation step."""
        pass
    
    def setup_tokenizer(self) -> PreTrainedTokenizer:
        """Setup tokenizer."""
        tokenizer_path = self.config.tokenizer_path or self.config.model_path
        
        if os.path.exists(tokenizer_path):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception:
                tokenizer = get_default_tokenizer()
        else:
            tokenizer = get_default_tokenizer()
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def setup_lora(self, model: PreTrainedModel) -> PeftModel:
        """Setup LoRA configuration."""
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
            inference_mode=False,
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        return model
    
    def setup_training(self):
        """Setup training infrastructure."""
        if self.config.use_qlora and not self.config.use_lora:
            raise ValueError("QLoRA requires LoRA adapters. Pass --use-lora together with --use-qlora.")

        # Setup tokenizer
        self.setup_tokenizer()
        
        # Setup model
        self.model = self.setup_model()
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self.model = self.setup_lora(self.model)
        
        # Setup data
        self.train_loader, self.eval_loader = self.setup_data()
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Setup accelerator
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
        )
        
        # Prepare with accelerator
        if self.eval_loader is not None:
            prepared = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.eval_loader
            )
            self.model, self.optimizer, self.train_loader, self.eval_loader = prepared
        else:
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader
            )
        
        # Initialize wandb
        if self.config.wandb_project and HAS_WANDB:
            wandb.init(project=self.config.wandb_project, config=vars(self.config))
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)
        model_to_save = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        
        model_to_save.save_pretrained(output_dir)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        
        # Save training state
        state = {
            "step": step,
            "epoch": self.current_epoch,
            "metrics": metrics,
        }
        with open(output_dir / "trainer_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        print(f"Saved checkpoint to {output_dir}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to console and wandb."""
        if self.accelerator.is_main_process:
            msg = f"{prefix}/" + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(msg)
            
            if self.config.wandb_project and HAS_WANDB:
                wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})
    
    def train(self):
        """Main training loop."""
        if not self.model:
            self.setup_training()
        
        self.model.train()
        
        num_epochs = self.config.epochs
        num_training_steps = len(self.train_loader) * num_epochs
        
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = {}
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                step_metrics = self.training_step(batch)
                
                # Aggregate metrics across batches
                for k, v in step_metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v
                
                # Log
                if self.global_step % self.config.log_interval == 0:
                    avg_metrics = {k: v / (batch_idx + 1) for k, v in epoch_metrics.items()}
                    self.log_metrics(avg_metrics, "train")
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(self.global_step, epoch_metrics)
                
                # Evaluate
                if self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.log_metrics(eval_metrics, "eval")
                    self.model.train()
                
                self.global_step += 1
                progress_bar.update(1)
            
            # End of epoch
            avg_metrics = {k: v / len(self.train_loader) for k, v in epoch_metrics.items()}
            self.log_metrics(avg_metrics, f"epoch_{epoch}")
        
        # Save final model
        self.save_checkpoint("final", {})
        
        progress_bar.close()
        
        if self.config.wandb_project and HAS_WANDB:
            wandb.finish()
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        if not self.eval_loader:
            return {}
        
        self.model.eval()
        eval_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating"):
                batch_metrics = self.evaluation_step(batch)
                for k, v in batch_metrics.items():
                    if k not in eval_metrics:
                        eval_metrics[k] = 0
                    eval_metrics[k] += v
        
        # Average metrics
        eval_metrics = {k: v / len(self.eval_loader) for k, v in eval_metrics.items()}
        
        return eval_metrics
    
    @staticmethod
    def add_trainer_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add common training arguments to parser."""
        parser.add_argument("--model-path", type=str, default="./checkpoints/10m_pretrain")
        parser.add_argument("--tokenizer-path", type=str, default=None)
        parser.add_argument("--output-dir", type=str, default="./checkpoints/finetuned")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--learning-rate", type=float, default=2e-4)
        parser.add_argument("--weight-decay", type=float, default=0.01)
        parser.add_argument("--warmup-steps", type=int, default=100)
        parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
        parser.add_argument("--max-seq-length", type=int, default=512)
        parser.add_argument("--max-grad-norm", type=float, default=1.0)
        
        # LoRA
        parser.add_argument("--use-lora", action="store_true")
        parser.add_argument("--lora-r", type=int, default=16)
        parser.add_argument("--lora-alpha", type=int, default=32)
        parser.add_argument("--lora-dropout", type=float, default=0.05)
        parser.add_argument("--lora-target-modules", nargs="+", 
                           default=["q_proj", "v_proj", "k_proj", "o_proj"])
        
        # QLoRA
        parser.add_argument("--use-qlora", action="store_true")
        
        # Optimization
        parser.add_argument("--use-gradient-checkpointing", action="store_true")
        parser.add_argument("--mixed-precision", type=str, default="bf16", 
                           choices=["no", "fp16", "bf16"])
        
        # Logging
        parser.add_argument("--log-interval", type=int, default=10)
        parser.add_argument("--eval-interval", type=int, default=500)
        parser.add_argument("--save-interval", type=int, default=1000)
        parser.add_argument("--wandb-project", type=str, default=None)
        
        # Data
        parser.add_argument("--train-data-path", type=str, default=None)
        parser.add_argument("--eval-data-path", type=str, default=None)
        
        return parser


class DummyDataset(Dataset):
    """Dummy dataset for testing when no data is available."""
    
    def __init__(self, size: int = 100, seq_len: int = 128):
        self.size = size
        self.seq_len = seq_len
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(0, 32000, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, 32000, (self.seq_len,)),
        }
