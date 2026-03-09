#!/usr/bin/env python3
"""
Base finetuner class with common training functionality.
"""

import argparse
import json
import math
import os
import shutil
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
    prepare_model_for_kbit_training,
)
import accelerate

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

    # Save mode
    save_mode: str = "adapter"  # "adapter" or "merged"
    
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

    # Hugging Face dataset options
    hf_dataset_name: Optional[str] = None
    hf_train_split: str = "train"
    hf_eval_split: Optional[str] = None
    hf_conversation_column: str = "conversation_a"
    hf_max_samples: Optional[int] = None


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
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            checkpoints_dir = Path("./checkpoints")
            available = []
            if checkpoints_dir.is_dir():
                for child in sorted(checkpoints_dir.iterdir()):
                    if child.is_dir() and (
                        (child / "model.pt").exists()
                        or (child / "config.json").exists()
                        or (child / "tokenizer.json").exists()
                    ):
                        available.append(str(child))
            hint = ""
            if available:
                hint = f" Available checkpoint directories: {', '.join(available)}."
            raise FileNotFoundError(
                f"Tokenizer not found at {tokenizer_path}."
                " Pass --model-path/--tokenizer-path explicitly or use a YAML config under config/."
                f"{hint}"
            )
        
        # Ensure padding token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.tokenizer = tokenizer
        return tokenizer
    
    def setup_lora(self, model: PreTrainedModel) -> PeftModel:
        """Setup LoRA/QLoRA configuration."""
        if self.config.use_qlora:
            if self.config.quantization_bits != 4:
                raise ValueError("QLoRA currently supports 4-bit quantization only")
            if not getattr(model, "is_loaded_in_4bit", False):
                raise ValueError("--use-qlora requires a 4-bit loaded model (HF quantized load path)")
            model = prepare_model_for_kbit_training(model)

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
    
    def _check_large_model_safety(self, model_path: str) -> None:
        """Check if model is large and warn about VRAM requirements."""
        try:
            config_file = Path(model_path) / "config.pt"
            hf_config_file = Path(model_path) / "config.json"
            
            param_count = None
            if config_file.exists():
                custom_cfg = torch.load(config_file, map_location="cpu")
                model_cfg = custom_cfg.get("target_config", custom_cfg)
                hidden_size = model_cfg.get("hidden_size", 256)
                num_layers = model_cfg.get("num_layers", 6)
                num_heads = model_cfg.get("num_heads", 4)
                vocab_size = model_cfg.get("vocab_size", 16001)
                
                mlp_dim = int(hidden_size * model_cfg.get("mlp_ratio", 4.0))
                attn_dim = num_heads * model_cfg.get("head_dim", hidden_size // max(1, num_heads))
                
                embed_params = vocab_size * hidden_size
                layer_params = num_layers * (4 * hidden_size * attn_dim + 3 * hidden_size * mlp_dim)
                head_params = 2 * hidden_size * vocab_size
                param_count = embed_params + layer_params + head_params
                
            elif hf_config_file.exists():
                import json
                with open(hf_config_file, "r") as f:
                    cfg = json.load(f)
                hidden_size = cfg.get("hidden_size", 256)
                num_layers = cfg.get("num_hidden_layers", 6)
                vocab_size = cfg.get("vocab_size", 32000)
                intermediate_size = cfg.get("intermediate_size", hidden_size * 4)
                
                embed_params = vocab_size * hidden_size
                layer_params = num_layers * (12 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size)
                head_params = hidden_size * vocab_size
                param_count = embed_params + layer_params + head_params
            
            if param_count is not None and param_count > 1e9:
                param_gb = param_count * 2 / (1024 ** 3)
                qlora_gb = param_count * 0.5 / (1024 ** 3)
                print("=" * 60)
                print(f"LARGE MODEL DETECTED: ~{param_count / 1e9:.2f}B parameters")
                print(f"  Full BF16 baseline: ~{param_gb:.1f} GB VRAM (weights only)")
                print(f"  QLoRA 4-bit: ~{qlora_gb:.1f} GB VRAM (recommended)")
                if not getattr(self.config, "use_qlora", False):
                    print("WARNING: QLoRA not enabled! Consider --use-qlora or use_lora=true + use_qlora=true in YAML")
                    print("         to significantly reduce VRAM usage on multi-billion parameter models.")
                print("=" * 60)
        except Exception as e:
            pass
    
    def setup_training(self):
        """Setup training infrastructure."""
        if Path(self.config.model_path).exists() and getattr(self.config, "use_qlora", False):
            self._check_large_model_safety(self.config.model_path)
        
        self.setup_tokenizer()
        
        # Setup model
        self.model = self.setup_model()
        
        # Apply LoRA if configured (custom DualMode wrapper handles its own LoRA path)
        if self.config.use_lora and not getattr(self.model, "_is_custom_dualmode", False) and not getattr(self.model, "_peft_adapter_configured", False):
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
            self.model, self.optimizer, self.train_loader, self.eval_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.eval_loader
            )
        else:
            self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader
            )
        
        # Initialize wandb
        if self.config.wandb_project and HAS_WANDB:
            wandb.init(project=self.config.wandb_project, config=vars(self.config))
            self.model.accelerator = self.accelerator
    
    def save_checkpoint(self, step: int, metrics: Dict[str, float]):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model_to_save = self.accelerator.unwrap_model(self.model) if self.accelerator is not None else self.model

        if self.config.use_lora and self.config.save_mode == "merged":
            if not isinstance(model_to_save, PeftModel):
                raise ValueError("save_mode=merged requires a LoRA/QLoRA PEFT model")

            # Keep intermediate checkpoints as adapters so training can continue safely.
            if step != "final":
                if self.accelerator is None or self.accelerator.is_main_process:
                    model_to_save.save_pretrained(output_dir)
            else:
                if self.accelerator is not None:
                    self.accelerator.wait_for_everyone()
                if self.accelerator is None or self.accelerator.is_main_process:
                    merged = model_to_save.merge_and_unload()
                    merged.save_pretrained(output_dir)
                    # Preserve tokenizer/config side files from base model path when available
                    src_model_dir = Path(self.config.model_path)
                    for name in ["config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.json", "vocab.json", "merges.txt"]:
                        src = src_model_dir / name
                        dst = output_dir / name
                        if src.exists() and not dst.exists():
                            shutil.copy2(src, dst)
        else:
            if self.accelerator is None or self.accelerator.is_main_process:
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
        num_training_steps = math.ceil(len(self.train_loader) / max(1, self.config.gradient_accumulation_steps)) * num_epochs
        
        progress_bar = tqdm(total=num_training_steps, desc="Training")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_metrics = {}
            
            for batch_idx, batch in enumerate(self.train_loader):
                with self.accelerator.accumulate(self.model):
                    step_metrics = self.training_step(batch)
                    stepped = self.accelerator.sync_gradients
                
                # Aggregate metrics across batches
                for k, v in step_metrics.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = 0
                    epoch_metrics[k] += v
                
                if not stepped:
                    continue

                self.global_step += 1
                progress_bar.update(1)

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
        parser.add_argument("--quantization-bits", type=int, default=4, choices=[4], help="QLoRA quantization bits")
        parser.add_argument("--quantization-compute-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="QLoRA compute dtype")

        # Save mode
        parser.add_argument("--save-mode", type=str, default="adapter", choices=["adapter", "merged"], help="Save adapter only or merge adapter into base model")
        
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
        parser.add_argument("--train-data-path", "--data-path", dest="train_data_path", type=str, default=None)
        parser.add_argument("--eval-data-path", type=str, default=None)

        # Hugging Face dataset options
        parser.add_argument("--hf-dataset-name", type=str, default=None, help="HF dataset name, e.g. lmsys/chatbot_arena_conversations")
        parser.add_argument("--hf-train-split", type=str, default="train", help="HF train split name")
        parser.add_argument("--hf-eval-split", type=str, default=None, help="HF eval split name")
        parser.add_argument("--hf-conversation-column", type=str, default="conversation_a", help="Column containing ShareGPT-style conversation list")
        parser.add_argument("--hf-max-samples", type=int, default=None, help="Optional max samples to load from HF splits")
        
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
