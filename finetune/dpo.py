#!/usr/bin/env python3
"""
Direct Preference Optimization (DPO) trainer.

DPO is a method for fine-tuning language models using preference data without requiring
a separate reward model.

Reference: "Direct Preference Optimization: Your Language Model is a Reward Model"
https://arxiv.org/abs/2305.18290
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .base import BaseFinetuner, TrainerConfig
from .utils import create_quantization_config, load_model_for_finetuning


@dataclass
class DPOConfig(TrainerConfig):
    """Configuration for DPO training."""
    
    # DPO specific
    beta: float = 0.1  # KL penalty coefficient
    reference_free: bool = False  # If True, don't use reference model
    label_smoothing: float = 0.0  # Label smoothing for reference
    
    # Preference data
    preferred_response: str = "preferred"
    rejected_response: str = "rejected"
    
    # Loss
    loss_type: str = "sigmoid"  # "sigmoid", "hinge", "ipo"
    
    # Reference model
    reference_model_path: Optional[str] = None


class PreferenceDataset(Dataset):
    """Dataset for preference (chosen/rejected) data."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        preferred_key: str = "preferred",
        rejected_key: str = "rejected",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.preferred_key = preferred_key
        self.rejected_key = rejected_key
        
        # Load data
        if not data_path or not os.path.exists(data_path):
            raise FileNotFoundError(f"DPO dataset not found: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            if data_path.endswith(".json"):
                self.data = json.load(f)
            else:
                self.data = [json.loads(line) for line in f]
        
        # Ensure data is a list
        if isinstance(self.data, dict) and "data" in self.data:
            self.data = self.data["data"]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_prompt(self, prompt: str) -> str:
        """Format prompt."""
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        prompt = example.get("prompt", "")
        preferred = example.get(self.preferred_key, example.get("chosen", ""))
        rejected = example.get(self.rejected_key, example.get("rejected", ""))
        
        # Format text with both responses (for computing logprobs)
        # We'll compute loss on both and use the difference
        format_prompt = self.format_prompt(prompt)
        
        # Tokenize
        prompt_ids = self.tokenizer(format_prompt, add_special_tokens=False)["input_ids"]
        preferred_ids = self.tokenizer(preferred, add_special_tokens=False)["input_ids"]
        rejected_ids = self.tokenizer(rejected, add_special_tokens=False)["input_ids"]
        
        # Combine with prompt
        # For preferred: prompt + response + eos
        # For rejected: prompt + response + eos
        
        preferred_full = prompt_ids + preferred_ids + [self.tokenizer.eos_token_id]
        rejected_full = prompt_ids + rejected_ids + [self.tokenizer.eos_token_id]
        
        # Truncate
        if len(preferred_full) > self.max_seq_length:
            preferred_full = preferred_full[:self.max_seq_length]
        if len(rejected_full) > self.max_seq_length:
            rejected_full = rejected_full[:self.max_seq_length]
        
        # Pad
        preferred_full = preferred_full + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(preferred_full))
        rejected_full = rejected_full + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(rejected_full))
        
        return {
            "prompt": prompt,
            "prompt_length": len(prompt_ids),
            "preferred": torch.tensor(preferred_full, dtype=torch.long),
            "rejected": torch.tensor(rejected_full, dtype=torch.long),
        }


class DPOTrainer(BaseFinetuner):
    """
    Direct Preference Optimization trainer.
    
    DPO optimizes the following objective:
    -log(sigmoid(beta * (log_pi(y|w,x) - log_ref(y|w,x))))
    
    Where:
    - pi is the policy model (being trained)
    - ref is the reference model
    - beta is the KL penalty coefficient
    - y is the response, w is the prompt
    """
    
    def __init__(self, config: DPOConfig):
        super().__init__(config)
        self.config: DPOConfig = config
        self.reference_model: Optional[PreTrainedModel] = None
    
    def setup_model(self) -> PreTrainedModel:
        """Setup model for DPO."""
        print(f"Loading model from {self.config.model_path}")
        
        model = load_model_for_finetuning(
            self.config.model_path,
            device=self.config.device,
            use_lora=False,
            use_quantization=self.config.use_qlora,
            quantization_config=create_quantization_config(
                bits=self.config.quantization_bits,
                compute_dtype=self.config.quantization_compute_dtype,
            ) if self.config.use_qlora else None,
            torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32,
            trust_remote_code=True,
        )
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    def setup_reference_model(self) -> PreTrainedModel:
        """Setup reference model for DPO."""
        if self.config.reference_model_path:
            print(f"Loading reference model from {self.config.reference_model_path}")
            ref_model = load_model_for_finetuning(
                self.config.reference_model_path,
                device=self.config.device,
                use_lora=False,
                use_quantization=False,
                torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32,
                trust_remote_code=True,
            )
        else:
            # Clone from policy model
            print("Cloning policy model as reference")
            ref_model = load_model_for_finetuning(
                self.config.model_path,
                device=self.config.device,
                use_lora=False,
                use_quantization=False,
                torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32,
                trust_remote_code=True,
            )
        
        # Freeze reference
        for param in ref_model.parameters():
            param.requires_grad = False
        
        return ref_model
    
    def setup_data(self) -> tuple:
        """Setup training and evaluation data."""
        # Create dataset
        train_dataset = PreferenceDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
            preferred_key=self.config.preferred_response,
            rejected_key=self.config.rejected_response,
        )
        
        eval_dataset = None
        if self.config.eval_data_path:
            eval_dataset = PreferenceDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
                preferred_key=self.config.preferred_response,
                rejected_key=self.config.rejected_response,
            )
        else:
            eval_dataset = train_dataset
        
        def collate_fn(batch):
            return {
                "prompt_length": torch.tensor([x["prompt_length"] for x in batch], dtype=torch.long),
                "preferred": torch.stack([x["preferred"] for x in batch]),
                "rejected": torch.stack([x["rejected"] for x in batch]),
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        return train_loader, eval_loader
    
    def get_logprobs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for each token.
        
        Returns:
            logprobs: [batch, seq_len] log probabilities for each token
        """
        # Forward pass
        outputs = model(input_ids, return_dict=True)
        logits = outputs.logits  # [batch, seq_len, vocab]
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get log prob for each token (shifted by 1 for prediction)
        # input_ids[:, 1:] is what we're predicting
        # log_probs[:, :-1] are the predictions for input_ids[:, 1:]
        input_ids_shifted = input_ids[:, 1:]
        log_probs_shifted = log_probs[:, :-1, :]
        
        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs_shifted,
            dim=-1,
            index=input_ids_shifted.unsqueeze(-1),
        ).squeeze(-1)  # [batch, seq_len - 1]

        positions = torch.arange(token_log_probs.shape[1], device=input_ids.device).unsqueeze(0)
        response_mask = positions >= (prompt_lengths.unsqueeze(1) - 1)
        if self.tokenizer.pad_token_id is not None:
            response_mask = response_mask & (input_ids_shifted != self.tokenizer.pad_token_id)
        token_log_probs = token_log_probs * response_mask
        
        return token_log_probs
    
    def compute_dpo_loss(
        self,
        policy_logprobs_preferred: torch.Tensor,
        policy_logprobs_rejected: torch.Tensor,
        ref_logprobs_preferred: torch.Tensor,
        ref_logprobs_rejected: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DPO loss.
        
        Args:
            policy_logprobs_preferred: [batch, seq]
            policy_logprobs_rejected: [batch, seq]
            ref_logprobs_preferred: [batch, seq]
            ref_logprobs_rejected: [batch, seq]
        
        Returns:
            loss: scalar tensor
        """
        # Compute log ratio differences
        # log(pi(y_preferred|x) / ref(y_preferred|x))
        # Sum over sequence
        policy_sum_preferred = policy_logprobs_preferred.sum(dim=-1)
        ref_sum_preferred = ref_logprobs_preferred.sum(dim=-1)
        
        policy_sum_rejected = policy_logprobs_rejected.sum(dim=-1)
        ref_sum_rejected = ref_logprobs_rejected.sum(dim=-1)
        
        # Log ratio
        policy_ratio_preferred = policy_sum_preferred - ref_sum_preferred
        policy_ratio_rejected = policy_sum_rejected - ref_sum_rejected
        
        # Difference
        log_diff = policy_ratio_preferred - policy_ratio_rejected
        
        if self.config.loss_type == "sigmoid":
            # Sigmoid loss (standard DPO)
            loss = -F.logsigmoid(self.config.beta * log_diff)
        elif self.config.loss_type == "hinge":
            # Hinge loss
            loss = F.relu(1 - self.config.beta * log_diff)
        elif self.config.loss_type == "ipo":
            # IPO loss
            beta = self.config.beta
            loss = (log_diff - 1/beta) ** 2
        else:
            # Default to sigmoid
            loss = -F.logsigmoid(self.config.beta * log_diff)
        
        return loss.mean()
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        prompt_lengths = batch["prompt_length"].to(self.model.device)
        preferred = batch["preferred"].to(self.model.device)
        rejected = batch["rejected"].to(self.model.device)
        
        # Get policy log probs
        policy_logprobs_preferred = self.get_logprobs(self.model, preferred, prompt_lengths)
        policy_logprobs_rejected = self.get_logprobs(self.model, rejected, prompt_lengths)
        
        # Get reference log probs
        if self.config.reference_free:
            # Reference-free: use policy as reference (beta effectively becomes 0)
            ref_logprobs_preferred = policy_logprobs_preferred.detach()
            ref_logprobs_rejected = policy_logprobs_rejected.detach()
        else:
            # Use reference model
            with torch.no_grad():
                ref_logprobs_preferred = self.get_logprobs(self.reference_model, preferred, prompt_lengths)
                ref_logprobs_rejected = self.get_logprobs(self.reference_model, rejected, prompt_lengths)
        
        # Compute loss
        loss = self.compute_dpo_loss(
            policy_logprobs_preferred,
            policy_logprobs_rejected,
            ref_logprobs_preferred,
            ref_logprobs_rejected,
        )
        
        # Backward
        self.accelerator.backward(loss)
        
        # Clip gradients and step
        if self.accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single evaluation step."""
        # Similar to training but without gradient
        prompt_lengths = batch["prompt_length"].to(self.model.device)
        preferred = batch["preferred"].to(self.model.device)
        rejected = batch["rejected"].to(self.model.device)
        
        with torch.no_grad():
            policy_logprobs_preferred = self.get_logprobs(self.model, preferred, prompt_lengths)
            policy_logprobs_rejected = self.get_logprobs(self.model, rejected, prompt_lengths)
            
            if self.config.reference_free:
                ref_logprobs_preferred = policy_logprobs_preferred
                ref_logprobs_rejected = policy_logprobs_rejected
            else:
                ref_logprobs_preferred = self.get_logprobs(self.reference_model, preferred, prompt_lengths)
                ref_logprobs_rejected = self.get_logprobs(self.reference_model, rejected, prompt_lengths)
            
            loss = self.compute_dpo_loss(
                policy_logprobs_preferred,
                policy_logprobs_rejected,
                ref_logprobs_preferred,
                ref_logprobs_rejected,
            )
        
        return {"loss": loss.item()}
    
    def setup_training(self):
        """Setup training with reference model."""
        # Call parent's setup
        super().setup_training()
        
        # Setup reference model
        if not self.config.reference_free:
            self.reference_model = self.setup_reference_model()
            self.reference_model = self.reference_model.to(self.model.device)


def main():
    parser = argparse.ArgumentParser(description="DPO Fine-tuning")
    parser = BaseFinetuner.add_trainer_args(parser)
    
    # DPO specific args
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient")
    parser.add_argument("--reference-free", action="store_true", help="Use reference-free DPO")
    parser.add_argument("--loss-type", type=str, default="sigmoid", choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--reference-model-path", type=str, default=None, help="Reference model path")
    
    args = parser.parse_args()
    
    # Create config
    config = DPOConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_seq_length=args.max_seq_length,
        max_grad_norm=args.max_grad_norm,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        use_qlora=args.use_qlora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        wandb_project=args.wandb_project,
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        beta=args.beta,
        reference_free=args.reference_free,
        loss_type=args.loss_type,
        reference_model_path=args.reference_model_path,
    )
    
    # Create trainer
    trainer = DPOTrainer(config)
    
    # Train
    trainer.train()
    
    print(f"\nDPO training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
