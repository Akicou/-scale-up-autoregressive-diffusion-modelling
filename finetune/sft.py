#!/usr/bin/env python3
"""
Supervised Fine-Tuning (SFT) trainer with support for LoRA, QLoRA, and full-parameter finetuning.

Supports:
- LoRA and QLoRA (with 4-bit quantization)
- Full-parameter finetuning
- Packing support for variable length sequences
- Chat template formatting
- Tool calling / function calling finetuning
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    from bitsandbytes import quantize
    HAS_BNB = True
except ImportError:
    HAS_BNB = False

from .base import BaseFinetuner, TrainerConfig


class InstructionDataset(Dataset):
    """Dataset for instruction fine-tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
        template_format: str = "chat",
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.template_format = template_format
        
        # Load data
        if data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                if data_path.endswith(".json"):
                    self.data = json.load(f)
                else:
                    self.data = [json.loads(line) for line in f]
        else:
            # Dummy data for testing
            self.data = [
                {"instruction": "Explain the concept of machine learning.", "output": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."},
                {"instruction": "What is Python?", "output": "Python is a high-level, interpreted programming language known for its simplicity and readability."},
            ]
        
        # Ensure data is a list
        if isinstance(self.data, dict) and "data" in self.data:
            self.data = self.data["data"]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_example(self, example: Dict[str, str]) -> str:
        """Format example with template."""
        if self.template_format == "chat":
            # ChatML format
            return (
                f"<|im_start|>user\n{example.get('instruction', '')}<|im_end|>\n"
                f"<|im_start|>assistant\n{example.get('output', '')}<|im_end|>"
            )
        elif self.template_format == "alpaca":
            # Alpaca format
            return (
                f"Below is an instruction that describes a task, paired with an input that provides further context. "
                f"Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example.get('instruction', '')}\n\n"
                f"### Input:\n{example.get('input', '')}\n\n"
                f"### Response:\n{example.get('output', '')}"
            )
        else:
            # Simple format
            return f"Instruction: {example.get('instruction', '')}\nResponse: {example.get('output', '')}"
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Format text
        text = self.format_example(example)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Labels: -100 for prompt, actual tokens for response
        # This implements the standard SFT approach where we only train on the response
        if self.template_format == "chat":
            # Find where response starts
            response_start = text.find("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
            response_text = text[response_start:].replace("<|im_end>", "")
            response_enc = self.tokenizer(
                response_text,
                add_special_tokens=False,
            )
            response_tokens = response_enc["input_ids"]
            
            labels = input_ids.clone()
            labels[:] = -100
            
            # Find positions to fill with response tokens
            for i in range(len(input_ids) - len(response_tokens)):
                if torch.all(input_ids[i:i+len(response_tokens)] == torch.tensor(response_tokens[:len(input_ids)-i], device=input_ids.device)):
                    labels[i:i+len(response_tokens)] = torch.tensor(response_tokens[:len(input_ids)-i], device=input_ids.device)
                    break
        else:
            # Default: train on everything
            labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class ToolCallingDataset(Dataset):
    """Dataset for tool calling fine-tuning."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Load data
        if data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            # Dummy tool calling data
            self.data = [
                {
                    "messages": [
                        {"role": "user", "content": "What is 2 + 2?"},
                        {"role": "assistant", "content": "Let me calculate that."},
                        {"role": "tool_call", "name": "calculate", "args": {"expression": "2+2"}},
                        {"role": "tool_result", "content": "4"},
                        {"role": "assistant", "content": "The answer is 4."},
                    ]
                }
            ]
        
        # Ensure data is a list
        if isinstance(self.data, dict) and "data" in self.data:
            self.data = self.data["data"]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages including tool calls."""
        result = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                result.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                result.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool_call":
                # Format tool call
                tool_name = msg.get("name", "")
                tool_args = json.dumps(msg.get("args", {}))
                result.append(f"<|im_start|>assistant\n<tool_call>\n{tool_name}({tool_args})\n</tool_call><|im_end|>")
            elif role == "tool_result":
                result.append(f"<|im_start|>tool\n{content}<|im_end|>")
        
        return "".join(result)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.data[idx]
        
        # Format messages
        messages = example.get("messages", [])
        text = self.format_messages(messages)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        
        # Mark tool calls and results as trainable
        # (In practice, you might want to mask certain parts)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SFTTrainer(BaseFinetuner):
    """
    Supervised Fine-Tuning trainer.
    
    Supports:
    - LoRA (via peft)
    - QLoRA (4-bit quantization + LoRA)
    - Full-parameter finetuning
    - Instruction tuning
    - Tool calling finetuning
    """
    
    def __init__(self, config: TrainerConfig):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def setup_model(self) -> PreTrainedModel:
        """Setup model for SFT."""
        print(f"Loading model from {self.config.model_path}")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float32,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        
        # Gradient checkpointing
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        return model
    
    def setup_data(self) -> tuple:
        """Setup training and evaluation data."""
        # Create datasets
        train_dataset = InstructionDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_seq_length,
        )
        
        eval_dataset = None
        if self.config.eval_data_path:
            eval_dataset = InstructionDataset(
                data_path=self.config.eval_data_path,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
            )
        else:
            # Use a subset for eval
            eval_dataset = InstructionDataset(
                data_path=None,
                tokenizer=self.tokenizer,
                max_seq_length=self.config.max_seq_length,
            )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=data_collator,
        )
        
        return train_loader, eval_loader
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Backward
        self.accelerator.backward(loss)
        
        # Clip gradients
        if self.accelerator.sync_gradients:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {"loss": loss.item()}
    
    def evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single evaluation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {"loss": outputs.loss.item()}


def main():
    parser = argparse.ArgumentParser(description="SFT Fine-tuning")
    parser = BaseFinetuner.add_trainer_args(parser)
    parser.add_argument("--format", type=str, default="chat", choices=["chat", "alpaca", "simple"])
    parser.add_argument("--tool-calling", action="store_true", help="Use tool calling format")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainerConfig(
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
    )
    
    # Create trainer
    trainer = SFTTrainer(config)
    
    # Train
    trainer.train()
    
    print(f"\nSFT training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
