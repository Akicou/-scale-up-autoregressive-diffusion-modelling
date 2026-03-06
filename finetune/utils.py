#!/usr/bin/env python3
"""
Shared utilities for finetuning.
"""

import os
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollator,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


def setup_tokenizer(
    tokenizer_path: str,
    add_special_tokens: Optional[Dict[str, Any]] = None,
) -> PreTrainedTokenizer:
    """
    Setup tokenizer from path.
    
    Args:
        tokenizer_path: Path to tokenizer
        add_special_tokens: Optional dict of special tokens
    
    Returns:
        Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add special tokens if provided
    if add_special_tokens:
        tokenizer.add_special_tokens(add_special_tokens)
    
    return tokenizer


def load_model_for_finetuning(
    model_path: str,
    device: str = "auto",
    use_lora: bool = False,
    lora_config: Optional[Dict[str, Any]] = None,
    use_quantization: bool = False,
    quantization_config: Optional[Dict[str, Any]] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    """
    Load model for finetuning with optional LoRA and quantization.
    
    Args:
        model_path: Path to model
        device: Device mapping
        use_lora: Whether to use LoRA
        lora_config: LoRA configuration
        use_quantization: Whether to use quantization
        quantization_config: Quantization configuration
        torch_dtype: Data type
        trust_remote_code: Trust remote code
    
    Returns:
        Model (possibly with LoRA applied)
    """
    # Load base model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device,
        "trust_remote_code": trust_remote_code,
    }
    
    if use_quantization and quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    # Apply LoRA
    if use_lora:
        if lora_config is None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                inference_mode=False,
            )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model


class PackingDataCollator(DataCollator):
    """
    Data collator that packs sequences together for efficiency.
    
    This is useful for instruction tuning with variable length sequences.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 512,
        pad_to_multiple_of: Optional[int] = None,
    ):
        super().__init__(tokenizer, pad_to_multiple_of=pad_to_multiple_of)
        self.max_length = max_length
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Pack sequences into fixed-length chunks."""
        # For now, just use default behavior
        # Packing is complex and typically handled at the dataset level
        return super().__call__(features)


def prepare_data_collator(
    tokenizer: PreTrainedTokenizer,
    packing: bool = False,
    max_length: int = 512,
    mlm: bool = False,
) -> DataCollator:
    """
    Prepare data collator for training.
    
    Args:
        tokenizer: Tokenizer
        packing: Whether to use packing
        max_length: Maximum sequence length
        mlm: Whether to use masked language modeling
    
    Returns:
        Data collator
    """
    if packing:
        return PackingDataCollator(tokenizer, max_length=max_length)
    else:
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=mlm,
            pad_to_multiple_of=8 if tokenizer._pad_token is not None else None,
        )


def create_chat_template(
    messages: List[Dict[str, str]],
    system_message: Optional[str] = None,
    template_format: str = "chatml",
) -> str:
    """
    Create a chat template string from messages.
    
    Args:
        messages: List of message dicts with "role" and "content"
        system_message: Optional system message
        template_format: Template format ("chatml", "alpaca", "vicuna")
    
    Returns:
        Formatted string
    """
    if template_format == "chatml":
        result = ""
        if system_message:
            result += f"<|im_start|>system\n{system_message}<|im_end|>\n"
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        result += "<|im_start|>assistant\n"
    
    elif template_format == "alpaca":
        result = "Below is an instruction that describes a task, paired with an input that provides further context. "
        result += "Write a response that appropriately completes the request.\n\n"
        
        instruction = ""
        input_text = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                if not instruction:
                    instruction = content
                else:
                    input_text = content
            elif role == "assistant":
                result += f"### Instruction:\n{instruction}\n\n"
                if input_text:
                    result += f"### Input:\n{input_text}\n\n"
                result += f"### Response:\n{content}"
    
    elif template_format == "vicuna":
        result = ""
        if system_message:
            result += f"SYSTEM: {system_message}\n"
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                result += f"USER: {content}\n"
            elif role == "assistant":
                result += f"ASSISTANT: {content}\n"
        
        result += "ASSISTANT: "
    
    else:
        # Simple format
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            result += f"{role}: {content}\n"
    
    return result


def parse_model_output_for_tools(
    output: str,
    format: str = "json",
) -> Optional[Dict[str, Any]]:
    """
    Parse model output to extract tool calls.
    
    Args:
        output: Model output string
        format: Expected format ("json", "xml", "python")
    
    Returns:
        Parsed tool call dict or None
    """
    import re
    import json
    
    if format == "json":
        # Look for JSON in output
        # Handle both {"tool": ...} and {"name": ...}
        patterns = [
            r'\{[^{}]*"tool"[^{}]*\}',
            r'\{[^{}]*"name"[^{}]*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, output)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    # Normalize fields
                    if "name" in parsed and "tool" not in parsed:
                        parsed["tool"] = parsed.pop("name")
                    if "arguments" in parsed and "args" not in parsed:
                        parsed["args"] = parsed.pop("arguments")
                    return parsed
                except json.JSONDecodeError:
                    continue
        
        # Try to find full JSON block
        json_match = re.search(r'\{.*?"tool".*?\}', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    
    elif format == "xml":
        # Look for <tool_call>...</tool_call>
        match = re.search(r'<tool_call\s+name="([^"]+)"[^>]*>(.*?)</tool_call>', output, re.DOTALL)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Parse arguments
            args = {}
            arg_pattern = r'<argument\s+name="([^"]+)">([^<]+)</argument>'
            for arg_match in re.finditer(arg_pattern, args_str):
                args[arg_match.group(1)] = arg_match.group(2)
            
            return {"tool": tool_name, "args": args}
    
    elif format == "python":
        # Look for tool::name(args)
        match = re.search(r'tool::([^(]+)\((.*?)\)', output)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Simple parsing for key="value" format
            args = {}
            for arg in args_str.split(","):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    args[key.strip()] = value.strip().strip('"').strip("'")
            
            return {"tool": tool_name, "args": args}
    
    return None


def format_tool_result(tool_name: str, result: Any) -> str:
    """
    Format tool result for insertion into conversation.
    
    Args:
        tool_name: Name of the tool
        result: Tool result
    
    Returns:
        Formatted result string
    """
    if isinstance(result, (dict, list)):
        result_str = json.dumps(result)
    else:
        result_str = str(result)
    
    return f"<|im_start|>tool\n<tool={tool_name}>\n{result_str}\n</tool><|im_end|>"
