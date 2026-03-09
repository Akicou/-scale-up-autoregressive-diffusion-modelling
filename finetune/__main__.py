#!/usr/bin/env python3
"""Unified entrypoint for finetuning with YAML config support."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from finetune.config import resolve_finetune_type, override_config_from_args, trainer_config_from_yaml
from finetune.base import BaseFinetuner, TrainerConfig
from finetune.dpo import DPOConfig


DEFAULT_MODEL_PATH = "./checkpoints/10m_pretrain"


def _is_finetune_checkpoint(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "model.pt").exists() and (path / "config.pt").exists():
        return True
    if (path / "config.json").exists():
        return True
    if list(path.glob("*.safetensors")):
        return True
    if list(path.glob("pytorch_model*.bin")):
        return True
    return False


def _checkpoint_priority(path: Path) -> tuple[int, float, str]:
    name = path.name.lower()
    if "scaled" in name:
        rank = 0
    elif "sft" in name:
        rank = 1
    elif "pretrain" in name:
        rank = 2
    else:
        rank = 3
    try:
        mtime = -path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (rank, mtime, name)


def _discover_checkpoint() -> str | None:
    env_model_path = os.getenv("FINETUNE_MODEL_PATH")
    if env_model_path and _is_finetune_checkpoint(Path(env_model_path)):
        return env_model_path

    candidates = [
        Path("./checkpoints/4b_scaled"),
        Path("./checkpoints/1b_scaled"),
        Path(DEFAULT_MODEL_PATH),
    ]
    for candidate in candidates:
        if _is_finetune_checkpoint(candidate):
            return str(candidate)

    checkpoints_dir = Path("./checkpoints")
    if not checkpoints_dir.is_dir():
        return None

    discovered = sorted(
        (path for path in checkpoints_dir.iterdir() if _is_finetune_checkpoint(path)),
        key=_checkpoint_priority,
    )
    if discovered:
        return str(discovered[0])
    return None


def _resolve_model_path(model_path: str | None) -> str:
    if model_path:
        return model_path

    discovered = _discover_checkpoint()
    if discovered:
        if discovered != DEFAULT_MODEL_PATH:
            print(f"No --model-path provided; using detected checkpoint at {discovered}")
        return discovered
    return DEFAULT_MODEL_PATH


def _resolve_config_model_path(model_path: str) -> str:
    if Path(model_path).exists():
        return model_path
    if model_path != DEFAULT_MODEL_PATH:
        return model_path

    discovered = _discover_checkpoint()
    if discovered and discovered != model_path:
        print(f"Configured model path {model_path} not found; using detected checkpoint at {discovered}")
        return discovered
    return model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finetune (SFT/DPO) with YAML or CLI config")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML config file")
    
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser = BaseFinetuner.add_trainer_args(base_parser)
    
    known, extra = base_parser.parse_known_args()
    
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--use-lora", action="store_true", default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--lora-target-modules", nargs="+", default=None)
    parser.add_argument("--use-qlora", action="store_true", default=None)
    parser.add_argument("--save-mode", type=str, default=None, choices=["adapter", "merged"])
    parser.add_argument("--use-gradient-checkpointing", action="store_true", default=None)
    parser.add_argument("--mixed-precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--train-data-path", "--data-path", dest="train_data_path", type=str, default=None)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--hf-dataset-name", type=str, default=None)
    parser.add_argument("--hf-train-split", type=str, default=None)
    parser.add_argument("--hf-eval-split", type=str, default=None)
    parser.add_argument("--hf-conversation-column", type=str, default=None)
    parser.add_argument("--hf-max-samples", type=int, default=None)
    
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--reference-free", action="store_true", default=None)
    parser.add_argument("--loss-type", type=str, default=None, choices=["sigmoid", "hinge", "ipo"])
    parser.add_argument("--reference-model-path", type=str, default=None)
    
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = parse_args()
    
    if args.config:
        finetune_type = resolve_finetune_type(args.config)
        if finetune_type == "dpo":
            config = trainer_config_from_yaml(args.config, DPOConfig)
        else:
            config = trainer_config_from_yaml(args.config, TrainerConfig)
        config = override_config_from_args(config, args, skip={"config_path"})
        config.model_path = _resolve_config_model_path(config.model_path)
    else:
        model_path = _resolve_model_path(args.model_path)
        if hasattr(args, "reference_free") and (args.reference_free or args.beta is not None or args.loss_type or args.reference_model_path):
            config = DPOConfig(
                model_path=model_path,
                output_dir=args.output_dir or "./checkpoints/finetuned",
                epochs=args.epochs or 3,
                batch_size=args.batch_size or 8,
                learning_rate=args.learning_rate or 2e-4,
                weight_decay=args.weight_decay or 0.01,
                warmup_steps=args.warmup_steps or 100,
                gradient_accumulation_steps=args.gradient_accumulation_steps or 1,
                max_seq_length=args.max_seq_length or 512,
                max_grad_norm=args.max_grad_norm or 1.0,
                use_lora=args.use_lora or False,
                lora_r=args.lora_r or 16,
                lora_alpha=args.lora_alpha or 32,
                lora_dropout=args.lora_dropout or 0.05,
                lora_target_modules=args.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
                use_qlora=args.use_qlora or False,
                save_mode=args.save_mode or "adapter",
                use_gradient_checkpointing=args.use_gradient_checkpointing or False,
                mixed_precision=args.mixed_precision or "bf16",
                log_interval=args.log_interval or 10,
                eval_interval=args.eval_interval or 500,
                save_interval=args.save_interval or 1000,
                train_data_path=args.train_data_path,
                eval_data_path=args.eval_data_path,
                hf_dataset_name=args.hf_dataset_name,
                hf_train_split=args.hf_train_split or "train",
                hf_eval_split=args.hf_eval_split,
                hf_conversation_column=args.hf_conversation_column or "conversation_a",
                hf_max_samples=args.hf_max_samples,
                beta=args.beta or 0.1,
                reference_free=args.reference_free or False,
                loss_type=args.loss_type or "sigmoid",
                reference_model_path=args.reference_model_path,
            )
        else:
            config = TrainerConfig(
                model_path=model_path,
                output_dir=args.output_dir or "./checkpoints/finetuned",
                epochs=args.epochs or 3,
                batch_size=args.batch_size or 8,
                learning_rate=args.learning_rate or 2e-4,
                weight_decay=args.weight_decay or 0.01,
                warmup_steps=args.warmup_steps or 100,
                gradient_accumulation_steps=args.gradient_accumulation_steps or 1,
                max_seq_length=args.max_seq_length or 512,
                max_grad_norm=args.max_grad_norm or 1.0,
                use_lora=args.use_lora or False,
                lora_r=args.lora_r or 16,
                lora_alpha=args.lora_alpha or 32,
                lora_dropout=args.lora_dropout or 0.05,
                lora_target_modules=args.lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
                use_qlora=args.use_qlora or False,
                save_mode=args.save_mode or "adapter",
                use_gradient_checkpointing=args.use_gradient_checkpointing or False,
                mixed_precision=args.mixed_precision or "bf16",
                log_interval=args.log_interval or 10,
                eval_interval=args.eval_interval or 500,
                save_interval=args.save_interval or 1000,
                train_data_path=args.train_data_path,
                eval_data_path=args.eval_data_path,
                hf_dataset_name=args.hf_dataset_name,
                hf_train_split=args.hf_train_split or "train",
                hf_eval_split=args.hf_eval_split,
                hf_conversation_column=args.hf_conversation_column or "conversation_a",
                hf_max_samples=args.hf_max_samples,
            )
    
    print(f"Finetuning config loaded. Mode={getattr(config, 'loss_type', None) and 'DPO' or 'SFT'}, model={config.model_path}, lora={config.use_lora}, qlora={config.use_qlora}")
    
    if config.use_qlora and not config.use_lora:
        config.use_lora = True
        print("QLoRA requires LoRA; enabling LoRA automatically.")
    
    if hasattr(config, "loss_type"):
        from finetune.dpo import DPOTrainer, DPOConfig as DPOC
        if not isinstance(config, DPOC):
            config = DPOConfig(**{k: v for k, v in vars(config).items() if k in [f.name for f in DPOConfig.__dataclass_fields__.values()]})
        trainer = DPOTrainer(config)
    else:
        from finetune.sft import SFTTrainer
        trainer = SFTTrainer(config)
    
    trainer.train()
    print(f"Finetuning complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
