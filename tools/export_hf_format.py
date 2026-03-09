#!/usr/bin/env python3
"""Export custom DualModeModel checkpoints to a self-contained HF format."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pretrain import DualModeModel, get_default_tokenizer

TEMPLATE_DIR = Path(__file__).resolve().parent / "hf_export_templates"
EXPORT_VERSION = 1
EXPORT_METADATA_FILE = "nayhein_export.json"
REMOTE_CONFIG_FILE = "configuration_nayhein_mini.py"
REMOTE_MODEL_FILE = "modeling_nayhein_mini.py"


def _file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
    }


def _load_checkpoint_config(checkpoint_dir: Path) -> Tuple[dict, dict]:
    config_dict = torch.load(checkpoint_dir / "config.pt", map_location="cpu")
    model_cfg = config_dict.get("target_config", config_dict)
    return config_dict, model_cfg


def _build_hf_config(output_dir: Path, config_dict: dict, model_cfg: dict) -> dict:
    base_vocab = model_cfg.get("original_vocab_size") or model_cfg.get("base_vocab_size") or model_cfg.get("vocab_size", 16001) - 1
    hidden_size = int(model_cfg.get("hidden_size", 256))
    num_layers = int(model_cfg.get("num_layers", 6))
    num_heads = int(model_cfg.get("num_heads", 4))
    head_dim = int(model_cfg.get("head_dim", max(1, hidden_size // max(1, num_heads))))
    max_seq_len = int(model_cfg.get("max_seq_len", 4096))
    mlp_ratio = float(model_cfg.get("mlp_ratio", 4.0))
    vocab_size = int(model_cfg.get("vocab_size", base_vocab + 1))
    mask_token_id = int(model_cfg.get("mask_token_id", vocab_size - 1))
    layer_norm_eps = float(model_cfg.get("layer_norm_eps", 1e-6))
    rope_theta = float(model_cfg.get("rope_theta", 10000.0))
    bos_token_id = int(config_dict.get("bos_token_id", 1))
    eos_token_id = int(config_dict.get("eos_token_id", 2))
    pad_token_id = int(config_dict.get("pad_token_id", 0))

    return {
        "model_type": "nayhein_mini",
        "architectures": ["NayheinMiniForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_nayhein_mini.NayheinMiniConfig",
            "AutoModelForCausalLM": "modeling_nayhein_mini.NayheinMiniForCausalLM",
        },
        "vocab_size": vocab_size,
        "original_vocab_size": base_vocab,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "intermediate_size": int(hidden_size * mlp_ratio),
        "max_position_embeddings": max_seq_len,
        "mask_token_id": mask_token_id,
        "hidden_act": "silu",
        "layer_norm_eps": layer_norm_eps,
        "tie_word_embeddings": False,
        "rope_theta": rope_theta,
        "use_cache": True,
        "pad_token_id": pad_token_id,
        "bos_token_id": bos_token_id,
        "eos_token_id": eos_token_id,
        "export_version": EXPORT_VERSION,
        "_name_or_path": str(output_dir.absolute()),
    }


def _copy_remote_code_templates(output_dir: Path) -> None:
    for filename in (REMOTE_CONFIG_FILE, REMOTE_MODEL_FILE):
        shutil.copyfile(TEMPLATE_DIR / filename, output_dir / filename)


def _map_state_dict(state_dict: dict) -> Tuple[dict, List[str]]:
    hf_state_dict: Dict[str, torch.Tensor] = {}
    ignored: List[str] = []
    unknown: List[str] = []

    for key, value in state_dict.items():
        if key == "token_embeddings.weight":
            hf_state_dict["model.embed_tokens.weight"] = value
        elif key == "position_embeddings.weight":
            hf_state_dict["model.position_embeddings.weight"] = value
        elif key == "embed_layernorm.weight":
            hf_state_dict["model.embed_layernorm.weight"] = value
        elif key == "embed_layernorm.bias":
            hf_state_dict["model.embed_layernorm.bias"] = value
        elif key == "final_layernorm.weight":
            hf_state_dict["model.norm.weight"] = value
        elif key == "final_layernorm.bias":
            hf_state_dict["model.norm.bias"] = value
        elif key == "ar_head.lm_head.weight":
            hf_state_dict["lm_head.weight"] = value
        elif key.startswith("layers."):
            parts = key.split(".")
            if len(parts) < 4:
                continue
            layer_idx = parts[1]
            if parts[2] == "attention" and len(parts) == 5 and parts[3] in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                hf_state_dict[f"model.layers.{layer_idx}.self_attn.{parts[3]}.{parts[4]}"] = value
            elif parts[2] == "attention" and len(parts) >= 4 and parts[3] == "rope":
                ignored.append(key)
            elif parts[2] in ("input_layernorm", "post_attention_layernorm") and len(parts) == 4:
                hf_state_dict[f"model.layers.{layer_idx}.{parts[2]}.{parts[3]}"] = value
            elif parts[2] == "mlp" and len(parts) == 5:
                hf_state_dict[f"model.layers.{layer_idx}.mlp.{parts[3]}.{parts[4]}"] = value
        elif key.startswith("diffusion_head.") or key.startswith("mtp_heads."):
            ignored.append(key)
        else:
            unknown.append(key)

    if unknown:
        raise ValueError(f"Unmapped checkpoint tensors during HF export: {', '.join(sorted(unknown))}")

    return hf_state_dict, ignored


def _build_template_state(model_cfg: dict) -> dict:
    base_vocab = model_cfg.get("original_vocab_size") or model_cfg.get("base_vocab_size") or model_cfg.get("vocab_size", 16001) - 1
    template_model = DualModeModel(
        vocab_size=base_vocab,
        hidden_size=int(model_cfg.get("hidden_size", 256)),
        num_layers=int(model_cfg.get("num_layers", 6)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        head_dim=int(model_cfg.get("head_dim", 64)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        max_seq_len=int(model_cfg.get("max_seq_len", 4096)),
        mtp_enabled=bool(model_cfg.get("mtp_enabled", False)),
        mtp_num_heads=int(model_cfg.get("mtp_num_heads", 0)),
        mtp_loss_weights=model_cfg.get("mtp_loss_weights", [1.0, 0.7, 0.5]),
    )
    return template_model.state_dict()


def _patch_state_dict_to_template(state_dict: dict, template_state: dict) -> dict:
    patched_state = {}
    for key, value in state_dict.items():
        template_value = template_state.get(key)
        if template_value is None or template_value.shape == value.shape:
            patched_state[key] = value
            continue
        patched = template_value.clone()
        slices = tuple(slice(0, min(dst, src)) for dst, src in zip(template_value.shape, value.shape))
        patched[slices] = value[slices]
        patched_state[key] = patched
    return patched_state


def _copy_or_create_tokenizer(checkpoint_dir: Path, output_dir: Path) -> None:
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
    ]
    copied_any = False
    for filename in tokenizer_files:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copyfile(src, output_dir / filename)
            copied_any = True
    if copied_any:
        return

    print("Tokenizer files missing in checkpoint; saving fallback tokenizer into HF export")
    tokenizer = get_default_tokenizer()
    tokenizer.save_pretrained(output_dir)


def _write_metadata(checkpoint_dir: Path, output_dir: Path, ignored: List[str]) -> None:
    metadata = {
        "export_version": EXPORT_VERSION,
        "source_checkpoint_dir": str(checkpoint_dir),
        "source_files": {
            "model.pt": _file_fingerprint(checkpoint_dir / "model.pt"),
            "config.pt": _file_fingerprint(checkpoint_dir / "config.pt"),
        },
        "ignored_keys": ignored,
    }
    (output_dir / EXPORT_METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def export_dualmode_to_hf(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    model_type: str = "nayhein_mini",
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_type != "nayhein_mini":
        raise ValueError("Custom checkpoint export currently supports only model_type='nayhein_mini'")
    if not (checkpoint_dir / "model.pt").exists() or not (checkpoint_dir / "config.pt").exists():
        raise FileNotFoundError(f"Invalid checkpoint: expected model.pt and config.pt in {checkpoint_dir}")

    config_dict, model_cfg = _load_checkpoint_config(checkpoint_dir)
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    state_dict = _patch_state_dict_to_template(state_dict, _build_template_state(model_cfg))

    print(
        f"Exporting {model_cfg.get('num_layers', 6)}L x {model_cfg.get('hidden_size', 256)}H model "
        f"({model_cfg.get('num_heads', 4)} heads, {model_cfg.get('head_dim', 64)} head_dim) to HF format..."
    )

    hf_config = _build_hf_config(output_dir, config_dict, model_cfg)
    (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")
    print("Created config.json")

    _copy_remote_code_templates(output_dir)

    hf_state_dict, ignored = _map_state_dict(state_dict)
    torch.save(hf_state_dict, output_dir / "pytorch_model.bin")
    print(f"Saved {len(hf_state_dict)} tensors to pytorch_model.bin")
    if ignored:
        ignored_categories = []
        if any(key.startswith("diffusion_head.") for key in ignored):
            ignored_categories.append("diffusion_head")
        if any(key.startswith("mtp_heads.") for key in ignored):
            ignored_categories.append("mtp_heads")
        if any(".attention.rope." in key for key in ignored):
            ignored_categories.append("attention_rope_buffers")
        print(f"Ignored {len(ignored)} non-causal tensors ({', '.join(ignored_categories)})")

    _copy_or_create_tokenizer(checkpoint_dir, output_dir)
    _write_metadata(checkpoint_dir, output_dir, ignored)

    print(f"Export complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DualModeModel checkpoint to HF format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input checkpoint directory")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output HF-format directory")
    parser.add_argument("--model-type", type=str, default="nayhein_mini", help="Model type identifier")

    args = parser.parse_args()
    export_dualmode_to_hf(args.input, args.output, args.model_type)
    print("Export finished successfully!")


if __name__ == "__main__":
    main()
