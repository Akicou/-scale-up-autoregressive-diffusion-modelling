from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from pretrain import DualModeModel

inference = importlib.import_module("inference")


def _peft_available() -> bool:
    return importlib.util.find_spec("peft") is not None


class InferenceLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.mkdtemp(prefix="nayhein_inference_test_")
        self.root = Path(self.tempdir)

    def tearDown(self) -> None:
        for temp_dir in inference._temp_dirs:
            temp_dir.cleanup()
        inference._temp_dirs.clear()
        inference.model = None
        inference.tokenizer = None
        inference.device = None
        inference.load_context = None
        inference.tool_registry = None
        inference.tool_parser = None
        shutil.rmtree(self.tempdir, ignore_errors=True)

    def _make_tokenizer(self, target_dir: Path) -> PreTrainedTokenizerFast:
        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "hello": 4,
            "world": 5,
            "test": 6,
            "adapter": 7,
        }
        tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
        tokenizer_backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_backend,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(target_dir)
        return tokenizer

    def _write_custom_checkpoint(self, checkpoint_dir: Path) -> None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model = DualModeModel(
            vocab_size=8,
            hidden_size=12,
            num_layers=2,
            num_heads=3,
            head_dim=4,
            max_seq_len=16,
            mtp_enabled=True,
        )
        torch.save(model.state_dict(), checkpoint_dir / "model.pt")
        torch.save(
            {
                "vocab_size": model.vocab_size,
                "original_vocab_size": model.original_vocab_size,
                "mask_token_id": model.mask_token_id,
                "hidden_size": model.hidden_size,
                "num_layers": model.num_layers,
                "num_heads": model.num_heads,
                "head_dim": model.head_dim,
                "max_seq_len": model.max_seq_len,
                "mtp_enabled": model.mtp_enabled,
                "mtp_num_heads": model.mtp_num_heads,
                "mtp_loss_weights": model.mtp_loss_weights,
            },
            checkpoint_dir / "config.pt",
        )
        self._make_tokenizer(checkpoint_dir)

    def _write_hf_model(self, model_dir: Path) -> Path:
        model_dir.mkdir(parents=True, exist_ok=True)
        config = GPT2Config(
            vocab_size=8,
            n_positions=32,
            n_ctx=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        model = GPT2LMHeadModel(config)
        model.save_pretrained(model_dir, safe_serialization=True)
        self._make_tokenizer(model_dir)
        return model_dir / "model.safetensors"

    def test_loads_custom_dir_via_model_path(self) -> None:
        checkpoint_dir = self.root / "custom_dir"
        self._write_custom_checkpoint(checkpoint_dir)

        inference.load_model(str(checkpoint_dir), device_str="cpu")

        self.assertEqual(inference.load_context.backend_type, "custom_native")
        self.assertTrue(inference.load_context.capabilities.supports_ar)
        self.assertTrue(inference.load_context.capabilities.supports_diffusion)
        self.assertTrue(inference.load_context.capabilities.supports_mtp)

    def test_loads_custom_file_with_explicit_config(self) -> None:
        checkpoint_dir = self.root / "custom_file"
        self._write_custom_checkpoint(checkpoint_dir)
        external_config = self.root / "external_config.pt"
        shutil.copy2(checkpoint_dir / "config.pt", external_config)
        (checkpoint_dir / "config.pt").unlink()

        inference.load_model(
            str(checkpoint_dir / "model.pt"),
            config_path=str(external_config),
            device_str="cpu",
        )

        self.assertEqual(inference.load_context.backend_type, "custom_native")
        self.assertEqual(inference.load_context.source_paths["config_path"], str(external_config))

    def test_loads_hf_dir_saved_as_safetensors(self) -> None:
        model_dir = self.root / "hf_dir"
        self._write_hf_model(model_dir)

        inference.load_model(str(model_dir), device_str="cpu", trust_remote_code=False)

        self.assertEqual(inference.load_context.backend_type, "hf_causallm")
        self.assertTrue(inference.load_context.capabilities.supports_ar)
        self.assertFalse(inference.load_context.capabilities.supports_diffusion)

    def test_loads_single_safetensors_file_with_config(self) -> None:
        model_dir = self.root / "hf_single"
        model_file = self._write_hf_model(model_dir)

        inference.load_model(
            str(model_file),
            config_path=str(model_dir / "config.json"),
            device_str="cpu",
            trust_remote_code=False,
        )

        self.assertEqual(inference.load_context.backend_type, "hf_causallm")
        self.assertEqual(inference.load_context.source_paths["model_path"], str(model_file))

    @unittest.skipUnless(_peft_available(), "peft is not installed")
    def test_loads_adapter_dir_with_metadata_base_path(self) -> None:
        from peft import LoraConfig, get_peft_model

        base_dir = self.root / "adapter_base"
        adapter_dir = self.root / "adapter"
        self._write_hf_model(base_dir)

        base_model = GPT2LMHeadModel.from_pretrained(base_dir)
        peft_model = get_peft_model(
            base_model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["c_attn"],
                task_type="CAUSAL_LM",
            ),
        )
        peft_model.save_pretrained(adapter_dir)

        config_path = adapter_dir / "adapter_config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["base_model_name_or_path"] = str(base_dir)
        config_path.write_text(json.dumps(config_data), encoding="utf-8")
        self._make_tokenizer(adapter_dir)

        inference.load_model(str(adapter_dir), device_str="cpu", trust_remote_code=False)

        self.assertEqual(inference.load_context.backend_type, "peft_adapter")
        self.assertEqual(inference.load_context.source_paths["base_model_path"], str(base_dir))

    @unittest.skipUnless(_peft_available(), "peft is not installed")
    def test_loads_adapter_dir_with_explicit_base_override(self) -> None:
        from peft import LoraConfig, get_peft_model

        base_dir = self.root / "adapter_override_base"
        adapter_dir = self.root / "adapter_override"
        self._write_hf_model(base_dir)

        base_model = GPT2LMHeadModel.from_pretrained(base_dir)
        peft_model = get_peft_model(
            base_model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                bias="none",
                target_modules=["c_attn"],
                task_type="CAUSAL_LM",
            ),
        )
        peft_model.save_pretrained(adapter_dir)

        config_path = adapter_dir / "adapter_config.json"
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        config_data["base_model_name_or_path"] = "missing-base-model"
        config_path.write_text(json.dumps(config_data), encoding="utf-8")

        inference.load_model(
            str(adapter_dir),
            base_model_path=str(base_dir),
            device_str="cpu",
            trust_remote_code=False,
        )

        self.assertEqual(inference.load_context.backend_type, "peft_adapter")
        self.assertEqual(inference.load_context.source_paths["base_model_path"], str(base_dir))

    def test_diffusion_mode_fails_for_hf_backend(self) -> None:
        model_dir = self.root / "hf_diffusion"
        self._write_hf_model(model_dir)
        inference.load_model(str(model_dir), device_str="cpu", trust_remote_code=False)

        with self.assertRaisesRegex(RuntimeError, "requires a native DualModeModel checkpoint"):
            inference.complete("hello", max_tokens=2, mode="diffusion", verbose=False)

    def test_medusa_falls_back_to_ar_for_hf_backend(self) -> None:
        model_dir = self.root / "hf_medusa"
        self._write_hf_model(model_dir)
        inference.load_model(str(model_dir), device_str="cpu", trust_remote_code=False)

        result = inference.complete("hello", max_tokens=2, mode="medusa", verbose=False)

        self.assertIn("text", result)
        self.assertLessEqual(result["total_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
