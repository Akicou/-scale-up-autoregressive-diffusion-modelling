# Auto-Regressive + Diffusion Hybrid Pretraining System

A production-ready pretraining framework for compact language model training with parameter scaling and finetuning capabilities.

## Overview

This system implements a dual-mode architecture that combines:

- **Auto-Regressive (AR) mode**: Standard causal language modeling for reasoning/thinking
- **Diffusion mode**: Masked token prediction for iterative output generation

### Features

- **Phase 1**: Pretrain a ~10M parameter model from scratch
- **Phase 2**: Scale up to larger targets (1B, 4B+ parameters)
- **Phase 3**: Finetune with SFT, DPO, LoRA, or QLoRA
- **Tool Calling**: Built-in support for function calling
- **Production Ready**: Type hints, error handling, logging

## Architecture

### Dual-Mode Model

```
Input Text
    │
    ▼
┌─────────────────────────────────────┐
│   Shared Transformer Backbone       │
│   (8-12 layers, RoPE, SwiGLU)       │
└─────────────────────────────────────┘
    │              │
    ▼              ▼
┌─────────┐   ┌─────────────┐
│AR Head  │   │Diffusion Head│
│(next-   │   │(masked token │
│token)   │   │prediction)   │
└─────────┘   └─────────────┘
    │              │
    ▼              ▼
AR Output    Diffusion Output
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Phase 1: Pretraining

```bash
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
```

### Dev Branch Changes (Diffusion + Multi-GPU + MTP)

The `dev` branch includes targeted fixes and training improvements:

- Added a dedicated mask token in model vocabulary (`mask_token_id`) and persisted it in checkpoint config
- Added diffusion noising during training via random token masking (`diffusion_mask_prob`)
- Changed diffusion loss to train only on masked positions (non-masked positions are ignored)
- Kept AR training causal while diffusion mode uses bidirectional context behavior
- Updated inference/checkpoint loading to correctly reconstruct original vocab + mask token handling
- Added optional multi-GPU pretraining support via Hugging Face Accelerate (`--use-accelerate`)
- Added distributed-safe logging/checkpointing/saving behavior (main process only)
- Added optional Medusa-style MTP for AR training with 3 heads (`--mtp-enabled --mtp-num-heads 3`)
- Added balanced default MTP per-head loss weights (`1.0,0.7,0.5`) with configurable global scaling
- Added `medusa` inference mode for speculative multi-token AR decoding

### MTP (Medusa-Style) for AR

Enable 3 MTP heads (recommended balanced weights):

```bash
python pretrain.py \
  --loss-mode ar \
  --mtp-enabled \
  --mtp-num-heads 3 \
  --mtp-loss-weights 1.0,0.7,0.5 \
  --mtp-loss-weight 1.0
```

Use Medusa-style decoding at inference:

```bash
python inference.py \
  --checkpoint checkpoints/10m_pretrain/checkpoint-epoch0-step0/model.pt \
  --config checkpoints/10m_pretrain/checkpoint-epoch0-step0/config.pt \
  --prompt "Hello" \
  --mode medusa \
  --max-tokens 100
```

### Multi-GPU Pretraining (Accelerate)

Run single-node multi-GPU pretraining with:

```bash
# 2 GPUs
accelerate launch --num_processes 2 pretrain.py --use-accelerate --mixed-precision bf16 --loss-mode both

# 4 GPUs
accelerate launch --num_processes 4 pretrain.py --use-accelerate --mixed-precision bf16 --loss-mode both
```

### Recommended Training Profiles (10M / 50M / 150M)

Use these as starting points for `pretrain.py`:

| Target Params | Suggested Core Args |
|---|---|
| ~10M | `--hidden-size 256 --num-layers 6 --num-heads 4 --head-dim 64 --batch-size 64 --seq-len 512 --lr 1e-4 --warmup-steps 200 --loss-mode both --gradient-checkpointing` |
| ~50M | `--hidden-size 512 --num-layers 10 --num-heads 8 --head-dim 64 --batch-size 32 --seq-len 1024 --lr 1e-4 --warmup-steps 500 --loss-mode both --gradient-checkpointing --accumulation-steps 2` |
| ~150M | `--hidden-size 768 --num-layers 14 --num-heads 12 --head-dim 64 --batch-size 16 --seq-len 2048 --lr 8e-5 --warmup-steps 1000 --loss-mode both --gradient-checkpointing --accumulation-steps 4` |

### Recommended Context Windows by Model Size

| Target Params | Recommended `--seq-len` | Notes |
|---|---:|---|
| ~10M | 512 | Most stable and efficient for small models |
| ~50M | 1024 | Good tradeoff between context and throughput |
| ~150M | 1024–2048 | Prefer 2048 if memory budget allows |

### Recommended Data Scale by Model Size

Scale dataset size with model size to avoid undertraining:

| Target Params | Recommended Training Tokens | Practical Dataset Size (approx.) |
|---|---:|---:|
| ~10M | 1B–3B tokens | 4–12 GB cleaned text |
| ~50M | 5B–15B tokens | 20–60 GB cleaned text |
| ~150M | 15B–40B tokens | 60–160 GB cleaned text |

Example invocations:

```bash
# ~10M profile
python pretrain.py --hidden-size 256 --num-layers 6 --num-heads 4 --head-dim 64 --batch-size 64 --seq-len 512 --lr 1e-4 --warmup-steps 200 --loss-mode both --gradient-checkpointing

# ~50M profile
python pretrain.py --hidden-size 512 --num-layers 10 --num-heads 8 --head-dim 64 --batch-size 32 --seq-len 1024 --lr 1e-4 --warmup-steps 500 --loss-mode both --gradient-checkpointing --accumulation-steps 2

# ~150M profile
python pretrain.py --hidden-size 768 --num-layers 14 --num-heads 12 --head-dim 64 --batch-size 16 --seq-len 2048 --lr 8e-5 --warmup-steps 1000 --loss-mode both --gradient-checkpointing --accumulation-steps 4
```

### Inference

```bash
# Basic completion (AR mode)
python inference.py \
  --checkpoint checkpoints/10m_pretrain/checkpoint-epoch0-step0/model.pt \
  --config checkpoints/10m_pretrain/checkpoint-epoch0-step0/config.pt \
  --prompt "Hello, how are you?" \
  --max-tokens 50 \
  --mode ar

# Diffusion mode
python inference.py \
  --checkpoint checkpoints/10m_pretrain/checkpoint-epoch0-step0/model.pt \
  --config checkpoints/10m_pretrain/checkpoint-epoch0-step0/config.pt \
  --prompt "Hello" \
  --mode diffusion

# Reasoning mode (think then answer)
python inference.py \
  --checkpoint checkpoints/10m_pretrain/checkpoint-epoch0-step0/model.pt \
  --config checkpoints/10m_pretrain/checkpoint-epoch0-step0/config.pt \
  --prompt "What is 2+2?" \
  --max-tokens 100 \
  --mode reasoning
```

### Generation Modes

- `ar` - Auto-regressive (standard next-token prediction)
- `diffusion` - Masked token prediction (can refine outputs)
- `combined` - Average of AR and diffusion logits
- `reasoning` - AR for thinking, switches to diffusion after `</thinking>`
- `medusa` - AR with MTP speculative multi-token decoding

### Phase 2: Scale Up

```bash
python scale_up.py \
  --input ./checkpoints/10m_pretrain \
  --output ./checkpoints/4b_scaled \
  --target-parameters 4000000000 \
  --method width+depth \
  --interpolate-pos-embeddings linear
```

### Phase 2b: Extend Context Length

Extend the context window from 8K to 32K, 64K, or longer using YaRN (Yet Another RoPE Extension).

```bash
# Extend from 8K to 64K context
python extend_context.py \
  --checkpoint ./checkpoints/4b_scaled/model.pt \
  --config ./checkpoints/4b_scaled/config.pt \
  --output ./checkpoints/4b_64k/model.pt \
  --target-context 65536 \
  --base-context 8192 \
  --method yarn
```

**Available context extensions:**

| Base | Target | RoPE Scale | RoPE Factor |
|------|--------|------------|-------------|
| 8K   | 16K    | 0.5        | 1.0         |
| 8K   | 32K    | 0.25       | 2.0         |
| 8K   | 64K    | 0.125      | 4.0         |
| 16K  | 32K    | 0.5        | 1.0         |
| 16K  | 64K    | 0.25       | 2.0         |
| 32K  | 64K    | 0.5        | 1.0         |

**Apply YaRN to a running model:**

```python
from extend_context import apply_yarn_to_model

# Apply YaRN to extend context to 64K
apply_yarn_to_model(model, target_context=65536, base_context=8192)

# Now model can handle sequences up to 64K tokens
output = model.generate(prompt, max_length=50000)
```

**How YaRN works:**
- **RoPE Scale** (rope_scale): Interpolates position embeddings to reduce their "frequency", allowing the model to represent more positions
- **RoPE Factor** (rope_factor): Scales attention logits to compensate for the dilution effect of interpolation on long-range dependencies

### Phase 3: SFT Finetuning

**Low-VRAM QLoRA Finetuning** (Recommended for 4B+ models)

Finetune your scaled model with QLoRA 4-bit quantization, reducing VRAM usage from ~70GB to ~3-5GB:

```bash
# Using YAML config (recommended)
python -m finetune --config config/finetune_4b_qlora.yaml

# Or with CLI flags
python -m finetune.sft \
  --model-path ./checkpoints/4b_scaled \
  --use-lora \
  --use-qlora \
  --lora-r 16 \
  --save-mode adapter \
  --use-gradient-checkpointing
```

This path now auto-exports custom checkpoints to a self-contained HF directory with bundled remote code and auto-regenerates stale `hf_format` exports when the source checkpoint changes.

**Standard SFT** (for smaller models with sufficient VRAM):

```bash
python -m finetune.sft \
  --model-path ./checkpoints/4b_scaled \
  --train-data-path ./data/instruction.json \
  --output-dir ./checkpoints/4b_sft \
  --use-lora \
  --lora-r 16 \
  --epochs 3
```

### Phase 4: DPO (Optional)

**Low-VRAM QLoRA DPO**:

```bash
# Using YAML config
python -m finetune --config config/finetune_dpo_qlora.yaml

# Or with CLI flags
python -m finetune.dpo \
  --model-path ./checkpoints/4b_sft \
  --train-data-path ./data/preference.json \
  --output-dir ./checkpoints/4b_dpo \
  --use-qlora \
  --beta 0.1 \
  --reference-free
```

**Standard DPO**:

```bash
python -m finetune.dpo \
  --model-path ./checkpoints/4b_sft \
  --train-data-path ./data/preference.json \
  --output-dir ./checkpoints/4b_dpo \
  --beta 0.1
```

## Tool Calling

### Registering Tools

```python
from tools import ToolRegistry

# Create registry
registry = ToolRegistry()

# Register a tool
registry.register(
    name="calculate",
    func=lambda expr: eval(expr),
    description="Evaluate a mathematical expression",
    parameters=[
        {"name": "expr", "type": "string", "description": "Expression to evaluate"}
    ]
)

# Get tool schemas for prompts
print(registry.get_schemas_text())
```

### Parsing Tool Calls

```python
from tools import ToolCallParser

parser = ToolCallParser()

# Parse from JSON format
output = 'Let me calculate: {"tool": "calculate", "args": {"expression": "2+2"}}'
tool_calls = parser.parse(output)

for call in tool_calls:
    print(f"Tool: {call.tool_name}")
    print(f"Args: {call.arguments}")
```

### Tool Calling in Generation

```python
from tools import ToolCallingMixin, get_default_registry

# Get default tools
registry = get_default_registry()

# Mixin for model
class MyModelWrapper(ToolCallingMixin):
    def __init__(self):
        super().__init__(tool_registry=registry)
    
    def generate(self, prompt: str) -> str:
        output = self.model.generate(prompt)
        
        # Handle any tool calls
        if detect_tool_calls(output):
            output = self.handle_tool_call(output)
        
        return output
```

## Configuration

See `config/` for example configuration files:

### YAML Configs (Recommended)

| Config | Purpose | VRAM Usage |
|--------|---------|------------|
| `finetune_4b_qlora.yaml` | QLoRA SFT for 4B models | ~3-5 GB |
| `finetune_dpo_qlora.yaml` | QLoRA DPO for 4B models | ~3-5 GB |
| `finetune_default.yaml` | Standard LoRA SFT | ~8-24 GB |

### JSON Configs (Legacy)

- `pretrain_10m.json` - 10M parameter pretraining config
- `scale_1b.json` - Scale to 1B config
- `scale_4b.json` - Scale to 4B config
- `sft_lora.json` - SFT with LoRA config
- `dpo.json` - DPO training config

## API Reference

### pretrain.py

Main pretraining script with dual-mode AR + Diffusion training.

**Arguments:**

- `--data-path`: Training data path
- `--val-data-path`: Validation data path
- `--output-dir`: Output directory
- `--epochs`: Number of epochs
- `--batch-size`: Batch size
- `--lr`: Learning rate
- `--warmup-steps`: Warmup steps
- `--loss-mode`: Loss mode (ar, diffusion, both)
- `--use-lora`: Enable LoRA
- `--wandb-project`: WandB project name

### scale_up.py

Model scaling script with support for width, depth, and combined scaling.

**Arguments:**

- `--input`: Input model path
- `--output`: Output model path
- `--target-parameters`: Target parameter count
- `--method`: Scaling method (width, depth, width+depth)
- `--interpolate-pos-embeddings`: Position embedding interpolation

### finetune/

Unified finetuning entrypoint:

```bash
# With YAML config
python -m finetune --config config/finetune_4b_qlora.yaml

# Or use specific trainers
python -m finetune.sft  # SFT
python -m finetune.dpo  # DPO
```

### finetune/sft.py

SFT trainer with LoRA/QLoRA support.

**Arguments:**

- `--model-path`: Model to finetune
- `--train-data-path` / `--data-path`: Training data
- `--use-lora`: Enable LoRA
- `--use-qlora`: Enable QLoRA (4-bit quantization)
- `--lora-r`: LoRA rank
- `--save-mode`: `adapter` (LoRA only) or `merged`
- `--config`: Path to YAML config file

### finetune/dpo.py

DPO trainer for preference optimization.

**Arguments:**

- `--model-path`: Model to finetune
- `--train-data-path` / `--data-path`: Preference data
- `--beta`: KL penalty coefficient
- `--loss-type`: Loss type (sigmoid, hinge, ipo)
- `--reference-free`: Skip reference model for VRAM efficiency
- `--use-qlora`: Enable QLoRA

## Data Formats

### Instruction Data (JSON)

```json
[
  {
    "instruction": "Explain machine learning",
    "output": "Machine learning is..."
  }
]
```

### Thinking/Reasoning Data (JSON)

For models that support reasoning (think then answer), use this format:

```json
[
  {
    "messages": [
      {"role": "user", "content": "What is 2+2?"},
      {"role": "assistant", "content": "<thinking>Let me calculate 2+2. This is basic addition. 2+2=4</thinking>4"}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "Explain AI"},
      {"role": "assistant", "content": "<thinking>AI stands for Artificial Intelligence. It's a field of computer science that focuses on creating intelligent machines. Key concepts include machine learning, neural networks, and deep learning.</thinking>Artificial Intelligence (AI) is a branch of computer science focused on creating intelligent machines that can perform tasks that typically require human intelligence."}
    ]
  }
]
```

The `<thinking>...</thinking>` tags contain the model's reasoning process, which is generated using AR mode. The actual answer comes after the closing tag and can be generated using diffusion mode for better quality.

### Preference Data (JSON)

```json
[
  {
    "prompt": "Explain machine learning",
    "preferred": "Detailed explanation...",
    "rejected": "Brief explanation..."
  }
]
```

## Utilities

### Export Custom Checkpoint to HF Format

Required for QLoRA on custom scaled models:

```bash
python tools/cli.py export \
  --input ./checkpoints/4b_scaled \
  --output ./checkpoints/4b_scaled_hf

# Force regeneration of an existing HF export
python tools/cli.py export \
  --input ./checkpoints/4b_scaled \
  --output ./checkpoints/4b_scaled_hf \
  --force
```

### Estimate VRAM Requirements

```bash
python tools/cli.py estimate \
  --model-path ./checkpoints/4b_scaled \
  --use-qlora
```

### VRAM Estimates by Model Size

| Model | Full BF16 | QLoRA 4-bit |
|-------|-----------|-------------|
| 30M   | ~0.2 GB   | ~0.02 GB    |
| 500M  | ~3 GB     | ~0.5 GB     |
| 4B    | ~24 GB    | ~2.5 GB     |

*With gradients + optimizer: 4B full = ~70GB, 4B QLoRA = ~3-5GB*

## Model Output Format

Models are saved in the following format:

### Custom Checkpoint Format

```
./output/
├── model.pt            # Model weights
├── config.pt           # Model configuration
├── tokenizer.json      # Tokenizer
└── tokenizer_config.json
```

### HuggingFace Format (after export)

```
./output_hf/
├── config.json
├── pytorch_model.bin
├── tokenizer.json
└── tokenizer_config.json
```

Scaled models include `scaling_config.json` with original + target architecture details.

For custom QLoRA exports, the HF directory also bundles `configuration_nayhein_mini.py`, `modeling_nayhein_mini.py`, and `nayhein_export.json`. The HF export contains only the AR/CausalLM path needed for QLoRA; diffusion and MTP heads are intentionally omitted.

## Development

### Running Tests

```bash
python -m unittest tests.test_custom_hf_export
```

### Low-VRAM Finetuning

For detailed information on low-VRAM finetuning with QLoRA, see `LOW_VRAM_FINETUNING.md`.

**Quick tips:**

- Use `--use-qlora` for models >1B parameters
- Set small `batch_size` (1-2) + high `gradient_accumulation_steps`
- Use YAML configs for easier management
- Export custom checkpoints to HF format first run (automatic)

### Code Structure

```
.
├── pretrain.py          # Pretraining script
├── scale_up.py          # Model scaling script
├── finetune/            # Finetuning module
│   ├── __init__.py
│   ├── base.py         # Base trainer class
│   ├── sft.py          # SFT trainer
│   ├── dpo.py          # DPO trainer
│   └── utils.py        # Utilities
├── tools/              # Tool calling module
│   ├── __init__.py
│   ├── registry.py     # Tool registry
│   └── parsing.py      # Tool call parsing
├── config/             # Configuration files
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## License

MIT License
# -scale-up-autoregressive-diffusion-modelling


