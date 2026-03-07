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

```bash
python -m finetune.sft \
  --model-path ./checkpoints/4b_scaled \
  --data-path ./data/instruction.json \
  --output-dir ./checkpoints/4b_sft \
  --use-lora \
  --lora-r 16 \
  --epochs 3
```

### Phase 4: DPO (Optional)

```bash
python -m finetune.dpo \
  --model-path ./checkpoints/4b_sft \
  --data-path ./data/preference.json \
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

### finetune/sft.py

SFT trainer with LoRA/QLoRA support.

**Arguments:**

- `--model-path`: Model to finetune
- `--data-path`: Training data
- `--use-lora`: Enable LoRA
- `--use-qlora`: Enable QLoRA
- `--lora-r`: LoRA rank

### finetune/dpo.py

DPO trainer for preference optimization.

**Arguments:**

- `--model-path`: Model to finetune
- `--data-path`: Preference data
- `--beta`: KL penalty coefficient
- `--loss-type`: Loss type (sigmoid, hinge, ipo)

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

## Model Output Format

Models are saved in HuggingFace format:

```
./output/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── scaling_config.json (for scaled models)
```

## Development

### Running Tests

```bash
# Placeholder for test command
python -c "from pretrain import DualModeModel; print('Import OK')"
```

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
#   - s c a l e - u p - a u t o r e g r e s s i v e - d i f f u s i o n - m o d e l l i n g  
 