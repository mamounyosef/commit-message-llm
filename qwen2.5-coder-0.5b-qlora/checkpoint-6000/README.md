---
base_model: Qwen/Qwen2.5-Coder-0.5B
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-Coder-0.5B
- lora
- transformers
- qlora
- commit-message-generation
- code-summarization
license: apache-2.0
language:
- en
---

# QLoRA Adapter for Commit Message Generation

Fine-tuned LoRA adapter for **Qwen2.5-Coder-0.5B** that generates clear, concise Git commit messages from code diffs.

## Model Details

### Model Description

This model is a **QLoRA (4-bit quantized LoRA)** adapter trained on the Qwen2.5-Coder-0.5B base model to automatically generate commit messages from Git diffs. The adapter learns to summarize code changes into human-readable descriptions, understanding programming patterns and translating technical modifications into natural language.

**Key characteristics:**
- Uses the **PT (Pretrained/Base)** version of Qwen2.5-Coder for cleaner, more controllable outputs
- Trained with 4-bit NF4 quantization for efficient fine-tuning on consumer hardware
- Only LoRA adapters are included (~few MB); requires base model for inference
- Optimized for diff-to-message generation, not chat or instruction following

- **Developed by:** Mamoun Yosef
- **Model type:** Causal Language Model (Decoder-only Transformer) with LoRA adapters
- **Language(s):** English
- **License:** Apache 2.0
- **Finetuned from model:** Qwen/Qwen2.5-Coder-0.5B

### Model Sources

- **Repository:** [[commit-message-llm]](https://github.com/mamounyosef/commit-message-llm)
- **Base Model:** [Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)

## Uses

### Direct Use

This adapter is designed for **automated commit message generation** from Git diffs. It can be used to:

- Generate commit messages for staged changes in Git repositories
- Suggest descriptive summaries for code modifications
- Automate documentation of code changes in CI/CD pipelines
- Assist developers in writing clear, consistent commit messages

**Example input (Git diff):**
```diff
diff --git a/src/utils.py b/src/utils.py
index abc123..def456 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -10,6 +10,9 @@ def process_data(data):
     return result

+def validate_input(data):
+    return data is not None and len(data) > 0
+
 def save_output(output, filename):
```

**Example output:**
```
Add input validation function
```

### Downstream Use

Can be integrated into:
- Git hooks (pre-commit, commit-msg)
- IDE extensions for code editors
- Code review tools
- Developer productivity applications

### Out-of-Scope Use

**Not suitable for:**
- General text generation or chat
- Generating code from descriptions (reverse direction)
- Diffs from non-programming languages
- Extremely large diffs (>8000 characters)
- Commit messages requiring deep domain knowledge beyond code structure

## Bias, Risks, and Limitations

**Limitations:**
- Trained only on English commit messages
- May struggle with very complex multi-file changes
- Limited to diff length of 50-8000 characters
- Performance depends on code quality and diff clarity
- May generate generic messages for trivial changes
- Does not understand business context or domain-specific terminology

**Risks:**
- Generated messages may not capture full intent of changes
- Should be reviewed by developers before committing
- May miss important security or breaking change implications

### Recommendations

- Always review generated commit messages before use
- Use as a suggestion tool, not fully automated solution
- Combine with manual editing for complex changes
- Test on your codebase to evaluate quality

## How to Get Started with the Model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model in 4-bit
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-0.5B",
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "path/to/checkpoint-6000")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

# Generate commit message
diff = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,4 @@
+import os
 def main():
     print("Hello")
"""

prompt = diff + "\n\nCommit message:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=False,
    num_beams=1,
    eos_token_id=tokenizer.eos_token_id,
)

message = tokenizer.decode(outputs[0], skip_special_tokens=True)
message = message[len(prompt):].strip()
print(message)
```

## Training Details

### Training Data

**Dataset:** [Maxscha/commitbench](https://huggingface.co/datasets/Maxscha/commitbench)

**Preprocessing:**
- Removed trivial messages (fix, update, wip, etc.)
- Filtered out reference-only commits (fix #123)
- Removed placeholder tokens (<HASH>, <URL>)
- Kept diffs between 50-8000 characters
- Required messages with semantic content (≥3 words)

**Final dataset sizes:**
- Training: 120,000 samples
- Validation: 15,000 samples  
- Test: 15,000 samples

### Training Procedure

**Format:**
```
{diff content}

Commit message:
{target message}<eos>
```

Prompt tokens (diff + separator) are masked with label `-100` so loss is computed only on the commit message generation.

#### Preprocessing

1. Normalize newlines (CRLF → LF)
2. Tokenize diff + separator + message
3. Mask prompt labels to `-100`
4. Truncate to max_length=512 tokens
5. Append EOS token to target

#### Training Hyperparameters

**QLoRA Configuration:**
- Quantization: 4-bit NF4
- Compute dtype: bfloat16
- LoRA rank (r): 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj

**Training Parameters:**
- Max sequence length: 512 tokens
- Per-device train batch size: 6
- Per-device eval batch size: 6
- Gradient accumulation steps: 8
- **Effective batch size: 48**
- Learning rate: 1.8e-4
- LR scheduler: Cosine with 4% warmup
- Total training steps: 6000
- Epochs: ~2
- Optimizer: paged_adamw_8bit
- Gradient clipping: 1.0
- **Training regime:** bf16 mixed precision

**Memory Optimizations:**
- Gradient checkpointing enabled
- SDPA (Scaled Dot-Product Attention) for efficient attention
- 8-bit paged optimizer
- Group by length for efficient batching

#### Speeds, Sizes, Times

- **Hardware:** NVIDIA RTX 4060 (8GB VRAM)
- **Total training time:** ~13 hours
- **Checkpoint size:** ~few MB (LoRA adapters only)
- **Peak VRAM usage:** <8GB
- **Training throughput:** ~2500 samples/hour

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

**Test split from Maxscha/commitbench:**
- 15,000 cleaned samples
- Same preprocessing as training data
- No overlap with training/validation sets

#### Metrics

- **Loss:** Cross-entropy loss on commit message tokens
- **Perplexity:** exp(loss), measures model confidence
  - Lower perplexity = better prediction quality
  - Perplexity ≈ 17 is strong for this task

### Results

| Split | Loss | Perplexity |
|-------|------|------------|
| Validation | 2.8583 | 17.43 |
| Test | 2.8501 | 17.29 |

**Qualitative Example:**
```diff
diff --git a/src/client/core/commands/menu.js
+    'core/settings'
+], function (_, hr, MenubarView, box, panels, tabs, session, localfs, settings) {
+    }).menuSection({
+        'id': "themes.settings",
+        'title': "Settings",
+        'action': function() {
+            settings.open("themes"...
```

- **Ground truth:** Add command to open themes settings in view menu
- **Model output:** Add theme settings to the menu

The model correctly identifies the purpose (menu settings addition) and generates a concise, accurate description.

## Environmental Impact

- **Hardware Type:** NVIDIA RTX 4060 (8GB VRAM)
- **Hours used:** ~13 hours
- **Cloud Provider:** N/A (local training)
- **Compute Region:** N/A
- **Carbon Emitted:** Minimal (single consumer GPU, short training time)

## Technical Specifications

### Model Architecture and Objective

- **Base Architecture:** Qwen2.5-Coder-0.5B (Decoder-only Transformer)
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **Objective:** Causal language modeling with masked prompts
- **Loss Function:** Cross-entropy on commit message tokens only

### Compute Infrastructure

#### Hardware

- GPU: NVIDIA RTX 4060
- VRAM: 8GB
- System RAM: 16GB
- Storage: SSD recommended for dataset loading

#### Software

- **Framework:** PyTorch, Hugging Face Transformers
- **PEFT Version:** 0.18.1
- **Key Libraries:**
  - `transformers` (model loading, training)
  - `peft` (LoRA adapters)
  - `bitsandbytes` (4-bit quantization)
  - `datasets` (data loading)
  - `torch` (deep learning backend)

## Model Card Authors

Mamoun Yosef

### Framework versions

- PEFT 0.18.1
- Transformers 4.x
- PyTorch 2.x
- bitsandbytes 0.x