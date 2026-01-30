# commit-message-llm: QLoRA Fine-Tuning for Commit Message Generation

Fine-tuning **Qwen2.5-Coder-0.5B** LLM using **QLoRA** (4-bit quantization + LoRA adapters) to generate clear, concise commit messages from Git diffs by learning to summarize code changes into human-like descriptions.

**Training Date:** January 28-29, 2026

---

## Task

**Input:** Git diff  
**Output:** Natural language commit message  

Training setup:
- Prompt = `diff + separator`
- Target = commit message
- Prompt tokens masked (`-100`), so loss is only on message generation
- EOS token appended

---

## Dataset

**Source:** `Maxscha/commitbench`

### Cleaning Rules
- Remove trivial messages (fix, update, etc.)
- Remove reference-only commits (fix #123)
- Remove placeholder tokens (`<HASH>`, `<URL>`)
- Diff length: 50–8000 chars
- Message must have real semantic content

### Final Size

| Split | Samples |
|-------|--------|
| Train | 120,000 |
| Val   | 15,000  |
| Test  | 15,000  |

---

## Model

**Base Model:** `Qwen/Qwen2.5-Coder-0.5B`

### Why the Coder Version?
- Pretrained specifically on code repositories and diffs
- Better understands programming syntax, patterns, and code structure
- Superior at interpreting code changes vs general-purpose LLMs

**Architecture:** Decoder-only Transformer

### PT vs IT Model Choice

Qwen models come in two variants:
- **PT (Pretrained / Base)**: Raw language model trained for next-token prediction
- **IT (Instruction-Tuned)**: Further aligned for chat and instruction following

We fine-tune the **PT (base) model** instead of the IT model because our task (diff → commit message) is a direct supervised generation task. Using the base model avoids chat-style or verbose outputs and gives cleaner, more controllable commit messages.

---

## QLoRA Fine-Tuning Strategy

**QLoRA** (Quantized Low-Rank Adaptation) enables efficient fine-tuning of large models on consumer hardware by combining two key techniques:

### 1. 4-bit Quantization
- Base model weights quantized to 4-bit NF4 (Normal Float 4) format
- Reduces memory footprint by ~75% compared to full 16-bit precision
- Computation performed in bfloat16 for stability

### 2. Low-Rank Adaptation (LoRA)
- Injects trainable low-rank matrices into specific model layers
- Only **LoRA adapters** are trained; base model stays frozen
- Drastically reduces trainable parameters while maintaining performance

### QLoRA Configuration

| Component | Setting |
|-----------|--------|
| Quantization | 4-bit NF4 |
| Compute dtype | bfloat16 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target layers | q_proj, k_proj, v_proj, o_proj |

### Hardware & Training Time

- **GPU:** NVIDIA RTX 4060 (8GB VRAM)
- **Total Training Time:** ~13 hours
- **Memory Usage:** Fits entirely in 8GB VRAM thanks to QLoRA

**Result:** Train a 0.5B parameter model on consumer-grade hardware with minimal memory usage.

---

## Training Techniques

### SDPA Attention
Uses PyTorch Scaled Dot-Product Attention for efficient attention computation.

### Gradient Checkpointing
Reduces memory usage by recomputing activations during backprop.

### Gradient Accumulation
Simulates larger batch size (effective batch = 48).

### 8-bit Optimizer
`paged_adamw_8bit` reduces optimizer memory footprint.

---

## Hyperparameters

| Parameter | Value |
|----------|------|
| Max length | 512 |
| Batch/device | 6 |
| Grad accumulation | 8 |
| LR | 1.8e-4 |
| Scheduler | Cosine |
| Warmup | 4% |
| Steps | 6000 |
| Epochs | ~2 |
| Precision | bf16 |
| Clip norm | 1.0 |

---

## Results

### Training Loss Curves

<img width="691" height="468" alt="image" src="https://github.com/user-attachments/assets/ce69015a-3e44-45e3-9439-176e723be13c" />

### Validation
**Loss:** 2.8583  
**Perplexity:** 17.43

### Test
**Loss:** 2.8501  
**Perplexity:** 17.29

---

## Qualitative Generated Example
```diff
diff --git a/src/client/core/commands/menu.js b/src/client/core/commands/menu.js
index <HASH>..<HASH> 100644
--- a/src/client/core/commands/menu.js
+++ b/src/client/core/commands/menu.js
@@ -6,14 +6,21 @@ define([
     'core/panels',
     'core/tabs',
     'core/session',
-    'core/localfs'
-], function (_, hr, MenubarView, box, panels, tabs, session, localfs) {
+    'core/localfs',
+    'core/settings'
+], function (_, hr, MenubarView, box, panels, tabs, session, localfs, settings) {
     // Collection for all menu commands
     var menu = new MenubarView();
     
     menu.register("view", {
         title: "View",
         position: 5
+    }).menuSection({
+        'id': "themes.settings",
+        'title': "Settings",
+        'action': function() {
+            settings.open("themes"...
```

**Ground Truth:**  
Add command to open themes settings in view menu

**Model Output:**  
Add theme settings to the menu

The model correctly:
- Recognizes a menu item insertion
- Understands purpose of change
- Generates concise commit-style description

---

## Model Checkpoint

The trained model is saved in: `qwen2.5-coder-0.5b-qlora/checkpoint-6000/`

This folder contains the standard checkpoint files created by the Hugging Face Transformers library:

### Key Files

| File | Description |
|------|-------------|
| `adapter_model.safetensors` | Trained LoRA adapter weights (the actual fine-tuned parameters) |
| `adapter_config.json` | LoRA configuration (rank, alpha, target modules, etc.) |
| `optimizer.pt` | Optimizer state for resuming training |
| `scheduler.pt` | Learning rate scheduler state |
| `trainer_state.json` | Training metrics, loss history, and step count |
| `training_args.bin` | All training hyperparameters used |
| `tokenizer_config.json` | Tokenizer configuration |
| `tokenizer.json` | Full tokenizer vocabulary and merges |

**Note:** The base model weights are NOT included—only the LoRA adapters. To use this checkpoint, load the base `Qwen2.5-Coder-0.5B` model and apply these adapters.

---

## Pipeline Overview

1. Load dataset  
2. Clean/filter samples  
3. Tokenize with masked prompt labels  
4. Load Qwen2.5-Coder in 4-bit  
5. Prepare k-bit training  
6. Apply LoRA adapters
7. Enable gradient checkpointing  
8. Train with cosine scheduler  
9. Evaluate loss + perplexity  
10. Generate qualitative samples  
11. Save LoRA checkpoint

---

## Key Takeaways

- **QLoRA** enables large-model tuning on consumer hardware (8GB VRAM) with minimal quality loss
- Dataset quality strongly affects performance  
- Perplexity ≈ 17 indicates strong modeling for this task
- Only LoRA adapters (~few MB) need to be saved, not the entire model
- Full training completed in ~13 hours on a single RTX 4060
