from datasets import load_dataset
from pynvml import *
import random
import re
from statistics import mean

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


print_gpu_utilization()

# ----------------------------
# Load + inspect dataset
# ----------------------------
ds = load_dataset("Maxscha/commitbench")

print("Splits:", list(ds.keys()))
for split in ds.keys():
    print(f"{split}: {len(ds[split])} rows, columns={ds[split].column_names}")


def show_random_examples(split_name, k=3, seed=42):
    rnd = random.Random(seed)
    split = ds[split_name]
    print(f"\nRandom examples from {split_name}:")
    for idx in rnd.sample(range(len(split)), k=min(k, len(split))):
        ex = split[idx]
        print(f"\nIndex {idx}")
        for key, val in ex.items():
            if isinstance(val, str):
                preview = val[:400].replace("\r\n", "\n").replace("\r", "\n")
                print(f"- {key}: {preview!r}")


for split in ds.keys():
    show_random_examples(split, k=3)


# Heuristic inference: diff vs commit message

def looks_like_diff(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r"^diff --git ",
        r"^@@ ",
        r"^\+\+\+ ",
        r"^--- ",
        r"^index ",
        r"^[+-]",
    ]
    return any(re.search(p, text, flags=re.MULTILINE) for p in patterns)


def looks_like_message(text: str) -> bool:
    if not text:
        return False
    if looks_like_diff(text):
        return False
    return len(text.split()) <= 200


def infer_columns(example: dict):
    scores = {}
    for k, v in example.items():
        if not isinstance(v, str):
            continue
        diff_score = 0
        msg_score = 0
        if looks_like_diff(v):
            diff_score += 2
        if looks_like_message(v):
            msg_score += 1
        if len(v) > 500:
            diff_score += 1
        if len(v) < 300:
            msg_score += 1
        scores[k] = (diff_score, msg_score)

    if not scores:
        return None, None

    diff_col = max(scores, key=lambda k: scores[k][0])
    msg_col = max(scores, key=lambda k: scores[k][1])

    if diff_col == msg_col:
        msg_candidates = sorted(scores.keys(), key=lambda k: scores[k][1], reverse=True)
        if len(msg_candidates) > 1:
            msg_col = msg_candidates[1]

    return diff_col, msg_col


sample_split = "train" if "train" in ds else list(ds.keys())[0]
diff_col, msg_col = infer_columns(ds[sample_split][0])
print(f"\nInferred columns -> diff: {diff_col}, message: {msg_col}")

# Use inferred columns when possible, otherwise fall back
DIFF_COL = diff_col or "diff"
MSG_COL = msg_col or "message"


# Basic length stats (raw)

def length_stats(split, col):
    vals = [len(x) for x in split[col] if isinstance(x, str)]
    return {
        "min": min(vals) if vals else 0,
        "mean": mean(vals) if vals else 0,
        "max": max(vals) if vals else 0,
    }


print("\nRaw length stats:")
for split_name, split in ds.items():
    if DIFF_COL in split.column_names:
        print(f"- {split_name} diff {length_stats(split, DIFF_COL)}")
    if MSG_COL in split.column_names:
        print(f"- {split_name} msg  {length_stats(split, MSG_COL)}")


# ----------------------------
# Clean + reduce dataset
# ----------------------------
MIN_DIFF_CHARS = 50
MAX_DIFF_CHARS = 8000
MIN_MSG_CHARS = 3

BAD_EXACT = {
    "update", "updated", "fix", "fixed", "wip", ".", "..", "...", "temp", "test",
}

REF_ONLY_RE = re.compile(
    r"^\s*(fixe[sd]?|close[sd]?|resolve[sd]?|ref[s]?)\s*#?\w+.*$",
    re.IGNORECASE
)


def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def is_bad_message(msg: str) -> bool:
    m = msg.strip()
    if not m:
        return True
    if m.lower() in BAD_EXACT:
        return True
    if len(m) < MIN_MSG_CHARS:
        return True
    if len(m.split()) < 3:
        return True
    if REF_ONLY_RE.match(m):
        return True
    return False


def keep_example(ex) -> bool:
    d = normalize_newlines(ex.get(DIFF_COL, "") or "")
    m = normalize_newlines(ex.get(MSG_COL, "") or "")

    if len(d) < MIN_DIFF_CHARS:
        return False
    if len(d) > MAX_DIFF_CHARS:
        return False
    if is_bad_message(m):
        return False
    return True


def preprocess(ex):
    ex[DIFF_COL] = normalize_newlines(ex.get(DIFF_COL, "") or "")
    ex[MSG_COL] = normalize_newlines(ex.get(MSG_COL, "") or "")
    return ex


ds_clean = ds.map(preprocess)
ds_clean = ds_clean.filter(keep_example)

print("After cleaning:")
for split in ds.keys():
    before = len(ds[split])
    after = len(ds_clean[split])
    pct = 100.0 * after / before if before else 0.0
    print(f"- {split}: {after}/{before} kept ({pct:.2f}%)")

# Reduce dataset size for faster training
TRAIN_SAMPLES = 120000
VAL_SAMPLES = 15000
TEST_SAMPLES = 15000

ds_clean["train"] = ds_clean["train"].shuffle(seed=42).select(range(TRAIN_SAMPLES))
ds_clean["validation"] = ds_clean["validation"].shuffle(seed=42).select(range(VAL_SAMPLES))
ds_clean["test"] = ds_clean["test"].shuffle(seed=42).select(range(TEST_SAMPLES))

print("\nReduced dataset:")
print(f"- train: {len(ds_clean['train'])} samples")
print(f"- validation: {len(ds_clean['validation'])} samples")
print(f"- test: {len(ds_clean['test'])} samples")


# ----------------------------
# Load model + tokenizer (QLoRA)
# ----------------------------
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",  # PyTorch SDPA (not FlashAttention2)
)

model.eval()


# Sanity check: approximate token counts using the tokenizer

def estimate_token_stats_raw(split, sample_size=20000, seed=42):
    rnd = random.Random(seed)
    n = len(split)
    if n == 0:
        return {"samples": 0, "avg_tokens": 0, "min_tokens": 0, "max_tokens": 0, "est_total_tokens": 0}

    idxs = rnd.sample(range(n), k=min(sample_size, n))
    lengths = []
    for i in idxs:
        diff = split[i][DIFF_COL]
        msg = split[i][MSG_COL]
        full_text = diff + "\n\n" + msg
        lengths.append(len(tokenizer.encode(full_text)))

    avg_tokens = sum(lengths) / len(lengths)
    return {
        "samples": len(lengths),
        "avg_tokens": round(avg_tokens, 2),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "est_total_tokens": int(avg_tokens * n),
    }


print("\nToken length sanity check (sampled, raw diff+msg):")
for split_name in ["train", "validation", "test"]:
    stats = estimate_token_stats_raw(ds_clean[split_name])
    print(f"- {split_name}: {stats}")

print_gpu_utilization()


# ----------------------------
# Tokenize for causal LM training (raw continuation)
# ----------------------------
MAX_LENGTH = 384

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SEP = "\n\n"

def tokenize_batch(batch):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for diff, msg in zip(batch[DIFF_COL], batch[MSG_COL]):
        prompt_text = diff
        full_text = diff + SEP + msg

        prompt_ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]

        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        prompt_len = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:len(input_ids)]

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


tokenized_train = ds_clean["train"].map(
    tokenize_batch,
    batched=True,
    remove_columns=ds_clean["train"].column_names,
)

tokenized_val = ds_clean["validation"].map(
    tokenize_batch,
    batched=True,
    remove_columns=ds_clean["validation"].column_names,
)

tokenized_test = ds_clean["test"].map(
    tokenize_batch,
    batched=True,
    remove_columns=ds_clean["test"].column_names,
)


# ----------------------------
# LoRA setup
# ----------------------------
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.gradient_checkpointing_enable()
model.config.use_cache = False  # required with gradient checkpointing


# ----------------------------
# Training setup
# ----------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

OUTPUT_DIR = "qwen2.5-coder-0.5b-qlora"
PER_DEVICE_TRAIN_BATCH = 4
PER_DEVICE_EVAL_BATCH = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=50,
    save_steps=700,
    eval_steps=700,
    eval_strategy="steps",
    save_strategy="steps",
    fp16=False,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=4,   
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=1.0,
    group_by_length=True,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)


# ----------------------------
# Full training run
# ----------------------------
train_result = trainer.train()
print(train_result)


# Save metrics
import json as _json
from pathlib import Path as _Path

metrics_path = _Path(OUTPUT_DIR) / "metrics.json"
metrics_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.write_text(_json.dumps(trainer.state.log_history, indent=2), encoding="utf-8")
print(f"Saved metrics to {metrics_path}")


# ----------------------------
# Quick inference test
# ----------------------------
model.eval()
model.config.use_cache = True

test_diff = """diff --git a/src/auth.py b/src/auth.py
index a1b2c3d..e4f5g6h 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -42,7 +42,11 @@ def validate_token(token):
-    if not token:
-        return False
+    if token is None:
+        return False
+
+    token = token.strip()
+    if token == "":
+        return False
 
     return token in ACTIVE_TOKENS

"""

inputs = tokenizer(test_diff, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
commit_message = generated_text[len(test_diff):].strip()
print("Generated Commit Message:")
print(commit_message)
