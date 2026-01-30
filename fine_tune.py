import math
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
DIFF_COL = "diff"
MSG_COL = "message"


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


PLACEHOLDER_RE = re.compile(r"<HASH>|<URL>|#<I>|\(#<I>\)")

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

def _cap(n, limit):
    return min(n, limit)

ds_clean["train"] = ds_clean["train"].shuffle(seed=9105).select(
    range(_cap(len(ds_clean["train"]), TRAIN_SAMPLES))
)
ds_clean["validation"] = ds_clean["validation"].shuffle(seed=9105).select(
    range(_cap(len(ds_clean["validation"]), VAL_SAMPLES))
)
ds_clean["test"] = ds_clean["test"].shuffle(seed=9105).select(
    range(_cap(len(ds_clean["test"]), TEST_SAMPLES))
)

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
    dtype=torch.bfloat16,
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
MAX_LENGTH = 512

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SEP = "\n\nCommit message:\n"

def tokenize_batch(batch):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for diff, msg in zip(batch[DIFF_COL], batch[MSG_COL]):
        prompt_text = diff + SEP
        full_text   = diff + SEP + msg + tokenizer.eos_token

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

def _assert_nonempty(split, name):
    n = len(split)
    if n == 0:
        raise ValueError(f"{name} split is empty after filtering/selection. Check thresholds or sample caps.")
    return n

_assert_nonempty(tokenized_train, "train")
_assert_nonempty(tokenized_val, "validation")

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj"
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
PER_DEVICE_TRAIN_BATCH = 6
PER_DEVICE_EVAL_BATCH = 6
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1.8e-4 # the starting lr was 2e-4
NUM_EPOCHS = 2

train_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    lr_scheduler_type="cosine",
    warmup_ratio=0.04,
    logging_steps=30,
    max_steps=6000,
    save_steps=300,
    eval_steps=300,
    eval_strategy="steps",
    save_strategy="steps",
    fp16=False,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=8,   
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


# Full training run
train_result = trainer.train()   
print(train_result)



# Evaluate validation
print("\nEvaluating on validation set...")
val_metrics = trainer.evaluate(eval_dataset=tokenized_val, metric_key_prefix="val")
val_loss = val_metrics.get("val_loss")
val_ppl = math.exp(val_loss) if val_loss is not None else None

# Evaluate test
print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(eval_dataset=tokenized_test, metric_key_prefix="test")
test_loss = test_metrics.get("test_loss")
test_ppl = math.exp(test_loss) if test_loss is not None else None

print("Validation metrics:", val_metrics)
print("Validation perplexity:", val_ppl)
print("Test metrics:", test_metrics)
print("Test perplexity:", test_ppl)


# Testing on random samples from the test sets
def sample_eval_tokenized(tokenized_test, tokenizer, model, n=5, max_new_tokens=64):
    print(f"\n\nSampling {n} examples from the test set for qualitative evaluation:")

    rnd = random.Random()
    idxs = rnd.sample(range(len(tokenized_test)), k=min(n, len(tokenized_test)))

    model.eval()
    for idx in idxs:
        ex = tokenized_test[idx]
        input_ids = ex["input_ids"]
        labels = ex["labels"]

        # prompt length = leading -100 labels
        prompt_len = 0
        for l in labels:
            if l == -100:
                prompt_len += 1
            else:
                break

        if prompt_len >= len(labels):
            print(f"Index {idx}: skipped (no target tokens)")
            continue

        prompt_ids = input_ids[:prompt_len]
        target_ids = [l for l in labels if l != -100]

        # Generate from prompt_ids directly
        inputs = {
            "input_ids": torch.tensor([prompt_ids], device=model.device),
            "attention_mask": torch.tensor([[1] * len(prompt_ids)], device=model.device),
        }
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,        # greedy decoding
                num_beams=1,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )


        gen_ids = outputs[0].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        pred_msg = gen_text[len(prompt_text):].strip()
        gt_msg = tokenizer.decode(target_ids, skip_special_tokens=True).strip()

        print("=" * 80)
        print(f"Index: {idx}")
        print("- Prompt (truncated):")
        print(prompt_text[:800] + ("..." if len(prompt_text) > 800 else ""))
        print("\n- Ground truth:")
        print(gt_msg)
        print("\n- Model output:")
        print(pred_msg)

# Usage:
sample_eval_tokenized(
    tokenized_test=tokenized_test,
    tokenizer=tokenizer,
    model=model,
    n=2,
    max_new_tokens=64,
)



