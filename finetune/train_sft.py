"""
Stage 1: Supervised Fine-Tuning (SFT) on Spider train set.

QLoRA on Qwen2.5-Coder-32B-Instruct using 4-GPU DDP via torchrun.
Auto-resumes from latest checkpoint in checkpoint_dir.

Launch via finetune_sol.sh or directly:
    torchrun --nproc_per_node=4 --master_port=29500 finetune/train_sft.py \
        --project_dir /scratch/$USER/ADAPT-SQL-Text2SQL \
        --checkpoint_dir /scratch/$USER/finetune_checkpoints \
        --hf_cache /scratch/$USER/hf_cache
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM


MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_SEQ_LEN = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--hf_cache", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    return p.parse_args()


# ── Schema helpers ──────────────────────────────────────────────────────────

def get_schema_string(db_path: str) -> str:
    """Return a compact CREATE TABLE representation for the database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        lines = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            lines.append(f"CREATE TABLE {table} ({col_defs});")
            # Foreign keys
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            for fk in cursor.fetchall():
                lines.append(f"-- FK: {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
        conn.close()
        return "\n".join(lines)
    except Exception:
        return ""


def build_prompt(question: str, schema: str) -> str:
    return (
        f"### Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )


# ── Dataset building ─────────────────────────────────────────────────────────

def load_spider_train(project_dir: str) -> list[dict]:
    spider_root = Path(project_dir) / "data" / "spider" / "spider_data"
    train_json = spider_root / "train_spider.json"
    db_dir = spider_root / "database"

    with open(train_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    records = []
    skipped = 0
    for ex in examples:
        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            skipped += 1
            continue
        schema = get_schema_string(str(db_path))
        if not schema:
            skipped += 1
            continue
        prompt = build_prompt(ex["question"], schema)
        full_text = prompt + ex["query"] + "\n"
        records.append({"text": full_text, "prompt": prompt, "completion": ex["query"] + "\n"})

    print(f"[SFT] Loaded {len(records)} training examples ({skipped} skipped, no DB)")
    return records


# ── Checkpoint resume ────────────────────────────────────────────────────────

def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    base = Path(checkpoint_dir) / "sft"
    if not base.exists():
        return None
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    sft_output = Path(args.checkpoint_dir) / "sft"
    sft_output.mkdir(parents=True, exist_ok=True)

    # ── Load dataset ────────────────────────────────────────────────────────
    if is_main:
        print(f"[SFT] Loading Spider training data...")
    records = load_spider_train(args.project_dir)
    dataset = Dataset.from_list(records)

    # ── Quantization config ─────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── Model + tokenizer ───────────────────────────────────────────────────
    if is_main:
        print(f"[SFT] Loading {MODEL_ID}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=args.hf_cache, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": local_rank},
        cache_dir=args.hf_cache,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA ────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # ── Data collator — train only on completions ────────────────────────────
    response_template = "### SQL:\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # ── Resume ──────────────────────────────────────────────────────────────
    resume_from = find_latest_checkpoint(args.checkpoint_dir)
    if resume_from and is_main:
        print(f"[SFT] Resuming from {resume_from}")

    # ── TrainingArguments ───────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(sft_output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        tf32=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    # ── Trainer ─────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    if is_main:
        final_path = sft_output / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"[SFT] Saved final model to {final_path}")


if __name__ == "__main__":
    main()
