"""
Stage 1: SFT on Intel Gaudi 2 (HPU) — no bitsandbytes.

Full bf16 LoRA on Qwen2.5-Coder-7B-Instruct using 4-card HPU DDP.
Gaudi 2 has enough HBM to fit the 7B model in bf16 without quantization.

Launch via finetune_gaudi.sh or directly:
    python gaudi_spawn.py --nproc_per_node 4 finetune/train_sft_gaudi.py \
        --project_dir /scratch/$USER/ADAPT-SQL-Text2SQL \
        --checkpoint_dir /scratch/$USER/finetune_checkpoints_gaudi \
        --hf_cache /scratch/$USER/hf_cache
"""
import argparse
import json
import os
import sqlite3
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LEN = 2048


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--hf_cache", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--data_file", default=None)
    return p.parse_args()


def get_schema_string(db_path: str) -> str:
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
            cursor.execute(f"PRAGMA foreign_key_list({table})")
            for fk in cursor.fetchall():
                lines.append(f"-- FK: {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
        conn.close()
        return "\n".join(lines)
    except Exception:
        return ""


def load_spider_train(project_dir: str) -> list[dict]:
    spider_root = Path(project_dir) / "data" / "spider" / "spider_data"
    with open(spider_root / "train_spider.json", "r", encoding="utf-8") as f:
        examples = json.load(f)
    db_dir = spider_root / "database"
    records, skipped = [], 0
    for ex in examples:
        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            skipped += 1
            continue
        schema = get_schema_string(str(db_path))
        if not schema:
            skipped += 1
            continue
        prompt = f"### Schema:\n{schema}\n\n### Question:\n{ex['question']}\n\n### SQL:\n"
        records.append({"text": prompt + ex["query"] + "\n"})
    print(f"[SFT] Loaded {len(records)} examples ({skipped} skipped)")
    return records


def find_latest_checkpoint(checkpoint_dir: str) -> str | None:
    base = Path(checkpoint_dir) / "sft"
    if not base.exists():
        return None
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    sft_output = Path(args.checkpoint_dir) / "sft"
    sft_output.mkdir(parents=True, exist_ok=True)

    # ── Detect device ────────────────────────────────────────────────────────
    use_bf16 = False
    try:
        import habana_frameworks.torch.core as htcore  # noqa: F401
        device = torch.device("hpu")   # Gaudi doesn't support hpu:N indexing
        use_bf16 = True
        if is_main:
            print("[SFT] Running on Intel Gaudi (HPU)")
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            use_bf16 = True
        else:
            device = torch.device("cpu")
            use_bf16 = False
        if is_main:
            print(f"[SFT] Habana not found, falling back to {device}")
    if is_main:
        print(f"[SFT] bf16={use_bf16}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    if args.data_file:
        if is_main:
            print(f"[SFT] Loading pipeline-format data from {args.data_file}...")
        with open(args.data_file) as f:
            records = json.load(f)
        if is_main:
            print(f"[SFT] Loaded {len(records)} records")
    else:
        records = load_spider_train(args.project_dir)
    dataset = Dataset.from_list(records)

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=args.hf_cache, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Model (bf16, no quantization) ────────────────────────────────────────
    if is_main:
        print(f"[SFT] Loading {MODEL_ID} in bf16...")
    # low_cpu_mem_usage=True loads shards one at a time — avoids each of the
    # 4 DDP processes holding a full 14GB copy in CPU RAM simultaneously.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=args.hf_cache,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if is_main:
        model.print_trainable_parameters()

    # ── Resume ───────────────────────────────────────────────────────────────
    resume_from = find_latest_checkpoint(args.checkpoint_dir)
    if resume_from and is_main:
        print(f"[SFT] Resuming from {resume_from}")

    # ── Training args ────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=str(sft_output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=use_bf16,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    if is_main:
        final_path = sft_output / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"[SFT] Saved final model to {final_path}")


if __name__ == "__main__":
    main()
