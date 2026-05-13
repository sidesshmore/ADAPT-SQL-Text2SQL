"""
Stage 2: GRPO (Group Relative Policy Optimization) fine-tuning.

Loads the SFT adapter from Stage 1 and continues training with a
4-component reward: format correctness, SQL executability, result
match against gold, and length penalty.

Launch via finetune_smore123.sh or directly:
    python finetune/train_grpo.py \
        --project_dir /scratch/$USER/ADAPT-SQL-Text2SQL \
        --checkpoint_dir /scratch/$USER/finetune_checkpoints \
        --hf_cache /scratch/$USER/hf_cache

NOTE: bitsandbytes 4-bit quantization is incompatible with DDP — quantized
weights cannot be synchronized across ranks. Run on a single GPU only.
"""
import argparse
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer


MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--hf_cache", required=True)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    return p.parse_args()


# ── Schema helpers (same as SFT) ────────────────────────────────────────────

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


def build_prompt(question: str, schema: str) -> str:
    return (
        f"### Schema:\n{schema}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )


# ── Dataset ──────────────────────────────────────────────────────────────────

def load_grpo_dataset(project_dir: str) -> list[dict]:
    """Load Spider train data for GRPO (prompt only — reward comes from execution)."""
    spider_root = Path(project_dir) / "data" / "spider" / "spider_data"
    train_json = spider_root / "train_spider.json"
    db_dir = spider_root / "database"

    with open(train_json, "r", encoding="utf-8") as f:
        examples = json.load(f)

    records = []
    for ex in examples:
        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            continue
        schema = get_schema_string(str(db_path))
        if not schema:
            continue
        records.append({
            "prompt": build_prompt(ex["question"], schema),
            "gold_sql": ex["query"],
            "db_path": str(db_path),
        })

    print(f"[GRPO] Loaded {len(records)} examples for RL training")
    return records


# ── Reward functions ─────────────────────────────────────────────────────────

def _extract_sql(text: str) -> str:
    """Pull the first SQL statement from generated text."""
    # Try markdown code block first
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: everything after ### SQL: marker or just take as-is
    if "### SQL:" in text:
        return text.split("### SQL:")[-1].strip().split("\n")[0].strip()
    return text.strip().split("\n")[0].strip()


def _execute_sql(sql: str, db_path: str, timeout: int = 10):
    """Execute SQL against SQLite DB. Returns (success, results)."""
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.execute("PRAGMA query_only = ON")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()
        return True, set(str(r) for r in rows)
    except Exception as e:
        return False, str(e)


def compute_rewards(completions: list[str], prompts: list[str], metadata: list[dict]) -> list[float]:
    """
    4-component reward per completion:
      1. format_reward    [0, 0.1]  — ends with a parseable SQL statement
      2. execute_reward   [0, 0.3]  — SQL executes without error
      3. result_reward    [0, 0.5]  — execution results match gold
      4. length_penalty   [-0.1, 0] — penalise unnecessarily long outputs
    """
    rewards = []
    for completion, meta in zip(completions, metadata):
        sql = _extract_sql(completion)
        db_path = meta["db_path"]
        gold_sql = meta["gold_sql"]

        # 1. Format reward
        has_select = bool(re.search(r"\bSELECT\b", sql, re.IGNORECASE))
        format_r = 0.1 if has_select else 0.0

        # 2. Execute reward
        ok, pred_rows = _execute_sql(sql, db_path)
        execute_r = 0.3 if ok else 0.0

        # 3. Result match reward
        result_r = 0.0
        if ok:
            _, gold_rows = _execute_sql(gold_sql, db_path)
            if isinstance(gold_rows, set) and pred_rows == gold_rows:
                result_r = 0.5

        # 4. Length penalty — penalise if >2× gold length
        gold_len = len(gold_sql.split())
        pred_len = len(sql.split())
        length_r = -0.1 if pred_len > 2 * gold_len else 0.0

        rewards.append(format_r + execute_r + result_r + length_r)

    return rewards


# ── Reward wrapper for GRPOTrainer ───────────────────────────────────────────

def reward_fn(completions, prompts=None, db_path=None, gold_sql=None, **kwargs):
    """
    Reward function for GRPOTrainer.
    TRL passes extra dataset columns (db_path, gold_sql) as kwargs per batch.
    """
    batch_size = len(completions)
    if db_path is None or gold_sql is None:
        return [0.0] * batch_size
    metadata = [{"db_path": db_path[i], "gold_sql": gold_sql[i]} for i in range(batch_size)]
    return compute_rewards(completions, prompts or [""] * batch_size, metadata)


# ── Resume helper ────────────────────────────────────────────────────────────

def find_latest_checkpoint(base: Path) -> str | None:
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    sft_final = Path(args.checkpoint_dir) / "sft" / "final"
    grpo_output = Path(args.checkpoint_dir) / "grpo"
    grpo_output.mkdir(parents=True, exist_ok=True)

    if is_main:
        print(f"[GRPO] Loading SFT adapter from {sft_final}")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        str(sft_final), trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # GRPO needs left padding for generation

    # ── Load model ───────────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=args.hf_cache,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, str(sft_final), is_trainable=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    records = load_grpo_dataset(args.project_dir)
    # db_path and gold_sql are kept as dataset columns so TRL forwards
    # them per-batch to reward_fn via **kwargs — fixes the metadata bug.
    dataset = Dataset.from_list(records)

    # ── GRPO config ──────────────────────────────────────────────────────────
    resume_from = find_latest_checkpoint(grpo_output)
    if resume_from and is_main:
        print(f"[GRPO] Resuming from {resume_from}")

    grpo_config = GRPOConfig(
        output_dir=str(grpo_output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        max_completion_length=MAX_NEW_TOKENS,
        temperature=0.7,
        num_generations=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    trainer.train(resume_from_checkpoint=resume_from)

    if is_main:
        final_path = grpo_output / "final"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        print(f"[GRPO] Saved final model to {final_path}")


if __name__ == "__main__":
    main()
