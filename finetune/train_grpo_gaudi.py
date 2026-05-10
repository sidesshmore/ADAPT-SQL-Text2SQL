"""
Stage 2: GRPO on Intel Gaudi 2 (HPU) — no bitsandbytes.

Full bf16 GRPO on Qwen2.5-Coder-7B-Instruct using 4-card HPU DDP.

Launch via finetune_gaudi.sh or directly:
    python gaudi_spawn.py --nproc_per_node 4 finetune/train_grpo_gaudi.py \
        --project_dir /scratch/$USER/ADAPT-SQL-Text2SQL \
        --checkpoint_dir /scratch/$USER/finetune_checkpoints_gaudi \
        --hf_cache /scratch/$USER/hf_cache
"""
import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    return f"### Schema:\n{schema}\n\n### Question:\n{question}\n\n### SQL:\n"


def load_grpo_dataset(project_dir: str) -> list[dict]:
    spider_root = Path(project_dir) / "data" / "spider" / "spider_data"
    with open(spider_root / "train_spider.json", "r", encoding="utf-8") as f:
        examples = json.load(f)
    db_dir = spider_root / "database"
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


def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    if "### SQL:" in text:
        return text.split("### SQL:")[-1].strip().split("\n")[0].strip()
    return text.strip().split("\n")[0].strip()


def _execute_sql(sql: str, db_path: str, timeout: int = 10):
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


def compute_rewards(completions, prompts, metadata):
    rewards = []
    for completion, meta in zip(completions, metadata):
        sql = _extract_sql(completion)
        db_path = meta["db_path"]
        gold_sql = meta["gold_sql"]

        has_select = bool(re.search(r"\bSELECT\b", sql, re.IGNORECASE))
        format_r = 0.1 if has_select else 0.0

        ok, pred_rows = _execute_sql(sql, db_path)
        execute_r = 0.3 if ok else 0.0

        result_r = 0.0
        if ok:
            _, gold_rows = _execute_sql(gold_sql, db_path)
            if isinstance(gold_rows, set) and pred_rows == gold_rows:
                result_r = 0.5

        gold_len = len(gold_sql.split())
        pred_len = len(sql.split())
        length_r = -0.1 if pred_len > 2 * gold_len else 0.0

        rewards.append(format_r + execute_r + result_r + length_r)
    return rewards


def reward_fn(completions, prompts=None, db_path=None, gold_sql=None, **kwargs):
    batch_size = len(completions)
    if db_path is None or gold_sql is None:
        return [0.0] * batch_size
    metadata = [{"db_path": db_path[i], "gold_sql": gold_sql[i]} for i in range(batch_size)]
    return compute_rewards(completions, prompts or [""] * batch_size, metadata)


def find_latest_checkpoint(base: Path) -> str | None:
    checkpoints = sorted(base.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1]) if checkpoints else None


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    os.environ["HF_HOME"] = args.hf_cache
    os.environ["TRANSFORMERS_CACHE"] = args.hf_cache

    sft_final = Path(args.checkpoint_dir) / "sft" / "final"
    grpo_output = Path(args.checkpoint_dir) / "grpo"
    grpo_output.mkdir(parents=True, exist_ok=True)

    # ── Detect device ────────────────────────────────────────────────────────
    use_bf16 = False
    try:
        import habana_frameworks.torch.core as htcore  # noqa: F401
        device = torch.device("hpu")   # Gaudi doesn't support hpu:N indexing
        use_bf16 = True
        if is_main:
            print("[GRPO] Running on Intel Gaudi (HPU)")
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            use_bf16 = True
        else:
            device = torch.device("cpu")
            use_bf16 = False
        if is_main:
            print(f"[GRPO] Habana not found, falling back to {device}")
    if is_main:
        print(f"[GRPO] bf16={use_bf16}")

    if is_main:
        print(f"[GRPO] Loading SFT adapter from {sft_final}")

    tokenizer = AutoTokenizer.from_pretrained(str(sft_final), trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=args.hf_cache,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model = base_model.to(device)
    model = PeftModel.from_pretrained(base_model, str(sft_final), is_trainable=True)

    records = load_grpo_dataset(args.project_dir)
    dataset = Dataset.from_list(records)

    resume_from = find_latest_checkpoint(grpo_output)
    if resume_from and is_main:
        print(f"[GRPO] Resuming from {resume_from}")

    grpo_config = GRPOConfig(
        output_dir=str(grpo_output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=use_bf16,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        num_generations=2,
        gradient_checkpointing=True,
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
