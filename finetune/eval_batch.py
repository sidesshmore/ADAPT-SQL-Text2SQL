"""
Standalone batch evaluator for fine-tuned adapt-sql-coder model.
No Streamlit dependency — runs as a plain Python script in sbatch jobs.

Usage:
    python finetune/eval_batch.py \
        --start 0 --num 259 \
        --model adapt-sql-coder \
        --ollama_host http://127.0.0.1:11437 \
        --checkpoint_dir ./batch_results/range_0_259 \
        --project_dir /scratch/smore123/ADAPT-SQL-Text2SQL
"""
import argparse
import json
import pickle  # required: must match format used by merge_checkpoints.py and batch_utils.py
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def get_schema_from_sqlite(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    schema_dict = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        schema_dict[table] = [
            {"column_name": row[1], "data_type": row[2], "is_nullable": "YES" if row[3] == 0 else "NO"}
            for row in cursor.fetchall()
        ]
    conn.close()
    return schema_dict


def get_foreign_keys_from_sqlite(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    foreign_keys = []
    for table in tables:
        cursor.execute(f"PRAGMA foreign_key_list({table})")
        for row in cursor.fetchall():
            foreign_keys.append({"from_table": table, "from_column": row[3], "to_table": row[2], "to_column": row[4]})
    conn.close()
    return foreign_keys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--num", type=int, required=True)
    p.add_argument("--model", default="adapt-sql-coder")
    p.add_argument("--ollama_host", default="http://127.0.0.1:11437")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--project_dir", required=True)
    p.add_argument("--checkpoint_every", type=int, default=25)
    return p.parse_args()


def save_checkpoint(results, checkpoint_dir, final=False):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"final_checkpoint_{ts}.pkl" if final else f"checkpoint_{len(results)}_{ts}.pkl"
    data = {
        "results": results,
        "timestamp": ts,
        "num_results": len(results),
        "saved_at": ts,
        "is_final": final,
    }
    path = checkpoint_dir / name
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[checkpoint] Saved {len(results)} results → {path}")
    return path


def main():
    args = parse_args()
    project_dir = Path(args.project_dir)
    sys.path.insert(0, str(project_dir))

    import os
    os.environ["OLLAMA_HOST"] = args.ollama_host

    from core.adapt_baseline import ADAPTBaseline

    spider_json = project_dir / "data/spider/dev.json"
    spider_db_dir = project_dir / "data/spider/spider_data/database"

    with open(spider_json) as f:
        spider_data = json.load(f)

    end = min(args.start + args.num, len(spider_data))
    total = end - args.start
    print(f"[eval] Range {args.start}–{end-1} ({total} examples) | model={args.model} | host={args.ollama_host}")

    adapt = ADAPTBaseline(
        model=args.model,
        ollama_host=args.ollama_host,
        vector_store_path=str(project_dir / "vector_store"),
    )

    # Resume from latest checkpoint if one exists
    results = []
    processed = set()
    checkpoint_dir = Path(args.checkpoint_dir)
    existing = sorted(checkpoint_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime) if checkpoint_dir.exists() else []
    if existing:
        with open(existing[-1], "rb") as f:
            saved = pickle.load(f)
        results = saved["results"]
        processed = {r["index"] for r in results}
        print(f"[resume] Loaded {len(results)} results from {existing[-1].name}")

    for i in range(args.start, end):
        if i in processed:
            continue

        example = spider_data[i]
        db_path = spider_db_dir / example["db_id"] / f"{example['db_id']}.sqlite"

        if not db_path.exists():
            print(f"[warn] DB not found: {example['db_id']}, skipping")
            continue

        try:
            schema_dict = get_schema_from_sqlite(str(db_path))
            foreign_keys = get_foreign_keys_from_sqlite(str(db_path))
        except Exception as e:
            print(f"[warn] Schema error for {example['db_id']}: {e}, skipping")
            continue

        try:
            result = adapt.run_full_pipeline(
                example["question"],
                schema_dict,
                foreign_keys,
                k_examples=3,
                enable_retry=True,
                db_path=str(db_path),
                gold_sql=example.get("query"),
                enable_execution=True,
                enable_evaluation=True,
                enable_multi_candidate=True,
            )
            results.append({"index": i, "example": example, "result": result, "retry_result": None})
            processed.add(i)

            n = len(results)
            done = i - args.start + 1
            ex = result.get("step11", {}).get("execution_accuracy", 0) if result.get("step11") else 0
            print(f"[{done}/{total}] idx={i} ex={ex} q={example['question'][:60]}")

            if n % args.checkpoint_every == 0:
                save_checkpoint(results, args.checkpoint_dir)

        except Exception as e:
            print(f"[error] idx={i}: {e}")

    save_checkpoint(results, args.checkpoint_dir, final=True)
    print(f"[eval] Done. {len(results)}/{total} examples completed.")


if __name__ == "__main__":
    main()
