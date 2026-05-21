"""
Local eval runner using ASU Voyager API.
Reads VOYAGER_API_KEY from .env automatically.

Usage:
    python eval_voyager.py --split dev --start 0 --num 1034
    python eval_voyager.py --split test --start 0 --num 2147
"""
import argparse
import json
import os
import pickle  # required: matches checkpoint format used by merge_checkpoints.py
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Load .env manually if VOYAGER_API_KEY not already in environment (local dev)
if not os.environ.get("VOYAGER_API_KEY"):
    env_file = PROJECT_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

os.environ.setdefault("VOYAGER_BASE_URL", "https://openai.rc.asu.edu/v1")
os.environ.setdefault("VOYAGER_MODEL", "qwen3-coder-30b-a3b-instruct")

SPIDER_DATA_DIR = PROJECT_DIR / "data/spider/spider_data"

SPLIT_CONFIG = {
    "dev": {
        "json": PROJECT_DIR / "data/spider/dev.json",
        "db_dir": SPIDER_DATA_DIR / "database",
    },
    "test": {
        "json": SPIDER_DATA_DIR / "test.json",
        "db_dir": SPIDER_DATA_DIR / "test_database",
    },
}


def get_schema(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t] = [
            {"column_name": r[1], "data_type": r[2], "is_nullable": "YES" if r[3] == 0 else "NO"}
            for r in cur.fetchall()
        ]
    conn.close()
    return schema


def get_foreign_keys(db_path: str) -> list:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    fks = []
    for t in tables:
        cur.execute(f"PRAGMA foreign_key_list({t})")
        for r in cur.fetchall():
            fks.append({"from_table": t, "from_column": r[3], "to_table": r[2], "to_column": r[4]})
    conn.close()
    return fks


def save_checkpoint(results, out_dir, final=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = f"final_checkpoint_{ts}.pkl" if final else f"checkpoint_{len(results)}_{ts}.pkl"
    with open(out_dir / name, "wb") as f:
        pickle.dump({"results": results, "timestamp": ts, "num_results": len(results), "is_final": final}, f)
    print(f"[checkpoint] {len(results)} results -> {out_dir / name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Which Spider split to evaluate (dev=1034, test=2147)")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num", type=int, default=50)
    parser.add_argument("--model", default=None,
                        help="Voyager model name (overrides VOYAGER_MODEL env var). "
                             "Auto-organizes checkpoints under eval_results_voyager/<model>/<split>/range_<start>_<end>/")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Override checkpoint dir (default: eval_results_voyager/<model>/<split>/range_<start>_<end>)")
    parser.add_argument("--checkpoint_every", type=int, default=25)
    parser.add_argument("--k_examples", type=int, default=10,
                        help="Few-shot examples from vector store (0 = skip embeddings)")
    parser.add_argument("--proxy", action="store_true",
                        help="Use data/spider/dev_proxy.json (200-query autoresearch subset)")
    args = parser.parse_args()

    if not os.environ.get("VOYAGER_API_KEY"):
        print("[error] VOYAGER_API_KEY not set. Check your .env file.")
        sys.exit(1)

    # --model overrides env var
    if args.model:
        os.environ["VOYAGER_MODEL"] = args.model

    from core.adapt_baseline import ADAPTBaseline
    from ui.enhanced_retry_engine import EnhancedRetryEngine

    if args.proxy:
        spider_json = PROJECT_DIR / "data/spider/dev_proxy.json"
        spider_db_dir = SPLIT_CONFIG["dev"]["db_dir"]
        print(f"[proxy-eval] Using 200-query proxy subset: {spider_json}")
    else:
        cfg = SPLIT_CONFIG[args.split]
        spider_json = cfg["json"]
        spider_db_dir = cfg["db_dir"]

    vector_store_path = str(PROJECT_DIR / "vector_store")

    with open(spider_json) as f:
        spider_data = json.load(f)

    if args.proxy:
        end = len(spider_data)
        total = end
        print(f"[proxy-eval] {total} examples (set-op wrong + other wrong + correct guard)")
    else:
        end = min(args.start + args.num, len(spider_data))
        total = end - args.start

    model = os.environ["VOYAGER_MODEL"]

    # Auto checkpoint dir: eval_results_voyager/<model-slug>/<split>/range_<start>_<end>
    if args.checkpoint_dir is not None:
        ckpt_dir = Path(args.checkpoint_dir)
    elif args.proxy:
        model_slug = model.replace("/", "___")
        ckpt_dir = PROJECT_DIR / "eval_results_voyager" / model_slug / "dev" / "proxy"
    else:
        model_slug = model.replace("/", "___")
        ckpt_dir = PROJECT_DIR / "eval_results_voyager" / model_slug / args.split / f"range_{args.start}_{end}"

    if args.proxy:
        print(f"[proxy-eval] Model: {model} @ {os.environ['VOYAGER_BASE_URL']}")
    else:
        print(f"[voyager-eval] Split={args.split} Range={args.start}-{end-1} ({total} examples)")
        print(f"[voyager-eval] Model: {model} @ {os.environ['VOYAGER_BASE_URL']}")
    print(f"[voyager-eval] Checkpoint dir: {ckpt_dir}")

    adapt = ADAPTBaseline(model=model, vector_store_path=vector_store_path)
    retry_engine = EnhancedRetryEngine(model=model, max_full_retries=2)

    # Resume from checkpoint if exists
    results = []
    processed = set()
    existing = sorted(ckpt_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime) if ckpt_dir.exists() else []
    if existing:
        saved = pickle.load(open(existing[-1], "rb"))
        results = saved["results"]
        processed = {r["index"] for r in results}
        print(f"[resume] {len(results)} results from {existing[-1].name}")

    if args.proxy:
        for pos, example in enumerate(spider_data):
            orig_idx = example["original_index"]
            if orig_idx in processed:
                continue
            db_path = spider_db_dir / example["db_id"] / f"{example['db_id']}.sqlite"
            if not db_path.exists():
                print(f"[warn] DB not found: {example['db_id']}, skipping")
                continue
            try:
                schema = get_schema(str(db_path))
                fks = get_foreign_keys(str(db_path))
            except Exception as e:
                print(f"[warn] Schema error {example['db_id']}: {e}")
                continue
            try:
                retry_output = retry_engine.run_with_full_retry(
                    adapt_baseline=adapt,
                    natural_query=example["question"],
                    schema_dict=schema,
                    foreign_keys=fks,
                    k_examples=args.k_examples,
                    db_path=str(db_path),
                    gold_sql=example.get("query"),
                )
                result = retry_output["final_result"]
                results.append({"index": orig_idx, "example": example, "result": result, "retry_result": retry_output})
                processed.add(orig_idx)

                done = pos + 1
                ex = result.get("step11", {}).get("execution_accuracy", 0) if result.get("step11") else 0
                correct = sum(r.get("result", {}).get("step11", {}).get("execution_accuracy", False) for r in results)
                print(f"[{done}/{total}] orig={orig_idx} ex={ex} EX={correct}/{len(results)}={correct/len(results)*100:.1f}% q={example['question'][:55]}")

                if len(results) % args.checkpoint_every == 0:
                    save_checkpoint(results, ckpt_dir)

            except Exception as e:
                print(f"[error] orig={orig_idx}: {e}")

        save_checkpoint(results, ckpt_dir, final=True)
        correct = sum(r.get("result", {}).get("step11", {}).get("execution_accuracy", False) for r in results)
        print(f"\n[done] Proxy EX: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")
        return

    for i in range(args.start, end):
        if i in processed:
            continue
        example = spider_data[i]
        db_path = spider_db_dir / example["db_id"] / f"{example['db_id']}.sqlite"
        if not db_path.exists():
            print(f"[warn] DB not found: {example['db_id']}, skipping")
            continue
        try:
            schema = get_schema(str(db_path))
            fks = get_foreign_keys(str(db_path))
        except Exception as e:
            print(f"[warn] Schema error {example['db_id']}: {e}")
            continue
        try:
            retry_output = retry_engine.run_with_full_retry(
                adapt_baseline=adapt,
                natural_query=example["question"],
                schema_dict=schema,
                foreign_keys=fks,
                k_examples=args.k_examples,
                db_path=str(db_path),
                gold_sql=example.get("query"),
            )
            result = retry_output["final_result"]
            results.append({"index": i, "example": example, "result": result, "retry_result": retry_output})
            processed.add(i)

            done = i - args.start + 1
            ex = result.get("step11", {}).get("execution_accuracy", 0) if result.get("step11") else 0
            correct = sum(r.get("result", {}).get("step11", {}).get("execution_accuracy", False) for r in results)
            print(f"[{done}/{total}] idx={i} ex={ex} EX={correct}/{len(results)}={correct/len(results)*100:.1f}% q={example['question'][:55]}")

            if len(results) % args.checkpoint_every == 0:
                save_checkpoint(results, ckpt_dir)

        except Exception as e:
            print(f"[error] idx={i}: {e}")

    save_checkpoint(results, ckpt_dir, final=True)
    correct = sum(r.get("result", {}).get("step11", {}).get("execution_accuracy", False) for r in results)
    print(f"\n[done] Split={args.split} EX: {correct}/{len(results)} = {correct/len(results)*100:.1f}%")


if __name__ == "__main__":
    main()
