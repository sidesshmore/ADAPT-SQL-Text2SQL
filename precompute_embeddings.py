"""
Pre-compute embeddings for Spider dev/test queries on local Mac (fast),
then upload to SOL so eval nodes skip slow CPU embedding.

Usage:
    python precompute_embeddings.py --split dev   # produces vector_store/dev_query_embeddings.pkl
    python precompute_embeddings.py --split test  # produces vector_store/test_query_embeddings.pkl
    python precompute_embeddings.py --split all   # both

Upload to SOL:
    scp vector_store/dev_query_embeddings.pkl  smore123@sol.rc.asu.edu:/scratch/smore123/ADAPT-SQL-Text2SQL/vector_store/
    scp vector_store/test_query_embeddings.pkl smore123@sol.rc.asu.edu:/scratch/smore123/ADAPT-SQL-Text2SQL/vector_store/
"""
import argparse
import json
import pickle  # matches checkpoint format used across the pipeline
import re
import time
from pathlib import Path

import numpy as np
import ollama

PROJECT_DIR = Path(__file__).parent
SPIDER_DATA_DIR = PROJECT_DIR / "data/spider/spider_data"

SPLIT_CONFIG = {
    "dev": {
        "json": PROJECT_DIR / "data/spider/dev.json",
        "out": PROJECT_DIR / "vector_store/dev_query_embeddings.pkl",
    },
    "test": {
        "json": SPIDER_DATA_DIR / "test.json",
        "out": PROJECT_DIR / "vector_store/test_query_embeddings.pkl",
    },
}

_SKELETON_KEYWORDS = {
    'SELECT', 'FROM', 'JOIN', 'LEFT', 'RIGHT', 'INNER', 'OUTER', 'CROSS',
    'WHERE', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'DISTINCT',
    'COUNT', 'SUM', 'AVG', 'MAX', 'MIN', 'NOT', 'IN', 'EXISTS',
    'UNION', 'INTERSECT', 'EXCEPT', 'AS', 'ON', 'AND', 'OR',
}

def extract_sql_skeleton(sql: str) -> str:
    tokens = re.findall(r'\b[A-Za-z_]\w*\b', sql.upper())
    return ' '.join(t for t in tokens if t in _SKELETON_KEYWORDS)

def get_embedding(text: str, model: str = "nomic-embed-text") -> np.ndarray:
    response = ollama.embeddings(model=model, prompt=text)
    return np.array(response['embedding'], dtype=np.float32)

def precompute(split: str):
    cfg = SPLIT_CONFIG[split]
    with open(cfg["json"]) as f:
        data = json.load(f)

    print(f"\n[{split}] Pre-computing embeddings for {len(data)} queries...")
    cache = {}
    errors = 0
    t0 = time.time()

    for i, example in enumerate(data):
        question = example["question"]
        gold_sql = example.get("query", "")
        skeleton = extract_sql_skeleton(gold_sql)
        embed_text = f"{question} {skeleton}".strip() if skeleton else question

        for key in [question, embed_text]:
            if key not in cache:
                try:
                    cache[key] = get_embedding(key)
                except Exception as e:
                    print(f"  [warn] idx={i}: {e}")
                    errors += 1

        if (i + 1) % 100 == 0 or i == len(data) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(data)}] {elapsed:.0f}s elapsed  ETA {eta:.0f}s  errors={errors}")

    out_path = cfg["out"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"[{split}] Saved {len(cache)} embeddings to {out_path}  (errors={errors})")
    return out_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all",
                        help="Which split to precompute (default: all)")
    args = parser.parse_args()

    splits = ["dev", "test"] if args.split == "all" else [args.split]
    paths = []
    for split in splits:
        paths.append(precompute(split))

    print("\nUpload to SOL:")
    for p in paths:
        print(f"  scp {p} smore123@sol.rc.asu.edu:/scratch/smore123/ADAPT-SQL-Text2SQL/vector_store/")

if __name__ == "__main__":
    main()
