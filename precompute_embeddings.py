"""
Pre-compute embeddings for all Spider dev queries on local Mac (fast GPU/CPU),
then upload the cache to SOL so eval nodes don't need to embed on CPU.

Usage:
    python precompute_embeddings.py
    # produces: vector_store/dev_query_embeddings.pkl
    # then: scp vector_store/dev_query_embeddings.pkl smore123@sol.rc.asu.edu:/scratch/smore123/ADAPT-SQL-Text2SQL/vector_store/
"""
import json
import os
import pickle  # required: matches checkpoint format used across the pipeline
import re
import time
from pathlib import Path

import numpy as np
import ollama

PROJECT_DIR = Path(__file__).parent

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

def main():
    spider_json = PROJECT_DIR / "data/spider/dev.json"
    out_path = PROJECT_DIR / "vector_store/dev_query_embeddings.pkl"

    with open(spider_json) as f:
        dev_data = json.load(f)

    print(f"Pre-computing embeddings for {len(dev_data)} dev queries...")
    cache = {}
    errors = 0
    t0 = time.time()

    for i, example in enumerate(dev_data):
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

        if (i + 1) % 50 == 0 or i == len(dev_data) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(dev_data) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(dev_data)}] {elapsed:.0f}s elapsed  ETA {eta:.0f}s  errors={errors}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cache, f)

    print(f"\nSaved {len(cache)} embeddings to {out_path}")
    print(f"Errors: {errors}")
    print(f"\nNext step — upload to SOL:")
    print(f"  scp {out_path} smore123@sol.rc.asu.edu:/scratch/smore123/ADAPT-SQL-Text2SQL/vector_store/")

if __name__ == "__main__":
    main()
