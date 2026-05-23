"""
Analyze the 104 remaining failures after all rescore passes.
Look for patterns we haven't tried yet.
"""
import pickle, sqlite3, re, os, sys
from pathlib import Path

CKPT_BASE = Path("eval_results_voyager/qwen3-235b-a22b-instruct-2507/dev")
DB_BASE = Path("data/spider/spider_data/database")

def load_all():
    records = []
    for rdir in sorted(CKPT_BASE.glob("range_*")):
        finals = sorted(rdir.glob("final_checkpoint_*.pkl"))
        if not finals:
            continue
        with open(finals[-1], 'rb') as f:
            data = pickle.load(f)
        records.extend(data if isinstance(data, list) else [data])
    return records

def exec_sql(db_path, sql, timeout=10):
    try:
        conn = sqlite3.connect(db_path, timeout=timeout)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql)
        rows = [tuple(r) for r in cur.fetchall()]
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)

def normalize_rows(rows):
    if rows is None:
        return None
    return set(str(r) for r in rows)

records = load_all()
print(f"Loaded {len(records)} records")

failures = []
for entry in records:
    result = entry.get("result", {})
    step11 = result.get("step11", {})
    ex = step11.get("execution_accuracy", False) if step11 else False
    if not ex:
        example = entry.get("example", {})
        final_sql = result.get("final_sql", "")
        db_id = example.get("db_id", "")
        question = example.get("question", "")
        gold = example.get("query", "")
        idx = entry.get("index", -1)
        failures.append({
            "idx": idx, "db_id": db_id, "question": question,
            "gold": gold, "pred": final_sql
        })

print(f"Failures: {len(failures)}")
print()

# Categorize failures
cats = {
    "pred_empty": [],
    "gold_0_rows": [],
    "pred_error": [],
    "cardinality_mismatch": [],
    "set_op_missing": [],
    "other": [],
}

for f in failures:
    db_path = str(DB_BASE / f["db_id"] / f"{f['db_id']}.sqlite")
    if not os.path.exists(db_path):
        cats["other"].append(f)
        continue
    
    gold_rows, gold_err = exec_sql(db_path, f["gold"])
    pred_rows, pred_err = exec_sql(db_path, f["pred"])
    
    f["gold_rows"] = gold_rows
    f["pred_rows"] = pred_rows
    f["pred_err"] = pred_err
    
    if not f["pred"]:
        cats["pred_empty"].append(f)
    elif pred_err:
        cats["pred_error"].append(f)
    elif gold_rows is not None and len(gold_rows) == 0:
        cats["gold_0_rows"].append(f)
    elif gold_rows is not None and pred_rows is not None:
        gn, pn = len(gold_rows), len(pred_rows)
        gold_norm = normalize_rows(gold_rows)
        pred_norm = normalize_rows(pred_rows)
        gold_sql_up = f["gold"].upper()
        if gold_norm == pred_norm:
            # Actually matches? Mark as oddity
            cats["other"].append({**f, "note": "rows_match_but_ex_false"})
        elif any(kw in gold_sql_up for kw in ['INTERSECT', 'EXCEPT', 'UNION']):
            cats["set_op_missing"].append(f)
        elif gn != pn:
            cats["cardinality_mismatch"].append({**f, "gold_n": gn, "pred_n": pn})
        else:
            cats["other"].append(f)
    else:
        cats["other"].append(f)

print("=== FAILURE CATEGORIES ===")
for cat, items in cats.items():
    print(f"  {cat}: {len(items)}")

print()
print("=== CARDINALITY MISMATCHES (sample) ===")
for f in cats["cardinality_mismatch"][:5]:
    print(f"  idx={f['idx']} db={f['db_id']} gold_n={f['gold_n']} pred_n={f['pred_n']}")
    print(f"  Q: {f['question'][:80]}")
    print(f"  PRED: {f['pred'][:120]}")
    print(f"  GOLD: {f['gold'][:120]}")
    print()

print()
print("=== PRED ERRORS ===")
for f in cats["pred_error"][:5]:
    print(f"  idx={f['idx']} db={f['db_id']} err={f['pred_err'][:80]}")
    print(f"  Q: {f['question'][:80]}")
    print()

print()
print("=== SET-OP MISSING ===")
for f in cats["set_op_missing"][:5]:
    print(f"  idx={f['idx']} db={f['db_id']}")
    print(f"  Q: {f['question'][:80]}")
    print(f"  GOLD SET-OP: {re.search(r'INTERSECT|UNION|EXCEPT', f['gold'].upper()).group() if re.search(r'INTERSECT|UNION|EXCEPT', f['gold'].upper()) else 'N/A'}")
    print()
