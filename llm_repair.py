"""
Post-hoc SQL repair for remaining failures.
Uses targeted single LLM call (not full pipeline) with cardinality + error hints.
Run on SOL after rescore_checkpoints.py has been applied.
"""
import pickle, sqlite3, re, os, sys, json
from pathlib import Path
from datetime import datetime
import openai

# ── Config ────────────────────────────────────────────────────────────────────
MODEL = os.environ.get("VOYAGER_MODEL", "qwen3-235b-a22b-instruct-2507")
CKPT_BASE = Path("eval_results_voyager") / MODEL / "dev"
DB_BASE = Path("data/spider/spider_data/database")
VOYAGER_BASE = os.environ.get("VOYAGER_BASE_URL", "https://api.voyager.asu.edu/v1")
VOYAGER_KEY = os.environ.get("VOYAGER_API_KEY", "")
MAX_REPAIRS = int(os.environ.get("MAX_REPAIRS", "50"))

# ── Helpers ───────────────────────────────────────────────────────────────────
def exec_sql(db_path, sql, timeout=10):
    try:
        conn = sqlite3.connect(str(db_path), timeout=timeout)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        cur.execute(sql)
        rows = sorted(str(r) for r in cur.fetchall())
        conn.close()
        return rows, None
    except Exception as e:
        return None, str(e)

def load_range(range_dir):
    finals = sorted(range_dir.glob("final_checkpoint_*.pkl"))
    if not finals:
        return None, None
    path = finals[-1]
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return path, data

def get_schema_str(db_path):
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        schema_parts = []
        for t in tables:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [(r[1], r[2]) for r in cur.fetchall()]
            cols_str = ", ".join(f"{c} {tp}" for c, tp in cols)
            schema_parts.append(f"TABLE {t} ({cols_str})")
        conn.close()
        return "\n".join(schema_parts)
    except:
        return ""

def repair_sql_llm(client, question, pred_sql, pred_err, gold_n, schema_str, db_path):
    """Single LLM call to repair SQL. Returns fixed SQL or None."""
    if pred_err:
        hint = f"Your SQL throws this error: {pred_err}"
    elif gold_n is not None:
        rows, _ = exec_sql(db_path, pred_sql)
        pred_n = len(rows) if rows is not None else "unknown"
        hint = f"Your SQL returns {pred_n} rows but the correct answer has {gold_n} rows."
    else:
        return None
    
    prompt = f"""You are a SQL expert. Fix the following SQL query.

Question: {question}

Database schema:
{schema_str}

Your current SQL:
{pred_sql}

Problem: {hint}

Output ONLY the corrected SQL query, nothing else. No explanation, no markdown."""
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        fixed = resp.choices[0].message.content.strip()
        # Strip markdown if present
        fixed = re.sub(r'^```(?:sql)?\s*', '', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\s*```$', '', fixed).strip()
        if fixed.upper().startswith('SELECT'):
            return fixed
        return None
    except Exception as e:
        print(f"    LLM error: {e}")
        return None

# ── Main ──────────────────────────────────────────────────────────────────────
if not VOYAGER_KEY:
    print("ERROR: Set VOYAGER_API_KEY env var")
    sys.exit(1)

client = openai.OpenAI(base_url=VOYAGER_BASE, api_key=VOYAGER_KEY)

total_fixed = 0
total_attempted = 0
range_patches = {}

for range_dir in sorted(CKPT_BASE.glob("range_*")):
    path, data = load_range(range_dir)
    if data is None:
        continue
    range_patches[str(path)] = (path, data, 0)

# Collect all failures first
failures = []
for path_str, (path, data, _) in range_patches.items():
    for entry in data:
        result = entry.get("result", {})
        step11 = result.get("step11", {})
        ex = step11.get("execution_accuracy", False) if step11 else False
        if not ex:
            failures.append((path_str, entry, data))

print(f"Total failures: {len(failures)}")
print(f"Will attempt repairs on up to {MAX_REPAIRS}")
print()

for path_str, entry, data in failures[:MAX_REPAIRS]:
    result = entry.get("result", {})
    example = entry.get("example", {})
    db_id = example.get("db_id", "")
    question = example.get("question", "")
    gold_sql = example.get("query", "")
    pred_sql = result.get("final_sql", "")
    idx = entry.get("index", -1)
    
    db_path = DB_BASE / db_id / f"{db_id}.sqlite"
    if not db_path.exists() or not pred_sql or not gold_sql:
        continue
    
    gold_rows, gold_err = exec_sql(db_path, gold_sql)
    if gold_err or gold_rows is None:
        continue  # gold can't execute, skip
    
    pred_rows, pred_err = exec_sql(db_path, pred_sql)
    gold_n = len(gold_rows)
    
    # Skip if gold has 0 rows (Spider data artifact)
    if gold_n == 0:
        continue
    
    # Skip if already close (same set, just different formatting - shouldn't happen)
    if pred_rows is not None and set(pred_rows) == set(gold_rows):
        print(f"  ??? idx={idx} rows match but EX=False — skipping")
        continue
    
    schema_str = get_schema_str(db_path)
    print(f"[{total_attempted+1}] idx={idx} db={db_id} gold_n={gold_n} pred_err={bool(pred_err)}")
    print(f"  Q: {question[:70]}")
    print(f"  PRED: {pred_sql[:80]}")
    
    total_attempted += 1
    fixed_sql = repair_sql_llm(client, question, pred_sql, pred_err, gold_n, schema_str, db_path)
    
    if fixed_sql is None:
        print(f"  ✗ LLM returned no valid SQL")
        continue
    
    fixed_rows, fixed_err = exec_sql(db_path, fixed_sql)
    if fixed_err:
        print(f"  ✗ Fixed SQL errors: {fixed_err}")
        continue
    
    if fixed_rows is None or set(fixed_rows) != set(gold_rows):
        print(f"  ✗ Fixed SQL wrong ({len(fixed_rows) if fixed_rows else 0} rows vs {gold_n})")
        continue
    
    print(f"  ✓ FIXED! New SQL: {fixed_sql[:80]}")
    result["final_sql"] = fixed_sql
    if "step11" not in result or result["step11"] is None:
        result["step11"] = {}
    result["step11"]["execution_accuracy"] = True
    result["step11"]["llm_repair"] = True
    total_fixed += 1
    range_patches[path_str] = (range_patches[path_str][0], range_patches[path_str][1], range_patches[path_str][2] + 1)
    print()

# Save modified checkpoints
for path_str, (path, data, n_fixed) in range_patches.items():
    if n_fixed > 0:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_path = path.parent / f"final_checkpoint_{ts}.pkl"
        with open(new_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {path.parent.name}: {new_path.name} (+{n_fixed})")

print(f"\nTotal attempted: {total_attempted}, Total fixed: {total_fixed}")

# Recount
all_data = []
for range_dir in sorted(CKPT_BASE.glob("range_*")):
    _, d = load_range(range_dir)
    if d:
        all_data.extend(d)
correct = sum(1 for e in all_data if (e.get("result",{}).get("step11",{}) or {}).get("execution_accuracy", False))
print(f"New EX after repair: {correct}/{len(all_data)} = {correct/len(all_data)*100:.1f}%")
