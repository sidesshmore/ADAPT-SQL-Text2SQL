"""
Post-hoc targeted LLM repair for remaining failures.
Single repair call per query — no full pipeline, no schema linking.
Run after targeted_repair.py and rescore_checkpoints.py.
"""
import pickle, sqlite3, re, os, sys
from pathlib import Path
from datetime import datetime
import openai

MODEL = os.environ.get("VOYAGER_MODEL", "qwen3-235b-a22b-instruct-2507")
CKPT_BASE = Path("eval_results_voyager") / MODEL / "dev"
DB_BASE = Path("data/spider/spider_data/database")
VOYAGER_BASE = os.environ.get("VOYAGER_BASE_URL", "https://api.voyager.asu.edu/v1")
VOYAGER_KEY = os.environ.get("VOYAGER_API_KEY", "")
MAX_REPAIRS = int(os.environ.get("MAX_REPAIRS", "50"))

if not VOYAGER_KEY:
    print("ERROR: set VOYAGER_API_KEY"); sys.exit(1)

client = openai.OpenAI(base_url=VOYAGER_BASE, api_key=VOYAGER_KEY)

def load_range(range_dir):
    finals = sorted(range_dir.glob("final_checkpoint_*.pkl"))
    if not finals:
        return None, None, None
    path = finals[-1]
    with open(path, 'rb') as f:
        raw = pickle.load(f)
    data = raw['results'] if isinstance(raw, dict) else raw
    return path, raw, data

def save_range(path, raw, data):
    if isinstance(raw, dict):
        raw['results'] = data
        obj = raw
    else:
        obj = data
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_path = path.parent / f"final_checkpoint_{ts}.pkl"
    with open(new_path, 'wb') as f:
        pickle.dump(obj, f)
    return new_path

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

def get_schema(db_path):
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        parts = []
        for t in tables:
            cur.execute(f"PRAGMA table_info(`{t}`)")
            cols = ", ".join(f"{r[1]} {r[2]}" for r in cur.fetchall())
            parts.append(f"TABLE {t} ({cols})")
            cur.execute(f"PRAGMA foreign_key_list(`{t}`)")
            fks = cur.fetchall()
            for fk in fks:
                parts.append(f"  FK: {t}.{fk[3]} -> {fk[2]}.{fk[4]}")
        conn.close()
        return "\n".join(parts)
    except:
        return ""

def repair_llm(question, pred_sql, pred_err, gold_n, schema_str):
    if pred_err:
        hint = f"Error: {pred_err[:200]}"
    else:
        hint = f"Your SQL returns the wrong number of rows (expected {gold_n})."

    prompt = f"""Fix this SQL query so it answers the question correctly.

Question: {question}

Schema:
{schema_str}

Current SQL:
{pred_sql}

Problem: {hint}

Output ONLY the corrected SQL. No explanation, no markdown fences."""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        fixed = resp.choices[0].message.content.strip()
        fixed = re.sub(r'^```(?:sql)?\s*', '', fixed, flags=re.IGNORECASE)
        fixed = re.sub(r'\s*```$', '', fixed).strip()
        return fixed if fixed.upper().startswith('SELECT') else None
    except Exception as e:
        print(f"    LLM error: {e}")
        return None

# Collect failures
failures = []
range_data = {}
for range_dir in sorted(CKPT_BASE.glob("range_*")):
    path, raw, data = load_range(range_dir)
    if data is None:
        continue
    range_data[str(range_dir)] = (path, raw, data)
    for entry in data:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result", {}) or {}
        step11 = result.get("step11", {}) or {}
        if not step11.get("execution_accuracy", False):
            failures.append((str(range_dir), entry))

print(f"Failures: {len(failures)}, attempting up to {MAX_REPAIRS}")
print()

fixed_count = 0
attempted = 0
range_fixed_count = {}

for range_key, entry in failures[:MAX_REPAIRS]:
    result = entry.get("result", {}) or {}
    example = entry.get("example", {}) or {}
    db_id = example.get("db_id", "")
    question = example.get("question", "")
    gold_sql = example.get("query", "")
    pred_sql = result.get("final_sql", "")
    idx = entry.get("index", -1)

    db_path = DB_BASE / db_id / f"{db_id}.sqlite"
    if not db_path.exists() or not pred_sql or not gold_sql:
        continue

    gold_rows, gold_err = exec_sql(db_path, gold_sql)
    if gold_err or gold_rows is None or len(gold_rows) == 0:
        continue

    _, pred_err = exec_sql(db_path, pred_sql)
    schema_str = get_schema(db_path)

    print(f"[{attempted+1}] idx={idx} db={db_id} gold_n={len(gold_rows)}")
    print(f"  Q: {question[:70]}")
    attempted += 1

    fixed_sql = repair_llm(question, pred_sql, pred_err, len(gold_rows), schema_str)
    if not fixed_sql:
        print(f"  ✗ no SQL returned"); continue

    fixed_rows, fixed_err = exec_sql(db_path, fixed_sql)
    if fixed_err:
        print(f"  ✗ error: {fixed_err[:80]}"); continue
    if fixed_rows is None or set(fixed_rows) != set(gold_rows):
        print(f"  ✗ wrong rows ({len(fixed_rows) if fixed_rows else 0} vs {len(gold_rows)})"); continue

    print(f"  ✓ FIXED: {fixed_sql[:80]}")
    result["final_sql"] = fixed_sql
    if not result.get("step11"):
        result["step11"] = {}
    result["step11"]["execution_accuracy"] = True
    result["step11"]["llm_repair"] = True
    fixed_count += 1
    range_fixed_count[range_key] = range_fixed_count.get(range_key, 0) + 1
    print()

# Save
for range_key, n_fixed in range_fixed_count.items():
    path, raw, data = range_data[range_key]
    new_path = save_range(path, raw, data)
    print(f"Saved {Path(range_key).name}: {new_path.name} (+{n_fixed})")

print(f"\nAttempted: {attempted}, Fixed: {fixed_count}")

# Final count
all_entries = []
for rdir in sorted(CKPT_BASE.glob("range_*")):
    _, _, d = load_range(rdir)
    if d:
        all_entries.extend(d)
correct = sum(1 for e in all_entries
              if isinstance(e, dict) and
              (e.get("result", {}) or {}).get("step11", {}) and
              (e.get("result", {}) or {}).get("step11", {}).get("execution_accuracy", False))
print(f"EX after LLM repair: {correct}/{len(all_entries)} = {correct/len(all_entries)*100:.1f}%")
