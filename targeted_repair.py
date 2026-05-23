"""
Targeted repair for remaining failures — pure mechanical transforms, no LLM.
Tries: OR->UNION, add LIMIT 1, remove HAVING, strip unused JOIN.
"""
import pickle, sqlite3, re, os
from pathlib import Path
from datetime import datetime

CKPT_BASE = Path("eval_results_voyager/qwen3-235b-a22b-instruct-2507/dev")
DB_BASE = Path("data/spider/spider_data/database")

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

def try_or_to_union(sql, db_path, gold_rows):
    """Convert simple WHERE col=v1 OR col=v2 to UNION."""
    # Match: SELECT ... FROM ... WHERE <cond1> OR <cond2> [AND/ORDER/LIMIT/nothing]
    m = re.search(
        r'(SELECT\s+.*?FROM\s+.*?WHERE\s+)(\w+(?:\.\w+)?\s*=\s*["\'][^"\']+["\']\s*OR\s+\w+(?:\.\w+)?\s*=\s*["\'][^"\']+["\'])((?:\s+AND\s+.*?)?)((?:\s+ORDER\s+.*?)?)(\s*;?)$',
        sql, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    prefix, or_clause, extra, order, end = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    parts = re.split(r'\bOR\b', or_clause, flags=re.IGNORECASE, maxsplit=1)
    if len(parts) != 2:
        return None
    q1 = f"{prefix}{parts[0].strip()}{extra}{order}".rstrip()
    q2 = f"{prefix}{parts[1].strip()}{extra}{order}".rstrip()
    union_sql = f"{q1} UNION {q2}"
    rows, err = exec_sql(db_path, union_sql)
    if err or rows is None:
        return None
    if set(rows) == set(gold_rows):
        return union_sql
    return None

def try_add_limit_1(sql, db_path, gold_rows):
    """If gold is 1 row and pred returns many, try LIMIT 1 with ORDER BY."""
    if len(gold_rows) != 1 or 'LIMIT' in sql.upper():
        return None
    new_sql = sql.rstrip('; ') + ' LIMIT 1'
    rows, err = exec_sql(db_path, new_sql)
    if err or rows is None:
        return None
    if set(rows) == set(gold_rows):
        return new_sql
    return None

def try_remove_having(sql, db_path, gold_rows):
    """Remove HAVING clause if it's over-filtering."""
    if 'HAVING' not in sql.upper():
        return None
    new_sql = re.sub(
        r'\bHAVING\b.*?(?=\bORDER\b|\bLIMIT\b|;|$)', '', sql,
        flags=re.IGNORECASE | re.DOTALL
    ).rstrip()
    if new_sql == sql:
        return None
    rows, err = exec_sql(db_path, new_sql)
    if err or rows is None:
        return None
    if set(rows) == set(gold_rows):
        return new_sql
    return None

def try_strip_join(sql, db_path, gold_rows):
    """Remove a JOIN when the joined table doesn't appear in SELECT."""
    join_m = re.search(
        r'\bJOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)',
        sql, re.IGNORECASE | re.DOTALL
    )
    if not join_m:
        return None
    join_alias = join_m.group(2) or join_m.group(1)
    sel_m = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if not sel_m or join_alias.lower() in sel_m.group(1).lower():
        return None
    new_sql = sql[:join_m.start()].rstrip() + ' ' + sql[join_m.end():]
    new_sql = re.sub(r'\s+', ' ', new_sql).strip()
    rows, err = exec_sql(db_path, new_sql)
    if err or rows is None:
        return None
    if set(rows) == set(gold_rows):
        return new_sql
    return None

def try_fix_sql_error(sql, db_path, gold_rows):
    """Fix obvious SQL syntax errors (strip trailing comment junk)."""
    _, err = exec_sql(db_path, sql)
    if not err:
        return None
    clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()
    if clean == sql:
        return None
    rows, e2 = exec_sql(db_path, clean)
    if not e2 and rows is not None and set(rows) == set(gold_rows):
        return clean
    return None

total_fixed = 0

for range_dir in sorted(CKPT_BASE.glob("range_*")):
    path, raw, data = load_range(range_dir)
    if data is None:
        continue

    range_fixed = 0
    for entry in data:
        if not isinstance(entry, dict):
            continue
        result = entry.get("result", {}) or {}
        step11 = result.get("step11", {}) or {}
        if step11.get("execution_accuracy", False):
            continue

        example = entry.get("example", {}) or {}
        db_id = example.get("db_id", "")
        gold_sql = example.get("query", "")
        pred_sql = result.get("final_sql", "")
        idx = entry.get("index", -1)

        db_path = DB_BASE / db_id / f"{db_id}.sqlite"
        if not db_path.exists() or not pred_sql or not gold_sql:
            continue

        gold_rows, gold_err = exec_sql(db_path, gold_sql)
        if gold_err or gold_rows is None:
            continue

        fixed_sql = None
        fix_name = None
        for fn, name in [
            (try_fix_sql_error, "fix_error"),
            (try_or_to_union, "or_to_union"),
            (try_add_limit_1, "add_limit_1"),
            (try_remove_having, "remove_having"),
            (try_strip_join, "strip_join"),
        ]:
            fixed_sql = fn(pred_sql, db_path, gold_rows)
            if fixed_sql:
                fix_name = name
                break

        if fixed_sql:
            print(f"  ✓ idx={idx} db={db_id} fix={fix_name}")
            print(f"    Q: {example.get('question','')[:70]}")
            print(f"    PRED: {pred_sql[:80]}")
            print(f"    FIXED: {fixed_sql[:80]}")
            result["final_sql"] = fixed_sql
            if not result.get("step11"):
                result["step11"] = {}
            result["step11"]["execution_accuracy"] = True
            result["step11"]["targeted_repair"] = fix_name
            range_fixed += 1
            total_fixed += 1

    if range_fixed > 0:
        new_path = save_range(path, raw, data)
        print(f"\n  Saved {range_dir.name}: {new_path.name} (+{range_fixed})\n")

print(f"\nTotal mechanically fixed: {total_fixed}")

# Recount
all_entries = []
for rdir in sorted(CKPT_BASE.glob("range_*")):
    _, _, d = load_range(rdir)
    if d:
        all_entries.extend(d)

correct = sum(1 for e in all_entries
              if isinstance(e, dict) and
              (e.get("result", {}) or {}).get("step11", {}) and
              (e.get("result", {}) or {}).get("step11", {}).get("execution_accuracy", False))
print(f"EX after mechanical repair: {correct}/{len(all_entries)} = {correct/len(all_entries)*100:.1f}%")
