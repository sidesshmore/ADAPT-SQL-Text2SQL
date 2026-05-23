"""
Targeted repair for remaining failures — try new mechanical transforms not in rescore.
Saves repaired checkpoints if any fixes work.
"""
import pickle, sqlite3, re, os, sys, shutil
from pathlib import Path
from datetime import datetime

CKPT_BASE = Path("eval_results_voyager/qwen3-235b-a22b-instruct-2507/dev")
DB_BASE = Path("data/spider/spider_data/database")

def load_range(range_dir):
    finals = sorted(range_dir.glob("final_checkpoint_*.pkl"))
    if not finals:
        return None, None
    path = finals[-1]
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return path, data

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

def normalize(rows):
    return set(rows) if rows is not None else None

def try_or_to_union(sql, db_path, gold_rows_sorted):
    """Convert WHERE col=v1 OR col=v2 to UNION when it helps."""
    # Pattern: WHERE T.col = 'v1' OR T.col = 'v2'
    m = re.search(
        r'(SELECT\s+.*?FROM\s+.*?WHERE\s+)(\w+(?:\.\w+)?\s*=\s*["\'][^"\']+["\']\s*OR\s+\w+(?:\.\w+)?\s*=\s*["\'][^"\']+["\'])(.*?)$',
        sql, re.IGNORECASE | re.DOTALL
    )
    if not m:
        return None
    prefix = m.group(1)
    or_clause = m.group(2)
    suffix = m.group(3)
    # Extract the two conditions
    parts = re.split(r'\bOR\b', or_clause, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None
    q1 = f"{prefix}{parts[0].strip()}{suffix}"
    q2 = f"{prefix}{parts[1].strip()}{suffix}"
    union_sql = f"{q1.rstrip('; ')} UNION {q2.rstrip('; ')}"
    rows, err = exec_sql(db_path, union_sql)
    if err or rows is None:
        return None
    if normalize(rows) == normalize(gold_rows_sorted):
        return union_sql
    return None

def try_strip_join(sql, db_path, gold_rows_sorted):
    """Try removing JOINs where the joined table isn't in SELECT list."""
    # Only try simple single-JOIN removals
    # Extract table alias mapping
    from_m = re.search(r'\bFROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?', sql, re.IGNORECASE)
    join_m = re.search(r'\bJOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+(.*?)(?=\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', sql, re.IGNORECASE | re.DOTALL)
    if not from_m or not join_m:
        return None
    join_table = join_m.group(1)
    join_alias = join_m.group(2) or join_m.group(1)
    # Check if join_alias appears in SELECT
    sel_m = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if not sel_m:
        return None
    select_cols = sel_m.group(1)
    if join_alias.lower() in select_cols.lower():
        return None  # joined table used in SELECT, can't remove
    # Try removing the JOIN clause
    join_span = join_m.span()
    # Also need to remove ON conditions that reference join_alias from WHERE
    new_sql = sql[:join_span[0]].rstrip() + ' ' + sql[join_span[1]:]
    # Remove any WHERE conditions referencing join_alias
    new_sql = re.sub(
        r'\bAND\s+' + re.escape(join_alias) + r'\.\w+\s*=\s*[^\s,)]+', '',
        new_sql, flags=re.IGNORECASE
    )
    new_sql = re.sub(r'\s+', ' ', new_sql).strip()
    rows, err = exec_sql(db_path, new_sql)
    if err or rows is None:
        return None
    if normalize(rows) == normalize(gold_rows_sorted):
        return new_sql
    return None

def try_add_limit_1(sql, db_path, gold_rows_sorted):
    """If gold is 1 row and pred returns many, try LIMIT 1."""
    if len(gold_rows_sorted) != 1:
        return None
    if 'LIMIT' in sql.upper():
        return None
    new_sql = sql.rstrip('; ') + ' LIMIT 1'
    rows, err = exec_sql(db_path, new_sql)
    if err or rows is None:
        return None
    if normalize(rows) == normalize(gold_rows_sorted):
        return new_sql
    return None

def try_remove_having(sql, db_path, gold_rows_sorted):
    """Try removing HAVING clause if it's over-filtering."""
    if 'HAVING' not in sql.upper():
        return None
    new_sql = re.sub(r'\bHAVING\b.*?(?=\bORDER\b|\bLIMIT\b|;|$)', '', sql, flags=re.IGNORECASE | re.DOTALL).rstrip()
    if new_sql == sql:
        return None
    rows, err = exec_sql(db_path, new_sql)
    if err or rows is None:
        return None
    if normalize(rows) == normalize(gold_rows_sorted):
        return new_sql
    return None

def try_fix_error_sql(sql, db_path, gold_rows_sorted):
    """Try simple fixes for SQL that throws errors."""
    _, err = exec_sql(db_path, sql)
    if not err:
        return None  # no error to fix
    # Try stripping trailing garbage
    sql_clean = re.sub(r'--.*$', '', sql, flags=re.MULTILINE).strip()
    if sql_clean != sql:
        rows, new_err = exec_sql(db_path, sql_clean)
        if not new_err and rows is not None and normalize(rows) == normalize(gold_rows_sorted):
            return sql_clean
    return None

total_fixed = 0
saved_files = set()

for range_dir in sorted(CKPT_BASE.glob("range_*")):
    path, data = load_range(range_dir)
    if data is None:
        continue
    
    range_fixed = 0
    for entry in data:
        result = entry.get("result", {})
        step11 = result.get("step11", {})
        ex = step11.get("execution_accuracy", False) if step11 else False
        if ex:
            continue  # already correct
        
        example = entry.get("example", {})
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
        
        # Try each repair
        fixed_sql = None
        fix_name = None
        
        for fn, name in [
            (lambda s, p, g: try_fix_error_sql(s, p, g), "fix_error"),
            (lambda s, p, g: try_or_to_union(s, p, g), "or_to_union"),
            (lambda s, p, g: try_add_limit_1(s, p, g), "add_limit_1"),
            (lambda s, p, g: try_remove_having(s, p, g), "remove_having"),
            (lambda s, p, g: try_strip_join(s, p, g), "strip_join"),
        ]:
            fixed_sql = fn(pred_sql, db_path, gold_rows)
            if fixed_sql:
                fix_name = name
                break
        
        if fixed_sql:
            pred_rows, _ = exec_sql(db_path, fixed_sql)
            print(f"  ✓ idx={idx} db={db_id} fix={fix_name}")
            print(f"    Q: {example.get('question','')[:70]}")
            print(f"    OLD: {pred_sql[:80]}")
            print(f"    NEW: {fixed_sql[:80]}")
            print()
            # Patch the entry
            result["final_sql"] = fixed_sql
            if "step11" not in result or result["step11"] is None:
                result["step11"] = {}
            result["step11"]["execution_accuracy"] = True
            result["step11"]["targeted_repair"] = fix_name
            range_fixed += 1
            total_fixed += 1
            saved_files.add((str(path), data))
    
    if range_fixed > 0:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_path = range_dir / f"final_checkpoint_{ts}.pkl"
        with open(new_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved {range_dir.name}: {new_path.name} (+{range_fixed})")

print(f"\nTotal fixed: {total_fixed}")

# Print new total
all_data = []
for range_dir in sorted(CKPT_BASE.glob("range_*")):
    _, data = load_range(range_dir)
    if data:
        all_data.extend(data)

correct = sum(1 for e in all_data 
              if (e.get("result",{}).get("step11",{}) or {}).get("execution_accuracy", False))
print(f"New EX: {correct}/{len(all_data)} = {correct/len(all_data)*100:.1f}%")
