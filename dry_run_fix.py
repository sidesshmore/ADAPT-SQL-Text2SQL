"""
Dry-run normalizer patches against existing checkpoints.

Applies proposed SQL fixes to every result, re-executes against SQLite,
and reports net EX delta (recovered - broken) without rerunning LLM eval.

Usage:
    python dry_run_fix.py
    python dry_run_fix.py --model qwen3-coder-30b-a3b-instruct --split dev
    python dry_run_fix.py --no-cast   # skip CAST strip
    python dry_run_fix.py --no-limit  # skip spurious LIMIT strip

Note: checkpoints are pickle files written by eval_voyager.py (trusted local files).
"""
import argparse
import pickle  # safe: loading our own eval checkpoint files
import re
import sqlite3
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
SPIDER_DB_DIR = PROJECT_DIR / "data" / "spider" / "spider_data" / "database"

SUPERLATIVE_WORDS = {
    "most", "least", "highest", "lowest", "largest", "smallest",
    "greatest", "fewest", "maximum", "minimum", "best", "worst",
    "top", "bottom", "first", "last", "oldest", "youngest",
    "cheapest", "expensive", "tallest", "shortest", "longest",
    "max", "min", "earliest", "latest",
}
NUMBER_WORDS = {"one", "single"}


# ── Proposed fixes ───────────────────────────────────────────────────────────

def strip_spurious_cast(sql: str) -> tuple:
    """Remove CAST(expr AS TYPE) → expr everywhere in the SQL."""
    new_sql = re.sub(
        r'\bCAST\s*\(\s*([^)]+?)\s+AS\s+\w+\s*\)',
        lambda m: m.group(1).strip(),
        sql,
        flags=re.IGNORECASE,
    )
    return new_sql, new_sql != sql


def strip_spurious_limit(sql: str, question: str) -> tuple:
    """Remove LIMIT 1 when question has no superlative or number-ranking word."""
    if not re.search(r'\bLIMIT\s+1\b', sql, re.IGNORECASE):
        return sql, False
    q_words = set(re.findall(r'\b\w+\b', question.lower()))
    if q_words & SUPERLATIVE_WORDS:
        return sql, False   # legitimate LIMIT 1
    if q_words & NUMBER_WORDS:
        return sql, False
    new_sql = re.sub(r'\s*\bLIMIT\s+1\b', '', sql, flags=re.IGNORECASE).rstrip('; \n')
    return new_sql, True


def apply_fixes(sql: str, question: str, fix_cast: bool, fix_limit: bool) -> tuple:
    changes = []
    if fix_cast:
        sql, changed = strip_spurious_cast(sql)
        if changed:
            changes.append("strip_cast")
    if fix_limit:
        sql, changed = strip_spurious_limit(sql, question)
        if changed:
            changes.append("strip_limit")
    return sql, changes


# ── Execution ────────────────────────────────────────────────────────────────

def execute_sql(sql: str, db_path: Path) -> tuple:
    """Return (success, sorted_rows) for order-independent result comparison."""
    if not db_path.exists():
        return False, None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        # Sort for order-independent comparison (same as Spider EX metric)
        return True, sorted(str(r) for r in rows)
    except Exception:
        return False, None


# ── Checkpoint loading ───────────────────────────────────────────────────────

def load_all_results(model_slug: str, split: str) -> list:
    base = PROJECT_DIR / "eval_results_voyager" / model_slug / split
    results = []
    for range_dir in sorted(base.glob("range_*")):
        files = list(range_dir.glob("checkpoint_*.pkl")) + list(range_dir.glob("final_checkpoint_*.pkl"))
        if not files:
            continue
        latest = max(files, key=lambda f: f.stat().st_mtime)
        try:
            data = pickle.load(open(latest, "rb"))
            results.extend(data.get("results", []))
        except Exception as e:
            print(f"  [warn] {latest}: {e}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-coder-30b-a3b-instruct")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--no-cast",  dest="fix_cast",  action="store_false")
    parser.add_argument("--no-limit", dest="fix_limit", action="store_false")
    parser.set_defaults(fix_cast=True, fix_limit=True)
    args = parser.parse_args()

    model_slug = args.model.replace("/", "___")
    print(f"\nDry-run: {model_slug}/{args.split}")
    print(f"  fixes: cast={args.fix_cast}  limit={args.fix_limit}\n")

    records = load_all_results(model_slug, args.split)
    if not records:
        print("No results found.")
        return

    recovered, broken = [], []
    skipped = unchanged = cast_stripped = limit_stripped = 0
    baseline_pass = 0

    for rec in records:
        r        = rec.get("result", rec)
        ex       = rec.get("example", {})
        question = ex.get("question", "")
        db_id    = ex.get("db_id", "")
        gold_sql = ex.get("query", "")
        pred_sql = r.get("final_sql", "")
        was_pass = bool(r.get("step11", {}).get("execution_accuracy", False))

        if was_pass:
            baseline_pass += 1

        if not pred_sql or not db_id:
            skipped += 1
            continue

        fixed_sql, changes = apply_fixes(pred_sql, question, args.fix_cast, args.fix_limit)
        if not changes:
            unchanged += 1
            continue

        if "strip_cast" in changes:
            cast_stripped += 1
        if "strip_limit" in changes:
            limit_stripped += 1

        db_path = SPIDER_DB_DIR / db_id / f"{db_id}.sqlite"
        pred_ok, pred_rows = execute_sql(fixed_sql, db_path)
        gold_ok, gold_rows = execute_sql(gold_sql, db_path)

        if not pred_ok or not gold_ok:
            skipped += 1
            continue

        now_pass = (pred_rows == gold_rows)

        if not was_pass and now_pass:
            recovered.append({"q": question, "db": db_id, "changes": changes,
                               "orig": pred_sql, "fixed": fixed_sql, "gold": gold_sql})
        elif was_pass and not now_pass:
            broken.append({"q": question, "db": db_id, "changes": changes,
                           "orig": pred_sql, "fixed": fixed_sql, "gold": gold_sql})

    total = len(records)
    net   = len(recovered) - len(broken)
    new_pass = baseline_pass + net

    print(f"{'='*65}")
    print(f"  Total records   : {total}")
    print(f"  Unchanged       : {unchanged}")
    print(f"  Skipped         : {skipped}")
    print(f"  CAST strips     : {cast_stripped}")
    print(f"  LIMIT strips    : {limit_stripped}")
    print(f"{'─'*65}")
    print(f"  Recovered (0→1) : {len(recovered)}")
    print(f"  Broken    (1→0) : {len(broken)}")
    print(f"  Net EX delta    : {net:+d}")
    print(f"  Baseline EX     : {baseline_pass}/{total} = {baseline_pass/total*100:.1f}%")
    print(f"  Projected EX    : {new_pass}/{total} = {new_pass/total*100:.1f}%")
    print(f"{'='*65}")

    if recovered:
        print(f"\nRecovered ({len(recovered)}):")
        for i, r in enumerate(recovered):
            print(f"\n  [{i+1}] db={r['db']}  fix={r['changes']}")
            print(f"    Q:     {r['q'][:80]}")
            print(f"    ORIG:  {r['orig'][:120]}")
            print(f"    FIXED: {r['fixed'][:120]}")
            print(f"    GOLD:  {r['gold'][:120]}")

    if broken:
        print(f"\nBroken ({len(broken)}):")
        for i, r in enumerate(broken):
            print(f"\n  [{i+1}] db={r['db']}  fix={r['changes']}")
            print(f"    Q:     {r['q'][:80]}")
            print(f"    ORIG:  {r['orig'][:120]}")
            print(f"    FIXED: {r['fixed'][:120]}")
            print(f"    GOLD:  {r['gold'][:120]}")


if __name__ == "__main__":
    main()
