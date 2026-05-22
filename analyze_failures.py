"""
Failure analysis for ADAPT-SQL eval results.
Loads all checkpoints for a model/split and categorizes EX=0 examples.

Usage:
    python analyze_failures.py
    python analyze_failures.py --model qwen3-coder-30b-a3b-instruct --split dev
    python analyze_failures.py --model qwen3-235b-a22b-instruct-2507 --split test --top 20
    python analyze_failures.py --full-sql          # show full SQL, not truncated
    python analyze_failures.py --buckets           # show mechanical failure bucket counts
"""
import argparse
import glob
import pickle  # checkpoints are our own files written by eval_voyager.py
import re
from collections import Counter, defaultdict
from pathlib import Path

SUPERLATIVE_WORDS = {
    "most", "least", "highest", "lowest", "largest", "smallest",
    "greatest", "fewest", "maximum", "minimum", "best", "worst",
    "top", "bottom", "first", "last", "oldest", "youngest",
    "cheapest", "expensive", "tallest", "shortest", "longest",
    "max", "min", "earliest", "latest",
}

PROJECT_DIR = Path(__file__).parent


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
            print(f"  [warn] Could not load {latest}: {e}")
    return results


def get_pipeline(record: dict) -> dict:
    """Return the inner pipeline result dict."""
    return record.get("result", record)


def get_example(record: dict) -> dict:
    """Return the Spider example dict (has question, query, db_id)."""
    return record.get("example", {})


def get_step(record: dict, step: str):
    return get_pipeline(record).get(step, {})


def classify_failure(record: dict) -> str:
    r = get_pipeline(record)
    s10 = r.get("step10_generated", {})

    if s10 and not s10.get("success", True):
        err = s10.get("error_message", "")
        if "no such table" in err.lower():
            return "exec_error:no_such_table"
        if "no such column" in err.lower():
            return "exec_error:no_such_column"
        if "ambiguous" in err.lower():
            return "exec_error:ambiguous_column"
        if "syntax" in err.lower():
            return "exec_error:syntax"
        return "exec_error:other"

    return "wrong_results"


def get_complexity(record: dict) -> str:
    r = get_pipeline(record)
    s2 = r.get("step2", {})
    val = s2.get("complexity_class", s2.get("complexity", s2.get("classification")))
    if val is None:
        return "UNKNOWN"
    return getattr(val, "value", str(val))


def get_strategy(record: dict) -> str:
    r = get_pipeline(record)
    s5 = r.get("step5", {})
    val = s5.get("strategy", s5.get("routing_decision"))
    if val is None:
        return "UNKNOWN"
    return getattr(val, "value", str(val)).replace("GenerationStrategy.", "")


def sql_features(sql: str) -> list:
    sql_up = sql.upper()
    features = []
    for kw in ["EXCEPT", "INTERSECT", "UNION", "NOT IN", "NOT EXISTS",
                "EXISTS", "HAVING", "GROUP BY", "ORDER BY", "LIMIT",
                "WITH ", "CASE "]:
        if kw in sql_up:
            features.append(kw.strip())
    return features


def extract_tables(sql: str) -> set:
    """Extract table names referenced after FROM and JOIN keywords."""
    tokens = re.findall(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)
    return {t.lower() for t in tokens}


# ── Mechanical failure buckets ──────────────────────────────────────────────

def bucket_missing_limit(rec: dict) -> bool:
    """Pred has ORDER BY but no LIMIT, and question contains a superlative word."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    question = get_example(rec).get("question", "") or ""
    pred_up = pred.upper()
    if "ORDER BY" not in pred_up:
        return False
    if "LIMIT" in pred_up:
        return False
    q_words = set(re.findall(r'\b\w+\b', question.lower()))
    return bool(q_words & SUPERLATIVE_WORDS)


def bucket_over_join(rec: dict) -> bool:
    """Pred joins more tables than the gold query."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    pred_tables = extract_tables(pred)
    gold_tables = extract_tables(gold)
    return len(pred_tables) > len(gold_tables)


def bucket_missing_group_by(rec: dict) -> bool:
    """Gold has GROUP BY but pred does not."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    return "GROUP BY" in gold.upper() and "GROUP BY" not in pred.upper()


def bucket_spurious_group_by(rec: dict) -> bool:
    """Pred has GROUP BY but gold does not."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    return "GROUP BY" in pred.upper() and "GROUP BY" not in gold.upper()


def bucket_set_op_mismatch(rec: dict) -> bool:
    """Gold uses EXCEPT/INTERSECT/UNION but pred does not (or vice versa)."""
    pred = (get_pipeline(rec).get("final_sql", "") or "").upper()
    gold = (get_example(rec).get("query", "") or "").upper()
    set_ops = {"EXCEPT", "INTERSECT", "UNION"}
    gold_has = any(op in gold for op in set_ops)
    pred_has = any(op in pred for op in set_ops)
    return gold_has != pred_has


def bucket_missing_having(rec: dict) -> bool:
    """Gold has HAVING but pred does not."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    return "HAVING" in gold.upper() and "HAVING" not in pred.upper()


def bucket_nested_missed(rec: dict) -> bool:
    """Gold uses subquery (nested SELECT) but pred is flat."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    gold_nested = len(re.findall(r'\bSELECT\b', gold, re.IGNORECASE)) > 1
    pred_nested = len(re.findall(r'\bSELECT\b', pred, re.IGNORECASE)) > 1
    return gold_nested and not pred_nested


def bucket_spurious_cast(rec: dict) -> bool:
    """Pred wraps columns in CAST() but gold does not — changes NULL/sort behavior."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    return bool(re.search(r'\bCAST\s*\(', pred, re.IGNORECASE)) and \
           not bool(re.search(r'\bCAST\s*\(', gold, re.IGNORECASE))


def bucket_spurious_limit(rec: dict) -> bool:
    """Pred has LIMIT but gold does not — over-constrains result set."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    return "LIMIT" in pred.upper() and "LIMIT" not in gold.upper()


def _count_select_cols(sql: str) -> int:
    """Rough count of columns in the outermost SELECT clause."""
    m = re.match(r'\s*SELECT\s+(.*?)\s+FROM\b', sql, re.IGNORECASE | re.DOTALL)
    if not m:
        return -1
    cols = m.group(1)
    # Strip nested parens to avoid counting commas inside functions
    depth, cleaned, buf = 0, [], []
    for ch in cols:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0:
            buf.append(ch)
    cleaned = ''.join(buf)
    return len(cleaned.split(','))


def bucket_column_count_mismatch(rec: dict) -> bool:
    """Pred and gold SELECT a different number of columns."""
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    n_pred = _count_select_cols(pred)
    n_gold = _count_select_cols(gold)
    return n_pred >= 0 and n_gold >= 0 and n_pred != n_gold


def bucket_wrong_aggregation(rec: dict) -> bool:
    """Pred uses a different aggregate function than gold (COUNT vs SUM vs AVG etc)."""
    agg_re = re.compile(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', re.IGNORECASE)
    pred = get_pipeline(rec).get("final_sql", "") or ""
    gold = get_example(rec).get("query", "") or ""
    pred_aggs = Counter(m.upper() for m in agg_re.findall(pred))
    gold_aggs = Counter(m.upper() for m in agg_re.findall(gold))
    return pred_aggs != gold_aggs


BUCKETS = [
    ("missing LIMIT (ORDER BY + superlative)", bucket_missing_limit),
    ("spurious LIMIT (pred has, gold lacks)",  bucket_spurious_limit),
    ("over-joining (more tables than gold)",   bucket_over_join),
    ("missing GROUP BY (gold has, pred lacks)", bucket_missing_group_by),
    ("spurious GROUP BY (pred has, gold lacks)", bucket_spurious_group_by),
    ("set-op mismatch (EXCEPT/INTERSECT/UNION)", bucket_set_op_mismatch),
    ("missing HAVING (gold has, pred lacks)",   bucket_missing_having),
    ("nested query missed (gold nested, pred flat)", bucket_nested_missed),
    ("spurious CAST (pred casts, gold doesn't)", bucket_spurious_cast),
    ("column count mismatch (SELECT arity)",   bucket_column_count_mismatch),
    ("wrong aggregation (COUNT/SUM/AVG/MIN/MAX)", bucket_wrong_aggregation),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-coder-30b-a3b-instruct")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--top", type=int, default=10, help="Show top N failure examples")
    parser.add_argument("--full-sql", action="store_true", help="Print full SQL (no truncation)")
    parser.add_argument("--buckets", action="store_true", help="Show mechanical failure bucket counts")
    args = parser.parse_args()
    sql_width = 0 if args.full_sql else 120  # 0 = no truncation

    model_slug = args.model.replace("/", "___")
    print(f"\nLoading: {model_slug}/{args.split}")
    results = load_all_results(model_slug, args.split)

    if not results:
        print("No results found.")
        return

    failures = [r for r in results if not get_step(r, "step11").get("execution_accuracy", False)]
    passes   = [r for r in results if     get_step(r, "step11").get("execution_accuracy", False)]

    total = len(results)
    n_fail = len(failures)
    n_pass = len(passes)
    print(f"\n{'='*65}")
    print(f"  Total: {total}  |  Pass: {n_pass} ({n_pass/total*100:.1f}%)  |  Fail: {n_fail} ({n_fail/total*100:.1f}%)")
    print(f"{'='*65}")

    # ── Failure type breakdown ──────────────────────────────────────────
    fail_types = Counter(classify_failure(r) for r in failures)
    print(f"\nFailure types ({n_fail} total):")
    for ftype, cnt in fail_types.most_common():
        print(f"  {ftype:<35} {cnt:>5}  ({cnt/n_fail*100:.1f}%)")

    # ── Complexity breakdown ────────────────────────────────────────────
    print(f"\nFailures by complexity class:")
    complexity_fail = Counter(get_complexity(r) for r in failures)
    complexity_all  = Counter(get_complexity(r) for r in results)
    for comp, cnt in complexity_fail.most_common():
        total_comp = complexity_all[comp]
        pct_fail = cnt / total_comp * 100 if total_comp else 0
        print(f"  {comp:<25} {cnt:>5}/{total_comp:<6} failed ({pct_fail:.1f}% fail rate)")

    # ── Routing strategy breakdown ──────────────────────────────────────
    print(f"\nFailures by routing strategy:")
    strategy_fail = Counter(get_strategy(r) for r in failures)
    strategy_all  = Counter(get_strategy(r) for r in results)
    for strat, cnt in strategy_fail.most_common():
        total_strat = strategy_all[strat]
        pct_fail = cnt / total_strat * 100 if total_strat else 0
        print(f"  {strat:<30} {cnt:>5}/{total_strat:<6} failed ({pct_fail:.1f}% fail rate)")

    # ── SQL keyword patterns in failures ───────────────────────────────
    print(f"\nSQL keywords in failed gold queries:")
    kw_counts = Counter()
    for rec in failures:
        gold = get_example(rec).get("query", "")
        if gold:
            for kw in sql_features(gold):
                kw_counts[kw] += 1
    for kw, cnt in kw_counts.most_common(12):
        pct = cnt / n_fail * 100
        print(f"  {kw:<20} {cnt:>5}  ({pct:.1f}% of failures)")

    # ── Exec error details ─────────────────────────────────────────────
    exec_errors = [rec for rec in failures if classify_failure(rec).startswith("exec_error")]
    if exec_errors:
        print(f"\nExecution errors ({len(exec_errors)}):")
        for rec in exec_errors[:5]:
            q   = get_example(rec).get("question", "?")[:70]
            err = get_pipeline(rec).get("step10_generated", {}).get("error_message", "")[:100]
            print(f"  Q:   {q}")
            print(f"  Err: {err}\n")

    # ── Mechanical failure buckets ─────────────────────────────────────
    if args.buckets:
        wrong_results = [r for r in failures if classify_failure(r) == "wrong_results"]
        print(f"\nMechanical failure buckets ({len(wrong_results)} wrong_results failures):")
        print(f"  (buckets are not mutually exclusive)")
        print(f"  {'Bucket':<48} {'N':>5}  {'% of WR':>8}  {'% of all fail':>13}")
        print(f"  {'─'*80}")
        for label, fn in BUCKETS:
            matched = [r for r in wrong_results if fn(r)]
            n = len(matched)
            pct_wr   = n / len(wrong_results) * 100 if wrong_results else 0
            pct_fail = n / n_fail * 100 if n_fail else 0
            print(f"  {label:<48} {n:>5}  {pct_wr:>7.1f}%  {pct_fail:>12.1f}%")
        print()
        # Uncategorized (hit none of the buckets)
        uncategorized = [r for r in wrong_results if not any(fn(r) for _, fn in BUCKETS)]
        print(f"  {'uncategorized (no bucket matched)':<48} {len(uncategorized):>5}  "
              f"{len(uncategorized)/len(wrong_results)*100 if wrong_results else 0:>7.1f}%")

    # ── Top N failure examples ─────────────────────────────────────────
    print(f"\nSample failures (top {args.top}):")
    print(f"{'─'*65}")
    for i, rec in enumerate(failures[:args.top]):
        ex    = get_example(rec)
        rr    = get_pipeline(rec)
        q     = ex.get("question", "?")
        gold  = ex.get("query", "N/A")
        pred  = rr.get("final_sql", "N/A")
        comp  = get_complexity(rec)
        ftype = classify_failure(rec)
        retry = rr.get("retry_count", 0)
        db    = ex.get("db_id", "?")
        # Which buckets this failure hits
        hit_buckets = [label for label, fn in BUCKETS if fn(rec)]
        bucket_tag  = f"  [{', '.join(hit_buckets)}]" if hit_buckets else ""
        print(f"\n[{i+1}] {comp} | {ftype} | retries={retry} | db={db}{bucket_tag}")
        print(f"  Q:    {q[:90]}")
        if sql_width:
            print(f"  PRED: {pred[:sql_width]}")
            print(f"  GOLD: {gold[:sql_width]}")
        else:
            print(f"  PRED: {pred}")
            print(f"  GOLD: {gold}")


if __name__ == "__main__":
    main()
