"""
Generate SFT training data using the pipeline's actual prompt format.

Produces two types of training records per example:
  1. NatSQL generation  : (schema + guide + few-shot + question)  → NatSQL
  2. NatSQL→SQL         : (schema + fk + natsql + few-shot)       → gold SQL

No Ollama required — runs on CPU / login node.

Usage:
    python finetune/build_pipeline_sft_data.py \
        --project_dir /scratch/smore123/ADAPT-SQL-Text2SQL \
        --out /scratch/smore123/pipeline_sft_data.json
"""
import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path


# ── Schema helpers ───────────────────────────────────────────────────────────

def get_schema(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t] = [{"column_name": r[1], "data_type": r[2]} for r in cur.fetchall()]
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


# ── NatSQL helpers (mirrors pipeline/few_shot.py) ────────────────────────────

def convert_sql_to_natsql(sql: str) -> str:
    if not sql:
        return "SELECT ..."
    sql_upper = sql.upper()
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
    if not select_match:
        return "SELECT ..."
    select_clause = select_match.group(1).strip()
    select_clause = re.sub(r'\bT\d+\.', '', select_clause, flags=re.IGNORECASE)
    select_clause = re.sub(r'\bAS\s+\w+', '', select_clause, flags=re.IGNORECASE)
    natsql = f"SELECT {select_clause}"
    if 'JOIN' in sql_upper:
        join_tables = re.findall(r'JOIN\s+(\w+)', sql_upper)
        if join_tables:
            natsql += f" WHERE @ JOIN {join_tables[0]}.*"
    elif 'WHERE' in sql_upper:
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1).strip()
            where_clause = re.sub(r'\bT\d+\.', '', where_clause, flags=re.IGNORECASE)
            natsql += f" WHERE {where_clause}"
    return natsql


def format_schema_simple(schema: dict) -> str:
    out = ""
    for table, cols in sorted(schema.items()):
        col_str = ", ".join(f"{table}.{c['column_name']}" for c in cols)
        out += f"**{table}**: {col_str}\n"
    return out


def format_schema_detailed(schema: dict) -> str:
    out = ""
    for table, cols in sorted(schema.items()):
        out += f"**{table}**\n"
        for col in cols:
            out += f"  • {col['column_name']} ({col['data_type']})\n"
        out += "\n"
    return out


# ── Prompt builders (mirrors pipeline prompts exactly) ───────────────────────

def build_natsql_gen_prompt(question: str, schema: dict, examples: list) -> str:
    prompt = "# Generate NatSQL Intermediate (DIN-SQL Format)\n\n"
    prompt += "## Database Schema\n\n"
    prompt += format_schema_simple(schema)
    prompt += "\n"
    prompt += "## NatSQL Format (DIN-SQL)\n\n"
    prompt += "- No explicit FROM clause (implicit from table.column references)\n"
    prompt += "- JOINs as: WHERE @ JOIN table.*\n"
    prompt += "- Aggregations: count(), avg(), sum(), max(), min()\n"
    prompt += "- Column format: table.column\n\n"
    prompt += "## Examples\n\n"
    for i, ex in enumerate(examples[:3], 1):
        sql = ex.get("query", "")
        natsql = convert_sql_to_natsql(sql)
        prompt += f"### Example {i}\n"
        prompt += f"Question: {ex.get('question', '')}\n"
        prompt += f"NatSQL: {natsql}\n"
        prompt += f"SQL: {sql}\n\n"
    prompt += "## Your Task\n\n"
    prompt += f"Question: {question}\n\n"
    prompt += "Before generating NatSQL, reason through:\n"
    prompt += "1. Which tables contain the needed data?\n"
    prompt += "2. Do tables need to be joined? If so, which foreign key connects them?\n"
    prompt += "3. What filter conditions (WHERE) does the question require?\n"
    prompt += "4. Is aggregation (count/sum/avg/max/min), GROUP BY, ORDER BY, or a subquery needed?\n"
    prompt += "\nNow generate NatSQL intermediate representation:\n"
    return prompt


def build_natsql_to_sql_prompt(natsql: str, schema: dict, fks: list, examples: list) -> str:
    prompt = "# Convert NatSQL to SQL\n\n"
    prompt += "## Database Schema\n\n"
    prompt += format_schema_detailed(schema)
    if fks:
        prompt += "## Foreign Key Relationships\n\n"
        for fk in fks:
            prompt += f"- {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        prompt += "\n"
    prompt += "## Conversion Examples\n\n"
    for i, ex in enumerate(examples[:2], 1):
        sql = ex.get("query", "")
        natsql_ex = convert_sql_to_natsql(sql)
        prompt += f"### Example {i}\n"
        prompt += f"NatSQL: {natsql_ex}\n"
        prompt += f"SQL:\n```sql\n{sql}\n```\n\n"
    prompt += "## Your Task\n\n"
    prompt += f"NatSQL: {natsql}\n\n"
    prompt += "Convert to SQL. Follow these rules:\n"
    prompt += "1. Add explicit FROM clause\n"
    prompt += "2. Convert WHERE @ JOIN to proper JOIN ON\n"
    prompt += "3. Use standard SQL syntax\n"
    prompt += "4. Match the style of ground truth examples\n\n"
    prompt += "Output ONLY the SQL query:\n"
    return prompt


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    project = Path(args.project_dir)

    train_json = project / "data/spider/spider_data/train_spider.json"
    db_dir = project / "data/spider/spider_data/database"

    with open(train_json) as f:
        examples = json.load(f)

    records = []
    skipped = 0

    for i, ex in enumerate(examples):
        db_path = db_dir / ex["db_id"] / f"{ex['db_id']}.sqlite"
        if not db_path.exists():
            skipped += 1
            continue

        try:
            schema = get_schema(str(db_path))
            fks = get_foreign_keys(str(db_path))
        except Exception:
            skipped += 1
            continue

        # Pick 3 random few-shot examples (different from current)
        pool = [e for e in examples if e is not ex]
        few_shot = random.sample(pool, min(3, len(pool)))

        gold_sql = ex["query"]
        gold_natsql = convert_sql_to_natsql(gold_sql)

        # Record 1: NatSQL generation
        prompt1 = build_natsql_gen_prompt(ex["question"], schema, few_shot)
        records.append({"text": prompt1 + gold_natsql + "\n"})

        # Record 2: NatSQL→SQL conversion
        prompt2 = build_natsql_to_sql_prompt(gold_natsql, schema, fks, few_shot)
        records.append({"text": prompt2 + f"```sql\n{gold_sql}\n```\n"})

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(examples)} processed, {len(records)} records so far")

    print(f"Done. {len(records)} records from {len(examples)-skipped} examples ({skipped} skipped).")

    with open(args.out, "w") as f:
        json.dump(records, f)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
