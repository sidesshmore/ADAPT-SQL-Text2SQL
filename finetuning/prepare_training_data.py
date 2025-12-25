"""
Prepare Spider dataset for fine-tuning Qwen3-Coder
Converts Spider examples to instruction-tuning format
"""

import json
import sqlite3
from pathlib import Path
from typing import List, Dict


def get_schema_string(db_path: str) -> str:
    """Extract schema from SQLite database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema_parts = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()

        col_defs = []
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            col_defs.append(f"{col_name} {col_type}")

        schema_parts.append(f"Table {table_name}:\n  " + "\n  ".join(col_defs))

    # Get foreign keys
    fk_parts = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        fks = cursor.fetchall()
        for fk in fks:
            fk_parts.append(f"{table_name}.{fk[3]} -> {fk[2]}.{fk[4]}")

    conn.close()

    schema_str = "\n\n".join(schema_parts)
    if fk_parts:
        schema_str += "\n\nForeign Keys:\n  " + "\n  ".join(fk_parts)

    return schema_str


def create_instruction_prompt(question: str, schema: str, sql: str = None) -> Dict:
    """Create instruction-tuning format"""
    system_msg = """You are an expert SQL generator. Given a database schema and a natural language question, generate the correct SQL query.

Rules:
1. Only use tables and columns that exist in the schema
2. Use proper SQL syntax (SQLite dialect)
3. Include necessary JOINs based on foreign keys
4. Use appropriate WHERE, GROUP BY, HAVING, ORDER BY clauses
5. Return only the SQL query without explanation"""

    user_msg = f"""Database Schema:
{schema}

Question: {question}

Generate the SQL query:"""

    if sql:  # Training format
        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": sql}
            ]
        }
    else:  # Inference format
        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
        }


def prepare_spider_data(
    spider_json_path: str,
    database_dir: str,
    output_path: str,
    max_examples: int = None
):
    """Convert Spider dataset to fine-tuning format"""

    with open(spider_json_path, 'r') as f:
        spider_data = json.load(f)

    if max_examples:
        spider_data = spider_data[:max_examples]

    training_examples = []
    skipped = 0

    for idx, example in enumerate(spider_data):
        db_id = example['db_id']
        question = example['question']
        sql = example['query']

        # Get database schema
        db_path = Path(database_dir) / db_id / f"{db_id}.sqlite"

        if not db_path.exists():
            print(f"Warning: Database not found for {db_id}, skipping...")
            skipped += 1
            continue

        try:
            schema = get_schema_string(str(db_path))
            instruction = create_instruction_prompt(question, schema, sql)
            training_examples.append(instruction)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(spider_data)} examples...")

        except Exception as e:
            print(f"Error processing example {idx} (db: {db_id}): {e}")
            skipped += 1
            continue

    # Save in JSONL format (one example per line)
    with open(output_path, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"Total examples: {len(training_examples)}")
    print(f"Skipped: {skipped}")
    print(f"Output saved to: {output_path}")
    print(f"{'='*60}\n")

    return training_examples


if __name__ == "__main__":
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    SPIDER_DATA_DIR = PROJECT_ROOT / "data" / "spider" / "spider_data"

    # Prepare training data (from train_spider.json)
    print("Preparing training data...")
    train_examples = prepare_spider_data(
        spider_json_path=str(SPIDER_DATA_DIR / "train_spider.json"),
        database_dir=str(SPIDER_DATA_DIR / "database"),
        output_path=str(PROJECT_ROOT / "finetuning" / "train_data.jsonl"),
        max_examples=None  # Use all training examples
    )

    # Prepare validation data (from dev.json)
    print("\nPreparing validation data...")
    val_examples = prepare_spider_data(
        spider_json_path=str(SPIDER_DATA_DIR / "dev.json"),
        database_dir=str(SPIDER_DATA_DIR / "database"),
        output_path=str(PROJECT_ROOT / "finetuning" / "val_data.jsonl"),
        max_examples=200  # Use 200 for validation
    )
