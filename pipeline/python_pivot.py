"""
Python Pivot (G') — Pi-SQL inspired.

For NESTED_COMPLEX queries, generate a "oracle SQL" that captures the
expected result shape (row count, column names), execute it, and return
a hint for the main SQL generator.  This anchors generation to a concrete
expected output, which Pi-SQL showed reduces NESTED failures.

Unlike the original Pi-SQL which uses actual Python/pandas, we generate
a deliberately simplified SQL (without correlated subqueries) so the model
can produce something executable, then use the shape as the hint.
"""
import re
from typing import Dict, List, Optional

import ollama


class PythonPivot:
    def __init__(self, model: str = "qwen3-coder", timeout_sec: int = 15):
        self.model = model
        self.timeout_sec = timeout_sec

    def get_hint(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        db_path: str,
        db_manager=None
    ) -> str:
        """
        Generate a simple oracle SQL, execute it, return a result-shape hint.
        Returns empty string on any failure — caller falls back to normal generation.
        """
        if not db_manager or not db_path:
            return ''
        try:
            oracle_sql = self._generate_oracle_sql(question, pruned_schema, foreign_keys)
            if not oracle_sql:
                return ''
            exec_result = db_manager.execute_query(oracle_sql, db_path)
            if not exec_result or not exec_result.get('success'):
                return ''
            return self._format_hint(exec_result)
        except Exception:
            return ''

    # ------------------------------------------------------------------
    # Step 1: generate oracle SQL
    # ------------------------------------------------------------------

    def _generate_oracle_sql(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> str:
        """Ask the model for a simple, probably-correct SQL — not necessarily the final one."""
        schema_str = self._format_schema(pruned_schema)
        fk_str = self._format_foreign_keys(foreign_keys)

        prompt = f"""Generate a SIMPLE SQLite query for this question. It does not need to be perfect.
Goal: get the right column names and an approximate row count.
Keep it simple — use JOINs instead of subqueries where possible.

Schema:
{schema_str}

Foreign keys:
{fk_str}

Question: {question}

Output ONLY the SQL query:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a SQL expert. Output only the SQL query, no explanation.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.1}
            )
            raw = response['message']['content'].strip()
            raw = re.sub(r'```sql\s*', '', raw, flags=re.IGNORECASE)
            raw = re.sub(r'```\s*', '', raw)
            lines = raw.split('\n')
            sql_lines = []
            in_sql = False
            for line in lines:
                if any(line.strip().upper().startswith(kw) for kw in ['SELECT', 'WITH']):
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                    if line.strip().endswith(';'):
                        break
            result = '\n'.join(sql_lines).strip() if sql_lines else raw
            if not result.endswith(';'):
                result += ';'
            return result
        except Exception:
            return ''

    # ------------------------------------------------------------------
    # Step 2: format hint from execution result
    # ------------------------------------------------------------------

    def _format_hint(self, exec_result: Dict) -> str:
        """Describe the oracle result shape for the SQL prompt."""
        rows = exec_result.get('result_rows', [])
        col_names = exec_result.get('column_names', [])

        n_rows = len(rows)
        if n_rows == 0:
            return 'Oracle hint: the answer is likely an empty result set (0 rows).'

        col_str = ', '.join(f'"{c}"' for c in col_names[:6]) if col_names else '(unknown columns)'
        if col_names and len(col_names) > 6:
            col_str += ', ...'

        return (
            f'Oracle hint: the answer likely has ~{n_rows} row(s) '
            f'with column(s) [{col_str}]. '
            'Use this as a sanity check — your SQL should produce a similar shape.'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_schema(self, pruned_schema: Dict[str, List[Dict]]) -> str:
        lines = []
        for table, cols in sorted(pruned_schema.items()):
            col_names = [c['column_name'] for c in cols]
            lines.append(f"  {table}: {', '.join(col_names)}")
        return '\n'.join(lines)

    def _format_foreign_keys(self, foreign_keys: List[Dict]) -> str:
        lines = []
        for fk in foreign_keys:
            ft = fk.get('from_table') or fk.get('source_table', '')
            fc = fk.get('from_column') or fk.get('source_column', '')
            tt = fk.get('to_table') or fk.get('target_table', '')
            tc = fk.get('to_column') or fk.get('target_column', '')
            if ft and tt:
                lines.append(f"  {ft}.{fc} → {tt}.{tc}")
        return '\n'.join(lines) if lines else '  (none)'
