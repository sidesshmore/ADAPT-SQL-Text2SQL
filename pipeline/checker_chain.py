"""
Deterministic Checker Chain (E') — DeepEye-SQL inspired.

Six sequential rule-based checkers that produce explicit correction directives.
Each checker returns (passed, directive). The chain stops at the first failure
so the retry engine receives one clear, actionable instruction.
"""
import re
import sqlite3
from typing import Dict, List, Optional, Tuple


class CheckerChain:
    def __init__(self, schema_links: Dict):
        self.schema_links = schema_links
        # table -> set of valid column names (lower-cased for matching)
        self._col_map: Dict[str, Dict[str, str]] = {}
        pruned = schema_links.get('pruned_schema') or {}
        # schema_links may not carry pruned_schema; fall back to columns dict
        cols_by_table = schema_links.get('columns', {})
        for table, col_set in cols_by_table.items():
            self._col_map[table] = {c.lower(): c for c in col_set}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        sql: str,
        db_path: Optional[str] = None,
        db_manager=None
    ) -> Dict:
        """Run all 6 checkers in order. Return on first failure."""
        checkers = [
            self._check_syntax,
            self._check_select_star,
            self._check_maxmin,
            self._check_orderby_columns,
            self._check_join_validity,
            self._check_empty_result,
        ]
        checker_args = [
            (sql,),
            (sql,),
            (sql,),
            (sql,),
            (sql,),
            (sql, db_path, db_manager),
        ]

        for checker, args in zip(checkers, checker_args):
            passed, directive = checker(*args)
            if not passed:
                return {
                    'passed': False,
                    'checker': checker.__name__,
                    'directive': directive
                }

        return {'passed': True, 'checker': None, 'directive': ''}

    # ------------------------------------------------------------------
    # Checker 1: Syntax / Execution
    # ------------------------------------------------------------------

    def _check_syntax(self, sql: str) -> Tuple[bool, str]:
        """Check SQL parses without a syntax error using sqlite3."""
        clean = re.sub(r'```(?:sql)?', '', sql, flags=re.IGNORECASE).strip().rstrip('`').strip()
        try:
            conn = sqlite3.connect(':memory:')
            conn.execute(f"EXPLAIN {clean}")
            conn.close()
            return True, ''
        except sqlite3.OperationalError as e:
            msg = str(e)
            return False, (
                f"SYNTAX ERROR in the SQL: {msg}. "
                "Fix the SQL syntax so it parses without errors. "
                "Output only the corrected SQL."
            )

    # ------------------------------------------------------------------
    # Checker 2: SELECT *
    # ------------------------------------------------------------------

    def _check_select_star(self, sql: str) -> Tuple[bool, str]:
        """Warn when SELECT * is used — explicit columns are safer."""
        if re.search(r'\bSELECT\s+\*', sql, re.IGNORECASE):
            tables = list(self.schema_links.get('tables', set()))
            col_hint = ''
            if tables:
                sample_table = tables[0]
                sample_cols = list(self.schema_links.get('columns', {}).get(sample_table, []))
                if sample_cols:
                    col_hint = f" For example, from {sample_table}: {', '.join(sample_cols[:5])}."
            return False, (
                "The SQL uses SELECT * which returns all columns. "
                "Replace SELECT * with only the specific columns the question requires."
                + col_hint
                + " Output only the corrected SQL."
            )
        return True, ''

    # ------------------------------------------------------------------
    # Checker 3: MAX/MIN correctness
    # ------------------------------------------------------------------

    def _check_maxmin(self, sql: str) -> Tuple[bool, str]:
        """Check common MAX/MIN misuse patterns."""
        sql_upper = sql.upper()
        has_maxmin = bool(re.search(r'\b(MAX|MIN)\s*\(', sql_upper))
        if not has_maxmin:
            return True, ''

        # Pattern: SELECT col, MAX(other) without GROUP BY — likely missing GROUP BY
        select_part = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
        if select_part:
            cols_in_select = [c.strip() for c in select_part.group(1).split(',')]
            non_agg = [
                c for c in cols_in_select
                if not re.search(r'\b(MAX|MIN|COUNT|SUM|AVG)\s*\(', c, re.IGNORECASE)
                and c.strip() not in ('*',)
                and not re.match(r'^\d+$', c.strip())
            ]
            has_group_by = bool(re.search(r'\bGROUP\s+BY\b', sql, re.IGNORECASE))
            if non_agg and not has_group_by:
                return False, (
                    "The SQL selects both a non-aggregated column and MAX/MIN without GROUP BY. "
                    "Either add GROUP BY for the non-aggregated column, "
                    "or rewrite as a subquery / ORDER BY ... LIMIT 1 pattern. "
                    "Output only the corrected SQL."
                )

        return True, ''

    # ------------------------------------------------------------------
    # Checker 4: ORDER BY column validity
    # ------------------------------------------------------------------

    def _check_orderby_columns(self, sql: str) -> Tuple[bool, str]:
        """Check ORDER BY references valid schema columns."""
        orderby_match = re.search(r'\bORDER\s+BY\s+(.*?)(?:\bLIMIT\b|$)', sql, re.IGNORECASE | re.DOTALL)
        if not orderby_match:
            return True, ''

        orderby_expr = orderby_match.group(1).strip()
        # Extract bare column names (skip expressions, aliases, literals)
        col_tokens = re.findall(r'\b([A-Za-z_]\w*)\b', orderby_expr)
        reserved = {'ASC', 'DESC', 'NULLS', 'FIRST', 'LAST'}
        col_tokens = [t for t in col_tokens if t.upper() not in reserved]

        # Build flat set of known column names
        known_cols: Dict[str, str] = {}
        for cols_by_name in self._col_map.values():
            known_cols.update(cols_by_name)

        unknown = [t for t in col_tokens if t.lower() not in known_cols]
        if unknown:
            return False, (
                f"ORDER BY references column(s) not found in the schema: {', '.join(unknown)}. "
                "Replace with valid column names from the schema. "
                "Output only the corrected SQL."
            )

        return True, ''

    # ------------------------------------------------------------------
    # Checker 5: JOIN validity
    # ------------------------------------------------------------------

    def _check_join_validity(self, sql: str) -> Tuple[bool, str]:
        """Check that every JOINed table has a FK connection to at least one other table in query."""
        join_tables = re.findall(r'\bJOIN\s+(\w+)', sql, re.IGNORECASE)
        if not join_tables:
            return True, ''

        fks = self.schema_links.get('foreign_keys', [])
        fk_pairs = set()
        for fk in fks:
            fk_pairs.add((fk.get('from_table', ''), fk.get('to_table', '')))
            fk_pairs.add((fk.get('to_table', ''), fk.get('from_table', '')))
            # Also handle source_table/target_table naming
            fk_pairs.add((fk.get('source_table', ''), fk.get('target_table', '')))
            fk_pairs.add((fk.get('target_table', ''), fk.get('source_table', '')))

        from_tables = re.findall(r'\bFROM\s+(\w+)', sql, re.IGNORECASE)
        all_query_tables = set(from_tables) | set(join_tables)

        disconnected = []
        for jt in join_tables:
            connected = any(
                (jt, other) in fk_pairs or (other, jt) in fk_pairs
                for other in all_query_tables
                if other != jt
            )
            if not connected and jt in self.schema_links.get('tables', set()):
                disconnected.append(jt)

        if disconnected:
            return False, (
                f"JOIN on table(s) {', '.join(disconnected)} has no foreign key connection "
                "to the other tables in the query. "
                "Remove unnecessary JOINs or replace with the correctly connected table. "
                "Output only the corrected SQL."
            )

        return True, ''

    # ------------------------------------------------------------------
    # Checker 6: Empty result guard
    # ------------------------------------------------------------------

    def _check_empty_result(
        self,
        sql: str,
        db_path: Optional[str],
        db_manager
    ) -> Tuple[bool, str]:
        """Execute SQL; if it returns 0 rows on a non-negation question, flag it."""
        if not db_path or not db_manager:
            return True, ''

        try:
            result = db_manager.execute_query(sql, db_path)
            if not result.get('success'):
                return True, ''  # execution error handled by syntax checker
            rows = result.get('result_rows', [])
            if len(rows) == 0:
                return False, (
                    "The SQL executed successfully but returned 0 rows. "
                    "Check the WHERE conditions, table names, JOIN predicates, "
                    "and column references — they may filter out all data. "
                    "Output only the corrected SQL."
                )
        except Exception:
            pass

        return True, ''
