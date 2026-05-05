"""
Execution-Based Candidate Selector — Phase C.

Given N SQL candidates for the same question, executes each against the DB
and picks the winner by majority result-set agreement (XiYan-SQL approach).

Algorithm:
1. Execute each candidate. Collect (sql, rows, success).
2. Hash each result set: frozenset of sorted row-tuples.
3. Group by hash. Majority group wins.
4. Tie-break: prefer lower index (primary / lower temperature).
5. If no candidate executes successfully: return candidates[0] unchanged.

Falls back silently — caller always gets a valid SQL string back.
"""
from typing import List, Optional


class CandidateSelector:
    def __init__(self, db_manager=None):
        self.db_manager = db_manager

    def select(
        self,
        candidates: List[str],
        db_path: str,
        db_manager=None
    ) -> dict:
        """
        Args:
            candidates: list of SQL strings (index 0 = primary / lowest temp)
            db_path: path to the SQLite database file
            db_manager: DatabaseManager instance (falls back to self.db_manager)

        Returns:
            {
                'winner_sql': str,
                'winner_index': int,
                'winner_reason': str,
                'exec_results': list[dict],  # one per candidate
                'group_counts': dict,        # hash → count
            }
        """
        mgr = db_manager or self.db_manager
        if not mgr or not db_path or not candidates:
            return {
                'winner_sql': candidates[0] if candidates else '',
                'winner_index': 0,
                'winner_reason': 'no_db_manager',
                'exec_results': [],
                'group_counts': {},
            }

        exec_results = []
        for sql in candidates:
            try:
                r = mgr.execute_query(sql, db_path)
                rows = r.get('result_rows', []) if r.get('success') else None
                exec_results.append({
                    'success': bool(r.get('success')),
                    'rows': rows,
                    'row_count': len(rows) if rows is not None else -1,
                })
            except Exception as e:
                exec_results.append({'success': False, 'rows': None, 'row_count': -1})

        # Group successful candidates by result-set identity
        groups: dict = {}  # hash_key → list of indices
        for i, res in enumerate(exec_results):
            if res['success'] and res['rows'] is not None:
                try:
                    key = frozenset(
                        tuple(row) if not isinstance(row, (list, tuple)) else tuple(row)
                        for row in res['rows']
                    )
                    groups.setdefault(key, []).append(i)
                except TypeError:
                    # unhashable row values — fall back to string hash
                    key = str(sorted(str(r) for r in res['rows']))
                    groups.setdefault(key, []).append(i)

        group_counts = {str(k): len(v) for k, v in groups.items()}

        if not groups:
            # No candidate executed successfully; return primary
            return {
                'winner_sql': candidates[0],
                'winner_index': 0,
                'winner_reason': 'no_successful_candidates',
                'exec_results': exec_results,
                'group_counts': group_counts,
            }

        # Majority group = group with most members; tie → lowest candidate index in group
        best_group_indices = max(groups.values(), key=lambda idxs: (len(idxs), -min(idxs)))
        winner_idx = min(best_group_indices)
        n_agree = len(best_group_indices)
        reason = f"majority_{n_agree}_of_{len(candidates)}" if n_agree > 1 else "only_successful"

        return {
            'winner_sql': candidates[winner_idx],
            'winner_index': winner_idx,
            'winner_reason': reason,
            'exec_results': exec_results,
            'group_counts': group_counts,
        }
