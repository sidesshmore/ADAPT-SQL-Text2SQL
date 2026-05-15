"""
DB Value Retrieval — DeepEye-SQL "semantic value retrieval" approach.

For each text column in the pruned schema, retrieves distinct DB values and
fuzzy-matches them against candidate filter terms extracted from the question.
Returns a dict of {table.column: matched_db_value} for any close matches.

Purpose: anchor WHERE clause literals to actual DB values when the question uses
a variant spelling (e.g. question says "France" but DB stores "FRA" or "france").
"""
import re
import sqlite3
from difflib import SequenceMatcher
from typing import Dict, List


def _extract_question_terms(question: str) -> set:
    """Extract candidate filter values from a question."""
    terms = set()
    # Quoted strings
    terms.update(re.findall(r'"([^"]+)"', question))
    terms.update(re.findall(r"'([^']+)'", question))
    # Capitalized words (likely proper nouns / named entities)
    for word in re.findall(r'\b[A-Z][a-z]{1,}\b', question):
        terms.add(word)
    # Multi-word capitalized phrases (up to 3 words)
    for phrase in re.findall(r'\b(?:[A-Z][a-z]+\s){1,2}[A-Z][a-z]+\b', question):
        terms.add(phrase.strip())
    return terms


def retrieve_db_values(
    question: str,
    pruned_schema: Dict[str, List[Dict]],
    db_path: str,
    max_vals: int = 50,
    min_similarity: float = 0.82,
) -> Dict[str, str]:
    """
    Retrieve DB values that closely match terms in the question.

    Args:
        question: Natural language question
        pruned_schema: {table: [{column_name, data_type, ...}]}
        db_path: Path to SQLite database
        max_vals: Max distinct values to fetch per column
        min_similarity: SequenceMatcher threshold (0.82 = very close match)

    Returns:
        {table.column: db_value} for columns where a close match was found
    """
    if not db_path or not pruned_schema:
        return {}

    question_terms = _extract_question_terms(question)
    if not question_terms:
        return {}

    matches: Dict[str, str] = {}

    try:
        conn = sqlite3.connect(db_path, timeout=5)
        for table, columns in pruned_schema.items():
            for col in columns:
                col_type = col.get('data_type', '').upper()
                # Only scan text-like columns — skip INT/REAL/NUMERIC
                if col_type and not any(t in col_type for t in ['TEXT', 'CHAR', 'CLOB', 'STRING', 'BLOB', '']):
                    if any(t in col_type for t in ['INT', 'REAL', 'NUMERIC', 'FLOAT', 'DOUBLE', 'BOOL']):
                        continue
                col_name = col['column_name']
                fqn = f"{table}.{col_name}"
                try:
                    rows = conn.execute(
                        f'SELECT DISTINCT "{col_name}" FROM "{table}"'
                        f' WHERE "{col_name}" IS NOT NULL LIMIT {max_vals}'
                    ).fetchall()
                    db_vals = [str(r[0]) for r in rows if r[0] is not None and str(r[0]).strip()]
                except Exception:
                    continue

                for term in question_terms:
                    term_lower = term.lower()
                    for dv in db_vals:
                        dv_lower = dv.lower()
                        if term_lower == dv_lower:
                            continue  # exact match — model will get it right anyway
                        # Containment: "France" matches "South France" or "france_region"
                        is_contained = (
                            len(term_lower) >= 3 and (
                                term_lower in dv_lower or dv_lower in term_lower
                            )
                        )
                        score = SequenceMatcher(None, term_lower, dv_lower).ratio()
                        if score >= min_similarity or is_contained:
                            # Prefer longer match (more specific) if multiple candidates
                            if fqn not in matches or len(dv) > len(matches[fqn]):
                                matches[fqn] = dv
        conn.close()
    except Exception:
        pass

    return matches
