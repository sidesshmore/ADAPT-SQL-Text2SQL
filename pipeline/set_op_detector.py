"""
Set Operation Detector — Phase B.

Detects when a natural language question likely requires a SQL set operation
(EXCEPT, INTERSECT, UNION) rather than a WHERE filter.

~3% of Spider dev queries need set ops; the system previously never generated them.
Signal-based detection with confidence scores; returns empty dict on no signal.
"""


class SetOpDetector:
    # Ordered from most specific to least to minimize false positives
    EXCEPT_SIGNALS = [
        'but not', 'except for', 'excluding', 'who are not', 'that are not',
        'which are not', 'not in common', 'who never', 'that never',
        'have not', 'has not', 'do not have', 'does not have',
        'without any', 'minus',
        # Additional Spider EXCEPT patterns
        "haven't", "hasn't",
        'never been', 'never had', 'never taught', 'never attended',
        'who did not', 'that did not', 'which did not',
        'not among', 'absent from',
        # Language / attribute negation (specific enough to be safe)
        'do not speak', 'does not speak',
    ]
    INTERSECT_SIGNALS = [
        'who are also', 'that are also', 'which are also',
        'in common', 'shared between', 'appear in both',
        'at the same time', 'simultaneously',
        # Additional Spider INTERSECT patterns
        'who has both', 'appear in all', 'member of both',
        # High-value new signals
        'in both',          # "played in both 2013 and 2016"
        'won both',         # "won both X and Y"
        'both have',        # "who both have friends and are liked"
        'are also',         # "have friends and are also liked"
        'from both',        # "teams from both leagues"
        'enrolled in both', # enrollment queries
    ]
    UNION_SIGNALS = [
        'combined with', 'together with', 'as well as',
        'or both', 'either or',
        # Additional Spider UNION patterns (kept narrow to avoid false positives)
        'or those who', 'or all those', 'union of',
        # 'either' triggers on "either X or Y" — UNION produces same result as WHERE OR
        'either',
    ]

    def detect(self, question: str) -> dict:
        """
        Returns dict with op and confidence, or empty dict if no signal found.
        {'op': 'EXCEPT'|'INTERSECT'|'UNION', 'confidence': float, 'signal': str}
        """
        q = question.lower()

        for sig in self.EXCEPT_SIGNALS:
            if sig in q:
                return {'op': 'EXCEPT', 'confidence': 0.85, 'signal': sig}

        for sig in self.INTERSECT_SIGNALS:
            if sig in q:
                return {'op': 'INTERSECT', 'confidence': 0.75, 'signal': sig}

        for sig in self.UNION_SIGNALS:
            if sig in q:
                return {'op': 'UNION', 'confidence': 0.65, 'signal': sig}

        return {}

    def make_hint(self, question: str) -> str:
        """
        Returns a prompt hint string if a set op is detected, else empty string.
        The hint is injected into generation prompts.
        """
        detection = self.detect(question)
        if not detection:
            return ''
        op = detection['op']
        sig = detection['signal']

        if op == 'EXCEPT':
            return (
                f"IMPORTANT: The phrase \"{sig}\" signals this question requires a SQL EXCEPT set operation.\n"
                "Structure your answer as:\n"
                "  SELECT col FROM table1 [WHERE ...]\n"
                "  EXCEPT\n"
                "  SELECT col FROM table2 [WHERE ...]\n"
                "Do NOT approximate this with WHERE filters or NOT IN — use EXCEPT.\n"
            )
        if op == 'INTERSECT':
            return (
                f"IMPORTANT: The phrase \"{sig}\" signals this question requires a SQL INTERSECT set operation.\n"
                "Structure your answer as:\n"
                "  SELECT col FROM table1 [WHERE condition1]\n"
                "  INTERSECT\n"
                "  SELECT col FROM table2 [WHERE condition2]\n"
                "Do NOT approximate this with JOIN — use INTERSECT.\n"
            )
        if op == 'UNION':
            return (
                f"IMPORTANT: The phrase \"{sig}\" signals this question requires a SQL UNION set operation.\n"
                "Structure your answer as:\n"
                "  SELECT col FROM table1 [WHERE condition1]\n"
                "  UNION\n"
                "  SELECT col FROM table2 [WHERE condition2]\n"
                "Do NOT approximate this with OR in WHERE — use UNION.\n"
            )
        return ''

    def verify_sql_uses_setop(self, sql: str, detected_op: str) -> bool:
        """
        Check that generated SQL actually contains the expected set operation keyword.
        Used for post-generation enforcement.
        """
        if not sql or not detected_op:
            return True  # no enforcement if no detection
        sql_upper = sql.upper()
        op = detected_op.upper()
        if op == 'EXCEPT':
            return 'EXCEPT' in sql_upper
        if op == 'INTERSECT':
            return 'INTERSECT' in sql_upper
        if op == 'UNION':
            return 'UNION' in sql_upper
        return True
