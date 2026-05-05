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
    ]
    INTERSECT_SIGNALS = [
        'who are also', 'that are also', 'which are also',
        'in common', 'shared between', 'appear in both',
        'both', 'at the same time', 'simultaneously',
    ]
    UNION_SIGNALS = [
        'combined with', 'together with', 'as well as',
        'or both', 'either or',
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
                "  SELECT ... FROM ... EXCEPT SELECT ... FROM ...\n"
                "Do NOT approximate this with WHERE filters or NOT IN — use EXCEPT.\n"
            )
        if op == 'INTERSECT':
            return (
                f"IMPORTANT: The phrase \"{sig}\" signals this question requires a SQL INTERSECT set operation.\n"
                "Structure your answer as:\n"
                "  SELECT ... FROM ... INTERSECT SELECT ... FROM ...\n"
                "Do NOT approximate this with JOIN — use INTERSECT.\n"
            )
        if op == 'UNION':
            return (
                f"IMPORTANT: The phrase \"{sig}\" signals this question requires a SQL UNION set operation.\n"
                "Structure your answer as:\n"
                "  SELECT ... FROM ... UNION SELECT ... FROM ...\n"
            )
        return ''
