"""
Enhanced Fuzzy Schema Validator with Aggressive Matching
Reduces false-positive schema errors through intelligent fuzzy matching
"""
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher
import re


class FuzzySchemaValidator:
    def __init__(
        self, 
        fuzzy_threshold: float = 0.7,
        substring_threshold: float = 0.8
    ):
        """
        Initialize fuzzy validator
        
        Args:
            fuzzy_threshold: Minimum similarity for fuzzy matches (0.7 = 70%)
            substring_threshold: Minimum ratio for substring matches
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.substring_threshold = substring_threshold
        
        # Common prefixes to strip
        self.common_prefixes = [
            'tbl_', 'table_', 't_', 'tb_',
            'col_', 'column_', 'c_', 'fld_'
        ]
        
        # Common suffixes to strip  
        self.common_suffixes = [
            '_id', '_name', '_value', '_data', '_info'
        ]
        
        # Plural/singular mappings
        self.plural_rules = [
            (r'ies$', 'y'),      # countries → country
            (r'es$', ''),        # classes → class
            (r's$', ''),         # students → student
        ]
    
    def normalize_name(self, name: str) -> str:
        """
        Normalize column/table name for fuzzy matching
        
        Examples:
            student_name → studentname
            tbl_students → students  
            Countries → country
        """
        normalized = name.lower().strip()
        
        # Strip common prefixes
        for prefix in self.common_prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        # Remove underscores and spaces
        normalized = normalized.replace('_', '').replace(' ', '')
        
        # Handle plural → singular
        for pattern, replacement in self.plural_rules:
            if re.search(pattern, normalized):
                normalized = re.sub(pattern, replacement, normalized)
                break
        
        return normalized
    
    def fuzzy_match_score(self, name1: str, name2: str) -> float:
        """
        Calculate fuzzy match score between two names
        
        Returns:
            Score between 0.0 and 1.0
        """
        # Exact match
        if name1.lower() == name2.lower():
            return 1.0
        
        # Normalized match
        norm1 = self.normalize_name(name1)
        norm2 = self.normalize_name(name2)
        
        if norm1 == norm2:
            return 0.95
        
        # Substring match (one contains the other)
        if norm1 in norm2 or norm2 in norm1:
            ratio = min(len(norm1), len(norm2)) / max(len(norm1), len(norm2))
            if ratio >= self.substring_threshold:
                return 0.90
        
        # Levenshtein-based similarity
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def find_best_match(
        self, 
        target: str, 
        candidates: Set[str],
        threshold: Optional[float] = None
    ) -> Optional[Tuple[str, float]]:
        """
        Find best matching candidate for target
        
        Returns:
            (best_match, score) or None if no match above threshold
        """
        if threshold is None:
            threshold = self.fuzzy_threshold
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = self.fuzzy_match_score(target, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate
        
        if best_score >= threshold:
            return (best_match, best_score)
        
        return None
    
    def validate_column_with_suggestions(
        self,
        column: str,
        table: str,
        schema_dict: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Validate column and provide suggestions if not found
        
        Returns:
            {
                'is_valid': bool,
                'severity': str,  # 'NONE', 'WARNING', 'ERROR'
                'message': str,
                'suggestion': str or None,
                'confidence': float
            }
        """
        # Check if table exists
        if table not in schema_dict:
            return {
                'is_valid': False,
                'severity': 'ERROR',
                'message': f'Table "{table}" does not exist',
                'suggestion': None,
                'confidence': 0.0
            }
        
        # Get table columns
        table_columns = {col['column_name'] for col in schema_dict[table]}
        
        # Exact match - all good
        if column in table_columns:
            return {
                'is_valid': True,
                'severity': 'NONE',
                'message': f'Column "{column}" exists in table "{table}"',
                'suggestion': None,
                'confidence': 1.0
            }
        
        # Try fuzzy match
        match_result = self.find_best_match(column, table_columns)
        
        if match_result:
            matched_col, score = match_result
            
            # High confidence match (>= 0.9) - treat as WARNING
            if score >= 0.90:
                return {
                    'is_valid': False,
                    'severity': 'WARNING',
                    'message': f'Column "{column}" not found, but "{matched_col}" is very similar',
                    'suggestion': matched_col,
                    'confidence': score
                }
            
            # Medium confidence match (0.7-0.9) - still WARNING
            elif score >= self.fuzzy_threshold:
                return {
                    'is_valid': False,
                    'severity': 'WARNING',
                    'message': f'Column "{column}" not found. Did you mean "{matched_col}"?',
                    'suggestion': matched_col,
                    'confidence': score
                }
        
        # No good match found - hard ERROR
        # But still provide top 3 candidates
        scored_candidates = [
            (col, self.fuzzy_match_score(column, col))
            for col in table_columns
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_3 = [col for col, _ in scored_candidates[:3]]
        
        return {
            'is_valid': False,
            'severity': 'ERROR',
            'message': f'Column "{column}" does not exist in table "{table}"',
            'suggestion': f"Similar columns: {', '.join(top_3)}",
            'confidence': 0.0
        }
    
    def validate_table_with_suggestions(
        self,
        table: str,
        schema_dict: Dict[str, List[Dict]]
    ) -> Dict:
        """
        Validate table and provide suggestions if not found
        
        Returns:
            Same format as validate_column_with_suggestions
        """
        all_tables = set(schema_dict.keys())
        
        # Exact match
        if table in all_tables:
            return {
                'is_valid': True,
                'severity': 'NONE',
                'message': f'Table "{table}" exists',
                'suggestion': None,
                'confidence': 1.0
            }
        
        # Try fuzzy match
        match_result = self.find_best_match(table, all_tables, threshold=0.75)
        
        if match_result:
            matched_table, score = match_result
            
            # High confidence
            if score >= 0.90:
                return {
                    'is_valid': False,
                    'severity': 'WARNING',
                    'message': f'Table "{table}" not found, but "{matched_table}" is very similar',
                    'suggestion': matched_table,
                    'confidence': score
                }
            
            # Medium confidence
            else:
                return {
                    'is_valid': False,
                    'severity': 'WARNING',
                    'message': f'Table "{table}" not found. Did you mean "{matched_table}"?',
                    'suggestion': matched_table,
                    'confidence': score
                }
        
        # No good match
        scored_candidates = [
            (tbl, self.fuzzy_match_score(table, tbl))
            for tbl in all_tables
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_3 = [tbl for tbl, _ in scored_candidates[:3]]
        
        return {
            'is_valid': False,
            'severity': 'ERROR',
            'message': f'Table "{table}" does not exist',
            'suggestion': f"Similar tables: {', '.join(top_3)}",
            'confidence': 0.0
        }


# ============================================================================
# Integration Example
# ============================================================================

def integrate_fuzzy_validator_example():
    """
    Example of how to integrate FuzzySchemaValidator into validate_sql.py
    """
    
    # Initialize validator
    fuzzy_validator = FuzzySchemaValidator(
        fuzzy_threshold=0.7,      # 70% similarity required
        substring_threshold=0.8    # 80% overlap for substring matches
    )
    
    # Example schema
    schema_dict = {
        'students': [
            {'column_name': 'student_id', 'data_type': 'INTEGER'},
            {'column_name': 'name', 'data_type': 'TEXT'},
            {'column_name': 'age', 'data_type': 'INTEGER'},
            {'column_name': 'gpa', 'data_type': 'REAL'}
        ],
        'courses': [
            {'column_name': 'course_id', 'data_type': 'INTEGER'},
            {'column_name': 'course_name', 'data_type': 'TEXT'},
            {'column_name': 'credits', 'data_type': 'INTEGER'}
        ]
    }
    
    # Test cases
    test_cases = [
        ('student_name', 'students'),    # Should suggest 'name'
        ('student_id', 'students'),      # Exact match
        ('studentname', 'students'),     # Should suggest 'name' 
        ('gpa', 'students'),             # Exact match
        ('invalid_col', 'students'),     # No match, show similar
        ('course', 'courses'),           # Should suggest 'course_name' or 'course_id'
    ]
    
    print("Fuzzy Schema Validation Examples:")
    print("=" * 60)
    
    for column, table in test_cases:
        result = fuzzy_validator.validate_column_with_suggestions(
            column, table, schema_dict
        )
        
        print(f"\nColumn: {table}.{column}")
        print(f"  Valid: {result['is_valid']}")
        print(f"  Severity: {result['severity']}")
        print(f"  Message: {result['message']}")
        if result['suggestion']:
            print(f"  Suggestion: {result['suggestion']}")
            print(f"  Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    integrate_fuzzy_validator_example()