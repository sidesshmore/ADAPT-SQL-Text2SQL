"""
Rule-Based Complexity Classification
Add deterministic rules BEFORE LLM classification for better accuracy
"""
import re
from typing import Dict, List, Set
from enum import Enum


class ComplexityClass(Enum):
    EASY = "EASY"
    NON_NESTED_COMPLEX = "NON_NESTED_COMPLEX"
    NESTED_COMPLEX = "NESTED_COMPLEX"


class RuleBasedComplexityClassifier:
    def __init__(self):
        """Initialize rule-based classifier"""
        
        # Nested query indicators (strong signals)
        self.nested_indicators = [
            # Comparison with aggregates
            r'more\s+than\s+(average|avg)',
            r'less\s+than\s+(average|avg)',
            r'greater\s+than\s+(average|avg)',
            r'higher\s+than\s+(average|avg)',
            r'lower\s+than\s+(average|avg)',
            
            # Exclusion patterns
            r'\bexcept\b',
            r'\bnot\s+in\b',
            r'\bno(?:t)?\s+\w+\s+that\b',
            
            # Existence patterns
            r'\bthat\s+(?:have|has|had)\b',
            r'\bwho\s+(?:have|has|had)\b',
            r'\bwhich\s+(?:have|has|had)\b',
            
            # Superlatives with filters
            r'most\s+\w+\s+(?:that|who|which)',
            r'least\s+\w+\s+(?:that|who|which)',
            
            # Nested logic
            r'\b(?:every|all|any)\s+\w+\s+(?:that|who|which)\b',
        ]
        
        # Complex query indicators (no nesting)
        self.complex_indicators = [
            # Multiple aggregations
            r'(count|sum|avg|max|min).*(?:and|,).*(count|sum|avg|max|min)',
            
            # Aggregation with grouping
            r'(?:each|every|per)\s+\w+',
            r'(?:for|by)\s+each',
            
            # Multiple conditions
            r'(?:and|or).*(?:and|or)',
            
            # Join keywords
            r'\b(?:with|from|in)\s+\w+\s+(?:and|with|from)\s+\w+\b',
        ]
        
        # Simple query indicators
        self.simple_indicators = [
            r'^(?:show|list|display|get|find|what)\s+(?:all|the)?\s+\w+\s*$',
            r'^(?:how many|count|total)\s+\w+\s*$',
        ]
    
    def apply_rules(
        self, 
        question: str,
        num_tables: int,
        has_aggregation: bool,
        aggregation_types: List[str]
    ) -> Dict:
        """
        Apply deterministic rules for classification
        
        Returns:
            {
                'classification': ComplexityClass or None,
                'confidence': float,
                'reasoning': str,
                'rule_matched': str or None
            }
        """
        question_lower = question.lower()
        
        # ================================================================
        # RULE 1: NESTED_COMPLEX Detection
        # ================================================================
        for i, pattern in enumerate(self.nested_indicators, 1):
            if re.search(pattern, question_lower, re.IGNORECASE):
                return {
                    'classification': ComplexityClass.NESTED_COMPLEX,
                    'confidence': 0.95,
                    'reasoning': f'Detected nested query pattern: "{pattern}"',
                    'rule_matched': f'NESTED_RULE_{i}'
                }
        
        # Additional nested checks
        if 'more than average' in question_lower or 'less than average' in question_lower:
            return {
                'classification': ComplexityClass.NESTED_COMPLEX,
                'confidence': 0.98,
                'reasoning': 'Comparison with aggregate (requires subquery)',
                'rule_matched': 'NESTED_RULE_AGGREGATE_COMPARISON'
            }
        
        # ================================================================
        # RULE 2: NON_NESTED_COMPLEX Detection  
        # ================================================================
        
        # Multiple tables + aggregation → complex
        if num_tables >= 2 and has_aggregation:
            return {
                'classification': ComplexityClass.NON_NESTED_COMPLEX,
                'confidence': 0.85,
                'reasoning': f'{num_tables} tables + aggregation ({", ".join(aggregation_types)})',
                'rule_matched': 'COMPLEX_RULE_MULTI_TABLE_AGG'
            }
        
        # Multiple aggregations → complex
        if len(aggregation_types) >= 2:
            return {
                'classification': ComplexityClass.NON_NESTED_COMPLEX,
                'confidence': 0.90,
                'reasoning': f'Multiple aggregations: {", ".join(aggregation_types)}',
                'rule_matched': 'COMPLEX_RULE_MULTI_AGG'
            }
        
        # Check complex patterns
        for i, pattern in enumerate(self.complex_indicators, 1):
            if re.search(pattern, question_lower, re.IGNORECASE):
                return {
                    'classification': ComplexityClass.NON_NESTED_COMPLEX,
                    'confidence': 0.80,
                    'reasoning': f'Complex pattern detected: "{pattern}"',
                    'rule_matched': f'COMPLEX_RULE_{i}'
                }
        
        # 3+ tables → complex
        if num_tables >= 3:
            return {
                'classification': ComplexityClass.NON_NESTED_COMPLEX,
                'confidence': 0.85,
                'reasoning': f'{num_tables} tables require multiple JOINs',
                'rule_matched': 'COMPLEX_RULE_MANY_TABLES'
            }
        
        # ================================================================
        # RULE 3: EASY Detection
        # ================================================================
        
        # Single table, no aggregation → easy
        if num_tables == 1 and not has_aggregation:
            return {
                'classification': ComplexityClass.EASY,
                'confidence': 0.95,
                'reasoning': 'Single table, no aggregation',
                'rule_matched': 'EASY_RULE_SINGLE_TABLE'
            }
        
        # Single table with simple aggregation → easy
        if num_tables == 1 and len(aggregation_types) == 1:
            return {
                'classification': ComplexityClass.EASY,
                'confidence': 0.85,
                'reasoning': f'Single table with {aggregation_types[0]}',
                'rule_matched': 'EASY_RULE_SIMPLE_AGG'
            }
        
        # Two tables, no aggregation → easy (simple JOIN)
        if num_tables == 2 and not has_aggregation:
            return {
                'classification': ComplexityClass.EASY,
                'confidence': 0.80,
                'reasoning': '2 tables, simple JOIN',
                'rule_matched': 'EASY_RULE_SIMPLE_JOIN'
            }
        
        # Check simple patterns
        for i, pattern in enumerate(self.simple_indicators, 1):
            if re.search(pattern, question_lower, re.IGNORECASE):
                return {
                    'classification': ComplexityClass.EASY,
                    'confidence': 0.90,
                    'reasoning': f'Simple query pattern: "{pattern}"',
                    'rule_matched': f'EASY_RULE_{i}'
                }
        
        # ================================================================
        # NO CLEAR RULE MATCH - Let LLM decide
        # ================================================================
        return {
            'classification': None,
            'confidence': 0.0,
            'reasoning': 'No definitive rule match - requires LLM analysis',
            'rule_matched': None
        }
    
    def classify_with_llm_fallback(
        self,
        question: str,
        num_tables: int,
        has_aggregation: bool,
        aggregation_types: List[str],
        llm_classification_func=None,
        **llm_kwargs
    ) -> Dict:
        """
        Apply rules first, fallback to LLM if needed
        
        Args:
            question: Natural language question
            num_tables: Number of tables involved
            has_aggregation: Whether query has aggregations
            aggregation_types: List of aggregation types (COUNT, AVG, etc.)
            llm_classification_func: Function to call for LLM classification
            **llm_kwargs: Arguments to pass to LLM function
            
        Returns:
            {
                'complexity_class': ComplexityClass,
                'confidence': float,
                'method': str,  # 'RULE_BASED' or 'LLM'
                'reasoning': str,
                'rule_matched': str or None
            }
        """
        # Apply rules first
        rule_result = self.apply_rules(
            question, num_tables, has_aggregation, aggregation_types
        )
        
        # If rules gave definitive answer with high confidence
        if rule_result['classification'] and rule_result['confidence'] >= 0.80:
            return {
                'complexity_class': rule_result['classification'],
                'confidence': rule_result['confidence'],
                'method': 'RULE_BASED',
                'reasoning': f"Rule-based: {rule_result['reasoning']}",
                'rule_matched': rule_result['rule_matched']
            }
        
        # Otherwise, use LLM
        if llm_classification_func:
            llm_result = llm_classification_func(**llm_kwargs)
            
            return {
                'complexity_class': llm_result['complexity_class'],
                'confidence': 0.70,  # LLM gets medium confidence
                'method': 'LLM',
                'reasoning': f"LLM-based (no clear rule): {llm_result.get('reasoning', 'N/A')}",
                'rule_matched': None
            }
        
        # Fallback to EASY if no LLM
        return {
            'complexity_class': ComplexityClass.EASY,
            'confidence': 0.50,
            'method': 'FALLBACK',
            'reasoning': 'No clear rule or LLM - defaulting to EASY',
            'rule_matched': None
        }


# ============================================================================
# Integration Example
# ============================================================================

def integrate_with_query_complexity():
    """
    Example of how to integrate into query_complexity.py
    """
    
    # In your classify_query method, ADD THIS BEFORE LLM call:
    
    rule_classifier = RuleBasedComplexityClassifier()
    
    # Extract info from question
    question = "Find students who have a GPA higher than the average GPA"
    num_tables = 1  # From schema_links
    has_aggregation = True  # From structural hints
    aggregation_types = ['avg']  # From structural hints
    
    # Apply rules
    classification_result = rule_classifier.classify_with_llm_fallback(
        question=question,
        num_tables=num_tables,
        has_aggregation=has_aggregation,
        aggregation_types=aggregation_types,
        llm_classification_func=None  # Your existing LLM function
    )
    
    print(f"Classification: {classification_result['complexity_class'].value}")
    print(f"Confidence: {classification_result['confidence']:.2f}")
    print(f"Method: {classification_result['method']}")
    print(f"Reasoning: {classification_result['reasoning']}")
    
    if classification_result['rule_matched']:
        print(f"Rule Matched: {classification_result['rule_matched']}")


if __name__ == "__main__":
    integrate_with_query_complexity()