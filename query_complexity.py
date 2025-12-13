"""
Query Complexity Classifier - STEP 2 (ENHANCED with Rule-Based Classification)
Classifies query complexity based on schema linking results
Now uses deterministic rules BEFORE LLM for better accuracy and speed
"""
import ollama
import re
from typing import Dict, List, Set, Optional
from enum import Enum


class ComplexityClass(Enum):
    """Query complexity classifications"""
    EASY = "EASY"
    NON_NESTED_COMPLEX = "NON_NESTED_COMPLEX"
    NESTED_COMPLEX = "NESTED_COMPLEX"


class RuleBasedComplexityClassifier:
    """Deterministic rule-based classifier for fast, accurate classification"""
    
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
            r'above\s+(average|avg)',
            r'below\s+(average|avg)',
            
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


class QueryComplexityClassifier:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.rule_classifier = RuleBasedComplexityClassifier()
    
    def classify_query(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict
    ) -> Dict:
        """
        STEP 2: Query Complexity Classification with Rule-Based Pre-filtering
        
        Flow:
        1. Extract structural hints
        2. Apply rule-based classification (FAST)
        3. If confident → return immediately
        4. If uncertain → fallback to LLM
        
        Args:
            question: Natural language question
            pruned_schema: Pruned schema from Step 1
            schema_links: Schema links from Step 1
            
        Returns:
            {
                'complexity_class': ComplexityClass,
                'required_tables': Set[str],
                'sub_questions': List[str],
                'reasoning': str,
                'needs_joins': bool,
                'needs_subqueries': bool,
                'aggregations': List[str],
                'has_grouping': bool,
                'classification_method': str,  # 'RULE_BASED' or 'LLM'
                'rule_matched': str or None,
                'rule_confidence': float
            }
        """
        print(f"\n{'='*60}")
        print("STEP 2: QUERY COMPLEXITY CLASSIFICATION (RULE-BASED + LLM)")
        print(f"{'='*60}\n")
        
        # Extract structural hints
        print("2.1: Extracting structural hints...")
        structural_hints = self._extract_structural_hints(question)
        print(f"   Aggregations: {', '.join(structural_hints['aggregation_types']) if structural_hints['aggregation_types'] else 'None'}")
        
        # Count tables
        num_tables = len(schema_links['tables'])
        has_aggregation = structural_hints['has_aggregation']
        aggregation_types = structural_hints['aggregation_types']
        
        # Apply rule-based classification first
        print("2.2: Applying rule-based classification...")
        
        rule_result = self.rule_classifier.apply_rules(
            question=question,
            num_tables=num_tables,
            has_aggregation=has_aggregation,
            aggregation_types=aggregation_types
        )
        
        classification = None
        classification_method = None
        rule_matched = None
        llm_classification = None
        
        # If rules give high-confidence answer, use it
        if rule_result['classification'] and rule_result['confidence'] >= 0.80:
            classification = rule_result['classification']
            classification_method = 'RULE_BASED'
            rule_matched = rule_result['rule_matched']
            
            print(f"   ✅ Rule-based: {classification.value}")
            print(f"   Confidence: {rule_result['confidence']:.2f}")
            print(f"   Rule: {rule_matched}")
            print(f"   Reasoning: {rule_result['reasoning']}")
            
            # Parse into standard format
            parsed_classification = {
                'complexity_class': classification,
                'needs_joins': num_tables > 1,
                'needs_subqueries': classification == ComplexityClass.NESTED_COMPLEX,
                'needs_set_operations': False,
                'aggregations': aggregation_types,
                'has_grouping': 'each' in question.lower() or 'per' in question.lower(),
                'has_ordering': 'order' in question.lower() or 'sort' in question.lower()
            }
            
        else:
            # Fallback to LLM for uncertain cases
            print("   ⚠️ No confident rule match - using LLM...")
            
            llm_classification = self._classify_with_llm(
                question, pruned_schema, schema_links, structural_hints
            )
            
            parsed_classification = self._parse_classification(
                llm_classification, schema_links
            )
            
            classification = parsed_classification['complexity_class']
            classification_method = 'LLM'
            rule_matched = None
            
            print(f"   ✅ LLM classification: {classification.value}")
        
        # Determine required tables
        required_tables = schema_links['tables']
        
        # Identify sub-questions if nested
        sub_questions = []
        if classification == ComplexityClass.NESTED_COMPLEX:
            print("2.4: Identifying sub-questions...")
            if llm_classification:
                sub_questions = self._identify_sub_questions(question, llm_classification)
            else:
                # Use heuristic if no LLM was called
                sub_questions = self._heuristic_sub_questions(question)
            print(f"   Found {len(sub_questions)} sub-questions")
        
        # Generate reasoning
        reasoning = self._generate_reasoning_enhanced(
            question=question,
            classification=parsed_classification,
            structural_hints=structural_hints,
            required_tables=required_tables,
            sub_questions=sub_questions,
            classification_method=classification_method,
            rule_matched=rule_matched,
            rule_reasoning=rule_result.get('reasoning'),
            llm_classification=llm_classification
        )
        
        print(f"\n{'='*60}")
        print("STEP 2 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'complexity_class': classification,
            'required_tables': required_tables,
            'sub_questions': sub_questions,
            'reasoning': reasoning,
            'needs_joins': parsed_classification['needs_joins'],
            'needs_subqueries': parsed_classification['needs_subqueries'],
            'needs_set_operations': parsed_classification['needs_set_operations'],
            'aggregations': parsed_classification['aggregations'],
            'has_grouping': parsed_classification['has_grouping'],
            'has_ordering': parsed_classification['has_ordering'],
            'structural_hints': structural_hints,
            # NEW FIELDS:
            'classification_method': classification_method,
            'rule_matched': rule_matched,
            'rule_confidence': rule_result.get('confidence', 0.0)
        }
    
    def _extract_structural_hints(self, question: str) -> Dict:
        """Extract structural hints from question"""
        question_lower = question.lower()
        
        hints = {
            'has_aggregation': False,
            'aggregation_types': [],
            'has_comparison': False,
            'has_superlative': False,
            'has_nested_logic': False
        }
        
        # Aggregations
        agg_keywords = {
            'count': ['how many', 'count', 'number of'],
            'avg': ['average', 'avg', 'mean'],
            'sum': ['sum', 'total'],
            'max': ['maximum', 'max', 'highest', 'most', 'top'],
            'min': ['minimum', 'min', 'lowest', 'least']
        }
        
        for agg_type, keywords in agg_keywords.items():
            if any(kw in question_lower for kw in keywords):
                hints['has_aggregation'] = True
                hints['aggregation_types'].append(agg_type)
        
        # Comparisons
        comparison_keywords = ['more than', 'less than', 'greater than', 'fewer than']
        hints['has_comparison'] = any(kw in question_lower for kw in comparison_keywords)
        
        # Superlatives
        superlative_keywords = ['most', 'least', 'highest', 'lowest', 'best', 'worst']
        hints['has_superlative'] = any(kw in question_lower for kw in superlative_keywords)
        
        # Nested logic
        nested_indicators = ['that have', 'who have', 'which have', 'except', 'not in']
        hints['has_nested_logic'] = any(ind in question_lower for ind in nested_indicators)
        
        return hints
    
    def _classify_with_llm(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        structural_hints: Dict
    ) -> str:
        """Use LLM to classify complexity"""
        
        # Create schema summary
        schema_str = ""
        for table, cols in pruned_schema.items():
            col_names = [c['column_name'] for c in cols]
            schema_str += f"  {table}: {', '.join(col_names)}\n"
        
        # FK info
        fk_str = ""
        if schema_links.get('foreign_keys'):
            fk_str = "\nForeign Keys:\n"
            for fk in schema_links['foreign_keys']:
                fk_str += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        
        prompt = f"""PRUNED SCHEMA:
{schema_str}{fk_str}

QUESTION: {question}

HINTS:
- Aggregations: {', '.join(structural_hints['aggregation_types']) if structural_hints['aggregation_types'] else 'None'}
- Has comparison: {structural_hints['has_comparison']}
- Has superlative: {structural_hints['has_superlative']}
- Nested logic: {structural_hints['has_nested_logic']}

Classify this query's SQL complexity:

A) EASY:
   - Single table OR simple JOIN
   - No subqueries/nesting
   - Example: "List all singers" or "Show concerts with their stadiums"

B) NON_NESTED_COMPLEX:
   - Multiple JOINs
   - No subqueries
   - May have aggregations with GROUP BY
   - Example: "Count concerts per stadium"

C) NESTED_COMPLEX:
   - Requires subqueries (IN, NOT IN, >/<, etc.)
   - Comparisons with aggregates ("more than average")
   - Example: "Singers who performed at more concerts than average"

ANALYZE:
1. How many tables needed?
2. Are JOINs required?
3. Are subqueries needed?
4. What aggregations (COUNT, AVG, MAX, etc.)?
5. Is GROUP BY needed?
6. Is ORDER BY needed?

CLASSIFICATION: [EASY/NON_NESTED_COMPLEX/NESTED_COMPLEX]

Provide concise analysis:"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a SQL complexity classifier. Analyze query requirements and classify accurately.'
                    },
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _parse_classification(
        self, 
        llm_classification: str,
        schema_links: Dict
    ) -> Dict:
        """Parse LLM classification"""
        
        result = {
            'complexity_class': ComplexityClass.EASY,
            'needs_joins': False,
            'needs_subqueries': False,
            'needs_set_operations': False,
            'aggregations': [],
            'has_grouping': False,
            'has_ordering': False
        }
        
        text_upper = llm_classification.upper()
        text_lower = llm_classification.lower()
        
        # Determine complexity
        if 'NESTED_COMPLEX' in text_upper or 'NESTED COMPLEX' in text_upper:
            result['complexity_class'] = ComplexityClass.NESTED_COMPLEX
        elif 'NON_NESTED_COMPLEX' in text_upper or 'NON-NESTED' in text_upper:
            result['complexity_class'] = ComplexityClass.NON_NESTED_COMPLEX
        elif 'EASY' in text_upper:
            result['complexity_class'] = ComplexityClass.EASY
        else:
            # Fallback
            if len(schema_links.get('tables', [])) > 1:
                result['complexity_class'] = ComplexityClass.NON_NESTED_COMPLEX
        
        # Check for JOINs
        result['needs_joins'] = any(ind in text_lower for ind in ['join', 'multiple tables'])
        
        # Check for subqueries
        result['needs_subqueries'] = any(ind in text_lower for ind in ['subquery', 'nested', 'sub-query'])
        
        # Check for set operations
        result['needs_set_operations'] = any(ind in text_lower for ind in ['union', 'intersect', 'except'])
        
        # Extract aggregations
        for agg in ['COUNT', 'AVG', 'SUM', 'MAX', 'MIN']:
            if agg in text_upper:
                result['aggregations'].append(agg)
        
        # Check for GROUP BY
        result['has_grouping'] = 'group by' in text_lower or 'grouping' in text_lower
        
        # Check for ORDER BY
        result['has_ordering'] = 'order by' in text_lower or 'sort' in text_lower
        
        return result
    
    def _identify_sub_questions(
        self,
        question: str,
        llm_classification: str
    ) -> List[str]:
        """Identify sub-questions for nested queries from LLM output"""
        sub_questions = []
        
        # Look for sub-question patterns
        patterns = [
            r'(?:sub-?questions?|sub-?queries?):\s*(.+?)(?:\n\n|\*\*|CLASSIFICATION|$)',
            r'(?:first|inner|outer) (?:query|question):\s*(.+?)(?:\n|$)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, llm_classification, re.IGNORECASE | re.DOTALL)
            if matches:
                text = matches[0]
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    line = re.sub(r'^[\d\-\*\•]+[\.\):]?\s*', '', line)
                    if len(line) > 10:
                        sub_questions.append(line)
        
        # Heuristic: look for clauses
        if not sub_questions:
            sub_questions = self._heuristic_sub_questions(question)
        
        return sub_questions[:3]  # Limit to 3
    
    def _heuristic_sub_questions(self, question: str) -> List[str]:
        """
        Generate sub-questions heuristically without LLM
        Used when rules confidently classify as NESTED_COMPLEX
        """
        sub_questions = []
        
        # Pattern 1: "X that have Y" → ["What are the Y?", "Which X have those Y?"]
        match = re.search(r'(\w+)\s+that\s+have\s+(.*?)(?:\?|$)', question, re.IGNORECASE)
        if match:
            entity = match.group(1)
            condition = match.group(2)
            sub_questions.append(f"What are the {condition}?")
            sub_questions.append(f"Which {entity} have those?")
            return sub_questions
        
        # Pattern 2: "more than average X" → ["What is the average X?", "Which are more than that?"]
        match = re.search(r'more\s+than\s+(?:the\s+)?average\s+(\w+)', question, re.IGNORECASE)
        if match:
            metric = match.group(1)
            sub_questions.append(f"What is the average {metric}?")
            sub_questions.append(f"Which entries have {metric} more than the average?")
            return sub_questions
        
        # Pattern 3: "except X" → ["What are all items?", "What are X?", "Remove X from all"]
        if 'except' in question.lower():
            sub_questions.append("Identify all items")
            sub_questions.append("Identify items to exclude")
            sub_questions.append("Return items not in exclusion set")
            return sub_questions
        
        # Generic fallback
        sub_questions.append("Inner query: " + question.split(' that ')[0] if ' that ' in question else question)
        sub_questions.append("Outer query: Apply condition from inner result")
        
        return sub_questions
    
    def _generate_reasoning_enhanced(
        self,
        question: str,
        classification: Dict,
        structural_hints: Dict,
        required_tables: Set[str],
        sub_questions: List[str],
        classification_method: str,
        rule_matched: Optional[str],
        rule_reasoning: Optional[str],
        llm_classification: Optional[str]
    ) -> str:
        """Generate enhanced reasoning with rule info"""
        reasoning = "STEP 2: COMPLEXITY CLASSIFICATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += f"Classification: {classification['complexity_class'].value}\n"
        reasoning += f"Method: {classification_method}\n"
        
        if rule_matched:
            reasoning += f"Rule Matched: {rule_matched}\n"
            reasoning += f"Rule Reasoning: {rule_reasoning}\n"
        
        reasoning += "\n"
        
        reasoning += "Analysis:\n"
        reasoning += f"  • Tables: {len(required_tables)} ({', '.join(sorted(required_tables))})\n"
        reasoning += f"  • Needs JOINs: {'Yes' if classification['needs_joins'] else 'No'}\n"
        reasoning += f"  • Needs subqueries: {'Yes' if classification['needs_subqueries'] else 'No'}\n"
        
        if classification['aggregations']:
            reasoning += f"  • Aggregations: {', '.join(classification['aggregations'])}\n"
        
        reasoning += f"  • GROUP BY: {'Yes' if classification['has_grouping'] else 'No'}\n"
        reasoning += f"  • ORDER BY: {'Yes' if classification['has_ordering'] else 'No'}\n"
        
        if sub_questions:
            reasoning += f"\nSub-questions ({len(sub_questions)}):\n"
            for i, sq in enumerate(sub_questions, 1):
                reasoning += f"  {i}. {sq}\n"
        
        if llm_classification:
            reasoning += "\n" + "-" * 50 + "\n"
            reasoning += "LLM Analysis:\n"
            reasoning += llm_classification + "\n"
        
        return reasoning
    
    def get_generation_strategy(self, complexity_class: ComplexityClass) -> str:
        """Return generation strategy for complexity class"""
        strategies = {
            ComplexityClass.EASY: "SIMPLE_FEW_SHOT",
            ComplexityClass.NON_NESTED_COMPLEX: "INTERMEDIATE_REPRESENTATION",
            ComplexityClass.NESTED_COMPLEX: "DECOMPOSED_GENERATION"
        }
        return strategies.get(complexity_class, "SIMPLE_FEW_SHOT")