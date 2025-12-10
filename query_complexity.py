"""
Query Complexity Classifier - STEP 2
Classifies query complexity based on schema linking results
"""
import ollama
import re
from typing import Dict, List, Set
from enum import Enum


class ComplexityClass(Enum):
    """Query complexity classifications"""
    EASY = "EASY"
    NON_NESTED_COMPLEX = "NON_NESTED_COMPLEX"
    NESTED_COMPLEX = "NESTED_COMPLEX"


class QueryComplexityClassifier:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
    
    def classify_query(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict
    ) -> Dict:
        """
        STEP 2: Query Complexity Classification
        
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
                'has_ordering': bool
            }
        """
        print(f"\n{'='*60}")
        print("STEP 2: QUERY COMPLEXITY CLASSIFICATION")
        print(f"{'='*60}\n")
        
        # Extract structural hints
        print("2.1: Extracting structural hints...")
        structural_hints = self._extract_structural_hints(question)
        print(f"   Aggregations: {', '.join(structural_hints['aggregation_types']) if structural_hints['aggregation_types'] else 'None'}")
        
        # Use LLM to classify
        print("2.2: Running LLM classification...")
        llm_classification = self._classify_with_llm(
            question, pruned_schema, schema_links, structural_hints
        )
        
        # Parse classification
        print("2.3: Parsing classification...")
        classification = self._parse_classification(llm_classification, schema_links)
        print(f"   Complexity: {classification['complexity_class'].value}")
        
        # Determine required tables
        required_tables = schema_links['tables']
        
        # Identify sub-questions if nested
        sub_questions = []
        if classification['complexity_class'] == ComplexityClass.NESTED_COMPLEX:
            print("2.4: Identifying sub-questions...")
            sub_questions = self._identify_sub_questions(question, llm_classification)
            print(f"   Found {len(sub_questions)} sub-questions")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question, classification, structural_hints, 
            required_tables, sub_questions, llm_classification
        )
        
        print(f"\n{'='*60}")
        print("STEP 2 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'complexity_class': classification['complexity_class'],
            'required_tables': required_tables,
            'sub_questions': sub_questions,
            'reasoning': reasoning,
            'needs_joins': classification['needs_joins'],
            'needs_subqueries': classification['needs_subqueries'],
            'needs_set_operations': classification['needs_set_operations'],
            'aggregations': classification['aggregations'],
            'has_grouping': classification['has_grouping'],
            'has_ordering': classification['has_ordering'],
            'structural_hints': structural_hints
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
        """Identify sub-questions for nested queries"""
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
            clauses = re.split(r'\s+(?:that|which|who|where)\s+', question, flags=re.IGNORECASE)
            if len(clauses) > 1:
                for clause in clauses[1:]:
                    if len(clause) > 10:
                        sub_questions.append(clause.strip())
        
        return sub_questions[:3]  # Limit to 3
    
    def _generate_reasoning(
        self,
        question: str,
        classification: Dict,
        structural_hints: Dict,
        required_tables: Set[str],
        sub_questions: List[str],
        llm_classification: str
    ) -> str:
        """Generate reasoning"""
        reasoning = "STEP 2: COMPLEXITY CLASSIFICATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += f"Classification: {classification['complexity_class'].value}\n\n"
        
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