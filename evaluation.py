"""
STEP 11: Evaluation
Evaluates generated SQL against ground truth using multiple metrics
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, List, Set


class Text2SQLEvaluator:
    def __init__(self):
        """Initialize evaluator"""
        pass
    
    def evaluate_example(
        self,
        question: str,
        generated_sql: str,
        gold_sql: str,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> Dict:
        """
        STEP 11: Evaluation
        
        Args:
            question: Natural language question
            generated_sql: Generated SQL query
            gold_sql: Ground truth SQL query
            generated_execution: Execution result of generated SQL (from Step 10)
            gold_execution: Execution result of gold SQL (from Step 10)
            
        Returns:
            {
                'execution_accuracy': bool,
                'exact_match': bool,
                'normalized_match': bool,
                'semantic_equivalence': float (0.0-1.0),
                'component_scores': Dict,
                'evaluation_score': float (0.0-1.0),
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 11: EVALUATION")
        print(f"{'='*60}\n")
        
        print(f"Question: {question}")
        
        # 1. Execution Accuracy
        print("11.1: Checking execution accuracy...")
        execution_accuracy = self._check_execution_accuracy(
            generated_execution, gold_execution
        )
        print(f"   Execution accuracy: {'✅ PASS' if execution_accuracy else '❌ FAIL'}")
        
        # 2. Exact Match
        print("11.2: Checking exact match...")
        exact_match = self._check_exact_match(generated_sql, gold_sql)
        print(f"   Exact match: {'✅ YES' if exact_match else '❌ NO'}")
        
        # 3. Normalized Match
        print("11.3: Checking normalized match...")
        normalized_match = self._check_normalized_match(generated_sql, gold_sql)
        print(f"   Normalized match: {'✅ YES' if normalized_match else '❌ NO'}")
        
        # 4. Semantic Equivalence
        print("11.4: Checking semantic equivalence...")
        semantic_equivalence = self._check_semantic_equivalence(
            generated_sql, gold_sql
        )
        print(f"   Semantic equivalence: {semantic_equivalence:.2f}")
        
        # 5. Component-level Scores
        print("11.5: Computing component-level scores...")
        component_scores = self._compute_component_scores(
            generated_sql, gold_sql
        )
        
        # 6. Overall Evaluation Score
        evaluation_score = self._compute_evaluation_score(
            execution_accuracy,
            exact_match,
            normalized_match,
            semantic_equivalence,
            component_scores
        )
        print(f"   Overall evaluation score: {evaluation_score:.2f}")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question,
            generated_sql,
            gold_sql,
            execution_accuracy,
            exact_match,
            normalized_match,
            semantic_equivalence,
            component_scores,
            evaluation_score,
            generated_execution,
            gold_execution
        )
        
        print(f"\n{'='*60}")
        print("STEP 11 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'execution_accuracy': execution_accuracy,
            'exact_match': exact_match,
            'normalized_match': normalized_match,
            'semantic_equivalence': semantic_equivalence,
            'component_scores': component_scores,
            'evaluation_score': evaluation_score,
            'reasoning': reasoning
        }
    
    def _check_execution_accuracy(
        self,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> bool:
        """Check if execution results match"""
        # Both must execute successfully
        if not generated_execution['success'] or not gold_execution['success']:
            return False
        
        gen_df = generated_execution['result_df']
        gold_df = gold_execution['result_df']
        
        # Check if DataFrames are equal
        try:
            # Handle empty DataFrames
            if len(gen_df) == 0 and len(gold_df) == 0:
                return True
            
            # Check shape first
            if gen_df.shape != gold_df.shape:
                return False
            
            # If only one is empty, they're different
            if len(gen_df) == 0 or len(gold_df) == 0:
                return False
            
            # Normalize column names (case-insensitive, strip whitespace)
            gen_df_copy = gen_df.copy()
            gold_df_copy = gold_df.copy()
            
            gen_df_copy.columns = [str(col).strip().lower() for col in gen_df_copy.columns]
            gold_df_copy.columns = [str(col).strip().lower() for col in gold_df_copy.columns]
            
            # Sort columns alphabetically
            gen_df_copy = gen_df_copy.sort_index(axis=1)
            gold_df_copy = gold_df_copy.sort_index(axis=1)
            
            # Sort rows by all columns
            try:
                gen_df_copy = gen_df_copy.sort_values(by=list(gen_df_copy.columns)).reset_index(drop=True)
                gold_df_copy = gold_df_copy.sort_values(by=list(gold_df_copy.columns)).reset_index(drop=True)
            except:
                # If sorting fails, just reset index
                gen_df_copy = gen_df_copy.reset_index(drop=True)
                gold_df_copy = gold_df_copy.reset_index(drop=True)
            
            # Convert all values to strings for comparison (handles type differences)
            gen_df_str = gen_df_copy.astype(str)
            gold_df_str = gold_df_copy.astype(str)
            
            # Compare using equals
            if gen_df_str.equals(gold_df_str):
                return True
            
            # If equals fails, try element-wise comparison with tolerance for floating point
            try:
                # Check if all values match (with tolerance for numeric values)
                for col in gen_df_copy.columns:
                    gen_col = gen_df_copy[col]
                    gold_col = gold_df_copy[col]
                    
                    # Try numeric comparison first
                    try:
                        gen_numeric = pd.to_numeric(gen_col, errors='coerce')
                        gold_numeric = pd.to_numeric(gold_col, errors='coerce')
                        
                        if not np.allclose(gen_numeric, gold_numeric, rtol=1e-5, atol=1e-8, equal_nan=True):
                            return False
                    except:
                        # Fall back to string comparison
                        if not gen_col.astype(str).equals(gold_col.astype(str)):
                            return False
                
                return True
            except:
                return False
                
        except Exception as e:
            print(f"   ⚠️ Error comparing DataFrames: {e}")
            return False
    
    def _check_exact_match(self, generated_sql: str, gold_sql: str) -> bool:
        """Check if SQL queries match exactly (case-insensitive, whitespace-normalized)"""
        gen_normalized = ' '.join(generated_sql.upper().split())
        gold_normalized = ' '.join(gold_sql.upper().split())
        
        return gen_normalized == gold_normalized
    
    def _check_normalized_match(self, generated_sql: str, gold_sql: str) -> bool:
        """Check if SQL queries match after normalization"""
        gen_normalized = self._normalize_sql(generated_sql)
        gold_normalized = self._normalize_sql(gold_sql)
        
        return gen_normalized == gold_normalized
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL query for comparison"""
        # Convert to uppercase
        sql = sql.upper()
        
        # Remove comments
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
        
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        
        # Remove semicolon at end
        sql = sql.rstrip(';')
        
        # Normalize quotes
        sql = sql.replace('"', "'")
        
        # Sort ORDER BY clauses
        sql = self._normalize_order_by(sql)
        
        return sql
    
    def _normalize_order_by(self, sql: str) -> str:
        """Normalize ORDER BY clause"""
        order_by_match = re.search(r'ORDER\s+BY\s+([^;]+)', sql, re.IGNORECASE)
        
        if order_by_match:
            order_clause = order_by_match.group(1)
            # Split by comma and sort
            parts = [p.strip() for p in order_clause.split(',')]
            parts_sorted = sorted(parts)
            
            sql = sql[:order_by_match.start(1)] + ', '.join(parts_sorted) + sql[order_by_match.end(1):]
        
        return sql
    
    def _check_semantic_equivalence(
        self,
        generated_sql: str,
        gold_sql: str
    ) -> float:
        """
        Check semantic equivalence using SQL component analysis
        Returns score from 0.0 to 1.0
        """
        gen_components = self._extract_sql_components(generated_sql)
        gold_components = self._extract_sql_components(gold_sql)
        
        scores = []
        
        # Compare SELECT columns
        if gen_components['select_columns'] and gold_components['select_columns']:
            select_score = self._jaccard_similarity(
                gen_components['select_columns'],
                gold_components['select_columns']
            )
            scores.append(select_score)
        
        # Compare FROM tables
        if gen_components['from_tables'] and gold_components['from_tables']:
            from_score = self._jaccard_similarity(
                gen_components['from_tables'],
                gold_components['from_tables']
            )
            scores.append(from_score)
        
        # Compare WHERE conditions
        if gen_components['where_conditions'] and gold_components['where_conditions']:
            where_score = self._jaccard_similarity(
                gen_components['where_conditions'],
                gold_components['where_conditions']
            )
            scores.append(where_score)
        
        # Compare JOINs
        if gen_components['joins'] and gold_components['joins']:
            join_score = self._jaccard_similarity(
                gen_components['joins'],
                gold_components['joins']
            )
            scores.append(join_score)
        
        # Average scores
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0
    
    def _extract_sql_components(self, sql: str) -> Dict:
        """Extract SQL components for comparison"""
        sql_upper = sql.upper()
        
        components = {
            'select_columns': set(),
            'from_tables': set(),
            'where_conditions': set(),
            'joins': set(),
            'group_by': set(),
            'order_by': set(),
            'aggregations': set()
        }
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            # Split by comma
            columns = [c.strip() for c in select_clause.split(',')]
            components['select_columns'] = set(columns)
        
        # Extract FROM tables
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            components['from_tables'].add(from_match.group(1))
        
        # Extract JOINs
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        components['from_tables'].update(join_matches)
        components['joins'] = set(join_matches)
        
        # Extract WHERE conditions
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Simple tokenization
            conditions = re.findall(r'\w+\s*[=<>!]+\s*\w+', where_clause)
            components['where_conditions'] = set(conditions)
        
        # Extract aggregations
        agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
        for agg in agg_functions:
            if agg in sql_upper:
                components['aggregations'].add(agg)
        
        return components
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_component_scores(
        self,
        generated_sql: str,
        gold_sql: str
    ) -> Dict:
        """Compute component-level scores"""
        gen_components = self._extract_sql_components(generated_sql)
        gold_components = self._extract_sql_components(gold_sql)
        
        scores = {}
        
        # SELECT clause score
        scores['select'] = self._jaccard_similarity(
            gen_components['select_columns'],
            gold_components['select_columns']
        )
        
        # FROM clause score
        scores['from'] = self._jaccard_similarity(
            gen_components['from_tables'],
            gold_components['from_tables']
        )
        
        # WHERE clause score
        scores['where'] = self._jaccard_similarity(
            gen_components['where_conditions'],
            gold_components['where_conditions']
        )
        
        # JOIN score
        scores['join'] = self._jaccard_similarity(
            gen_components['joins'],
            gold_components['joins']
        )
        
        # Aggregation score
        scores['aggregation'] = self._jaccard_similarity(
            gen_components['aggregations'],
            gold_components['aggregations']
        )
        
        return scores
    
    def _compute_evaluation_score(
        self,
        execution_accuracy: bool,
        exact_match: bool,
        normalized_match: bool,
        semantic_equivalence: float,
        component_scores: Dict
    ) -> float:
        """
        Compute overall evaluation score
        
        Weights (revised for better accuracy):
        - Execution accuracy: 60% (most important!)
        - Exact/normalized match: 15%
        - Semantic equivalence: 15%
        - Component scores: 10%
        """
        score = 0.0
        
        # Execution accuracy (60%) - This is the most important metric
        if execution_accuracy:
            score += 0.60
        
        # Exact/normalized match (15%)
        if exact_match:
            score += 0.15
        elif normalized_match:
            score += 0.12  # Slightly less if only normalized match
        
        # Semantic equivalence (15%)
        score += semantic_equivalence * 0.15
        
        # Component scores (10%)
        if component_scores:
            avg_component = sum(component_scores.values()) / len(component_scores)
            score += avg_component * 0.10
        
        return min(score, 1.0)
    
    def _generate_reasoning(
        self,
        question: str,
        generated_sql: str,
        gold_sql: str,
        execution_accuracy: bool,
        exact_match: bool,
        normalized_match: bool,
        semantic_equivalence: float,
        component_scores: Dict,
        evaluation_score: float,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> str:
        """Generate evaluation reasoning"""
        reasoning = "STEP 11: EVALUATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += "Generated SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += "Ground Truth SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += gold_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += "EVALUATION METRICS\n"
        reasoning += "=" * 50 + "\n\n"
        
        # 1. Execution Accuracy
        reasoning += f"1. Execution Accuracy: {'✅ PASS' if execution_accuracy else '❌ FAIL'}\n"
        if execution_accuracy:
            reasoning += "   Both queries executed successfully and produced identical results.\n"
            gen_shape = generated_execution['result_df'].shape
            reasoning += f"   Result shape: {gen_shape[0]} rows × {gen_shape[1]} columns\n"
        else:
            if not generated_execution['success']:
                reasoning += f"   Generated SQL failed: {generated_execution['error_message']}\n"
            elif not gold_execution['success']:
                reasoning += f"   Ground truth SQL failed: {gold_execution['error_message']}\n"
            else:
                reasoning += "   Both executed but produced different results.\n"
                gen_shape = generated_execution['result_df'].shape
                gold_shape = gold_execution['result_df'].shape
                reasoning += f"   Generated result shape: {gen_shape}\n"
                reasoning += f"   Ground truth result shape: {gold_shape}\n"
                
                # Show first few rows for debugging
                if len(generated_execution['result_df']) > 0:
                    reasoning += "\n   Generated result (first 3 rows):\n"
                    reasoning += generated_execution['result_df'].head(3).to_string(index=False) + "\n"
                
                if len(gold_execution['result_df']) > 0:
                    reasoning += "\n   Ground truth result (first 3 rows):\n"
                    reasoning += gold_execution['result_df'].head(3).to_string(index=False) + "\n"
        reasoning += "\n"
        
        # 2. Exact Match
        reasoning += f"2. Exact Match: {'✅ YES' if exact_match else '❌ NO'}\n"
        if exact_match:
            reasoning += "   Queries are identical (case-insensitive, whitespace-normalized).\n"
        else:
            reasoning += "   Queries differ in structure or formatting.\n"
        reasoning += "\n"
        
        # 3. Normalized Match
        reasoning += f"3. Normalized Match: {'✅ YES' if normalized_match else '❌ NO'}\n"
        if normalized_match:
            reasoning += "   Queries are semantically identical after normalization.\n"
        else:
            reasoning += "   Queries differ even after normalization.\n"
        reasoning += "\n"
        
        # 4. Semantic Equivalence
        reasoning += f"4. Semantic Equivalence: {semantic_equivalence:.2f}\n"
        if semantic_equivalence >= 0.8:
            reasoning += "   Very high semantic similarity.\n"
        elif semantic_equivalence >= 0.6:
            reasoning += "   Moderate semantic similarity.\n"
        else:
            reasoning += "   Low semantic similarity.\n"
        reasoning += "\n"
        
        # 5. Component Scores
        reasoning += "5. Component-Level Scores:\n"
        for component, score in sorted(component_scores.items()):
            icon = "✅" if score >= 0.8 else "⚠️" if score >= 0.5 else "❌"
            reasoning += f"   {icon} {component.upper()}: {score:.2f}\n"
        reasoning += "\n"
        
        # Overall Score
        reasoning += "=" * 50 + "\n"
        reasoning += f"OVERALL EVALUATION SCORE: {evaluation_score:.2f}\n"
        reasoning += "=" * 50 + "\n\n"
        
        # Grade with explanation
        if evaluation_score >= 0.9:
            reasoning += "Grade: A (Excellent)\n"
            reasoning += "  All metrics passed with high scores.\n"
        elif evaluation_score >= 0.7:
            reasoning += "Grade: B (Good)\n"
            reasoning += "  Most metrics passed, minor differences in SQL structure.\n"
        elif evaluation_score >= 0.5:
            reasoning += "Grade: C (Fair)\n"
            reasoning += "  Execution passed but SQL structure differs significantly.\n"
        else:
            reasoning += "Grade: D (Needs Improvement)\n"
            if not execution_accuracy:
                reasoning += "  Critical: Execution results do not match ground truth.\n"
            else:
                reasoning += "  SQL structure needs significant improvement.\n"
        
        # Key insight
        reasoning += "\nKey Insight:\n"
        if execution_accuracy and evaluation_score < 0.7:
            reasoning += "  ✅ Execution is correct, but SQL structure could be optimized.\n"
        elif not execution_accuracy:
            reasoning += "  ❌ Execution failed or results differ - this is the critical issue.\n"
        
        return reasoning
    
    def compute_batch_metrics(
        self,
        evaluation_results: List[Dict]
    ) -> Dict:
        """
        Compute aggregate metrics for batch evaluation
        
        Args:
            evaluation_results: List of evaluation results from evaluate_example
            
        Returns:
            {
                'execution_accuracy': float,
                'exact_match_accuracy': float,
                'normalized_match_accuracy': float,
                'avg_semantic_equivalence': float,
                'avg_evaluation_score': float,
                'total_examples': int
            }
        """
        if not evaluation_results:
            return {
                'execution_accuracy': 0.0,
                'exact_match_accuracy': 0.0,
                'normalized_match_accuracy': 0.0,
                'avg_semantic_equivalence': 0.0,
                'avg_evaluation_score': 0.0,
                'total_examples': 0
            }
        
        total = len(evaluation_results)
        
        return {
            'execution_accuracy': sum(r['execution_accuracy'] for r in evaluation_results) / total,
            'exact_match_accuracy': sum(r['exact_match'] for r in evaluation_results) / total,
            'normalized_match_accuracy': sum(r['normalized_match'] for r in evaluation_results) / total,
            'avg_semantic_equivalence': sum(r['semantic_equivalence'] for r in evaluation_results) / total,
            'avg_evaluation_score': sum(r['evaluation_score'] for r in evaluation_results) / total,
            'total_examples': total
        }