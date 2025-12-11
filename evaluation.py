"""
STEP 11: Evaluation (Aligned with DAIL-SQL and DIN-SQL Papers)
Evaluates generated SQL using official Spider metrics:
- Execution Accuracy (EX) - Primary metric
- Exact-Set-Match Accuracy (EM) - Secondary metric
"""
import pandas as pd
import numpy as np
import re
from typing import Dict, Optional, List, Set, Tuple
from collections import defaultdict


class Text2SQLEvaluator:
    def __init__(self):
        """Initialize evaluator with Spider-compatible metrics"""
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'LIMIT', 'INTERSECT',
            'UNION', 'EXCEPT', 'JOIN', 'ON', 'AS', 'DISTINCT', 'HAVING'
        }
        
        self.agg_ops = {'COUNT', 'SUM', 'AVG', 'MAX', 'MIN'}
        self.cond_ops = {'=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IN', 'BETWEEN'}
    
    def evaluate_example(
        self,
        question: str,
        generated_sql: str,
        gold_sql: str,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> Dict:
        """
        STEP 11: Evaluation (Spider-compatible)
        
        Primary metrics from DAIL-SQL and DIN-SQL papers:
        - Execution Accuracy (EX): Most important - checks if results match
        - Exact-Set-Match (EM): Checks if SQL structure matches
        
        Args:
            question: Natural language question
            generated_sql: Generated SQL query
            gold_sql: Ground truth SQL query
            generated_execution: Execution result of generated SQL (from Step 10)
            gold_execution: Execution result of gold SQL (from Step 10)
            
        Returns:
            {
                'execution_accuracy': bool (EX),
                'exact_set_match': bool (EM),
                'component_match': Dict (detailed breakdown),
                'evaluation_score': float (0.0-1.0),
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 11: EVALUATION (Spider Metrics)")
        print(f"{'='*60}\n")
        
        print(f"Question: {question}")
        
        # ================================================================
        # METRIC 1: EXECUTION ACCURACY (PRIMARY - FROM PAPERS)
        # ================================================================
        print("\n11.1: Computing Execution Accuracy (EX)...")
        execution_accuracy = self._compute_execution_accuracy(
            generated_execution, gold_execution
        )
        print(f"   EX = {'✅ 1' if execution_accuracy else '❌ 0'}")
        
        # ================================================================
        # METRIC 2: EXACT-SET-MATCH ACCURACY (SECONDARY - FROM PAPERS)
        # ================================================================
        print("\n11.2: Computing Exact-Set-Match Accuracy (EM)...")
        exact_set_match, component_match = self._compute_exact_set_match(
            generated_sql, gold_sql
        )
        print(f"   EM = {'✅ 1' if exact_set_match else '❌ 0'}")
        
        # Component-level details
        print("\n11.3: Component-level Match:")
        for component, matches in component_match.items():
            status = "✅" if matches else "❌"
            print(f"   {status} {component}")
        
        # ================================================================
        # COMPOSITE EVALUATION SCORE (for retry decisions)
        # ================================================================
        # Weighted according to papers: EX >> EM
        # EX: 80%, EM: 20%
        evaluation_score = (execution_accuracy * 0.80) + (exact_set_match * 0.20)
        
        print(f"\n11.4: Composite Evaluation Score: {evaluation_score:.2f}")
        print(f"   (80% EX + 20% EM)")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question,
            generated_sql,
            gold_sql,
            execution_accuracy,
            exact_set_match,
            component_match,
            evaluation_score,
            generated_execution,
            gold_execution
        )
        
        print(f"\n{'='*60}")
        print("STEP 11 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'execution_accuracy': execution_accuracy,
            'exact_set_match': exact_set_match,
            'component_match': component_match,
            'evaluation_score': evaluation_score,
            'reasoning': reasoning
        }
    
    # ====================================================================
    # EXECUTION ACCURACY (PRIMARY METRIC)
    # ====================================================================
    
    def _compute_execution_accuracy(
        self,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> bool:
        """
        Compute Execution Accuracy (EX) - PRIMARY metric
        
        From papers: "compares the execution output of the predicted SQL 
        query with that of the ground truth SQL query on database instances"
        
        Returns True if execution results are identical
        """
        # Both must execute successfully
        if not generated_execution['success'] or not gold_execution['success']:
            return False
        
        gen_df = generated_execution['result_df']
        gold_df = gold_execution['result_df']
        
        try:
            # Check if DataFrames are equal
            return self._dataframes_equal(gen_df, gold_df)
                
        except Exception as e:
            print(f"   ⚠️ Error comparing DataFrames: {e}")
            return False
    
    def _dataframes_equal(self, df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
        """
        Compare DataFrames for equality (handles various edge cases)
        """
        # Handle empty DataFrames
        if len(df1) == 0 and len(df2) == 0:
            return True
        
        # Check shape
        if df1.shape != df2.shape:
            return False
        
        # If only one is empty
        if len(df1) == 0 or len(df2) == 0:
            return False
        
        # Normalize for comparison
        df1_norm = df1.copy()
        df2_norm = df2.copy()
        
        # Normalize column names (case-insensitive)
        df1_norm.columns = [str(col).strip().lower() for col in df1_norm.columns]
        df2_norm.columns = [str(col).strip().lower() for col in df2_norm.columns]
        
        # Sort columns alphabetically
        df1_norm = df1_norm.sort_index(axis=1)
        df2_norm = df2_norm.sort_index(axis=1)
        
        # Sort rows by all columns
        try:
            df1_norm = df1_norm.sort_values(
                by=list(df1_norm.columns)
            ).reset_index(drop=True)
            df2_norm = df2_norm.sort_values(
                by=list(df2_norm.columns)
            ).reset_index(drop=True)
        except:
            df1_norm = df1_norm.reset_index(drop=True)
            df2_norm = df2_norm.reset_index(drop=True)
        
        # Convert to strings for comparison
        df1_str = df1_norm.astype(str)
        df2_str = df2_norm.astype(str)
        
        # Compare
        if df1_str.equals(df2_str):
            return True
        
        # Try numeric comparison with tolerance
        try:
            for col in df1_norm.columns:
                col1 = pd.to_numeric(df1_norm[col], errors='coerce')
                col2 = pd.to_numeric(df2_norm[col], errors='coerce')
                
                if not np.allclose(col1, col2, rtol=1e-5, atol=1e-8, equal_nan=True):
                    return False
            
            return True
        except:
            return False
    
    # ====================================================================
    # EXACT-SET-MATCH ACCURACY (SECONDARY METRIC)
    # ====================================================================
    
    def _compute_exact_set_match(
        self, 
        generated_sql: str, 
        gold_sql: str
    ) -> Tuple[bool, Dict]:
        """
        Compute Exact-Set-Match Accuracy (EM) - SECONDARY metric
        
        From papers: "treats each clause as a set and compares the prediction 
        for each clause to its corresponding clause in the reference query"
        
        Returns (exact_match: bool, component_match: Dict)
        """
        # Parse both queries into components
        gen_components = self._parse_sql_components(generated_sql)
        gold_components = self._parse_sql_components(gold_sql)
        
        # Compare each component as sets
        component_match = {}
        
        # SELECT clause
        component_match['SELECT'] = self._sets_equal(
            gen_components['select_cols'],
            gold_components['select_cols']
        )
        
        # FROM clause (tables)
        component_match['FROM'] = self._sets_equal(
            gen_components['from_tables'],
            gold_components['from_tables']
        )
        
        # WHERE clause
        component_match['WHERE'] = self._sets_equal(
            gen_components['where_conds'],
            gold_components['where_conds']
        )
        
        # GROUP BY clause
        component_match['GROUP_BY'] = self._sets_equal(
            gen_components['group_by'],
            gold_components['group_by']
        )
        
        # ORDER BY clause
        component_match['ORDER_BY'] = self._sets_equal(
            gen_components['order_by'],
            gold_components['order_by']
        )
        
        # HAVING clause
        component_match['HAVING'] = self._sets_equal(
            gen_components['having'],
            gold_components['having']
        )
        
        # Keywords (JOIN, DISTINCT, LIMIT, etc.)
        component_match['KEYWORDS'] = self._sets_equal(
            gen_components['keywords'],
            gold_components['keywords']
        )
        
        # Exact match = ALL components match
        exact_match = all(component_match.values())
        
        return exact_match, component_match
    
    def _parse_sql_components(self, sql: str) -> Dict:
        """
        Parse SQL into components for set-based comparison
        Following Spider evaluation methodology
        """
        sql_upper = sql.upper()
        sql_normalized = ' '.join(sql_upper.split())
        
        components = {
            'select_cols': set(),
            'from_tables': set(),
            'where_conds': set(),
            'group_by': set(),
            'order_by': set(),
            'having': set(),
            'keywords': set()
        }
        
        # Extract SELECT columns
        select_match = re.search(
            r'SELECT\s+(.*?)\s+FROM', 
            sql_normalized, 
            re.DOTALL
        )
        if select_match:
            select_clause = select_match.group(1)
            # Split by comma and normalize
            cols = [c.strip() for c in select_clause.split(',')]
            components['select_cols'] = set(self._normalize_items(cols))
        
        # Extract FROM tables
        from_match = re.search(r'FROM\s+(.*?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)', sql_normalized)
        if from_match:
            from_clause = from_match.group(1)
            # Extract table names (handle JOINs)
            tables = re.findall(r'\b([A-Z_][A-Z0-9_]*)\b', from_clause)
            # Filter out keywords
            tables = [t for t in tables if t not in self.sql_keywords]
            components['from_tables'] = set(tables)
        
        # Extract WHERE conditions
        where_match = re.search(
            r'WHERE\s+(.*?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', 
            sql_normalized
        )
        if where_match:
            where_clause = where_match.group(1)
            conds = self._extract_conditions(where_clause)
            components['where_conds'] = set(conds)
        
        # Extract GROUP BY
        group_match = re.search(
            r'GROUP\s+BY\s+(.*?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|$)', 
            sql_normalized
        )
        if group_match:
            group_clause = group_match.group(1)
            cols = [c.strip() for c in group_clause.split(',')]
            components['group_by'] = set(self._normalize_items(cols))
        
        # Extract ORDER BY
        order_match = re.search(
            r'ORDER\s+BY\s+(.*?)(?:\s+LIMIT|$)', 
            sql_normalized
        )
        if order_match:
            order_clause = order_match.group(1)
            cols = [c.strip() for c in order_clause.split(',')]
            components['order_by'] = set(self._normalize_items(cols))
        
        # Extract HAVING
        having_match = re.search(
            r'HAVING\s+(.*?)(?:\s+ORDER|\s+LIMIT|$)', 
            sql_normalized
        )
        if having_match:
            having_clause = having_match.group(1)
            conds = self._extract_conditions(having_clause)
            components['having'] = set(conds)
        
        # Extract keywords
        if 'DISTINCT' in sql_upper:
            components['keywords'].add('DISTINCT')
        if 'JOIN' in sql_upper:
            components['keywords'].add('JOIN')
        if 'LIMIT' in sql_upper:
            components['keywords'].add('LIMIT')
        if 'UNION' in sql_upper:
            components['keywords'].add('UNION')
        if 'INTERSECT' in sql_upper:
            components['keywords'].add('INTERSECT')
        if 'EXCEPT' in sql_upper:
            components['keywords'].add('EXCEPT')
        
        return components
    
    def _normalize_items(self, items: List[str]) -> List[str]:
        """Normalize SQL items for comparison"""
        normalized = []
        for item in items:
            # Remove extra whitespace
            item = ' '.join(item.split())
            # Remove ASC/DESC
            item = re.sub(r'\s+(ASC|DESC)\s*$', '', item)
            normalized.append(item)
        return normalized
    
    def _extract_conditions(self, clause: str) -> List[str]:
        """Extract conditions from WHERE or HAVING clause"""
        # Split by AND/OR
        conds = re.split(r'\s+(AND|OR)\s+', clause)
        # Keep only conditions (not AND/OR keywords)
        conds = [c.strip() for c in conds if c.strip() not in ('AND', 'OR')]
        return self._normalize_items(conds)
    
    def _sets_equal(self, set1: Set, set2: Set) -> bool:
        """Compare two sets for equality"""
        # Empty sets are equal
        if not set1 and not set2:
            return True
        
        return set1 == set2
    
    # ====================================================================
    # REASONING GENERATION
    # ====================================================================
    
    def _generate_reasoning(
        self,
        question: str,
        generated_sql: str,
        gold_sql: str,
        execution_accuracy: bool,
        exact_set_match: bool,
        component_match: Dict,
        evaluation_score: float,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> str:
        """Generate comprehensive reasoning"""
        reasoning = "STEP 11: EVALUATION (Spider Metrics)\n"
        reasoning += "=" * 60 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += "Generated SQL:\n"
        reasoning += "-" * 60 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 60 + "\n\n"
        
        reasoning += "Ground Truth SQL:\n"
        reasoning += "-" * 60 + "\n"
        reasoning += gold_sql + "\n"
        reasoning += "-" * 60 + "\n\n"
        
        reasoning += "=" * 60 + "\n"
        reasoning += "OFFICIAL SPIDER METRICS\n"
        reasoning += "=" * 60 + "\n\n"
        
        # ================================================================
        # METRIC 1: EXECUTION ACCURACY (PRIMARY)
        # ================================================================
        reasoning += "1. EXECUTION ACCURACY (EX) - Primary Metric\n"
        reasoning += "-" * 60 + "\n"
        reasoning += f"   Result: {'✅ PASS (1.0)' if execution_accuracy else '❌ FAIL (0.0)'}\n\n"
        
        if execution_accuracy:
            reasoning += "   Both queries executed successfully and produced IDENTICAL results.\n"
            gen_shape = generated_execution['result_df'].shape
            reasoning += f"   Result shape: {gen_shape[0]} rows × {gen_shape[1]} columns\n\n"
            
            reasoning += "   This is the GOLD STANDARD for Text-to-SQL evaluation.\n"
            reasoning += "   Multiple different SQL queries can be correct as long as\n"
            reasoning += "   they produce the same results.\n"
        else:
            reasoning += "   Execution results DO NOT MATCH.\n\n"
            
            if not generated_execution['success']:
                reasoning += f"   ❌ Generated SQL failed to execute:\n"
                reasoning += f"      {generated_execution['error_message']}\n\n"
            elif not gold_execution['success']:
                reasoning += f"   ❌ Ground truth SQL failed to execute:\n"
                reasoning += f"      {gold_execution['error_message']}\n\n"
            else:
                reasoning += "   Both queries executed but produced DIFFERENT results:\n\n"
                
                gen_shape = generated_execution['result_df'].shape
                gold_shape = gold_execution['result_df'].shape
                
                reasoning += f"   Generated result:  {gen_shape[0]} rows × {gen_shape[1]} columns\n"
                reasoning += f"   Ground truth result: {gold_shape[0]} rows × {gold_shape[1]} columns\n\n"
                
                # Show sample results
                if len(generated_execution['result_df']) > 0:
                    reasoning += "   Generated result (first 3 rows):\n"
                    reasoning += "   " + "\n   ".join(
                        generated_execution['result_df'].head(3).to_string(index=False).split('\n')
                    ) + "\n\n"
                
                if len(gold_execution['result_df']) > 0:
                    reasoning += "   Ground truth result (first 3 rows):\n"
                    reasoning += "   " + "\n   ".join(
                        gold_execution['result_df'].head(3).to_string(index=False).split('\n')
                    ) + "\n\n"
        
        reasoning += "\n"
        
        # ================================================================
        # METRIC 2: EXACT-SET-MATCH (SECONDARY)
        # ================================================================
        reasoning += "2. EXACT-SET-MATCH ACCURACY (EM) - Secondary Metric\n"
        reasoning += "-" * 60 + "\n"
        reasoning += f"   Result: {'✅ PASS (1.0)' if exact_set_match else '❌ FAIL (0.0)'}\n\n"
        
        reasoning += "   Component-Level Breakdown:\n"
        for component, matches in sorted(component_match.items()):
            status = "✅" if matches else "❌"
            reasoning += f"   {status} {component:12s}\n"
        
        reasoning += "\n"
        
        if exact_set_match:
            reasoning += "   SQL structure matches ground truth exactly.\n"
            reasoning += "   All clauses (SELECT, FROM, WHERE, etc.) are identical when\n"
            reasoning += "   treated as sets (order doesn't matter).\n"
        else:
            reasoning += "   SQL structure differs from ground truth.\n"
            reasoning += "   Note: Different SQL structures can still be correct\n"
            reasoning += "   if they produce the same execution results (EX metric).\n"
        
        reasoning += "\n"
        
        # ================================================================
        # COMPOSITE SCORE
        # ================================================================
        reasoning += "=" * 60 + "\n"
        reasoning += "COMPOSITE EVALUATION SCORE\n"
        reasoning += "=" * 60 + "\n"
        reasoning += f"Score: {evaluation_score:.2f} / 1.00\n"
        reasoning += f"Formula: (EX × 0.80) + (EM × 0.20)\n"
        reasoning += f"       = ({int(execution_accuracy)} × 0.80) + ({int(exact_set_match)} × 0.20)\n"
        reasoning += f"       = {evaluation_score:.2f}\n\n"
        
        # ================================================================
        # FINAL ASSESSMENT
        # ================================================================
        reasoning += "=" * 60 + "\n"
        reasoning += "FINAL ASSESSMENT\n"
        reasoning += "=" * 60 + "\n"
        
        if execution_accuracy and exact_set_match:
            reasoning += "Grade: A+ (Perfect Match)\n"
            reasoning += "  ✅ Execution results match (EX = 1.0)\n"
            reasoning += "  ✅ SQL structure matches (EM = 1.0)\n"
            reasoning += "  This is the ideal outcome.\n"
        
        elif execution_accuracy and not exact_set_match:
            reasoning += "Grade: A (Correct but Different Structure)\n"
            reasoning += "  ✅ Execution results match (EX = 1.0) - MOST IMPORTANT\n"
            reasoning += "  ❌ SQL structure differs (EM = 0.0)\n"
            reasoning += "  This is still CORRECT - multiple valid SQL queries exist.\n"
        
        elif not execution_accuracy and exact_set_match:
            reasoning += "Grade: C (Structure Match but Wrong Results)\n"
            reasoning += "  ❌ Execution results differ (EX = 0.0) - CRITICAL ISSUE\n"
            reasoning += "  ✅ SQL structure matches (EM = 1.0)\n"
            reasoning += "  This is INCORRECT despite matching structure.\n"
        
        else:
            reasoning += "Grade: F (Failed)\n"
            reasoning += "  ❌ Execution results differ (EX = 0.0)\n"
            reasoning += "  ❌ SQL structure differs (EM = 0.0)\n"
            reasoning += "  Query needs significant correction.\n"
        
        reasoning += "\n"
        
        # Key insight
        reasoning += "Key Insight:\n"
        if execution_accuracy:
            reasoning += "  ✅ The PRIMARY metric (Execution Accuracy) passed.\n"
            reasoning += "     This query correctly answers the question.\n"
            if not exact_set_match:
                reasoning += "     The SQL structure difference is acceptable.\n"
        else:
            reasoning += "  ❌ The PRIMARY metric (Execution Accuracy) FAILED.\n"
            reasoning += "     This is the critical issue that must be fixed.\n"
            if exact_set_match:
                reasoning += "     Despite matching SQL structure, results are wrong.\n"
        
        return reasoning
    
    # ====================================================================
    # BATCH EVALUATION
    # ====================================================================
    
    def compute_batch_metrics(
        self,
        evaluation_results: List[Dict]
    ) -> Dict:
        """
        Compute aggregate metrics for batch evaluation
        Following Spider benchmark reporting standards
        
        Returns:
            {
                'execution_accuracy': float (primary),
                'exact_set_match': float (secondary),
                'total_examples': int
            }
        """
        if not evaluation_results:
            return {
                'execution_accuracy': 0.0,
                'exact_set_match': 0.0,
                'total_examples': 0
            }
        
        total = len(evaluation_results)
        
        return {
            'execution_accuracy': sum(
                r['execution_accuracy'] for r in evaluation_results
            ) / total,
            'exact_set_match': sum(
                r['exact_set_match'] for r in evaluation_results
            ) / total,
            'total_examples': total
        }