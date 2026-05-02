"""
STEP 8: Validation-Feedback Retry
Regenerates SQL based on validation errors and suggestions

Usage:
    from validation_feedback_retry import ValidationFeedbackRetry
    
    retry_engine = ValidationFeedbackRetry(model="qwen3-coder", max_retries=2)
    result = retry_engine.retry_with_feedback(...)
"""
import os
import ollama
import re
from difflib import get_close_matches
from typing import Dict, List

# Cache one client per host so the 90s timeout applies without re-creating SSL contexts.
_client_cache: dict = {}

def _get_ollama_client():
    host = os.environ.get("OLLAMA_HOST", "")
    key = host or "default"
    if key not in _client_cache:
        kwargs = {'timeout': 90}
        if host:
            kwargs['host'] = host
        _client_cache[key] = ollama.Client(**kwargs)
    return _client_cache[key]


class ValidationFeedbackRetry:
    def __init__(self, model: str = "qwen3-coder", max_retries: int = 3):
        """
        Initialize validation feedback retry system
        
        Args:
            model: Ollama model name
            max_retries: Maximum number of retry attempts
        """
        self.model = model
        self.max_retries = max_retries
    
    def retry_with_feedback(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        generated_sql: str,
        validation_result: Dict,
        generation_strategy: str,
        original_examples: List[Dict] = None
    ) -> Dict:
        """
        STEP 8: Validation-Feedback Retry
        
        Args:
            question: Natural language question
            pruned_schema: Schema from Step 1
            schema_links: Schema links from Step 1
            generated_sql: SQL from Step 6
            validation_result: Validation results from Step 7
            generation_strategy: Strategy used in Step 6 (for context)
            original_examples: Examples used in Step 6 (optional)
            
        Returns:
            {
                'final_sql': str,
                'is_valid': bool,
                'retry_count': int,
                'validation_history': List[Dict],
                'improvements': List[str],
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 8: VALIDATION-FEEDBACK RETRY")
        print(f"{'='*60}\n")
        
        # Check if retry is needed
        if validation_result['is_valid']:
            print("✅ SQL is valid - no retry needed")
            
            return {
                'final_sql': generated_sql,
                'is_valid': True,
                'retry_count': 0,
                'validation_history': [validation_result],
                'improvements': ['SQL was valid on first attempt'],
                'reasoning': self._generate_reasoning(
                    question, generated_sql, [], 0, True
                )
            }
        
        print(f"⚠️  SQL has validation errors - attempting retry (max: {self.max_retries})")
        
        # Initialize retry loop
        current_sql = generated_sql
        all_sqls = [generated_sql]
        validation_history = [validation_result]
        improvements = []
        retry_count = 0
        
        # Retry loop
        while retry_count < self.max_retries:
            retry_count += 1
            
            print(f"\n{'─'*60}")
            print(f"RETRY ATTEMPT {retry_count}/{self.max_retries}")
            print(f"{'─'*60}\n")
            
            # Get current validation errors and suggestions
            current_validation = validation_history[-1]
            errors = current_validation['errors']
            warnings = current_validation['warnings']
            suggestions = current_validation['suggestions']
            
            print(f"8.{retry_count}.1: Analyzing validation feedback...")
            print(f"   Errors: {len(errors)}")
            print(f"   Warnings: {len(warnings)}")
            print(f"   Suggestions: {len(suggestions)}")
            
            # Generate corrected SQL
            print(f"8.{retry_count}.2: Regenerating SQL with feedback...")
            corrected_sql = self._regenerate_with_validation_feedback(
                question=question,
                pruned_schema=pruned_schema,
                schema_links=schema_links,
                failed_sql=current_sql,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                generation_strategy=generation_strategy,
                retry_number=retry_count,
                original_examples=original_examples
            )
            
            print(f"   Generated: {len(corrected_sql)} characters")
            
            # Validate the corrected SQL
            print(f"8.{retry_count}.3: Validating corrected SQL...")
            from pipeline.validate_sql import SQLValidator
            validator = SQLValidator()
            
            new_validation = validator.validate_sql_enhanced(
                corrected_sql,
                pruned_schema,
                schema_links
            )
            
            validation_history.append(new_validation)
            
            print(f"   Valid: {new_validation['is_valid']}")
            print(f"   Score: {new_validation['validation_score']:.2f}")
            
            # Track improvements
            improvement = self._calculate_improvement(
                current_validation, 
                new_validation
            )
            improvements.append(improvement)
            
            print(f"   Improvement: {improvement}")
            
            # Update current SQL
            current_sql = corrected_sql
            all_sqls.append(corrected_sql)
            
            # Check if valid now
            if new_validation['is_valid']:
                print(f"\n✅ SQL corrected successfully after {retry_count} attempt(s)!")
                
                reasoning = self._generate_reasoning(
                    question, current_sql, validation_history, 
                    retry_count, True, improvements
                )
                
                print(f"\n{'='*60}")
                print("STEP 8 COMPLETED ✓ - SQL CORRECTED")
                print(f"{'='*60}\n")
                
                return {
                    'final_sql': current_sql,
                    'is_valid': True,
                    'retry_count': retry_count,
                    'validation_history': validation_history,
                    'improvements': improvements,
                    'reasoning': reasoning
                }
            
            # Check if we should continue
            if retry_count >= self.max_retries:
                print(f"\n⚠️  Maximum retries ({self.max_retries}) reached")
                break
            
            # Check if we're making progress
            if len(validation_history) >= 2:
                prev_score = validation_history[-2]['validation_score']
                curr_score = validation_history[-1]['validation_score']
                
                if curr_score <= prev_score:
                    print(f"   ⚠️  No improvement in validation score")
                    print(f"   Previous: {prev_score:.2f}, Current: {curr_score:.2f}")
        
        # Max retries reached without success
        print(f"\n⚠️  Could not fully correct SQL after {retry_count} attempts")
        
        # Select best attempt from all tracked SQLs
        best_sql, best_validation = self._select_best_attempt(all_sqls, validation_history)

        reasoning = self._generate_reasoning(
            question, best_sql, validation_history,
            retry_count, False, improvements
        )

        print(f"\n{'='*60}")
        print("STEP 8 COMPLETED - BEST ATTEMPT SELECTED")
        print(f"{'='*60}\n")

        return {
            'final_sql': best_sql,  # Best attempt by validation score
            'is_valid': False,
            'retry_count': retry_count,
            'validation_history': validation_history,
            'improvements': improvements,
            'reasoning': reasoning
        }
    
    def _regenerate_with_validation_feedback(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        failed_sql: str,
        errors: List[Dict],
        warnings: List[Dict],
        suggestions: List[str],
        generation_strategy: str,
        retry_number: int,
        original_examples: List[Dict] = None
    ) -> str:
        """Regenerate SQL based on validation feedback"""
        
        prompt = self._build_retry_prompt(
            question=question,
            pruned_schema=pruned_schema,
            schema_links=schema_links,
            failed_sql=failed_sql,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            generation_strategy=generation_strategy,
            retry_number=retry_number,
            original_examples=original_examples
        )
        
        try:
            client = _get_ollama_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert SQL debugger and corrector. Your task is to fix SQL queries based on validation errors and suggestions. Generate ONLY the corrected SQL query without explanations.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.1}
            )

            corrected_sql = response['message']['content'].strip()
            corrected_sql = self._clean_sql(corrected_sql)

            return corrected_sql

        except Exception as e:
            print(f"   ⚠️  Regeneration failed ({type(e).__name__}): {e}")
            return failed_sql  # Return original if regeneration fails or times out
    
    def _build_retry_prompt(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        failed_sql: str,
        errors: List[Dict],
        warnings: List[Dict],
        suggestions: List[str],
        generation_strategy: str,
        retry_number: int,
        original_examples: List[Dict] = None
    ) -> str:
        """Build comprehensive retry prompt with validation feedback"""
        
        prompt = f"# SQL Query Correction - Attempt {retry_number}\n\n"
        
        # Add context
        prompt += "## Original Task\n\n"
        prompt += f"Question: {question}\n"
        prompt += f"Generation Strategy: {generation_strategy}\n\n"
        
        # Add schema
        prompt += "## Database Schema\n\n"
        for table_name, columns in sorted(pruned_schema.items()):
            prompt += f"**{table_name}**\n"
            
            relevant_cols = schema_links.get('columns', {}).get(table_name, set())
            
            for col in columns:
                col_name = col['column_name']
                data_type = col.get('data_type', '')
                
                if col_name in relevant_cols:
                    prompt += f"  ⭐ {col_name} ({data_type}) - RELEVANT\n"
                else:
                    prompt += f"  • {col_name} ({data_type})\n"
            
            prompt += "\n"
        
        # Add foreign keys
        if schema_links.get('foreign_keys'):
            prompt += "## Foreign Key Relationships\n\n"
            for fk in schema_links['foreign_keys']:
                prompt += f"- {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
            prompt += "\n"
        
        # Add examples if provided
        if original_examples:
            prompt += "## Reference Examples\n\n"
            for i, ex in enumerate(original_examples[:3], 1):
                prompt += f"### Example {i}\n"
                prompt += f"Question: {ex.get('question', 'N/A')}\n"
                prompt += f"```sql\n{ex.get('query', 'N/A')}\n```\n\n"
        
        # Add failed SQL
        prompt += "## Failed SQL Query\n\n"
        prompt += "```sql\n"
        prompt += failed_sql + "\n"
        prompt += "```\n\n"
        
        # Add validation errors
        if errors:
            prompt += f"## Validation Errors ({len(errors)})\n\n"
            
            # Group errors by severity
            critical_errors = [e for e in errors if e['severity'] == 'CRITICAL']
            high_errors = [e for e in errors if e['severity'] == 'HIGH']
            medium_errors = [e for e in errors if e['severity'] == 'MEDIUM']
            
            if critical_errors:
                prompt += "### 🔴 CRITICAL Errors (MUST FIX)\n\n"
                for i, error in enumerate(critical_errors, 1):
                    prompt += f"{i}. **{error['type']}**: {error['message']}\n"
                    if 'table' in error:
                        prompt += f"   - Table: `{error['table']}`\n"
                    if 'column' in error:
                        prompt += f"   - Column: `{error['column']}`\n"
                prompt += "\n"
            
            if high_errors:
                prompt += "### 🟠 HIGH Priority Errors\n\n"
                for i, error in enumerate(high_errors, 1):
                    prompt += f"{i}. **{error['type']}**: {error['message']}\n"
                    if 'table' in error:
                        prompt += f"   - Table: `{error['table']}`\n"
                    if 'column' in error:
                        prompt += f"   - Column: `{error['column']}`\n"
                prompt += "\n"
            
            if medium_errors:
                prompt += "### 🟡 MEDIUM Priority Errors\n\n"
                for i, error in enumerate(medium_errors, 1):
                    prompt += f"{i}. {error['message']}\n"
                prompt += "\n"
        
        # Add warnings
        if warnings:
            prompt += f"## Warnings ({len(warnings)})\n\n"
            for i, warning in enumerate(warnings, 1):
                prompt += f"{i}. [{warning['severity']}] {warning['message']}\n"
            prompt += "\n"
        
        # Add suggestions
        if suggestions:
            prompt += f"## Suggestions for Correction\n\n"
            for i, suggestion in enumerate(suggestions, 1):
                prompt += f"{i}. {suggestion}\n"
            prompt += "\n"
        
        # Add correction guidelines
        prompt += "## Correction Guidelines\n\n"
        prompt += "1. **Fix all CRITICAL errors first** - these prevent SQL execution\n"
        prompt += "2. **Use ONLY tables and columns from the schema above**\n"
        prompt += "3. **Check table and column names for typos**\n"
        prompt += "4. **Ensure JOIN conditions use foreign key relationships**\n"
        prompt += "5. **Validate aggregations have proper GROUP BY**\n"
        prompt += "6. **Check subquery structure and syntax**\n"
        prompt += "7. **Maintain the original query intent** - answer the same question\n"
        prompt += "8. **Prefer plain JOINs over subqueries**: If the failed SQL used a JOIN, keep it as a JOIN. Do NOT rewrite as `WHERE col IN (SELECT...)` or `WHERE EXISTS (SELECT...)` unless the question explicitly requires set difference or correlated logic.\n\n"
        
        # Add specific fixes based on error types
        if errors:
            prompt += "## Required Fixes\n\n"
            
            for error in errors[:5]:  # Top 5 errors
                if error['type'] == 'SCHEMA_ERROR' and 'table' in error:
                    table = error['table']
                    available = ', '.join(sorted(pruned_schema.keys()))
                    prompt += f"- Replace invalid table `{table}` with correct table from: {available}\n"
                
                elif error['type'] == 'SCHEMA_ERROR' and 'column' in error:
                    column = error['column']
                    table  = error.get('table', '')
                    # Find closest valid column name so the LLM knows exactly what to use
                    all_cols = []
                    if table and table in pruned_schema:
                        all_cols = [c['column_name'] for c in pruned_schema[table]]
                    else:
                        for cols in pruned_schema.values():
                            all_cols.extend(c['column_name'] for c in cols)
                    closest = get_close_matches(column, all_cols, n=1, cutoff=0.5)
                    if closest:
                        prompt += f"- Replace invalid column `{column}` with `{closest[0]}` (closest valid column in schema)\n"
                    else:
                        prompt += f"- Fix invalid column `{column}` - use only column names listed in the schema above\n"
                
                elif error['type'] == 'JOIN_ERROR':
                    prompt += "- Fix JOIN condition to use proper foreign key relationship\n"
                
                elif error['type'] == 'SYNTAX_ERROR':
                    prompt += f"- Fix syntax error: {error['message']}\n"
                
                elif error['type'] == 'AGGREGATION_WARNING':
                    prompt += "- Add GROUP BY clause or remove non-aggregated columns from SELECT\n"
            
            prompt += "\n"
        
        # Final instruction
        prompt += "## Your Task\n\n"
        prompt += "Generate the **CORRECTED SQL query** that:\n"
        prompt += "- Fixes all errors listed above\n"
        prompt += "- Uses only valid tables and columns from the schema\n"
        prompt += "- Maintains the original query intent\n"
        prompt += "- Follows SQL best practices\n\n"
        prompt += "**Output ONLY the corrected SQL query (no explanations, no markdown):**\n"
        
        return prompt
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL output from LLM"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        # Extract SQL content
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_upper = line.strip().upper()
            
            if any(line_upper.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                in_sql = True
            
            if in_sql:
                sql_lines.append(line)
                if line.strip().endswith(';'):
                    break
        
        if sql_lines:
            sql = '\n'.join(sql_lines)
        
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _calculate_improvement(
        self, 
        old_validation: Dict, 
        new_validation: Dict
    ) -> str:
        """Calculate improvement between two validation results"""
        old_score = old_validation['validation_score']
        new_score = new_validation['validation_score']
        
        old_errors = len(old_validation['errors'])
        new_errors = len(new_validation['errors'])
        
        score_diff = new_score - old_score
        error_diff = old_errors - new_errors
        
        if new_validation['is_valid']:
            return "✅ SQL now valid"
        elif score_diff > 0.1:
            return f"📈 Improved score by {score_diff:.2f}, fixed {error_diff} error(s)"
        elif score_diff > 0:
            return f"📊 Slight improvement (+{score_diff:.2f})"
        elif score_diff == 0:
            return "➡️  No change in validation score"
        else:
            return f"📉 Score decreased by {abs(score_diff):.2f}"
    
    def _select_best_attempt(
        self, 
        all_sqls: List[str], 
        validation_history: List[Dict]
    ) -> tuple:
        """Select the best SQL attempt based on validation scores"""
        best_idx = 0
        best_score = validation_history[0]['validation_score']
        
        for i, validation in enumerate(validation_history):
            if validation['validation_score'] > best_score:
                best_score = validation['validation_score']
                best_idx = i
        
        return all_sqls[min(best_idx, len(all_sqls)-1)], validation_history[best_idx]
    
    def retry_with_execution_feedback(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        current_sql: str,
        generation_strategy: str,
        db_path: str = None,
        db_manager=None,
        max_exec_retries: int = 2
    ) -> Dict:
        """
        Execution-driven retry: fix SQL that executes but returns 0 rows.
        Inspired by LitE-SQL execution-guided self-correction.
        Only called when the question is NOT a negation/exclusion query.
        """
        print(f"\n{'='*60}")
        print("STEP 8 (EXEC): EXECUTION-DRIVEN RETRY")
        print(f"{'='*60}\n")
        print("⚠️  SQL returned 0 rows — attempting execution-guided fix...")

        best_sql = current_sql

        for attempt in range(1, max_exec_retries + 1):
            print(f"\n{'─'*60}")
            print(f"EXEC RETRY ATTEMPT {attempt}/{max_exec_retries}")
            print(f"{'─'*60}\n")

            corrected_sql = self._regenerate_for_empty_result(
                question=question,
                pruned_schema=pruned_schema,
                schema_links=schema_links,
                failed_sql=best_sql,
                generation_strategy=generation_strategy,
                attempt_number=attempt
            )

            if not corrected_sql or corrected_sql == best_sql:
                print("   No change produced — stopping exec retry")
                break

            if db_manager and db_path:
                exec_result = db_manager.execute_query(corrected_sql, db_path)
                rows = exec_result.get('result_rows', [])
                if exec_result.get('success') and len(rows) > 0:
                    print(f"   ✅ Fixed! New SQL returns {len(rows)} row(s)")
                    best_sql = corrected_sql
                    break
                elif not exec_result.get('success'):
                    print("   ⚠️  Corrected SQL fails to execute — keeping previous SQL")
                    break
                else:
                    print("   Still 0 rows after correction — continuing...")
                    best_sql = corrected_sql
            else:
                best_sql = corrected_sql

        changed = best_sql != current_sql
        print(f"\n{'='*60}")
        print(f"STEP 8 (EXEC) COMPLETED — {'SQL updated' if changed else 'no change'}")
        print(f"{'='*60}\n")

        return {
            'final_sql': best_sql,
            'retry_count': max_exec_retries,
            'exec_retried': True,
            'sql_changed': changed
        }

    def retry_with_plausibility_feedback(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        current_sql: str,
        generation_strategy: str,
        plausibility_issue: str,
        db_path: str = None,
        db_manager=None
    ) -> Dict:
        """Fix SQL whose result shape is implausible (e.g., scalar agg returning multiple rows)."""
        print(f"\n{'='*60}")
        print("STEP 8 (PLAUSIBILITY): SHAPE-MISMATCH RETRY")
        print(f"{'='*60}\n")
        print(f"⚠️  Plausibility issue: {plausibility_issue}")

        corrected_sql = self._regenerate_for_implausible_result(
            question=question,
            pruned_schema=pruned_schema,
            schema_links=schema_links,
            failed_sql=current_sql,
            plausibility_issue=plausibility_issue
        )

        changed = corrected_sql and corrected_sql != current_sql
        if changed and db_manager and db_path:
            exec_result = db_manager.execute_query(corrected_sql, db_path)
            if not exec_result.get('success'):
                print("   ⚠️  Plausibility-fixed SQL fails to execute — reverting")
                corrected_sql = current_sql
                changed = False

        final_sql = corrected_sql if changed else current_sql
        print(f"\n{'='*60}")
        print(f"STEP 8 (PLAUSIBILITY) COMPLETED — {'SQL updated' if changed else 'no change'}")
        print(f"{'='*60}\n")

        return {'final_sql': final_sql, 'sql_changed': changed}

    def _regenerate_for_implausible_result(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        failed_sql: str,
        plausibility_issue: str
    ) -> str:
        """Prompt LLM to fix a SQL whose result shape is implausible."""
        schema_str = ""
        for table_name, columns in sorted(pruned_schema.items()):
            col_list = ", ".join(
                f"{c['column_name']} ({c.get('data_type', '')})" for c in columns
            )
            schema_str += f"  {table_name}: {col_list}\n"

        fk_str = ""
        for fk in schema_links.get('foreign_keys', []):
            fk_str += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        if not fk_str:
            fk_str = "  (none)\n"

        prompt = f"""# SQL Shape-Mismatch Fix

## Problem
{plausibility_issue}

## Question
{question}

## Schema
{schema_str}
## Foreign Keys
{fk_str}
## Current SQL
```sql
{failed_sql}
```

## Fix Instructions
- If the aggregation (COUNT/SUM/AVG/MAX/MIN) applies to groups, add a GROUP BY clause.
- If the aggregation is scalar (single result expected), remove any stray joins or WHERE conditions that cause row duplication.
- Keep the question's intent unchanged.
- Output ONLY the corrected SQL (no explanations, no markdown fences):"""

        try:
            client = _get_ollama_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert SQL debugger. Fix SQL with incorrect result shapes. Output ONLY the corrected SQL.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.15}
            )
            corrected = response['message']['content'].strip()
            return self._clean_sql(corrected)
        except Exception as e:
            print(f"   ⚠️  Plausibility retry LLM call failed: {e}")
            return failed_sql

    def _regenerate_for_empty_result(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        failed_sql: str,
        generation_strategy: str,
        attempt_number: int
    ) -> str:
        """Prompt LLM to fix a SQL that executed successfully but returned 0 rows."""
        schema_str = ""
        for table_name, columns in sorted(pruned_schema.items()):
            col_list = ", ".join(
                f"{c['column_name']} ({c.get('data_type', '')})" for c in columns
            )
            schema_str += f"  {table_name}: {col_list}\n"

        fk_str = ""
        for fk in schema_links.get('foreign_keys', []):
            fk_str += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        if not fk_str:
            fk_str = "  (none)\n"

        prompt = f"""# SQL Execution-Driven Fix (Attempt {attempt_number})

## Task
The SQL below executed without errors but returned **0 rows**.
Fix it so it returns the correct non-empty result for this question.

## Question
{question}

## Schema
{schema_str}
## Foreign Keys
{fk_str}
## SQL That Returned 0 Rows
```sql
{failed_sql}
```

## Common Causes of 0 Rows
1. Over-strict WHERE filter — literal string that doesn't match DB content (case, spacing, abbreviation)
2. Wrong JOIN condition — joining on wrong columns or mismatched types
3. Missing JOIN — a required table is not joined in
4. Wrong column used in filter — filtering a column that doesn't hold the target value
5. HAVING clause excludes all groups — relax or remove the HAVING condition

## Instructions
- Identify the most likely cause of the 0-row result
- Fix only that condition; keep the rest of the query intact
- Do NOT change the question's intent
- Output ONLY the corrected SQL (no explanations, no markdown fences):"""

        try:
            client = _get_ollama_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert SQL debugger. Fix SQL that returns 0 rows. Output ONLY the corrected SQL.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.15}
            )
            corrected = response['message']['content'].strip()
            return self._clean_sql(corrected)
        except Exception as e:
            print(f"   ⚠️  Exec retry LLM call failed: {e}")
            return failed_sql

    def _generate_reasoning(
        self,
        question: str,
        final_sql: str,
        validation_history: List[Dict],
        retry_count: int,
        is_valid: bool,
        improvements: List[str] = None
    ) -> str:
        """Generate reasoning for Step 8"""
        reasoning = "STEP 8: VALIDATION-FEEDBACK RETRY\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += f"Retry Attempts: {retry_count}\n"
        reasoning += f"Final Status: {'✅ VALID' if is_valid else '⚠️  INVALID'}\n\n"
        
        if retry_count == 0:
            reasoning += "No retry needed - SQL was valid on first attempt.\n\n"
        else:
            reasoning += "Validation History:\n"
            reasoning += "-" * 50 + "\n"
            
            for i, validation in enumerate(validation_history):
                attempt = "Initial" if i == 0 else f"Retry {i}"
                reasoning += f"\n{attempt}:\n"
                reasoning += f"  • Valid: {validation['is_valid']}\n"
                reasoning += f"  • Score: {validation['validation_score']:.2f}\n"
                reasoning += f"  • Errors: {len(validation['errors'])}\n"
                reasoning += f"  • Warnings: {len(validation['warnings'])}\n"
                
                if i > 0 and improvements:
                    reasoning += f"  • Improvement: {improvements[i-1]}\n"
            
            reasoning += "\n" + "-" * 50 + "\n\n"
        
        reasoning += "Final SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += final_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        if is_valid:
            reasoning += "✅ SQL successfully corrected and validated!\n"
        else:
            reasoning += "⚠️  Could not fully correct SQL within retry limit.\n"
            reasoning += "Consider:\n"
            reasoning += "  • Reviewing the question for ambiguity\n"
            reasoning += "  • Checking if schema has required information\n"
            reasoning += "  • Manually refining the SQL\n"
        
        return reasoning