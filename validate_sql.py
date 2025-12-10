"""
STEP 7: SQL Validation
Validates generated SQL for syntax, schema compliance, and logical correctness
"""
import re
import sqlparse
from typing import Dict, List, Set, Tuple


class SQLValidator:
    def __init__(self):
        """Initialize SQL validator"""
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'ON', 'GROUP', 'BY', 'HAVING', 'ORDER', 'ASC', 'DESC', 'LIMIT',
            'UNION', 'INTERSECT', 'EXCEPT', 'AS', 'DISTINCT', 'COUNT', 'SUM',
            'AVG', 'MAX', 'MIN', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE',
            'BETWEEN', 'IS', 'NULL', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END'
        }
    
    def validate_sql_enhanced(
        self,
        generated_sql: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict = None
    ) -> Dict:
        """
        STEP 7: Enhanced SQL Validation
        
        Args:
            generated_sql: SQL query to validate
            pruned_schema: Schema from Step 1
            schema_links: Schema links from Step 1 (optional)
            
        Returns:
            {
                'is_valid': bool,
                'errors': List[Dict],
                'warnings': List[Dict],
                'suggestions': List[str],
                'validation_score': float,
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 7: SQL VALIDATION")
        print(f"{'='*60}\n")
        
        errors = []
        warnings = []
        suggestions = []
        
        # Skip validation if SQL contains error message
        if generated_sql.startswith('--'):
            errors.append({
                'type': 'GENERATION_ERROR',
                'message': 'SQL generation failed',
                'severity': 'CRITICAL'
            })
            
            return {
                'is_valid': False,
                'errors': errors,
                'warnings': warnings,
                'suggestions': ['Retry generation with different examples'],
                'validation_score': 0.0,
                'reasoning': 'SQL generation produced an error message instead of valid SQL'
            }
        
        # 1. Check basic syntax
        print("7.1: Checking basic syntax...")
        syntax_errors = self._check_syntax(generated_sql)
        errors.extend(syntax_errors)
        print(f"   Syntax errors: {len(syntax_errors)}")
        
        # 2. Verify table existence
        print("7.2: Verifying table existence...")
        table_errors = self._verify_tables(generated_sql, pruned_schema)
        errors.extend(table_errors)
        print(f"   Table errors: {len(table_errors)}")
        
        # 3. Verify column existence
        print("7.3: Verifying column existence...")
        column_errors = self._verify_columns(generated_sql, pruned_schema)
        errors.extend(column_errors)
        print(f"   Column errors: {len(column_errors)}")
        
        # 4. Validate JOIN conditions
        print("7.4: Validating JOIN conditions...")
        join_errors, join_warnings = self._validate_joins(
            generated_sql, pruned_schema, schema_links
        )
        errors.extend(join_errors)
        warnings.extend(join_warnings)
        print(f"   JOIN errors: {len(join_errors)}, warnings: {len(join_warnings)}")
        
        # 5. Check aggregation usage
        print("7.5: Checking aggregation usage...")
        agg_warnings = self._check_aggregations(generated_sql)
        warnings.extend(agg_warnings)
        print(f"   Aggregation warnings: {len(agg_warnings)}")
        
        # 6. Validate subqueries
        print("7.6: Validating subqueries...")
        subquery_errors = self._validate_subqueries(generated_sql)
        errors.extend(subquery_errors)
        print(f"   Subquery errors: {len(subquery_errors)}")
        
        # 7. Check for common mistakes
        print("7.7: Checking for common mistakes...")
        mistake_warnings = self._check_common_mistakes(generated_sql, pruned_schema)
        warnings.extend(mistake_warnings)
        print(f"   Common mistake warnings: {len(mistake_warnings)}")
        
        # Generate suggestions
        print("7.8: Generating suggestions...")
        suggestions = self._generate_suggestions(errors, warnings, generated_sql, pruned_schema)
        print(f"   Suggestions: {len(suggestions)}")
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(errors, warnings)
        
        # Determine if valid
        is_valid = len([e for e in errors if e['severity'] == 'CRITICAL']) == 0
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            generated_sql, errors, warnings, suggestions, validation_score, is_valid
        )
        
        print(f"\nValidation Score: {validation_score:.2f}")
        print(f"Is Valid: {is_valid}")
        
        print(f"\n{'='*60}")
        print("STEP 7 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'suggestions': suggestions,
            'validation_score': validation_score,
            'reasoning': reasoning
        }
    
    def _check_syntax(self, sql: str) -> List[Dict]:
        """Check basic SQL syntax"""
        errors = []
        
        # Check for empty SQL
        if not sql or sql.strip() == '' or sql.strip() == ';':
            errors.append({
                'type': 'SYNTAX_ERROR',
                'message': 'Empty SQL query',
                'severity': 'CRITICAL'
            })
            return errors
        
        # Check for SELECT keyword
        if 'SELECT' not in sql.upper():
            errors.append({
                'type': 'SYNTAX_ERROR',
                'message': 'Missing SELECT keyword',
                'severity': 'CRITICAL'
            })
        
        # Check for FROM keyword (unless it's a simple SELECT without tables)
        sql_upper = sql.upper()
        if 'FROM' not in sql_upper and not re.search(r'SELECT\s+\d+', sql_upper):
            errors.append({
                'type': 'SYNTAX_ERROR',
                'message': 'Missing FROM clause',
                'severity': 'CRITICAL'
            })
        
        # Check for balanced parentheses
        if sql.count('(') != sql.count(')'):
            errors.append({
                'type': 'SYNTAX_ERROR',
                'message': 'Unbalanced parentheses',
                'severity': 'CRITICAL'
            })
        
        # Check for unclosed quotes
        single_quotes = sql.count("'")
        if single_quotes % 2 != 0:
            errors.append({
                'type': 'SYNTAX_ERROR',
                'message': 'Unclosed single quote',
                'severity': 'CRITICAL'
            })
        
        # Try to parse with sqlparse
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                errors.append({
                    'type': 'SYNTAX_ERROR',
                    'message': 'Unable to parse SQL',
                    'severity': 'HIGH'
                })
        except Exception as e:
            errors.append({
                'type': 'SYNTAX_ERROR',
                'message': f'Parse error: {str(e)}',
                'severity': 'HIGH'
            })
        
        return errors
    
    def _verify_tables(self, sql: str, pruned_schema: Dict) -> List[Dict]:
        """Verify that all referenced tables exist in schema"""
        errors = []
        
        # Extract table names from SQL
        referenced_tables = self._extract_table_names(sql)
        
        # Check each table
        for table in referenced_tables:
            if table not in pruned_schema:
                errors.append({
                    'type': 'SCHEMA_ERROR',
                    'message': f'Table "{table}" does not exist in schema',
                    'severity': 'CRITICAL',
                    'table': table
                })
        
        return errors
    
    def _extract_table_names(self, sql: str) -> Set[str]:
        """Extract table names from SQL query"""
        tables = set()
        
        # Pattern for FROM clause
        from_pattern = r'\bFROM\s+([A-Za-z_][A-Za-z0-9_]*)'
        from_matches = re.findall(from_pattern, sql, re.IGNORECASE)
        tables.update(from_matches)
        
        # Pattern for JOIN clauses
        join_pattern = r'\bJOIN\s+([A-Za-z_][A-Za-z0-9_]*)'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        tables.update(join_matches)
        
        return tables
    
    def _verify_columns(self, sql: str, pruned_schema: Dict) -> List[Dict]:
        """Verify that all referenced columns exist in their tables"""
        errors = []
        
        # Extract column references
        column_refs = self._extract_column_references(sql)
        
        # Build a map of all columns by table
        schema_columns = {}
        for table, columns in pruned_schema.items():
            schema_columns[table] = {col['column_name'].lower() for col in columns}
        
        # Check each column reference
        for table, column in column_refs:
            if table:
                # Qualified column (table.column)
                if table not in schema_columns:
                    continue  # Table error already caught
                
                if column.lower() not in schema_columns[table]:
                    errors.append({
                        'type': 'SCHEMA_ERROR',
                        'message': f'Column "{column}" does not exist in table "{table}"',
                        'severity': 'HIGH',
                        'table': table,
                        'column': column
                    })
            else:
                # Unqualified column - check if it exists in any table
                found = False
                for table_name, cols in schema_columns.items():
                    if column.lower() in cols:
                        found = True
                        break
                
                if not found and column != '*':
                    errors.append({
                        'type': 'SCHEMA_ERROR',
                        'message': f'Column "{column}" does not exist in any table',
                        'severity': 'MEDIUM',
                        'column': column
                    })
        
        return errors
    
    def _extract_column_references(self, sql: str) -> List[Tuple[str, str]]:
        """Extract column references as (table, column) tuples"""
        column_refs = []
        
        # Pattern for table.column
        qualified_pattern = r'\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*|\*)'
        qualified_matches = re.findall(qualified_pattern, sql)
        column_refs.extend(qualified_matches)
        
        # Pattern for SELECT columns (unqualified)
        select_pattern = r'SELECT\s+(.*?)\s+FROM'
        select_matches = re.findall(select_pattern, sql, re.IGNORECASE | re.DOTALL)
        
        for match in select_matches:
            # Remove aggregations and functions
            match = re.sub(r'(COUNT|SUM|AVG|MAX|MIN|DISTINCT)\s*\(', '', match, flags=re.IGNORECASE)
            match = match.replace(')', '')
            
            # Split by comma
            parts = [p.strip() for p in match.split(',')]
            
            for part in parts:
                # Skip if already qualified
                if '.' in part:
                    continue
                
                # Extract column name (handle AS alias)
                col_match = re.match(r'([A-Za-z_][A-Za-z0-9_]*)', part)
                if col_match:
                    column_refs.append((None, col_match.group(1)))
        
        return column_refs
    
    def _validate_joins(
        self, 
        sql: str, 
        pruned_schema: Dict,
        schema_links: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """Validate JOIN conditions"""
        errors = []
        warnings = []
        
        # Extract JOIN conditions
        join_pattern = r'JOIN\s+([A-Za-z_][A-Za-z0-9_]*)\s+ON\s+([^WHERE|GROUP|ORDER|UNION|;]+)'
        join_matches = re.findall(join_pattern, sql, re.IGNORECASE)
        
        if not join_matches:
            return errors, warnings
        
        # Check each JOIN
        for table, condition in join_matches:
            # Check if condition has equality
            if '=' not in condition:
                errors.append({
                    'type': 'JOIN_ERROR',
                    'message': f'JOIN on table "{table}" missing equality condition',
                    'severity': 'HIGH',
                    'table': table
                })
                continue
            
            # Extract columns from condition
            col_pattern = r'([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)'
            cols = re.findall(col_pattern, condition)
            
            if len(cols) < 2:
                warnings.append({
                    'type': 'JOIN_WARNING',
                    'message': f'JOIN on table "{table}" has unusual condition format',
                    'severity': 'LOW',
                    'table': table
                })
                continue
            
            # If we have schema_links, check if this is a valid foreign key
            if schema_links and schema_links.get('foreign_keys'):
                fks = schema_links['foreign_keys']
                
                is_valid_fk = False
                for fk in fks:
                    fk_pair = {
                        (fk['from_table'], fk['from_column']),
                        (fk['to_table'], fk['to_column'])
                    }
                    
                    condition_pair = {(cols[0][0], cols[0][1]), (cols[1][0], cols[1][1])}
                    
                    if fk_pair == condition_pair:
                        is_valid_fk = True
                        break
                
                if not is_valid_fk:
                    warnings.append({
                        'type': 'JOIN_WARNING',
                        'message': f'JOIN on table "{table}" does not use a foreign key relationship',
                        'severity': 'MEDIUM',
                        'table': table
                    })
        
        return errors, warnings
    
    def _check_aggregations(self, sql: str) -> List[Dict]:
        """Check aggregation function usage"""
        warnings = []
        
        sql_upper = sql.upper()
        
        # Check if aggregations are present
        has_agg = any(agg in sql_upper for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])
        has_group_by = 'GROUP BY' in sql_upper
        
        if has_agg and not has_group_by:
            # Check if SELECT has non-aggregated columns
            select_pattern = r'SELECT\s+(.*?)\s+FROM'
            select_match = re.search(select_pattern, sql, re.IGNORECASE | re.DOTALL)
            
            if select_match:
                select_clause = select_match.group(1)
                
                # Check for non-aggregated columns
                has_plain_columns = False
                parts = select_clause.split(',')
                
                for part in parts:
                    part_clean = part.strip()
                    # Skip if it's an aggregation
                    if any(agg in part_clean.upper() for agg in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN(']):
                        continue
                    # Skip if it's a constant or expression
                    if re.match(r'^[\d\s\+\-\*\/\(\)\'\"]+$', part_clean):
                        continue
                    # If it contains a column reference, it's a plain column
                    if re.search(r'[A-Za-z_]', part_clean):
                        has_plain_columns = True
                        break
                
                if has_plain_columns:
                    warnings.append({
                        'type': 'AGGREGATION_WARNING',
                        'message': 'Mixing aggregated and non-aggregated columns without GROUP BY',
                        'severity': 'MEDIUM'
                    })
        
        return warnings
    
    def _validate_subqueries(self, sql: str) -> List[Dict]:
        """Validate subquery structure"""
        errors = []
        
        # Count SELECT statements
        select_count = sql.upper().count('SELECT')
        
        if select_count > 1:
            # Has subqueries - check for proper structure
            
            # Check for subqueries in WHERE with IN/NOT IN
            in_pattern = r'(NOT\s+)?IN\s*\(\s*SELECT'
            in_matches = re.findall(in_pattern, sql, re.IGNORECASE)
            
            # Check for comparison subqueries
            comp_pattern = r'(>|<|>=|<=|=|!=)\s*\(\s*SELECT'
            comp_matches = re.findall(comp_pattern, sql, re.IGNORECASE)
            
            # Check for EXISTS subqueries
            exists_pattern = r'EXISTS\s*\(\s*SELECT'
            exists_matches = re.findall(exists_pattern, sql, re.IGNORECASE)
            
            total_subquery_uses = len(in_matches) + len(comp_matches) + len(exists_matches)
            
            # Rough check: should have subquery uses matching SELECT count - 1
            if total_subquery_uses == 0 and select_count > 1:
                errors.append({
                    'type': 'SUBQUERY_ERROR',
                    'message': 'Multiple SELECT statements detected but no clear subquery structure',
                    'severity': 'MEDIUM'
                })
        
        return errors
    
    def _check_common_mistakes(self, sql: str, pruned_schema: Dict) -> List[Dict]:
        """Check for common SQL mistakes"""
        warnings = []
        
        sql_upper = sql.upper()
        
        # 1. SELECT * warning
        if 'SELECT *' in sql_upper or 'SELECT  *' in sql_upper:
            warnings.append({
                'type': 'STYLE_WARNING',
                'message': 'Using SELECT * - consider specifying columns explicitly',
                'severity': 'LOW'
            })
        
        # 2. Missing WHERE with JOIN
        if 'JOIN' in sql_upper and 'WHERE' not in sql_upper:
            warnings.append({
                'type': 'LOGIC_WARNING',
                'message': 'JOIN without WHERE clause - ensure filtering is intentional',
                'severity': 'LOW'
            })
        
        # 3. HAVING without GROUP BY
        if 'HAVING' in sql_upper and 'GROUP BY' not in sql_upper:
            warnings.append({
                'type': 'LOGIC_WARNING',
                'message': 'HAVING clause without GROUP BY',
                'severity': 'MEDIUM'
            })
        
        # 4. ORDER BY without LIMIT on aggregations
        if 'ORDER BY' in sql_upper and any(agg in sql_upper for agg in ['COUNT(', 'SUM(', 'MAX(', 'MIN(']):
            if 'LIMIT' not in sql_upper:
                warnings.append({
                    'type': 'STYLE_WARNING',
                    'message': 'ORDER BY on aggregation without LIMIT - consider if all results are needed',
                    'severity': 'LOW'
                })
        
        return warnings
    
    def _generate_suggestions(
        self, 
        errors: List[Dict], 
        warnings: List[Dict],
        sql: str,
        pruned_schema: Dict
    ) -> List[str]:
        """Generate actionable suggestions based on errors and warnings"""
        suggestions = []
        
        # Suggestions for specific error types
        for error in errors:
            if error['type'] == 'SCHEMA_ERROR' and 'table' in error:
                table = error['table']
                # Suggest similar table names
                similar = self._find_similar_names(table, pruned_schema.keys())
                if similar:
                    suggestions.append(f"Did you mean table '{similar[0]}' instead of '{table}'?")
            
            elif error['type'] == 'SCHEMA_ERROR' and 'column' in error:
                column = error['column']
                # Suggest similar column names
                all_columns = []
                for table, cols in pruned_schema.items():
                    all_columns.extend([col['column_name'] for col in cols])
                
                similar = self._find_similar_names(column, all_columns)
                if similar:
                    suggestions.append(f"Did you mean column '{similar[0]}' instead of '{column}'?")
            
            elif error['type'] == 'JOIN_ERROR':
                suggestions.append("Review the JOIN condition to ensure it uses proper foreign key relationships")
            
            elif error['type'] == 'SYNTAX_ERROR':
                suggestions.append("Check SQL syntax for typos, missing keywords, or unbalanced parentheses")
        
        # Suggestions for warnings
        if len(warnings) > 3:
            suggestions.append("Multiple warnings detected - review query logic carefully")
        
        # General suggestions
        if not suggestions and (errors or warnings):
            suggestions.append("Review the generated SQL against the schema and question requirements")
        
        if not errors and not warnings:
            suggestions.append("SQL passed all validation checks!")
        
        return suggestions
    
    def _find_similar_names(self, target: str, candidates: List[str], max_distance: int = 2) -> List[str]:
        """Find similar names using simple edit distance"""
        target_lower = target.lower()
        
        similar = []
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Simple similarity check
            if target_lower in candidate_lower or candidate_lower in target_lower:
                similar.append(candidate)
            elif self._levenshtein_distance(target_lower, candidate_lower) <= max_distance:
                similar.append(candidate)
        
        return similar[:3]  # Return top 3
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_validation_score(self, errors: List[Dict], warnings: List[Dict]) -> float:
        """Calculate overall validation score (0.0 to 1.0)"""
        score = 1.0
        
        # Deduct for errors
        for error in errors:
            if error['severity'] == 'CRITICAL':
                score -= 0.3
            elif error['severity'] == 'HIGH':
                score -= 0.15
            elif error['severity'] == 'MEDIUM':
                score -= 0.08
        
        # Deduct for warnings
        for warning in warnings:
            if warning['severity'] == 'MEDIUM':
                score -= 0.05
            elif warning['severity'] == 'LOW':
                score -= 0.02
        
        return max(score, 0.0)
    
    def _generate_reasoning(
        self,
        sql: str,
        errors: List[Dict],
        warnings: List[Dict],
        suggestions: List[str],
        validation_score: float,
        is_valid: bool
    ) -> str:
        """Generate reasoning for validation results"""
        reasoning = "STEP 7: SQL VALIDATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += "SQL Query:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Validation Score: {validation_score:.2f}\n"
        reasoning += f"Is Valid: {is_valid}\n\n"
        
        if errors:
            reasoning += f"Errors ({len(errors)}):\n"
            for i, error in enumerate(errors, 1):
                reasoning += f"  {i}. [{error['severity']}] {error['type']}: {error['message']}\n"
            reasoning += "\n"
        
        if warnings:
            reasoning += f"Warnings ({len(warnings)}):\n"
            for i, warning in enumerate(warnings, 1):
                reasoning += f"  {i}. [{warning['severity']}] {warning['type']}: {warning['message']}\n"
            reasoning += "\n"
        
        if suggestions:
            reasoning += f"Suggestions ({len(suggestions)}):\n"
            for i, suggestion in enumerate(suggestions, 1):
                reasoning += f"  {i}. {suggestion}\n"
            reasoning += "\n"
        
        # Overall assessment
        reasoning += "Assessment:\n"
        if is_valid:
            reasoning += "  ✓ SQL query is valid and ready for execution\n"
        else:
            reasoning += "  ⚠ SQL query has critical errors and needs correction\n"
        
        return reasoning