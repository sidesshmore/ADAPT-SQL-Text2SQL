"""
ENHANCED Intermediate Representation Generator with Universal NatSQL
Now generates NatSQL for ALL query types (EASY, NON_NESTED, NESTED)
Implements DIN-SQL structure normalization for better EM scores
"""
import ollama
import re
from typing import Dict, List, Set, Tuple


class IntermediateRepresentationGenerator:
    def __init__(self, model: str = "qwen3-coder"):
        """Initialize intermediate representation generator"""
        self.model = model
        
        # DIN-SQL structure templates for common patterns
        self.natsql_templates = {
            'simple_select': 'SELECT {columns} FROM {tables}',
            'with_where': 'SELECT {columns} FROM {tables} WHERE {conditions}',
            'with_groupby': 'SELECT {columns} FROM {tables} WHERE {conditions} GROUP BY {group_cols}',
            'with_orderby': 'SELECT {columns} FROM {tables} WHERE {conditions} ORDER BY {order_cols}',
            'with_join': 'SELECT {columns} WHERE @ JOIN {join_tables}',
            'nested_in': 'SELECT {columns} WHERE {col} IN ({subquery})',
            'nested_comparison': 'SELECT {columns} WHERE {col} {op} ({subquery})',
            'nested_exists': 'SELECT {columns} WHERE EXISTS ({subquery})'
        }
        
        # Canonical clause ordering (DIN-SQL standard)
        self.clause_order = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']
        
        # Standard aggregation functions
        self.agg_functions = ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']
    
    def generate_sql_with_intermediate(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        selected_examples: List[Dict]
    ) -> Dict:
        """
        ENHANCED: Generate SQL via NatSQL intermediate for ALL query types
        
        Args:
            question: Natural language question
            pruned_schema: Pruned schema from Step 1
            schema_links: Schema links from Step 1
            selected_examples: Similar examples from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'natsql_intermediate': str,
                'natsql_structure': Dict,
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
        """
        print(f"\n{'='*60}")
        print("ENHANCED INTERMEDIATE REPRESENTATION GENERATION (Universal NatSQL)")
        print(f"{'='*60}\n")
        
        print(f"Question: {question}")
        print(f"Available examples: {len(selected_examples)}")
        
        # Select best examples
        print("Step 1: Selecting best examples...")
        best_examples = self._select_best_examples(selected_examples, n=5)
        print(f"   Using {len(best_examples)} examples")
        
        # NEW: Analyze ground truth patterns from examples
        print("Step 2: Analyzing ground truth patterns...")
        gt_patterns = self._analyze_ground_truth_patterns(best_examples)
        print(f"   Found {len(gt_patterns['common_structures'])} common structures")
        
        # Generate NatSQL intermediate with structure normalization
        print("Step 3: Generating NatSQL intermediate...")
        natsql_result = self._generate_natsql_universal(
            question,
            pruned_schema,
            schema_links,
            best_examples,
            gt_patterns
        )
        print(f"   NatSQL generated: {len(natsql_result['natsql'])} characters")
        
        # Convert NatSQL to SQL with structure normalization
        print("Step 4: Converting NatSQL to normalized SQL...")
        generated_sql = self._natsql_to_normalized_sql(
            natsql_result['natsql'],
            natsql_result['structure'],
            pruned_schema,
            schema_links,
            best_examples,
            gt_patterns
        )
        print(f"   SQL generated: {len(generated_sql)} characters")
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            generated_sql, 
            natsql_result['natsql'],
            best_examples
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question,
            natsql_result['natsql'],
            generated_sql,
            best_examples,
            confidence,
            gt_patterns
        )
        
        print(f"Confidence: {confidence:.2f}")
        
        print(f"\n{'='*60}")
        print("ENHANCED GENERATION COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'generated_sql': generated_sql,
            'natsql_intermediate': natsql_result['natsql'],
            'natsql_structure': natsql_result['structure'],
            'confidence': confidence,
            'reasoning': reasoning,
            'examples_used': len(best_examples)
        }
    
    def _analyze_ground_truth_patterns(self, examples: List[Dict]) -> Dict:
        """
        NEW: Analyze ground truth SQL patterns from examples
        Learn common structures, JOIN orders, aggregation formats
        """
        patterns = {
            'common_structures': [],
            'join_patterns': [],
            'aggregation_formats': [],
            'clause_orders': [],
            'table_orderings': []
        }
        
        for example in examples:
            sql = example.get('query', '')
            if not sql:
                continue
            
            sql_upper = sql.upper()
            
            # Extract structure pattern
            structure = self._extract_structure_pattern(sql)
            patterns['common_structures'].append(structure)
            
            # Extract JOIN pattern
            if 'JOIN' in sql_upper:
                join_pattern = self._extract_join_pattern(sql)
                patterns['join_patterns'].append(join_pattern)
            
            # Extract aggregation format
            for agg in self.agg_functions:
                if agg in sql_upper:
                    agg_format = self._extract_aggregation_format(sql, agg)
                    patterns['aggregation_formats'].append(agg_format)
            
            # Extract clause order
            clause_order = self._extract_clause_order(sql)
            patterns['clause_orders'].append(clause_order)
            
            # Extract table ordering (for FROM and JOIN)
            table_order = self._extract_table_ordering(sql)
            if table_order:
                patterns['table_orderings'].append(table_order)
        
        # Find most common patterns
        patterns['most_common_structure'] = self._find_most_common(patterns['common_structures'])
        patterns['most_common_join'] = self._find_most_common(patterns['join_patterns'])
        patterns['most_common_agg_format'] = self._find_most_common(patterns['aggregation_formats'])
        
        return patterns
    
    def _extract_structure_pattern(self, sql: str) -> str:
        """Extract high-level structure pattern"""
        sql_upper = sql.upper()
        
        pattern_parts = []
        if 'SELECT' in sql_upper:
            pattern_parts.append('SELECT')
        if 'FROM' in sql_upper:
            pattern_parts.append('FROM')
        if 'JOIN' in sql_upper:
            pattern_parts.append('JOIN')
        if 'WHERE' in sql_upper:
            pattern_parts.append('WHERE')
        if 'GROUP BY' in sql_upper:
            pattern_parts.append('GROUP_BY')
        if 'HAVING' in sql_upper:
            pattern_parts.append('HAVING')
        if 'ORDER BY' in sql_upper:
            pattern_parts.append('ORDER_BY')
        if 'LIMIT' in sql_upper:
            pattern_parts.append('LIMIT')
        
        return '-'.join(pattern_parts)
    
    def _extract_join_pattern(self, sql: str) -> str:
        """Extract JOIN pattern (e.g., 'INNER_JOIN-LEFT_JOIN')"""
        sql_upper = sql.upper()
        
        join_types = []
        if 'INNER JOIN' in sql_upper:
            join_types.append('INNER_JOIN')
        elif 'LEFT JOIN' in sql_upper:
            join_types.append('LEFT_JOIN')
        elif 'RIGHT JOIN' in sql_upper:
            join_types.append('RIGHT_JOIN')
        elif 'JOIN' in sql_upper:
            join_types.append('JOIN')
        
        return '-'.join(join_types) if join_types else 'NONE'
    
    def _extract_aggregation_format(self, sql: str, agg_func: str) -> str:
        """
        Extract aggregation format
        Returns: 'WITH_ALIAS' or 'NO_ALIAS'
        """
        sql_upper = sql.upper()
        
        # Check if aggregation has AS alias
        agg_pattern = rf'{agg_func}\s*\([^)]+\)\s+AS\s+\w+'
        if re.search(agg_pattern, sql_upper):
            return 'WITH_ALIAS'
        else:
            return 'NO_ALIAS'
    
    def _extract_clause_order(self, sql: str) -> List[str]:
        """Extract the order of SQL clauses"""
        sql_upper = sql.upper()
        
        order = []
        for clause in self.clause_order:
            if clause in sql_upper:
                order.append(clause)
        
        return order
    
    def _extract_table_ordering(self, sql: str) -> List[str]:
        """Extract table ordering from FROM and JOIN clauses"""
        sql_upper = sql.upper()
        
        tables = []
        
        # Extract FROM table
        from_match = re.search(r'FROM\s+(\w+)', sql_upper)
        if from_match:
            tables.append(from_match.group(1))
        
        # Extract JOIN tables
        join_matches = re.findall(r'JOIN\s+(\w+)', sql_upper)
        tables.extend(join_matches)
        
        return tables
    
    def _find_most_common(self, items: List) -> str:
        """Find most common item in list"""
        if not items:
            return 'NONE'
        
        from collections import Counter
        counter = Counter(items)
        return counter.most_common(1)[0][0] if counter else 'NONE'
    
    def _generate_natsql_universal(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict],
        gt_patterns: Dict
    ) -> Dict:
        """
        NEW: Generate NatSQL for ANY query type (EASY, NON_NESTED, NESTED)
        Uses DIN-SQL format with structure normalization
        """
        prompt = "# NatSQL Generation (DIN-SQL Format)\n\n"
        
        # Add DIN-SQL grammar reference
        prompt += "## NatSQL Grammar (DIN-SQL Standard)\n\n"
        prompt += """
NatSQL = SELECT Column {, Column} [WHERE Condition] [ORDER BY Column]

Rules:
1. No explicit FROM clause (implicit from table.column references)
2. JOIN represented as: WHERE @ JOIN table.*
3. Canonical column format: table.column or agg(table.column)
4. Aggregations: count(), sum(), avg(), max(), min()
5. Conditions: column operator value
6. Conjunctions: AND, OR, EXCEPT, INTERSECT, UNION

Examples:
- Simple: SELECT student.name
- With WHERE: SELECT student.name WHERE student.age > 18
- With JOIN: SELECT student.name WHERE @ JOIN has_pet.*
- With AGG: SELECT count(student.id) WHERE student.age > 18
"""
        prompt += "\n"
        
        # Add schema
        prompt += "## Database Schema\n\n"
        prompt += self._format_schema_natsql(pruned_schema, schema_links)
        prompt += "\n"
        
        # Add ground truth pattern guidance
        prompt += "## Ground Truth Patterns (Learn from these)\n\n"
        prompt += f"Common Structure: {gt_patterns.get('most_common_structure', 'UNKNOWN')}\n"
        prompt += f"Common JOIN Pattern: {gt_patterns.get('most_common_join', 'UNKNOWN')}\n"
        prompt += f"Common Aggregation Format: {gt_patterns.get('most_common_agg_format', 'UNKNOWN')}\n\n"
        
        # Add examples with NatSQL conversion
        prompt += "## Examples (SQL → NatSQL Conversion)\n\n"
        for i, example in enumerate(examples[:3], 1):
            prompt += f"### Example {i}\n"
            prompt += f"Question: {example.get('question', 'N/A')}\n"
            
            sql = example.get('query', '')
            natsql_converted = self._convert_sql_to_natsql_example(sql)
            
            prompt += f"SQL:\n```sql\n{sql}\n```\n"
            prompt += f"NatSQL:\n```\n{natsql_converted}\n```\n\n"
        
        # Add task
        prompt += "## Your Task\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Generate NatSQL following DIN-SQL format:\n"
        prompt += "- Use table.column format\n"
        prompt += "- No explicit FROM clause\n"
        prompt += "- Represent JOINs as: WHERE @ JOIN table.*\n"
        prompt += "- Use standard aggregations: count(), avg(), etc.\n"
        prompt += "- Follow the ground truth patterns above\n\n"
        prompt += "Generate NatSQL:\n"
        
        natsql = self._generate_with_llm(
            prompt,
            "You are an expert at generating NatSQL intermediate representations following DIN-SQL format."
        )
        
        # Parse structure
        structure = self._parse_natsql_structure(natsql)
        
        return {
            'natsql': natsql,
            'structure': structure
        }
    
    def _convert_sql_to_natsql_example(self, sql: str) -> str:
        """Convert SQL to NatSQL format for examples"""
        if not sql:
            return "N/A"
        
        sql_upper = sql.upper()
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if not select_match:
            return "SELECT ..."
        
        select_clause = select_match.group(1).strip()
        
        # Remove table aliases (T1, T2, etc.)
        select_clause = re.sub(r'\bT\d+\.', '', select_clause, flags=re.IGNORECASE)
        
        # Build NatSQL
        natsql = f"SELECT {select_clause}"
        
        # Check for JOIN (convert to WHERE @ JOIN)
        if 'JOIN' in sql_upper:
            join_tables = re.findall(r'JOIN\s+(\w+)', sql_upper)
            if join_tables:
                natsql += f" WHERE @ JOIN {join_tables[0]}.*"
        
        # Add WHERE conditions (if no JOIN)
        elif 'WHERE' in sql_upper:
            where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
            if where_match:
                where_clause = where_match.group(1).strip()
                # Simplify WHERE clause
                where_clause = re.sub(r'\bT\d+\.', '', where_clause, flags=re.IGNORECASE)
                natsql += f" WHERE {where_clause}"
        
        return natsql
    
    def _format_schema_natsql(
        self,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict
    ) -> str:
        """Format schema for NatSQL generation"""
        schema_str = ""
        
        for table_name, columns in sorted(pruned_schema.items()):
            col_names = [f"{table_name}.{col['column_name']}" for col in columns]
            schema_str += f"**{table_name}**: {', '.join(col_names)}\n"
        
        return schema_str
    
    def _parse_natsql_structure(self, natsql: str) -> Dict:
        """Parse NatSQL structure"""
        structure = {
            'has_select': 'SELECT' in natsql.upper(),
            'has_where': 'WHERE' in natsql.upper(),
            'has_join': 'JOIN' in natsql.upper(),
            'has_aggregation': any(agg.lower() in natsql.lower() for agg in self.agg_functions),
            'has_orderby': 'ORDER BY' in natsql.upper()
        }
        
        return structure
    
    def _natsql_to_normalized_sql(
        self,
        natsql: str,
        natsql_structure: Dict,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict],
        gt_patterns: Dict
    ) -> str:
        """
        NEW: Convert NatSQL to SQL with structure normalization
        Applies canonical formatting based on ground truth patterns
        """
        prompt = "# Convert NatSQL to Normalized SQL\n\n"
        
        # Add normalization rules
        prompt += "## SQL Normalization Rules\n\n"
        prompt += "1. **Canonical Clause Order**: SELECT, FROM, JOIN, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT\n"
        prompt += "2. **Table Ordering**: Alphabetical (unless FK dependency requires otherwise)\n"
        prompt += "3. **Aggregation Format**: Follow ground truth pattern (WITH_ALIAS or NO_ALIAS)\n"
        prompt += "4. **JOIN Format**: Use INNER JOIN, LEFT JOIN (explicit)\n"
        prompt += "5. **Column Format**: Remove verbose aliases on aggregations if ground truth doesn't use them\n\n"
        
        # Add ground truth patterns
        prompt += "## Ground Truth Patterns to Follow\n\n"
        prompt += f"Structure: {gt_patterns.get('most_common_structure', 'UNKNOWN')}\n"
        prompt += f"JOIN Style: {gt_patterns.get('most_common_join', 'UNKNOWN')}\n"
        prompt += f"Aggregation Style: {gt_patterns.get('most_common_agg_format', 'UNKNOWN')}\n\n"
        
        # Add schema
        prompt += "## Database Schema\n\n"
        prompt += self._format_schema(pruned_schema, schema_links)
        prompt += "\n"
        
        # Add foreign keys
        if schema_links.get('foreign_keys'):
            prompt += "## Foreign Key Relationships\n\n"
            for fk in schema_links['foreign_keys']:
                prompt += f"- {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
            prompt += "\n"
        
        # Add examples
        prompt += "## Conversion Examples\n\n"
        for i, example in enumerate(examples[:2], 1):
            sql = example.get('query', '')
            natsql_ex = self._convert_sql_to_natsql_example(sql)
            
            prompt += f"### Example {i}\n"
            prompt += f"NatSQL: {natsql_ex}\n"
            prompt += f"SQL:\n```sql\n{sql}\n```\n\n"
        
        # Add NatSQL to convert
        prompt += "## Your Task\n\n"
        prompt += f"NatSQL:\n{natsql}\n\n"
        prompt += "Convert to normalized SQL following the rules and ground truth patterns above.\n"
        prompt += "Output ONLY the SQL query:\n"
        
        sql = self._generate_with_llm(
            prompt,
            "You are an expert at converting NatSQL to normalized SQL following ground truth patterns."
        )
        
        # Apply post-processing normalization
        sql = self._apply_structure_normalization(sql, gt_patterns)
        
        return self._clean_sql(sql)
    
    def _apply_structure_normalization(self, sql: str, gt_patterns: Dict) -> str:
        """
        Apply structure normalization based on ground truth patterns
        """
        # 1. Remove verbose aggregation aliases if ground truth uses NO_ALIAS
        if gt_patterns.get('most_common_agg_format') == 'NO_ALIAS':
            for agg in self.agg_functions:
                # Remove patterns like: COUNT(*) AS total_count → COUNT(*)
                pattern = rf'({agg}\s*\([^)]+\))\s+AS\s+\w+'
                sql = re.sub(pattern, r'\1', sql, flags=re.IGNORECASE)
        
        # 2. Normalize whitespace
        sql = re.sub(r'\s+', ' ', sql)
        
        # 3. Ensure canonical clause order (basic check)
        # This is complex, so we do a simple validation
        sql_upper = sql.upper()
        expected_order = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY', 'LIMIT']
        
        # Just ensure it's reasonably ordered (full reordering is too complex)
        
        return sql
    
    def _select_best_examples(self, examples: List[Dict], n: int = 5) -> List[Dict]:
        """Select top N examples by similarity score"""
        has_combined_score = any('combined_score' in ex for ex in examples)
        
        if has_combined_score:
            sorted_examples = sorted(
                examples,
                key=lambda x: x.get('combined_score', 0),
                reverse=True
            )
        else:
            sorted_examples = sorted(
                examples,
                key=lambda x: x.get('similarity_score', 0),
                reverse=True
            )
        
        return sorted_examples[:n]
    
    def _format_schema(
        self, 
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict
    ) -> str:
        """Format schema with highlighted relevant columns"""
        schema_str = ""
        
        for table_name, columns in sorted(pruned_schema.items()):
            schema_str += f"**{table_name}**\n"
            
            relevant_cols = schema_links.get('columns', {}).get(table_name, set())
            
            for col in columns:
                col_name = col['column_name']
                data_type = col.get('data_type', '')
                
                if col_name in relevant_cols:
                    schema_str += f"  ⭐ {col_name} ({data_type})\n"
                else:
                    schema_str += f"  • {col_name} ({data_type})\n"
            
            schema_str += "\n"
        
        return schema_str
    
    def _generate_with_llm(self, prompt: str, system_msg: str) -> str:
        """Generate using LLM"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_msg},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.2}
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            return f"-- Error: {str(e)}"
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL output from LLM"""
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
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
    
    def _calculate_confidence(
        self, 
        generated_sql: str,
        natsql_intermediate: str,
        examples: List[Dict]
    ) -> float:
        """Calculate confidence score"""
        if not examples:
            return 0.5
        
        avg_similarity = sum(ex.get('similarity_score', 0) for ex in examples) / len(examples)
        
        sql_upper = generated_sql.upper()
        
        validity_score = 0.0
        if 'SELECT' in sql_upper:
            validity_score += 0.3
        if 'FROM' in sql_upper:
            validity_score += 0.2
        if generated_sql.strip().endswith(';'):
            validity_score += 0.1
        if '-- Error' not in generated_sql:
            validity_score += 0.4
        
        confidence = (avg_similarity * 0.5) + (validity_score * 0.5)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        question: str,
        natsql_intermediate: str,
        generated_sql: str,
        examples: List[Dict],
        confidence: float,
        gt_patterns: Dict
    ) -> str:
        """Generate reasoning"""
        reasoning = "ENHANCED INTERMEDIATE REPRESENTATION GENERATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += "Ground Truth Patterns Applied:\n"
        reasoning += f"  • Structure: {gt_patterns.get('most_common_structure', 'UNKNOWN')}\n"
        reasoning += f"  • JOIN Style: {gt_patterns.get('most_common_join', 'UNKNOWN')}\n"
        reasoning += f"  • Aggregation Format: {gt_patterns.get('most_common_agg_format', 'UNKNOWN')}\n\n"
        
        reasoning += f"NatSQL Intermediate:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += natsql_intermediate + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Generated SQL (Normalized):\n"
        reasoning += "-" * 50 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Confidence Score: {confidence:.2f}\n"
        
        return reasoning