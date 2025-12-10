"""
Preliminary SQL Prediction - STEP 3
Generates rough SQL skeleton for example matching
"""
import ollama
import re
from typing import Dict, List


class PreliminaryPredictor:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
    
    def predict_sql_skeleton(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict = None
    ) -> Dict:
        """
        STEP 3: Preliminary SQL Prediction
        
        Args:
            question: Natural language question
            pruned_schema: Pruned schema from Step 1
            schema_links: Schema links from Step 1 (optional)
            
        Returns:
            {
                'predicted_sql': str,
                'sql_skeleton': str,
                'sql_keywords': List[str],
                'sql_structure': Dict,
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 3: PRELIMINARY SQL PREDICTION")
        print(f"{'='*60}\n")
        
        # Generate rough SQL using LLM
        print("3.1: Generating preliminary SQL...")
        predicted_sql = self._generate_preliminary_sql(
            question, pruned_schema, schema_links
        )
        print(f"   SQL generated: {len(predicted_sql)} characters")
        
        # Extract SQL skeleton
        print("3.2: Extracting SQL skeleton...")
        sql_skeleton = self._extract_sql_skeleton(predicted_sql)
        print(f"   Skeleton: {sql_skeleton}")
        
        # Extract SQL keywords
        print("3.3: Extracting SQL keywords...")
        sql_keywords = self._extract_sql_keywords(predicted_sql)
        print(f"   Keywords: {', '.join(sql_keywords)}")
        
        # Analyze SQL structure
        print("3.4: Analyzing SQL structure...")
        sql_structure = self._analyze_sql_structure(predicted_sql)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question, predicted_sql, sql_skeleton, 
            sql_keywords, sql_structure
        )
        
        print(f"\n{'='*60}")
        print("STEP 3 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'predicted_sql': predicted_sql,
            'sql_skeleton': sql_skeleton,
            'sql_keywords': sql_keywords,
            'sql_structure': sql_structure,
            'reasoning': reasoning
        }
    
    def _generate_preliminary_sql(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict = None
    ) -> str:
        """Generate rough SQL using LLM"""
        
        # Create schema summary
        schema_str = "SCHEMA:\n"
        for table, cols in pruned_schema.items():
            col_list = ', '.join([c['column_name'] for c in cols])
            schema_str += f"  {table}({col_list})\n"
        
        # Add foreign keys if available
        fk_str = ""
        if schema_links and schema_links.get('foreign_keys'):
            fk_str = "\nFOREIGN KEYS:\n"
            for fk in schema_links['foreign_keys']:
                fk_str += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        
        prompt = f"""{schema_str}{fk_str}

QUESTION: {question}

Generate a preliminary SQL query to answer this question.

REQUIREMENTS:
- Use ONLY the tables and columns from the schema above
- Focus on correct SQL structure and logic
- Use proper JOIN conditions if multiple tables
- Include appropriate WHERE, GROUP BY, HAVING, ORDER BY as needed
- Use standard SQL syntax

Generate the SQL query (just the query, no explanations):"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a SQL query generator. Generate syntactically correct SQL queries based on the given schema.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.3  # Lower temperature for more consistent SQL
                }
            )
            
            sql = response['message']['content'].strip()
            
            # Clean up response
            sql = self._clean_sql_response(sql)
            
            return sql
            
        except Exception as e:
            return f"-- Error generating SQL: {str(e)}"
    
    def _clean_sql_response(self, sql: str) -> str:
        """Clean up SQL response from LLM"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove explanatory text before/after SQL
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_upper = line.strip().upper()
            
            # Check if line starts SQL
            if any(line_upper.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                in_sql = True
            
            if in_sql:
                sql_lines.append(line)
                
                # Check if line ends SQL (semicolon)
                if line.strip().endswith(';'):
                    break
        
        if sql_lines:
            sql = '\n'.join(sql_lines)
        
        # Ensure it ends with semicolon
        sql = sql.strip()
        if not sql.endswith(';'):
            sql += ';'
        
        return sql
    
    def _extract_sql_skeleton(self, sql: str) -> str:
        """
        Extract SQL skeleton (keywords only)
        Example: "SELECT ... FROM ... JOIN ... WHERE ... GROUP BY ... HAVING ... ORDER BY"
        """
        sql_upper = sql.upper()
        
        # SQL clause keywords in order
        keywords = [
            'SELECT',
            'FROM',
            'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
            'WHERE',
            'GROUP BY',
            'HAVING',
            'ORDER BY',
            'LIMIT'
        ]
        
        skeleton_parts = []
        
        for keyword in keywords:
            if keyword in sql_upper:
                # Handle JOIN variations
                if 'JOIN' in keyword:
                    if keyword != 'JOIN':
                        if keyword in sql_upper:
                            skeleton_parts.append(keyword)
                    else:
                        # Check if it's a plain JOIN (not LEFT/RIGHT/INNER/OUTER)
                        if re.search(r'\bJOIN\b(?!\s*BY)', sql_upper):
                            if 'LEFT JOIN' not in sql_upper and 'RIGHT JOIN' not in sql_upper:
                                skeleton_parts.append('JOIN')
                else:
                    skeleton_parts.append(keyword)
        
        # Check for subqueries
        if sql_upper.count('SELECT') > 1:
            skeleton_parts.append('SUBQUERY')
        
        # Check for set operations
        if 'UNION' in sql_upper:
            skeleton_parts.append('UNION')
        if 'INTERSECT' in sql_upper:
            skeleton_parts.append('INTERSECT')
        if 'EXCEPT' in sql_upper:
            skeleton_parts.append('EXCEPT')
        
        skeleton = ' '.join(skeleton_parts)
        
        return skeleton if skeleton else 'SELECT FROM'
    
    def _extract_sql_keywords(self, sql: str) -> List[str]:
        """Extract all SQL keywords used in the query"""
        sql_upper = sql.upper()
        
        # Comprehensive SQL keywords
        all_keywords = [
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN', 
            'RIGHT JOIN', 'OUTER JOIN', 'ON', 'GROUP BY', 'HAVING', 
            'ORDER BY', 'ASC', 'DESC', 'LIMIT', 'OFFSET',
            'DISTINCT', 'AS', 'AND', 'OR', 'NOT', 'IN', 'EXISTS',
            'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
            'UNION', 'INTERSECT', 'EXCEPT',
            'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'LIKE', 'BETWEEN', 'IS NULL', 'IS NOT NULL'
        ]
        
        found_keywords = []
        
        for keyword in all_keywords:
            if keyword in sql_upper:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _analyze_sql_structure(self, sql: str) -> Dict:
        """Analyze the structure of the SQL query"""
        sql_upper = sql.upper()
        
        structure = {
            'num_selects': sql_upper.count('SELECT'),
            'num_joins': len(re.findall(r'\bJOIN\b', sql_upper)),
            'num_tables': len(re.findall(r'\bFROM\s+(\w+)', sql_upper)) + 
                         len(re.findall(r'\bJOIN\s+(\w+)', sql_upper)),
            'has_subquery': sql_upper.count('SELECT') > 1,
            'has_aggregation': any(agg in sql_upper for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']),
            'has_groupby': 'GROUP BY' in sql_upper,
            'has_having': 'HAVING' in sql_upper,
            'has_orderby': 'ORDER BY' in sql_upper,
            'has_limit': 'LIMIT' in sql_upper,
            'has_distinct': 'DISTINCT' in sql_upper,
            'has_set_operation': any(op in sql_upper for op in ['UNION', 'INTERSECT', 'EXCEPT'])
        }
        
        # Determine query type
        if structure['has_subquery']:
            structure['query_type'] = 'NESTED'
        elif structure['num_joins'] > 0:
            structure['query_type'] = 'JOIN'
        else:
            structure['query_type'] = 'SIMPLE'
        
        # Calculate complexity score
        complexity_score = 0
        complexity_score += structure['num_selects'] * 2
        complexity_score += structure['num_joins'] * 3
        complexity_score += 2 if structure['has_subquery'] else 0
        complexity_score += 1 if structure['has_aggregation'] else 0
        complexity_score += 1 if structure['has_groupby'] else 0
        complexity_score += 1 if structure['has_having'] else 0
        complexity_score += 2 if structure['has_set_operation'] else 0
        
        structure['complexity_score'] = complexity_score
        
        return structure
    
    def _generate_reasoning(
        self,
        question: str,
        predicted_sql: str,
        sql_skeleton: str,
        sql_keywords: List[str],
        sql_structure: Dict
    ) -> str:
        """Generate reasoning for Step 3"""
        reasoning = "STEP 3: PRELIMINARY SQL PREDICTION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += "Generated SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += predicted_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"SQL Skeleton: {sql_skeleton}\n\n"
        
        reasoning += f"Keywords ({len(sql_keywords)}): {', '.join(sql_keywords)}\n\n"
        
        reasoning += "Structure Analysis:\n"
        reasoning += f"  • Query Type: {sql_structure['query_type']}\n"
        reasoning += f"  • Number of SELECTs: {sql_structure['num_selects']}\n"
        reasoning += f"  • Number of JOINs: {sql_structure['num_joins']}\n"
        reasoning += f"  • Number of Tables: {sql_structure['num_tables']}\n"
        reasoning += f"  • Has Subquery: {'Yes' if sql_structure['has_subquery'] else 'No'}\n"
        reasoning += f"  • Has Aggregation: {'Yes' if sql_structure['has_aggregation'] else 'No'}\n"
        reasoning += f"  • Has GROUP BY: {'Yes' if sql_structure['has_groupby'] else 'No'}\n"
        reasoning += f"  • Has HAVING: {'Yes' if sql_structure['has_having'] else 'No'}\n"
        reasoning += f"  • Has ORDER BY: {'Yes' if sql_structure['has_orderby'] else 'No'}\n"
        reasoning += f"  • Complexity Score: {sql_structure['complexity_score']}\n"
        
        return reasoning