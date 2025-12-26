"""
MODIFIED Few-Shot Generation - Now uses NatSQL for ALL EASY queries
Ensures consistent structure across all complexity levels
"""
import ollama
import re
from typing import Dict, List


class FewShotGenerator:
    def __init__(self, model: str = "qwen3-coder"):
        """Initialize few-shot generator"""
        self.model = model
    
    def generate_sql_easy(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        selected_examples: List[Dict]
    ) -> Dict:
        """
        MODIFIED: Generate SQL via NatSQL intermediate (even for EASY queries)
        
        Benefits:
        - Consistent structure across all query types
        - Better EM alignment with ground truth
        - Normalized output format
        
        Args:
            question: Natural language question
            pruned_schema: Pruned schema from Step 1
            schema_links: Schema links from Step 1
            selected_examples: Similar examples from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'natsql_intermediate': str,  # NEW
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
        """
        print(f"\n{'='*60}")
        print("STEP 6a: ENHANCED FEW-SHOT GENERATION (via NatSQL)")
        print(f"{'='*60}\n")
        
        print(f"Question: {question}")
        print(f"Available examples: {len(selected_examples)}")
        
        # Select best examples
        print("6a.1: Selecting best examples...")
        best_examples = self._select_best_examples(selected_examples, n=5)
        print(f"   Using {len(best_examples)} examples")
        
        # NEW: Generate NatSQL intermediate first
        print("6a.2: Generating NatSQL intermediate...")
        natsql_intermediate = self._generate_natsql_intermediate(
            question, pruned_schema, schema_links, best_examples
        )
        print(f"   NatSQL: {natsql_intermediate[:80]}...")
        
        # Convert NatSQL to normalized SQL
        print("6a.3: Converting NatSQL to SQL...")
        generated_sql = self._natsql_to_sql(
            natsql_intermediate,
            pruned_schema,
            schema_links,
            best_examples
        )
        print(f"   SQL: {len(generated_sql)} characters")
        
        # Calculate confidence
        confidence = self._calculate_confidence(generated_sql, best_examples)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question, natsql_intermediate, generated_sql, 
            best_examples, confidence
        )
        
        print(f"Confidence: {confidence:.2f}")
        
        print(f"\n{'='*60}")
        print("STEP 6a COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'generated_sql': generated_sql,
            'natsql_intermediate': natsql_intermediate,  # NEW
            'confidence': confidence,
            'reasoning': reasoning,
            'examples_used': len(best_examples)
        }
    
    def _generate_natsql_intermediate(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict]
    ) -> str:
        """
        NEW: Generate NatSQL intermediate representation
        Uses DIN-SQL format
        """
        prompt = "# Generate NatSQL Intermediate (DIN-SQL Format)\n\n"
        
        # Add schema
        prompt += "## Database Schema\n\n"
        for table_name, columns in sorted(pruned_schema.items()):
            col_names = [f"{table_name}.{col['column_name']}" for col in columns]
            prompt += f"**{table_name}**: {', '.join(col_names)}\n"
        prompt += "\n"
        
        # Add NatSQL format guide
        prompt += "## NatSQL Format (DIN-SQL)\n\n"
        prompt += "- No explicit FROM clause (implicit from table.column references)\n"
        prompt += "- JOINs as: WHERE @ JOIN table.*\n"
        prompt += "- Aggregations: count(), avg(), sum(), max(), min()\n"
        prompt += "- Column format: table.column\n\n"
        
        # Add examples with NatSQL conversion
        prompt += "## Examples\n\n"
        for i, example in enumerate(examples[:3], 1):
            sql = example.get('query', '')
            natsql = self._convert_sql_to_natsql(sql)
            
            prompt += f"### Example {i}\n"
            prompt += f"Question: {example.get('question', 'N/A')}\n"
            prompt += f"NatSQL: {natsql}\n"
            prompt += f"SQL: {sql}\n\n"
        
        # Add task
        prompt += "## Your Task\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Generate NatSQL intermediate representation:\n"
        
        natsql = self._generate_with_llm(
            prompt,
            "You are an expert at generating NatSQL intermediate representations."
        )
        
        return natsql.strip()
    
    def _convert_sql_to_natsql(self, sql: str) -> str:
        """Convert SQL to NatSQL format for examples"""
        if not sql:
            return "SELECT ..."
        
        sql_upper = sql.upper()
        
        # Extract SELECT columns
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if not select_match:
            return "SELECT ..."
        
        select_clause = select_match.group(1).strip()
        
        # Remove aliases
        select_clause = re.sub(r'\bT\d+\.', '', select_clause, flags=re.IGNORECASE)
        select_clause = re.sub(r'\bAS\s+\w+', '', select_clause, flags=re.IGNORECASE)
        
        natsql = f"SELECT {select_clause}"
        
        # Handle JOIN
        if 'JOIN' in sql_upper:
            join_tables = re.findall(r'JOIN\s+(\w+)', sql_upper)
            if join_tables:
                natsql += f" WHERE @ JOIN {join_tables[0]}.*"
        
        # Handle WHERE
        elif 'WHERE' in sql_upper:
            where_match = re.search(r'WHERE\s+(.*?)(?:GROUP|ORDER|LIMIT|$)', sql_upper, re.DOTALL)
            if where_match:
                where_clause = where_match.group(1).strip()
                where_clause = re.sub(r'\bT\d+\.', '', where_clause, flags=re.IGNORECASE)
                natsql += f" WHERE {where_clause}"
        
        return natsql
    
    def _natsql_to_sql(
        self,
        natsql: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict]
    ) -> str:
        """
        NEW: Convert NatSQL to normalized SQL
        """
        prompt = "# Convert NatSQL to SQL\n\n"
        
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
            natsql_ex = self._convert_sql_to_natsql(sql)
            
            prompt += f"### Example {i}\n"
            prompt += f"NatSQL: {natsql_ex}\n"
            prompt += f"SQL:\n```sql\n{sql}\n```\n\n"
        
        # Add task
        prompt += "## Your Task\n\n"
        prompt += f"NatSQL: {natsql}\n\n"
        prompt += "Convert to SQL. Follow these rules:\n"
        prompt += "1. Add explicit FROM clause\n"
        prompt += "2. Convert WHERE @ JOIN to proper JOIN ON\n"
        prompt += "3. Use standard SQL syntax\n"
        prompt += "4. Match the style of ground truth examples\n\n"
        prompt += "Output ONLY the SQL query:\n"
        
        sql = self._generate_with_llm(
            prompt,
            "You are an expert SQL generator. Convert NatSQL to standard SQL."
        )
        
        return self._clean_sql(sql)
    
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
    
    def _generate_with_llm(self, prompt: str, system_msg: str = None) -> str:
        """Generate using LLM"""
        if system_msg is None:
            system_msg = 'You are an expert SQL query generator.'
        
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
    
    def _calculate_confidence(self, generated_sql: str, examples: List[Dict]) -> float:
        """Calculate confidence score"""
        if not examples:
            return 0.5
        
        avg_similarity = sum(ex.get('similarity_score', 0) for ex in examples) / len(examples)
        
        sql_upper = generated_sql.upper()
        
        validity_score = 0.0
        if 'SELECT' in sql_upper:
            validity_score += 0.4
        if 'FROM' in sql_upper:
            validity_score += 0.3
        if generated_sql.strip().endswith(';'):
            validity_score += 0.1
        if '-- Error' not in generated_sql:
            validity_score += 0.2
        
        confidence = (avg_similarity * 0.6) + (validity_score * 0.4)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        question: str,
        natsql_intermediate: str,
        generated_sql: str,
        examples: List[Dict],
        confidence: float
    ) -> str:
        """Generate reasoning"""
        reasoning = "STEP 6a: ENHANCED FEW-SHOT GENERATION (via NatSQL)\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += f"Examples Used: {len(examples)}\n"
        has_structural = any('structural_similarity' in ex for ex in examples)
        
        for i, ex in enumerate(examples, 1):
            if has_structural:
                sem_score = ex.get('similarity_score', 0)
                struct_score = ex.get('structural_similarity', 0)
                combined_score = ex.get('combined_score', 0)
                
                reasoning += f"  {i}. Combined: {combined_score:.4f} "
                reasoning += f"(Sem: {sem_score:.4f}, Struct: {struct_score:.4f})\n"
                reasoning += f"     Question: {ex.get('question', 'N/A')[:50]}...\n"
            else:
                score = ex.get('similarity_score', 0)
                reasoning += f"  {i}. Similarity: {score:.4f} - {ex.get('question', 'N/A')[:50]}...\n"
        
        reasoning += f"\nNatSQL Intermediate:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += natsql_intermediate + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Generated SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Confidence Score: {confidence:.2f}\n\n"
        
        reasoning += "Benefits of NatSQL Intermediate:\n"
        reasoning += "  ✓ Consistent structure across all query types\n"
        reasoning += "  ✓ Better EM alignment with ground truth\n"
        reasoning += "  ✓ Normalized output format\n"
        
        return reasoning