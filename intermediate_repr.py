"""
STEP 6b: Generation with Intermediate Representation (NON_NESTED_COMPLEX)
Uses NatSQL-style intermediate representation for complex JOIN queries
"""
import ollama
import re
from typing import Dict, List, Set


class IntermediateRepresentationGenerator:
    def __init__(self, model: str = "llama3.2"):
        """Initialize intermediate representation generator"""
        self.model = model
    
    def generate_sql_with_intermediate(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        selected_examples: List[Dict]
    ) -> Dict:
        """
        STEP 6b: Generation with Intermediate Representation
        
        Args:
            question: Natural language question
            pruned_schema: Pruned schema from Step 1
            schema_links: Schema links from Step 1
            selected_examples: Similar examples from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'natsql_intermediate': str,
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
        """
        print(f"\n{'='*60}")
        print("STEP 6b: INTERMEDIATE REPRESENTATION GENERATION")
        print(f"{'='*60}\n")
        
        print(f"Question: {question}")
        print(f"Available examples: {len(selected_examples)}")
        
        # Select best examples (top 3-5)
        print("6b.0: Selecting best examples...")
        best_examples = self._select_best_examples(selected_examples, n=5)
        print(f"   Using {len(best_examples)} examples")
        
        # Sub-step 6b.1: Generate NatSQL intermediate representation
        print("6b.1: Generating NatSQL intermediate representation...")
        natsql_intermediate = self._generate_natsql_intermediate(
            question,
            pruned_schema,
            schema_links,
            best_examples
        )
        print(f"   NatSQL generated: {len(natsql_intermediate)} characters")
        
        # Sub-step 6b.2: Convert NatSQL to SQL
        print("6b.2: Converting NatSQL to SQL...")
        generated_sql = self._natsql_to_sql(
            natsql_intermediate,
            pruned_schema,
            schema_links,
            best_examples
        )
        print(f"   SQL generated: {len(generated_sql)} characters")
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            generated_sql, 
            natsql_intermediate,
            best_examples
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question,
            natsql_intermediate,
            generated_sql,
            best_examples,
            confidence
        )
        
        print(f"Confidence: {confidence:.2f}")
        
        print(f"\n{'='*60}")
        print("STEP 6b COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'generated_sql': generated_sql,
            'natsql_intermediate': natsql_intermediate,
            'confidence': confidence,
            'reasoning': reasoning,
            'examples_used': len(best_examples)
        }
    
    def _select_best_examples(
        self, 
        examples: List[Dict], 
        n: int = 5
    ) -> List[Dict]:
        """Select top N examples by similarity score (combined if available)"""
        
        # Check if structural reranking was applied
        has_combined_score = any('combined_score' in ex for ex in examples)
        
        # Sort by combined score if available, else semantic score
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
        
        # Take top N
        return sorted_examples[:n]
    
    def _generate_natsql_intermediate(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict]
    ) -> str:
        """
        Sub-step 6b.1: Generate NatSQL Intermediate Representation
        """
        prompt = "# NatSQL Intermediate Representation Generation\n\n"
        
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
        
        # Add examples with NatSQL-style intermediate
        prompt += "## Examples (Question → Intermediate → SQL)\n\n"
        for i, example in enumerate(examples[:3], 1):
            prompt += f"### Example {i}\n"
            prompt += f"Question: {example.get('question', 'N/A')}\n\n"
            
            # Generate simplified intermediate for example
            intermediate = self._create_intermediate_for_example(example)
            prompt += f"Intermediate:\n{intermediate}\n\n"
            
            prompt += f"SQL:\n```sql\n{example.get('query', 'N/A')}\n```\n\n"
        
        # Add instructions
        prompt += "## Your Task\n\n"
        prompt += f"Question: {question}\n\n"
        
        prompt += "Create an intermediate representation that:\n"
        prompt += "1. Identifies tables to SELECT from and JOIN\n"
        prompt += "2. Specifies JOIN conditions using foreign keys\n"
        prompt += "3. Lists WHERE conditions\n"
        prompt += "4. Specifies any aggregations (COUNT, AVG, etc.)\n"
        prompt += "5. Identifies GROUP BY and ORDER BY requirements\n\n"
        
        prompt += "Format:\n"
        prompt += "SELECT: [columns or aggregations]\n"
        prompt += "FROM: [main table]\n"
        prompt += "JOIN: [joined tables with conditions]\n"
        prompt += "WHERE: [filter conditions]\n"
        prompt += "GROUP BY: [grouping columns if needed]\n"
        prompt += "ORDER BY: [sorting if needed]\n\n"
        
        prompt += "Generate the intermediate representation:\n"
        
        return self._generate_with_llm(prompt, system_msg="You are an expert at creating structured query representations.")
    
    def _create_intermediate_for_example(self, example: Dict) -> str:
        """Create a simplified intermediate representation for an example"""
        sql = example.get('query', '').upper()
        
        intermediate = []
        
        # Extract SELECT
        if 'SELECT' in sql:
            select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.DOTALL)
            if select_match:
                intermediate.append(f"SELECT: {select_match.group(1).strip()}")
        
        # Extract FROM
        if 'FROM' in sql:
            from_match = re.search(r'FROM\s+(\w+)', sql)
            if from_match:
                intermediate.append(f"FROM: {from_match.group(1)}")
        
        # Extract JOINs
        join_matches = re.findall(r'((?:INNER |LEFT |RIGHT )?JOIN\s+\w+\s+ON\s+[^\s]+\s*=\s*[^\s]+)', sql)
        if join_matches:
            intermediate.append(f"JOIN: {' ; '.join(join_matches)}")
        
        # Extract WHERE
        if 'WHERE' in sql:
            where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql, re.DOTALL)
            if where_match:
                intermediate.append(f"WHERE: {where_match.group(1).strip()}")
        
        # Extract GROUP BY
        if 'GROUP BY' in sql:
            group_match = re.search(r'GROUP BY\s+(.*?)(?:HAVING|ORDER BY|LIMIT|$)', sql, re.DOTALL)
            if group_match:
                intermediate.append(f"GROUP BY: {group_match.group(1).strip()}")
        
        # Extract ORDER BY
        if 'ORDER BY' in sql:
            order_match = re.search(r'ORDER BY\s+(.*?)(?:LIMIT|$)', sql, re.DOTALL)
            if order_match:
                intermediate.append(f"ORDER BY: {order_match.group(1).strip()}")
        
        return '\n'.join(intermediate) if intermediate else "Simple query"
    
    def _natsql_to_sql(
        self,
        natsql_intermediate: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict]
    ) -> str:
        """
        Sub-step 6b.2: Convert NatSQL Intermediate to SQL
        """
        prompt = "# Convert Intermediate Representation to SQL\n\n"
        
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
        
        # Add conversion examples
        prompt += "## Conversion Examples\n\n"
        for i, example in enumerate(examples[:2], 1):
            prompt += f"### Example {i}\n"
            intermediate = self._create_intermediate_for_example(example)
            prompt += f"Intermediate:\n{intermediate}\n\n"
            prompt += f"SQL:\n```sql\n{example.get('query', 'N/A')}\n```\n\n"
        
        # Add the intermediate to convert
        prompt += "## Your Task\n\n"
        prompt += "Convert this intermediate representation to SQL:\n\n"
        prompt += natsql_intermediate + "\n\n"
        
        prompt += "Generate syntactically correct SQL using the schema above.\n"
        prompt += "Include proper JOIN conditions, WHERE clauses, and aggregations as specified.\n"
        prompt += "Output ONLY the SQL query:\n"
        
        sql = self._generate_with_llm(
            prompt, 
            system_msg="You are an expert SQL generator. Convert intermediate representations to valid SQL."
        )
        
        return self._clean_sql(sql)
    
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
    
    def _calculate_confidence(
        self, 
        generated_sql: str,
        natsql_intermediate: str,
        examples: List[Dict]
    ) -> float:
        """Calculate confidence score"""
        if not examples:
            return 0.5
        
        # Base confidence from example similarity
        avg_similarity = sum(ex.get('similarity_score', 0) for ex in examples) / len(examples)
        
        # Check SQL validity
        sql_upper = generated_sql.upper()
        
        validity_score = 0.0
        if 'SELECT' in sql_upper:
            validity_score += 0.3
        if 'FROM' in sql_upper:
            validity_score += 0.2
        if 'JOIN' in sql_upper:
            validity_score += 0.2
        if generated_sql.strip().endswith(';'):
            validity_score += 0.1
        if '-- Error' not in generated_sql:
            validity_score += 0.2
        
        # Check intermediate representation quality
        intermediate_score = 0.0
        if 'SELECT:' in natsql_intermediate:
            intermediate_score += 0.3
        if 'FROM:' in natsql_intermediate:
            intermediate_score += 0.3
        if 'JOIN:' in natsql_intermediate or 'WHERE:' in natsql_intermediate:
            intermediate_score += 0.4
        
        # Combine scores
        confidence = (avg_similarity * 0.4) + (validity_score * 0.4) + (intermediate_score * 0.2)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        question: str,
        natsql_intermediate: str,
        generated_sql: str,
        examples: List[Dict],
        confidence: float
    ) -> str:
        """Generate reasoning for Step 6b"""
        reasoning = "STEP 6b: INTERMEDIATE REPRESENTATION GENERATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += f"Examples Used: {len(examples)}\n"
        for i, ex in enumerate(examples, 1):
            score = ex.get('similarity_score', 0)
            reasoning += f"  {i}. Similarity: {score:.4f} - {ex.get('question', 'N/A')[:50]}...\n"
        
        reasoning += f"\nIntermediate Representation (NatSQL-style):\n"
        reasoning += "-" * 50 + "\n"
        reasoning += natsql_intermediate + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Generated SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Confidence Score: {confidence:.2f}\n\n"
        
        reasoning += "Confidence Factors:\n"
        if confidence >= 0.8:
            reasoning += "  ✓ High confidence - Strong intermediate representation and SQL generation\n"
        elif confidence >= 0.6:
            reasoning += "  ⚠ Medium confidence - Acceptable intermediate and SQL quality\n"
        else:
            reasoning += "  ⚠ Low confidence - Limited similarity or structural issues\n"
        
        return reasoning