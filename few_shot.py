"""
STEP 6a: Simple Few-Shot Generation (EASY queries)
Direct SQL generation using similar examples
"""
import ollama
import re
from typing import Dict, List


class FewShotGenerator:
    def __init__(self, model: str = "llama3.2"):
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
        STEP 6a: Simple Few-Shot Generation (EASY queries)
        
        Args:
            question: Natural language question
            pruned_schema: Pruned schema from Step 1
            schema_links: Schema links from Step 1
            selected_examples: Similar examples from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
        """
        print(f"\n{'='*60}")
        print("STEP 6a: SIMPLE FEW-SHOT GENERATION")
        print(f"{'='*60}\n")
        
        print(f"Question: {question}")
        print(f"Available examples: {len(selected_examples)}")
        
        # Select best examples (top 3-5)
        print("6a.1: Selecting best examples...")
        best_examples = self._select_best_examples(selected_examples, n=5)
        print(f"   Using {len(best_examples)} examples")
        
        # Build few-shot prompt
        print("6a.2: Building few-shot prompt...")
        prompt = self._build_few_shot_prompt(
            question, pruned_schema, schema_links, best_examples
        )
        
        # Generate SQL
        print("6a.3: Generating SQL...")
        generated_sql = self._generate_with_llm(prompt)
        
        # Calculate confidence
        confidence = self._calculate_confidence(generated_sql, best_examples)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question, generated_sql, best_examples, confidence
        )
        
        print(f"Generated SQL: {len(generated_sql)} characters")
        print(f"Confidence: {confidence:.2f}")
        
        print(f"\n{'='*60}")
        print("STEP 6a COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'generated_sql': generated_sql,
            'confidence': confidence,
            'reasoning': reasoning,
            'examples_used': len(best_examples)
        }
    
    def _select_best_examples(
        self, 
        examples: List[Dict], 
        n: int = 5
    ) -> List[Dict]:
        """Select top N examples by similarity score"""
        # Sort by similarity score (descending)
        sorted_examples = sorted(
            examples,
            key=lambda x: x.get('similarity_score', 0),
            reverse=True
        )
        
        # Take top N
        return sorted_examples[:n]
    
    def _build_few_shot_prompt(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict]
    ) -> str:
        """Build few-shot prompt with schema and examples"""
        
        prompt = "# SQL Generation Task\n\n"
        
        # Add schema with highlighted relevant columns
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
        prompt += "## Examples\n\n"
        for i, example in enumerate(examples, 1):
            prompt += f"### Example {i}\n"
            prompt += f"Question: {example.get('question', 'N/A')}\n"
            prompt += f"```sql\n{example.get('query', 'N/A')}\n```\n\n"
        
        # Add target question
        prompt += "## Your Task\n\n"
        prompt += f"Question: {question}\n\n"
        
        # Add schema links hint
        prompt += "Relevant Schema Elements:\n"
        prompt += f"- Tables: {', '.join(sorted(schema_links['tables']))}\n"
        if schema_links.get('columns'):
            for table, cols in sorted(schema_links['columns'].items()):
                if cols:
                    prompt += f"- {table} columns: {', '.join(sorted(cols))}\n"
        
        prompt += "\nGenerate the SQL query (provide ONLY the SQL query, no explanations):\n"
        
        return prompt
    
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
                
                # Highlight relevant columns
                if col_name in relevant_cols:
                    schema_str += f"  ⭐ {col_name} ({data_type})\n"
                else:
                    schema_str += f"  • {col_name} ({data_type})\n"
            
            schema_str += "\n"
        
        return schema_str
    
    def _generate_with_llm(self, prompt: str) -> str:
        """Generate SQL using LLM"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert SQL query generator. Generate syntactically correct SQL queries based on the schema and examples provided. Output ONLY the SQL query without any explanations or markdown.'
                    },
                    {'role': 'user', 'content': prompt}
                ],
                options={
                    'temperature': 0.2  # Low temperature for consistent SQL
                }
            )
            
            sql = response['message']['content'].strip()
            sql = self._clean_sql(sql)
            
            return sql
            
        except Exception as e:
            return f"-- Error generating SQL: {str(e)}"
    
    def _clean_sql(self, sql: str) -> str:
        """Clean SQL output from LLM"""
        # Remove markdown code blocks
        sql = re.sub(r'```sql\s*', '', sql, flags=re.IGNORECASE)
        sql = re.sub(r'```\s*', '', sql)
        
        # Remove explanatory text
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False
        
        for line in lines:
            line_upper = line.strip().upper()
            
            # Start of SQL
            if any(line_upper.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                in_sql = True
            
            if in_sql:
                sql_lines.append(line)
                
                # End of SQL
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
        examples: List[Dict]
    ) -> float:
        """Calculate confidence score based on similarity to examples"""
        if not examples:
            return 0.5
        
        # Use average similarity score of examples
        avg_similarity = sum(ex.get('similarity_score', 0) for ex in examples) / len(examples)
        
        # Adjust based on SQL validity
        sql_upper = generated_sql.upper()
        
        # Check for basic SQL structure
        has_select = 'SELECT' in sql_upper
        has_from = 'FROM' in sql_upper
        has_semicolon = generated_sql.strip().endswith(';')
        
        validity_score = 0.0
        if has_select:
            validity_score += 0.4
        if has_from:
            validity_score += 0.3
        if has_semicolon:
            validity_score += 0.1
        
        # No obvious errors
        if '-- Error' not in generated_sql:
            validity_score += 0.2
        
        # Combine scores
        confidence = (avg_similarity * 0.6) + (validity_score * 0.4)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        question: str,
        generated_sql: str,
        examples: List[Dict],
        confidence: float
    ) -> str:
        """Generate reasoning for Step 6a"""
        reasoning = "STEP 6a: SIMPLE FEW-SHOT GENERATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += f"Examples Used: {len(examples)}\n"
        for i, ex in enumerate(examples, 1):
            score = ex.get('similarity_score', 0)
            reasoning += f"  {i}. Similarity: {score:.4f} - {ex.get('question', 'N/A')[:50]}...\n"
        
        reasoning += f"\nGenerated SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Confidence Score: {confidence:.2f}\n\n"
        
        reasoning += "Confidence Factors:\n"
        if confidence >= 0.8:
            reasoning += "  ✓ High confidence - Similar examples with high similarity scores\n"
        elif confidence >= 0.6:
            reasoning += "  ⚠ Medium confidence - Moderate similarity to examples\n"
        else:
            reasoning += "  ⚠ Low confidence - Limited similarity to examples\n"
        
        return reasoning