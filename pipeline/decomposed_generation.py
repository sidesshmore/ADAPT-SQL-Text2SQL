"""
STEP 6c: Decomposed Generation with Subquery Handling (NESTED_COMPLEX)
Breaks down nested queries into sub-questions and composes final SQL
"""
import ollama
import re
from typing import Dict, List, Set


class DecomposedGenerator:
    def __init__(self, model: str = "qwen3-coder"):
        """Initialize decomposed generator"""
        self.model = model

    # NEW: Nested query structure templates (DIN-SQL style)
        self.nested_templates = {
            'IN_SELECT': {
                'pattern': 'SELECT {outer_cols} WHERE {col} IN (SELECT {inner_cols} WHERE {inner_cond})',
                'description': 'Check if value exists in subquery results',
                'examples': [
                    'students IN (SELECT student_id FROM enrollment)',
                    'singers IN (SELECT singer_id FROM concerts WHERE year > 2000)'
                ]
            },
            'NOT_IN_SELECT': {
                'pattern': 'SELECT {outer_cols} WHERE {col} NOT IN (SELECT {inner_cols} WHERE {inner_cond})',
                'description': 'Check if value does NOT exist in subquery results',
                'examples': [
                    'students NOT IN (SELECT student_id FROM failed_courses)'
                ]
            },
            'EXISTS_SELECT': {
                'pattern': 'SELECT {outer_cols} WHERE EXISTS (SELECT * WHERE {inner_cond})',
                'description': 'Check if subquery returns any results',
                'examples': [
                    'EXISTS (SELECT * FROM has_pet WHERE student.id = has_pet.student_id)'
                ]
            },
            'COMPARISON_WITH_AGG': {
                'pattern': 'SELECT {outer_cols} WHERE {col} {op} (SELECT {agg}({inner_col}) WHERE {inner_cond})',
                'description': 'Compare with aggregated subquery result',
                'examples': [
                    'age > (SELECT AVG(age) FROM students)',
                    'salary >= (SELECT MAX(salary) FROM employees WHERE dept = "IT")'
                ]
            },
            'EXCEPT': {
                'pattern': 'SELECT {cols} WHERE {cond1} EXCEPT SELECT {cols} WHERE {cond2}',
                'description': 'Set difference between two queries',
                'examples': [
                    'all students EXCEPT students who failed'
                ]
            }
        }

    # Method to identify nested pattern:

    def _identify_nested_pattern(
        self,
        question: str,
        sub_questions: List[str]
    ) -> str:
        """
        NEW: Identify which nested query pattern to use
        Returns pattern key from self.nested_templates
        """
        question_lower = question.lower()
        
        # Pattern 1: "more/less than average" → COMPARISON_WITH_AGG
        if any(phrase in question_lower for phrase in [
            'more than average', 'less than average', 
            'greater than average', 'higher than average',
            'above average', 'below average'
        ]):
            return 'COMPARISON_WITH_AGG'
        
        # Pattern 2: "except", "but not", "exclude" → EXCEPT or NOT_IN_SELECT
        if any(phrase in question_lower for phrase in ['except', 'but not', 'exclude', 'not in']):
            return 'NOT_IN_SELECT'
        
        # Pattern 3: "that have", "who have", "with" → EXISTS or IN_SELECT
        if any(phrase in question_lower for phrase in ['that have', 'who have', 'that had']):
            return 'IN_SELECT'
        
        # Pattern 4: "does not have", "without" → NOT_IN_SELECT or NOT EXISTS
        if any(phrase in question_lower for phrase in ['does not have', 'without', 'no']):
            return 'NOT_IN_SELECT'
        
        # Default: IN_SELECT (most common)
        return 'IN_SELECT'

    
    def generate_sql_decomposed(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        sub_questions: List[str],
        selected_examples: List[Dict],
        few_shot_generator=None,
        intermediate_generator=None
    ) -> Dict:
        """
        ENHANCED: Decomposed Generation with Subquery Handling
        Now uses structure templates for better EM alignment
        """
        print(f"\n{'='*60}")
        print("STEP 6c: ENHANCED DECOMPOSED GENERATION")
        print(f"{'='*60}\n")
        
        print(f"Main Question: {question}")
        print(f"Sub-questions: {len(sub_questions)}")
        print(f"Available examples: {len(selected_examples)}")
        
        # Select best examples
        print("6c.0: Selecting best examples...")
        best_examples = self._select_best_examples(selected_examples, n=5)
        print(f"   Using {len(best_examples)} examples")
        
        # NEW: Identify nested pattern
        pattern_key = self._identify_nested_pattern(question, sub_questions)
        print(f"\n6c.0.5: Identified nested pattern: {pattern_key}")
        
        # Sub-step 6c.1: Generate Sub-SQLs
        print("\n6c.1: Generating sub-SQLs for each sub-question...")
        sub_sql_list = self._generate_sub_sqls(
            sub_questions,
            pruned_schema,
            schema_links,
            best_examples,
            few_shot_generator,
            intermediate_generator
        )
        print(f"   Generated {len(sub_sql_list)} sub-SQLs")
        
        # Sub-step 6c.2: Generate NatSQL with Sub-queries (using template)
        print("\n6c.2: Generating NatSQL with sub-queries (template-based)...")
        natsql_intermediate = self._generate_natsql_with_subqueries(
            question,
            sub_questions,
            sub_sql_list,
            pruned_schema,
            schema_links,
            best_examples
        )
        print(f"   NatSQL generated: {len(natsql_intermediate)} characters")
        
        # Sub-step 6c.3: Convert to Final SQL
        print("\n6c.3: Converting to final SQL...")
        generated_sql = self._natsql_to_sql(
            natsql_intermediate,
            pruned_schema,
            schema_links,
            sub_sql_list,
            best_examples
        )
        print(f"   SQL generated: {len(generated_sql)} characters")
        
        # NEW: Validate structure
        is_valid_structure = self._validate_nested_structure(generated_sql, pattern_key)
        if not is_valid_structure:
            print(f"   ⚠️ Warning: Generated SQL may not follow {pattern_key} pattern")
        else:
            print(f"   ✓ SQL follows {pattern_key} pattern")
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            generated_sql,
            sub_sql_list,
            natsql_intermediate,
            best_examples
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question,
            sub_questions,
            sub_sql_list,
            natsql_intermediate,
            generated_sql,
            best_examples,
            confidence,
            pattern_key  # NEW
        )
        
        print(f"\nConfidence: {confidence:.2f}")
        
        print(f"\n{'='*60}")
        print("STEP 6c COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'generated_sql': generated_sql,
            'sub_sql_list': sub_sql_list,
            'natsql_intermediate': natsql_intermediate,
            'nested_pattern': pattern_key,  # NEW
            'pattern_valid': is_valid_structure,  # NEW
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
    
    def _generate_sub_sqls(
        self,
        sub_questions: List[str],
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict],
        few_shot_generator,
        intermediate_generator
    ) -> List[Dict]:
        """
        Sub-step 6c.1: Generate SQL for each sub-question
        Recursively calls appropriate generators based on sub-question complexity
        """
        sub_sql_list = []
        
        for i, sub_question in enumerate(sub_questions, 1):
            print(f"   6c.1.{i}: Processing sub-question: {sub_question[:50]}...")
            
            # Classify sub-question complexity
            complexity = self._classify_sub_question(sub_question, pruned_schema)
            print(f"          Complexity: {complexity}")
            
            # Generate SQL based on complexity
            if complexity == "EASY":
                # Use simple few-shot
                if few_shot_generator:
                    result = few_shot_generator.generate_sql_easy(
                        sub_question,
                        pruned_schema,
                        schema_links,
                        examples[:3]
                    )
                    sub_sql = result['generated_sql']
                else:
                    sub_sql = self._simple_generate(sub_question, pruned_schema, schema_links)
            else:
                # Use intermediate representation
                if intermediate_generator:
                    result = intermediate_generator.generate_sql_with_intermediate(
                        sub_question,
                        pruned_schema,
                        schema_links,
                        examples[:3]
                    )
                    sub_sql = result['generated_sql']
                else:
                    sub_sql = self._simple_generate(sub_question, pruned_schema, schema_links)
            
            sub_sql_list.append({
                'sub_question': sub_question,
                'complexity': complexity,
                'sql': sub_sql
            })
            print(f"          Generated: {sub_sql[:60]}...")
        
        return sub_sql_list
    
    def _classify_sub_question(
        self,
        sub_question: str,
        pruned_schema: Dict[str, List[Dict]]
    ) -> str:
        """Classify sub-question complexity (simplified)"""
        sub_lower = sub_question.lower()
        
        # Check for aggregations or multiple tables
        has_agg = any(kw in sub_lower for kw in ['count', 'average', 'sum', 'max', 'min', 'total'])
        has_join_words = any(kw in sub_lower for kw in ['and', 'with', 'from', 'in'])
        
        # Check for multiple table references
        table_matches = 0
        for table in pruned_schema.keys():
            if table.lower() in sub_lower:
                table_matches += 1
        
        if table_matches > 1 or (has_agg and has_join_words):
            return "NON_NESTED_COMPLEX"
        else:
            return "EASY"
    
    def _simple_generate(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict
    ) -> str:
        """Simple SQL generation fallback"""
        schema_str = ""
        for table, cols in pruned_schema.items():
            col_names = [c['column_name'] for c in cols]
            schema_str += f"  {table}: {', '.join(col_names)}\n"
        
        prompt = f"""SCHEMA:
{schema_str}

QUESTION: {question}

Generate a simple SQL query to answer this question.
Output ONLY the SQL query:"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are an expert SQL generator.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.2}
            )
            return self._clean_sql(response['message']['content'].strip())
        except:
            return "SELECT * FROM table;"
    
    def _generate_natsql_with_subqueries(
        self,
        question: str,
        sub_questions: List[str],
        sub_sql_list: List[Dict],
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        examples: List[Dict]
    ) -> str:
        """
        ENHANCED: Generate NatSQL with sub-queries using structure templates
        """
        # NEW: Identify appropriate nested pattern
        pattern_key = self._identify_nested_pattern(question, sub_questions)
        template = self.nested_templates[pattern_key]
        
        print(f"   Using nested pattern: {pattern_key}")
        print(f"   Pattern: {template['description']}")
        
        prompt = "# NatSQL Generation with Nested Queries\n\n"
        
        # Add template guidance
        prompt += f"## Nested Query Pattern: {pattern_key}\n\n"
        prompt += f"Description: {template['description']}\n"
        prompt += f"Pattern: {template['pattern']}\n\n"
        prompt += "Examples:\n"
        for ex in template['examples']:
            prompt += f"  • {ex}\n"
        prompt += "\n"
        
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
        
        # Add nested query examples from training data
        prompt += "## Nested Query Examples from Ground Truth\n\n"
        nested_examples = [ex for ex in examples if 'SELECT' in ex.get('query', '').upper() 
                        and ex.get('query', '').upper().count('SELECT') > 1]
        
        for i, example in enumerate(nested_examples[:2], 1):
            prompt += f"### Example {i}\n"
            prompt += f"Question: {example.get('question', 'N/A')}\n"
            prompt += f"SQL:\n```sql\n{example.get('query', 'N/A')}\n```\n\n"
        
        # Add sub-questions and their SQLs
        prompt += "## Sub-questions and Their SQLs\n\n"
        for i, sub_sql_info in enumerate(sub_sql_list, 1):
            prompt += f"### Sub-question {i}\n"
            prompt += f"Question: {sub_sql_info['sub_question']}\n"
            prompt += f"SQL:\n```sql\n{sub_sql_info['sql']}\n```\n\n"
        
        # Add main task with pattern guidance
        prompt += "## Your Task\n\n"
        prompt += f"Main Question: {question}\n\n"
        prompt += f"Use the **{pattern_key}** pattern to combine the sub-queries.\n\n"
        
        if pattern_key == 'COMPARISON_WITH_AGG':
            prompt += "Create structure:\n"
            prompt += "1. OUTER QUERY: Select items to check\n"
            prompt += "2. INNER QUERY: Calculate aggregate (AVG, MAX, MIN, etc.)\n"
            prompt += "3. COMPARISON: Use >, <, >=, <= to compare outer with inner result\n\n"
        
        elif pattern_key == 'IN_SELECT':
            prompt += "Create structure:\n"
            prompt += "1. OUTER QUERY: Select items to return\n"
            prompt += "2. INNER QUERY: Get IDs/values that match condition\n"
            prompt += "3. WHERE CLAUSE: Use IN (subquery) to filter outer query\n\n"
        
        elif pattern_key == 'NOT_IN_SELECT':
            prompt += "Create structure:\n"
            prompt += "1. OUTER QUERY: Select all items\n"
            prompt += "2. INNER QUERY: Get IDs/values to exclude\n"
            prompt += "3. WHERE CLAUSE: Use NOT IN (subquery) to filter\n\n"
        
        elif pattern_key == 'EXISTS_SELECT':
            prompt += "Create structure:\n"
            prompt += "1. OUTER QUERY: Select items to return\n"
            prompt += "2. INNER QUERY: Check existence with WHERE EXISTS\n"
            prompt += "3. CORRELATION: Link outer and inner with table.id = subquery.id\n\n"
        
        prompt += "Format (NatSQL-style):\n"
        prompt += "OUTER: [main selection]\n"
        prompt += "INNER: [subquery]\n"
        prompt += "PATTERN: [how they connect]\n\n"
        
        prompt += "Generate the intermediate representation:\n"
        
        return self._generate_with_llm(
            prompt, 
            f"You are an expert at composing nested SQL queries using the {pattern_key} pattern."
        )
    
    # Also add method to validate nested structure:

    def _validate_nested_structure(
        self,
        generated_sql: str,
        pattern_key: str
    ) -> bool:
        """
        NEW: Validate that generated SQL follows expected nested pattern
        """
        sql_upper = generated_sql.upper()
        
        if pattern_key == 'COMPARISON_WITH_AGG':
            # Should have: SELECT ... WHERE col op (SELECT agg(...)
            has_comparison = any(op in sql_upper for op in ['> (SELECT', '< (SELECT', '>= (SELECT', '<= (SELECT'])
            has_agg = any(agg in sql_upper for agg in ['AVG(', 'MAX(', 'MIN(', 'COUNT(', 'SUM('])
            return has_comparison and has_agg
        
        elif pattern_key == 'IN_SELECT':
            # Should have: WHERE ... IN (SELECT ...)
            return 'IN (SELECT' in sql_upper or 'IN ( SELECT' in sql_upper
        
        elif pattern_key == 'NOT_IN_SELECT':
            # Should have: WHERE ... NOT IN (SELECT ...)
            return 'NOT IN (SELECT' in sql_upper or 'NOT IN ( SELECT' in sql_upper
        
        elif pattern_key == 'EXISTS_SELECT':
            # Should have: WHERE EXISTS (SELECT ...)
            return 'EXISTS (SELECT' in sql_upper or 'EXISTS ( SELECT' in sql_upper
        
        elif pattern_key == 'EXCEPT':
            # Should have: SELECT ... EXCEPT SELECT ...
            return 'EXCEPT' in sql_upper
        
        return True  # Default: assume valid


    def _natsql_to_sql(
        self,
        natsql_intermediate: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        sub_sql_list: List[Dict],
        examples: List[Dict]
    ) -> str:
        """
        Sub-step 6c.3: Convert NatSQL with sub-queries to Final SQL
        """
        prompt = "# Convert Intermediate with Sub-queries to SQL\n\n"
        
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
        
        # Add sub-SQLs
        prompt += "## Available Sub-queries\n\n"
        for i, sub_sql_info in enumerate(sub_sql_list, 1):
            prompt += f"Sub-query {i}:\n"
            prompt += f"```sql\n{sub_sql_info['sql']}\n```\n\n"
        
        # Add nested query examples
        prompt += "## Nested Query Examples\n\n"
        nested_examples = [ex for ex in examples if ex.get('query', '').upper().count('SELECT') > 1]
        
        for i, example in enumerate(nested_examples[:2], 1):
            prompt += f"### Example {i}\n"
            prompt += f"```sql\n{example.get('query', 'N/A')}\n```\n\n"
        
        # Add intermediate to convert
        prompt += "## Intermediate Representation\n\n"
        prompt += natsql_intermediate + "\n\n"
        
        prompt += "## Your Task\n\n"
        prompt += "Convert the intermediate representation into a complete SQL query.\n"
        prompt += "Integrate the sub-queries appropriately (as subqueries in WHERE, IN, comparison).\n"
        prompt += "Generate syntactically correct nested SQL.\n\n"
        prompt += "Output ONLY the SQL query:\n"
        
        sql = self._generate_with_llm(
            prompt,
            "You are an expert SQL generator specializing in nested queries."
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
        sub_sql_list: List[Dict],
        natsql_intermediate: str,
        examples: List[Dict]
    ) -> float:
        """Calculate confidence score for decomposed generation"""
        if not examples:
            return 0.5
        
        # Base confidence from example similarity
        avg_similarity = sum(ex.get('similarity_score', 0) for ex in examples) / len(examples)
        
        # Check SQL validity
        sql_upper = generated_sql.upper()
        
        validity_score = 0.0
        if 'SELECT' in sql_upper:
            validity_score += 0.2
        if 'FROM' in sql_upper:
            validity_score += 0.15
        if sql_upper.count('SELECT') > 1:  # Has subqueries
            validity_score += 0.25
        if any(op in sql_upper for op in ['IN (SELECT', 'NOT IN', 'EXISTS', '> (SELECT', '< (SELECT']):
            validity_score += 0.2
        if generated_sql.strip().endswith(';'):
            validity_score += 0.1
        if '-- Error' not in generated_sql:
            validity_score += 0.1
        
        # Check sub-query generation quality
        sub_sql_score = 0.0
        if sub_sql_list:
            valid_subs = sum(1 for s in sub_sql_list if 'SELECT' in s['sql'].upper())
            sub_sql_score = valid_subs / len(sub_sql_list)
        
        # Check intermediate quality
        intermediate_score = 0.0
        if 'MAIN QUERY:' in natsql_intermediate or 'SELECT' in natsql_intermediate:
            intermediate_score += 0.5
        if 'SUBQUERY' in natsql_intermediate.upper():
            intermediate_score += 0.5
        
        # Combine scores
        confidence = (avg_similarity * 0.3) + (validity_score * 0.35) + (sub_sql_score * 0.2) + (intermediate_score * 0.15)
        
        return min(confidence, 1.0)
    
    def _generate_reasoning(
        self,
        question: str,
        sub_questions: List[str],
        sub_sql_list: List[Dict],
        natsql_intermediate: str,
        generated_sql: str,
        examples: List[Dict],
        confidence: float,
        pattern_key: str = None  # NEW
    ) -> str:
        """Generate reasoning for Step 6c"""
        reasoning = "STEP 6c: ENHANCED DECOMPOSED GENERATION\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Main Question: {question}\n\n"
        
        # NEW: Show pattern used
        if pattern_key:
            reasoning += f"Nested Pattern Used: {pattern_key}\n"
            reasoning += f"Pattern Description: {self.nested_templates[pattern_key]['description']}\n\n"
        
        reasoning += f"Sub-questions ({len(sub_questions)}):\n"
        for i, sq in enumerate(sub_questions, 1):
            reasoning += f"  {i}. {sq}\n"
        reasoning += "\n"
        
        reasoning += f"Sub-SQLs Generated ({len(sub_sql_list)}):\n"
        for i, sub_info in enumerate(sub_sql_list, 1):
            reasoning += f"  {i}. [{sub_info['complexity']}] {sub_info['sub_question'][:50]}...\n"
            reasoning += f"     SQL: {sub_info['sql'][:70]}...\n"
        reasoning += "\n"
        
        reasoning += f"Examples Used: {len(examples)}\n"
        for i, ex in enumerate(examples[:3], 1):
            score = ex.get('similarity_score', 0)
            reasoning += f"  {i}. Similarity: {score:.4f} - {ex.get('question', 'N/A')[:50]}...\n"
        reasoning += "\n"
        
        reasoning += f"Intermediate Representation:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += natsql_intermediate + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Generated SQL:\n"
        reasoning += "-" * 50 + "\n"
        reasoning += generated_sql + "\n"
        reasoning += "-" * 50 + "\n\n"
        
        reasoning += f"Confidence Score: {confidence:.2f}\n\n"
        
        reasoning += "Confidence Factors:\n"
        if confidence >= 0.75:
            reasoning += "  ✓ High confidence - Strong sub-query decomposition and composition\n"
            reasoning += f"  ✓ Used {pattern_key} pattern for nested structure\n"
        elif confidence >= 0.55:
            reasoning += "  ⚠ Medium confidence - Acceptable decomposition quality\n"
        else:
            reasoning += "  ⚠ Low confidence - Complex nested structure may have issues\n"
        
        return reasoning