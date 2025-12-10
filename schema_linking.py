"""
Enhanced Schema Linking Module - STEP 1 ONLY
Identifies relevant tables, columns, and foreign keys for a given question
"""
import ollama
import re
from typing import Dict, List, Set, Tuple, Optional


class EnhancedSchemaLinker:
    def __init__(self, model: str = "llama3.2"):
        self.model = model
    
    def link_schema(
        self, 
        question: str, 
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> Dict:
        """
        STEP 1: Enhanced Schema Linking
        
        Args:
            question: Natural language question
            schema_dict: Full database schema {table_name: [columns]}
            foreign_keys: List of FK relationships
            
        Returns:
            {
                'pruned_schema': Dict[str, List[Dict]],
                'schema_links': {
                    'tables': Set[str],
                    'columns': Dict[str, Set[str]],
                    'foreign_keys': List[Dict],
                    'join_paths': List[List[str]]
                },
                'reasoning': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 1: ENHANCED SCHEMA LINKING")
        print(f"{'='*60}\n")
        
        # Extract entities and keywords from question
        print("1.1: Extracting entities and keywords...")
        entities = self._extract_entities(question)
        print(f"   Detected operations: {', '.join(entities['operations'])}")
        
        # Create schema summary for LLM
        print("1.2: Creating schema summary...")
        schema_summary = self._create_schema_summary(schema_dict, foreign_keys)
        
        # Use LLM with chain-of-thought reasoning
        print("1.3: Running LLM chain-of-thought analysis...")
        llm_analysis = self._analyze_with_llm(question, schema_summary, entities)
        
        # Parse LLM analysis
        print("1.4: Parsing LLM analysis...")
        relevant_elements = self._parse_llm_analysis(llm_analysis, schema_dict)
        print(f"   Identified {len(relevant_elements['tables'])} relevant tables")
        
        # Identify critical foreign keys
        print("1.5: Identifying critical foreign keys...")
        critical_fks = self._identify_critical_foreign_keys(
            relevant_elements['tables'], foreign_keys
        )
        print(f"   Found {len(critical_fks)} critical foreign keys")
        
        # Find JOIN paths
        print("1.6: Finding JOIN paths...")
        join_paths = self._find_join_paths(relevant_elements['tables'], critical_fks)
        print(f"   Identified {len(join_paths)} possible join paths")
        
        # Create pruned schema
        print("1.7: Creating pruned schema...")
        pruned_schema = self._prune_schema(
            schema_dict, 
            relevant_elements['tables'],
            relevant_elements['columns']
        )
        total_cols = sum(len(cols) for cols in pruned_schema.values())
        print(f"   Pruned to {len(pruned_schema)} tables, {total_cols} columns\n")
        
        # Build schema links
        schema_links = {
            'tables': relevant_elements['tables'],
            'columns': relevant_elements['columns'],
            'foreign_keys': critical_fks,
            'join_paths': join_paths,
            'entity_values': relevant_elements.get('entity_values', {})
        }
        
        # Generate reasoning (cleaner format)
        reasoning = self._generate_reasoning(
            question, entities, relevant_elements, critical_fks, 
            pruned_schema, llm_analysis
        )
        
        print(f"{'='*60}")
        print("STEP 1 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'pruned_schema': pruned_schema,
            'schema_links': schema_links,
            'reasoning': reasoning
        }
    
    def _extract_entities(self, question: str) -> Dict:
        """Extract entities and keywords from question"""
        question_lower = question.lower()
        
        operation_keywords = {
            'count': ['how many', 'count', 'number of', 'total'],
            'average': ['average', 'avg', 'mean'],
            'sum': ['sum', 'total'],
            'max': ['maximum', 'max', 'highest', 'largest', 'most', 'top'],
            'min': ['minimum', 'min', 'lowest', 'smallest', 'least'],
            'select': ['show', 'list', 'display', 'get', 'find', 'what', 'which'],
            'filter': ['where', 'with', 'having', 'that have', 'whose'],
            'join': ['and', 'with', 'in', 'from', 'for', 'of'],
            'group': ['each', 'every', 'per', 'by'],
            'order': ['order', 'sort', 'rank']
        }
        
        detected_operations = []
        for op, keywords in operation_keywords.items():
            if any(kw in question_lower for kw in keywords):
                detected_operations.append(op)
        
        words = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', question)
        
        return {
            'operations': detected_operations,
            'keywords': words,
            'question_lower': question_lower
        }
    
    def _create_schema_summary(
        self, 
        schema_dict: Dict[str, List[Dict]], 
        foreign_keys: List[Dict]
    ) -> str:
        """Create schema summary for LLM"""
        summary = "DATABASE SCHEMA:\n\n"
        
        for table_name, columns in schema_dict.items():
            summary += f"Table: {table_name}\n"
            col_names = [col['column_name'] for col in columns]
            summary += f"  Columns: {', '.join(col_names)}\n\n"
        
        if foreign_keys:
            summary += "FOREIGN KEY RELATIONSHIPS:\n"
            for fk in foreign_keys:
                summary += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        
        return summary
    
    def _analyze_with_llm(
        self, 
        question: str, 
        schema_summary: str, 
        entities: Dict
    ) -> str:
        """Use LLM with chain-of-thought to identify relevant schema elements"""
        
        prompt = f"""{schema_summary}

QUESTION: {question}

Analyze this question to identify ONLY the relevant schema elements needed.

STEP 1: What is being asked?
- Identify the main goal (count, list values, aggregation, comparison)
- What entities/concepts are mentioned?

STEP 2: Identify relevant tables
- Which tables contain the needed data?
- List exact table names from schema

STEP 3: Identify relevant columns
- Which columns are needed for SELECT, WHERE, aggregations, or JOINs?
- Format: table1: col1, col2; table2: col3, col4

STEP 4: Identify required foreign keys (if multiple tables)
- Which foreign keys connect the tables?
- Format: table1.col → table2.col

Provide concise analysis:"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a database schema analyzer. Identify ONLY the minimum required schema elements. Be precise and concise.'
                    },
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _parse_llm_analysis(
        self, 
        llm_analysis: str, 
        schema_dict: Dict[str, List[Dict]]
    ) -> Dict:
        """Parse LLM's analysis"""
        relevant_tables = set()
        relevant_columns = {}
        entity_values = {}
        
        # Parse tables
        table_pattern = r'Tables?:\s*([^\n]+)'
        table_matches = re.findall(table_pattern, llm_analysis, re.IGNORECASE)
        
        for match in table_matches:
            tables = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', match)
            for table in tables:
                if table in schema_dict:
                    relevant_tables.add(table)
        
        # Parse columns
        column_pattern = r'([A-Za-z_][A-Za-z0-9_]*):\s*([^\n;]+)'
        column_matches = re.findall(column_pattern, llm_analysis)
        
        for table, cols_str in column_matches:
            if table in schema_dict:
                cols = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', cols_str)
                table_col_names = [c['column_name'] for c in schema_dict[table]]
                
                if table not in relevant_columns:
                    relevant_columns[table] = set()
                
                for col in cols:
                    if col in table_col_names:
                        relevant_columns[table].add(col)
        
        # Fallback
        if not relevant_tables:
            for table_name in schema_dict.keys():
                if table_name.lower() in llm_analysis.lower():
                    relevant_tables.add(table_name)
        
        # Ensure all tables have columns
        for table in relevant_tables:
            if table not in relevant_columns:
                relevant_columns[table] = set(
                    col['column_name'] for col in schema_dict[table]
                )
        
        return {
            'tables': relevant_tables,
            'columns': relevant_columns,
            'entity_values': entity_values
        }
    
    def _identify_critical_foreign_keys(
        self, 
        relevant_tables: Set[str], 
        foreign_keys: List[Dict]
    ) -> List[Dict]:
        """Identify foreign keys needed for joining"""
        critical_fks = []
        
        for fk in foreign_keys:
            if fk['from_table'] in relevant_tables and fk['to_table'] in relevant_tables:
                critical_fks.append(fk)
        
        return critical_fks
    
    def _find_join_paths(
        self, 
        relevant_tables: Set[str], 
        critical_fks: List[Dict]
    ) -> List[List[str]]:
        """Find JOIN paths between tables"""
        if len(relevant_tables) <= 1:
            return []
        
        graph = {table: [] for table in relevant_tables}
        
        for fk in critical_fks:
            from_table = fk['from_table']
            to_table = fk['to_table']
            if from_table in graph:
                graph[from_table].append(to_table)
            if to_table in graph:
                graph[to_table].append(from_table)
        
        paths = []
        tables_list = list(relevant_tables)
        
        for i in range(len(tables_list)):
            for j in range(i + 1, len(tables_list)):
                path = self._bfs_path(graph, tables_list[i], tables_list[j])
                if path:
                    paths.append(path)
        
        return paths
    
    def _bfs_path(
        self, 
        graph: Dict[str, List[str]], 
        start: str, 
        end: str
    ) -> Optional[List[str]]:
        """BFS to find shortest path"""
        if start == end:
            return [start]
        
        visited = set()
        queue = [(start, [start])]
        
        while queue:
            node, path = queue.pop(0)
            
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor == end:
                    return path + [neighbor]
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def _prune_schema(
        self,
        schema_dict: Dict[str, List[Dict]],
        relevant_tables: Set[str],
        relevant_columns: Dict[str, Set[str]]
    ) -> Dict[str, List[Dict]]:
        """Create pruned schema"""
        pruned = {}
        
        for table_name in relevant_tables:
            if table_name not in schema_dict:
                continue
            
            cols_to_include = relevant_columns.get(
                table_name, 
                set(col['column_name'] for col in schema_dict[table_name])
            )
            
            pruned_columns = [
                col for col in schema_dict[table_name]
                if col['column_name'] in cols_to_include
            ]
            
            pruned[table_name] = pruned_columns
        
        return pruned
    
    def _generate_reasoning(
        self,
        question: str,
        entities: Dict,
        relevant_elements: Dict,
        critical_fks: List[Dict],
        pruned_schema: Dict,
        llm_analysis: str
    ) -> str:
        """Generate concise reasoning"""
        reasoning = "STEP 1: SCHEMA LINKING\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += "Operations: " + ", ".join(entities['operations']) + "\n\n"
        
        reasoning += f"Tables ({len(relevant_elements['tables'])}):\n"
        for table in sorted(relevant_elements['tables']):
            reasoning += f"  • {table}\n"
        
        reasoning += f"\nColumns ({sum(len(c) for c in relevant_elements['columns'].values())}):\n"
        for table, cols in sorted(relevant_elements['columns'].items()):
            reasoning += f"  • {table}: {', '.join(sorted(cols))}\n"
        
        if critical_fks:
            reasoning += f"\nForeign Keys ({len(critical_fks)}):\n"
            for fk in critical_fks:
                reasoning += f"  • {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        
        reasoning += "\n" + "-" * 50 + "\n"
        reasoning += "LLM Analysis:\n"
        reasoning += llm_analysis + "\n"
        
        return reasoning