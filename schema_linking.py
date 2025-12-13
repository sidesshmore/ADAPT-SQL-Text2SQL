"""
Enhanced Schema Linking Module - STEP 1 with Three-Layer Approach
Layer 1: String Matching Pre-filter
Layer 2: LLM Analysis with hints
Layer 3: Post-Validation

This replaces schema_linking.py
"""
import ollama
import re
from typing import Dict, List, Set, Tuple, Optional
from difflib import SequenceMatcher
from collections import defaultdict


class EnhancedSchemaLinker:
    def __init__(self, model: str = "qwen3-coder"):
        self.model = model
        # Thresholds for fuzzy matching
        self.table_match_threshold = 0.6
        self.column_match_threshold = 0.5
    
    def link_schema(
        self, 
        question: str, 
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> Dict:
        """
        STEP 1: Enhanced Schema Linking with Three-Layer Approach
        
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
                'reasoning': str,
                'layer_details': Dict  # Diagnostic info from each layer
            }
        """
        print(f"\n{'='*60}")
        print("STEP 1: ENHANCED SCHEMA LINKING (THREE-LAYER APPROACH)")
        print(f"{'='*60}\n")
        
        layer_details = {}
        
        # =================================================================
        # LAYER 1: STRING MATCHING PRE-FILTER
        # =================================================================
        print("LAYER 1: String Matching Pre-filter")
        print("-" * 60)
        
        layer1_candidates = self._layer1_string_matching(
            question, schema_dict, foreign_keys
        )
        layer_details['layer1'] = layer1_candidates
        
        print(f"✓ Pre-filtered to {len(layer1_candidates['tables'])} candidate tables")
        print(f"✓ Pre-filtered to {sum(len(cols) for cols in layer1_candidates['columns'].values())} candidate columns\n")
        
        # =================================================================
        # LAYER 2: LLM ANALYSIS WITH HINTS
        # =================================================================
        print("LAYER 2: LLM Analysis with Pre-filter Hints")
        print("-" * 60)
        
        # Extract basic entities for context
        entities = self._extract_entities(question)
        
        # Create schema summary with hints from Layer 1
        schema_summary = self._create_schema_summary_with_hints(
            schema_dict, foreign_keys, layer1_candidates
        )
        
        # Run LLM analysis with pre-filter hints
        llm_analysis = self._layer2_llm_analysis(
            question, schema_summary, entities, layer1_candidates
        )
        
        # Parse LLM results
        llm_elements = self._parse_llm_analysis(llm_analysis, schema_dict)
        layer_details['layer2'] = {
            'llm_analysis': llm_analysis,
            'parsed_elements': llm_elements
        }
        
        print(f"✓ LLM identified {len(llm_elements['tables'])} tables")
        print(f"✓ LLM identified {sum(len(cols) for cols in llm_elements['columns'].values())} columns\n")
        
        # =================================================================
        # LAYER 3: POST-VALIDATION
        # =================================================================
        print("LAYER 3: Post-Validation")
        print("-" * 60)
        
        validated_elements = self._layer3_post_validation(
            llm_elements, layer1_candidates, schema_dict, foreign_keys, question
        )
        layer_details['layer3'] = validated_elements
        
        print(f"✓ Final validated: {len(validated_elements['tables'])} tables")
        print(f"✓ Final validated: {sum(len(cols) for cols in validated_elements['columns'].values())} columns\n")
        
        # =================================================================
        # FINALIZATION
        # =================================================================
        
        # Identify critical foreign keys
        print("Identifying critical foreign keys...")
        critical_fks = self._identify_critical_foreign_keys(
            validated_elements['tables'], foreign_keys
        )
        print(f"✓ Found {len(critical_fks)} critical foreign keys\n")
        
        # Find JOIN paths
        print("Finding JOIN paths...")
        join_paths = self._find_join_paths(validated_elements['tables'], critical_fks)
        print(f"✓ Identified {len(join_paths)} possible join paths\n")
        
        # Create pruned schema
        print("Creating pruned schema...")
        pruned_schema = self._prune_schema(
            schema_dict, 
            validated_elements['tables'],
            validated_elements['columns']
        )
        total_cols = sum(len(cols) for cols in pruned_schema.values())
        print(f"✓ Pruned to {len(pruned_schema)} tables, {total_cols} columns\n")
        
        # Build schema links
        schema_links = {
            'tables': validated_elements['tables'],
            'columns': validated_elements['columns'],
            'foreign_keys': critical_fks,
            'join_paths': join_paths,
            'entity_values': validated_elements.get('entity_values', {})
        }
        
        # Generate comprehensive reasoning
        reasoning = self._generate_reasoning(
            question, entities, layer_details, validated_elements, 
            critical_fks, pruned_schema
        )
        
        print(f"{'='*60}")
        print("STEP 1 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'pruned_schema': pruned_schema,
            'schema_links': schema_links,
            'reasoning': reasoning,
            'layer_details': layer_details
        }
    
    # =====================================================================
    # LAYER 1: STRING MATCHING PRE-FILTER
    # =====================================================================
    
    def _layer1_string_matching(
        self,
        question: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> Dict:
        """
        Layer 1: Use fuzzy string matching to pre-filter candidate tables and columns
        """
        question_lower = question.lower()
        question_tokens = self._tokenize(question_lower)
        
        candidate_tables = set()
        candidate_columns = defaultdict(set)
        match_details = {
            'table_matches': [],
            'column_matches': []
        }
        
        # Match tables
        for table_name in schema_dict.keys():
            table_tokens = self._tokenize(table_name.lower())
            
            # Check for exact token matches
            exact_match = any(token in question_tokens for token in table_tokens)
            
            # Check for fuzzy matches
            fuzzy_score = self._fuzzy_match_score(table_name.lower(), question_lower)
            
            if exact_match or fuzzy_score >= self.table_match_threshold:
                candidate_tables.add(table_name)
                match_details['table_matches'].append({
                    'table': table_name,
                    'exact_match': exact_match,
                    'fuzzy_score': fuzzy_score,
                    'reason': 'exact' if exact_match else 'fuzzy'
                })
        
        # Match columns
        for table_name, columns in schema_dict.items():
            for col in columns:
                col_name = col['column_name']
                col_tokens = self._tokenize(col_name.lower())
                
                # Check for exact token matches
                exact_match = any(token in question_tokens for token in col_tokens)
                
                # Check for fuzzy matches
                fuzzy_score = self._fuzzy_match_score(col_name.lower(), question_lower)
                
                # Check for semantic indicators (e.g., "name" for person_name)
                semantic_match = self._check_semantic_match(col_name.lower(), question_tokens)
                
                if exact_match or fuzzy_score >= self.column_match_threshold or semantic_match:
                    # Add the table containing this column
                    candidate_tables.add(table_name)
                    candidate_columns[table_name].add(col_name)
                    
                    match_details['column_matches'].append({
                        'table': table_name,
                        'column': col_name,
                        'exact_match': exact_match,
                        'fuzzy_score': fuzzy_score,
                        'semantic_match': semantic_match,
                        'reason': 'exact' if exact_match else ('semantic' if semantic_match else 'fuzzy')
                    })
        
        # Add connected tables via foreign keys
        connected_tables = self._find_connected_tables(candidate_tables, foreign_keys)
        candidate_tables.update(connected_tables)
        
        # If no matches, fall back to all tables (let LLM decide)
        if not candidate_tables:
            print("   ⚠️  No matches found in Layer 1, passing all tables to LLM")
            candidate_tables = set(schema_dict.keys())
            for table_name, columns in schema_dict.items():
                candidate_columns[table_name] = {col['column_name'] for col in columns}
        
        # Ensure all candidate tables have their columns included
        for table in candidate_tables:
            if table not in candidate_columns or not candidate_columns[table]:
                # Include all columns for this table
                candidate_columns[table] = {col['column_name'] for col in schema_dict[table]}
        
        return {
            'tables': candidate_tables,
            'columns': candidate_columns,
            'match_details': match_details,
            'method': 'string_matching'
        }
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into words, handling common separators"""
        # Split on non-alphanumeric characters
        tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
        
        # Also split camelCase and snake_case
        expanded_tokens = []
        for token in tokens:
            # Split camelCase
            split_camel = re.sub('([a-z])([A-Z])', r'\1 \2', token).lower().split()
            # Split snake_case and numbers
            for part in split_camel:
                expanded_tokens.extend(part.split('_'))
        
        return set(expanded_tokens)
    
    def _fuzzy_match_score(self, text1: str, text2: str) -> float:
        """Calculate fuzzy match score between two strings"""
        # Use SequenceMatcher for fuzzy matching
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _check_semantic_match(self, column_name: str, question_tokens: Set[str]) -> bool:
        """Check for semantic matches (e.g., 'age' matches 'old', 'young')"""
        semantic_map = {
            'age': {'old', 'young', 'age', 'years'},
            'name': {'name', 'called', 'named'},
            'date': {'date', 'when', 'time', 'year', 'day'},
            'count': {'many', 'number', 'total', 'count'},
            'price': {'cost', 'price', 'expensive', 'cheap'},
            'email': {'email', 'contact'},
            'phone': {'phone', 'telephone', 'contact'},
            'address': {'address', 'location', 'where'},
            'description': {'description', 'about', 'details'}
        }
        
        for key, synonyms in semantic_map.items():
            if key in column_name:
                if any(syn in question_tokens for syn in synonyms):
                    return True
        
        return False
    
    def _find_connected_tables(
        self, 
        candidate_tables: Set[str], 
        foreign_keys: List[Dict]
    ) -> Set[str]:
        """Find tables connected to candidates via foreign keys"""
        connected = set()
        
        for fk in foreign_keys:
            if fk['from_table'] in candidate_tables:
                connected.add(fk['to_table'])
            if fk['to_table'] in candidate_tables:
                connected.add(fk['from_table'])
        
        return connected
    
    # =====================================================================
    # LAYER 2: LLM ANALYSIS WITH HINTS
    # =====================================================================
    
    def _layer2_llm_analysis(
        self,
        question: str,
        schema_summary: str,
        entities: Dict,
        layer1_candidates: Dict
    ) -> str:
        """
        Layer 2: LLM analysis with hints from Layer 1
        """
        prompt = f"""{schema_summary}

QUESTION: {question}

PRE-FILTER HINTS (from string matching):
Candidate Tables: {', '.join(sorted(layer1_candidates['tables']))}
Candidate Columns: {self._format_candidate_columns(layer1_candidates['columns'])}

Analyze this question to identify the MINIMUM required schema elements.
Use the pre-filter hints as guidance, but you can include or exclude elements as needed.

STEP 1: What is being asked?
- Identify the main goal (count, list values, aggregation, comparison)
- What entities/concepts are mentioned?

STEP 2: Identify relevant tables
- Which tables contain the needed data?
- Consider the pre-filtered candidates but validate their necessity
- List exact table names from schema

STEP 3: Identify relevant columns
- Which columns are needed for SELECT, WHERE, aggregations, or JOINs?
- Format: table1: col1, col2; table2: col3, col4
- Only include columns that are actually needed

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
                        'content': 'You are a database schema analyzer. Use pre-filter hints as guidance, but critically evaluate each element. Identify ONLY the minimum required schema elements.'
                    },
                    {'role': 'user', 'content': prompt}
                ]
            )
            
            return response['message']['content']
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _format_candidate_columns(self, columns_dict: Dict[str, Set[str]]) -> str:
        """Format candidate columns for display"""
        if not columns_dict:
            return "None"
        
        parts = []
        for table, cols in sorted(columns_dict.items()):
            if cols:
                parts.append(f"{table}: {', '.join(sorted(cols))}")
        
        return '; '.join(parts) if parts else "None"
    
    # =====================================================================
    # LAYER 3: POST-VALIDATION
    # =====================================================================
    
    def _layer3_post_validation(
        self,
        llm_elements: Dict,
        layer1_candidates: Dict,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        question: str
    ) -> Dict:
        """
        Layer 3: Post-validate LLM results
        """
        validated_tables = set()
        validated_columns = defaultdict(set)
        validation_log = []
        
        # Validate tables
        for table in llm_elements['tables']:
            if table in schema_dict:
                validated_tables.add(table)
                validation_log.append({
                    'type': 'table',
                    'element': table,
                    'status': 'valid',
                    'in_layer1': table in layer1_candidates['tables']
                })
            else:
                # Try to find closest match
                closest = self._find_closest_match(table, schema_dict.keys())
                if closest:
                    validated_tables.add(closest)
                    validation_log.append({
                        'type': 'table',
                        'element': table,
                        'status': 'corrected',
                        'corrected_to': closest,
                        'in_layer1': closest in layer1_candidates['tables']
                    })
        
        # Validate columns
        for table, cols in llm_elements['columns'].items():
            if table not in validated_tables:
                continue
            
            if table not in schema_dict:
                continue
            
            schema_col_names = {col['column_name'] for col in schema_dict[table]}
            
            for col in cols:
                if col in schema_col_names:
                    validated_columns[table].add(col)
                    validation_log.append({
                        'type': 'column',
                        'element': f"{table}.{col}",
                        'status': 'valid',
                        'in_layer1': col in layer1_candidates['columns'].get(table, set())
                    })
                else:
                    # Try to find closest match
                    closest = self._find_closest_match(col, schema_col_names)
                    if closest:
                        validated_columns[table].add(closest)
                        validation_log.append({
                            'type': 'column',
                            'element': f"{table}.{col}",
                            'status': 'corrected',
                            'corrected_to': closest,
                            'in_layer1': closest in layer1_candidates['columns'].get(table, set())
                        })
        
        # Ensure connectivity via foreign keys
        validated_tables, validated_columns = self._ensure_connectivity(
            validated_tables, validated_columns, foreign_keys, schema_dict
        )
        
        # Ensure minimum columns per table
        for table in validated_tables:
            if table not in validated_columns or not validated_columns[table]:
                # Include primary key columns at minimum
                pk_cols = self._get_key_columns(table, schema_dict[table])
                validated_columns[table].update(pk_cols)
        
        return {
            'tables': validated_tables,
            'columns': validated_columns,
            'validation_log': validation_log,
            'method': 'post_validation'
        }
    
    def _find_closest_match(
        self, 
        target: str, 
        candidates: Set[str],
        threshold: float = 0.7
    ) -> Optional[str]:
        """Find closest matching name from candidates"""
        target_lower = target.lower()
        
        best_match = None
        best_score = 0.0
        
        for candidate in candidates:
            score = SequenceMatcher(None, target_lower, candidate.lower()).ratio()
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = candidate
        
        return best_match
    
    def _ensure_connectivity(
        self,
        tables: Set[str],
        columns: Dict[str, Set[str]],
        foreign_keys: List[Dict],
        schema_dict: Dict[str, List[Dict]]
    ) -> Tuple[Set[str], Dict[str, Set[str]]]:
        """Ensure selected tables can be connected via foreign keys"""
        if len(tables) <= 1:
            return tables, columns
        
        # Build connectivity graph - FIXED: store only table names
        graph = defaultdict(list)
        for fk in foreign_keys:
            graph[fk['from_table']].append(fk['to_table'])
            graph[fk['to_table']].append(fk['from_table'])
        
        # Check if all tables are connected
        if not self._are_tables_connected(tables, graph):
            # Find minimum spanning set that connects all tables
            connected_tables = self._find_connecting_tables(tables, graph)
            
            # Add connecting tables
            for table in connected_tables:
                if table not in tables:
                    tables.add(table)
                    # Add key columns for connecting table
                    key_cols = self._get_key_columns(table, schema_dict[table])
                    columns[table].update(key_cols)
        
        return tables, columns
    
    def _are_tables_connected(self, tables: Set[str], graph: Dict) -> bool:
        """Check if all tables are connected via foreign keys"""
        if len(tables) <= 1:
            return True
        
        # BFS to check connectivity - FIXED: no tuple unpacking
        visited = set()
        queue = [next(iter(tables))]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in graph.get(current, []):
                if neighbor in tables and neighbor not in visited:
                    queue.append(neighbor)
        
        return len(visited) == len(tables)
    
    def _find_connecting_tables(self, tables: Set[str], graph: Dict) -> Set[str]:
        """Find minimum set of tables to connect all target tables"""
        connecting = set()
        
        # Simple approach: add all tables on paths between target tables
        tables_list = list(tables)
        for i in range(len(tables_list)):
            for j in range(i + 1, len(tables_list)):
                path = self._bfs_path(graph, tables_list[i], tables_list[j])
                if path:
                    connecting.update(path)
        
        return connecting
    
    def _get_key_columns(
        self, 
        table: str, 
        columns: List[Dict]
    ) -> Set[str]:
        """Get key columns (id, primary key, etc.)"""
        key_cols = set()
        
        for col in columns:
            col_name = col['column_name'].lower()
            # Common patterns for key columns
            if 'id' in col_name or col_name == 'name' or 'key' in col_name:
                key_cols.add(col['column_name'])
        
        # If no key columns found, add first column
        if not key_cols and columns:
            key_cols.add(columns[0]['column_name'])
        
        return key_cols
    
    # =====================================================================
    # HELPER METHODS
    # =====================================================================
    
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
    
    def _create_schema_summary_with_hints(
        self, 
        schema_dict: Dict[str, List[Dict]], 
        foreign_keys: List[Dict],
        layer1_candidates: Dict
    ) -> str:
        """Create schema summary highlighting pre-filtered candidates"""
        summary = "DATABASE SCHEMA:\n\n"
        
        for table_name, columns in schema_dict.items():
            is_candidate = table_name in layer1_candidates['tables']
            marker = "⭐ " if is_candidate else "   "
            
            summary += f"{marker}Table: {table_name}\n"
            col_names = [col['column_name'] for col in columns]
            summary += f"  Columns: {', '.join(col_names)}\n\n"
        
        if foreign_keys:
            summary += "FOREIGN KEY RELATIONSHIPS:\n"
            for fk in foreign_keys:
                summary += f"  {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        
        summary += "\n⭐ = Pre-filtered candidate from string matching\n"
        
        return summary
    
    def _parse_llm_analysis(
        self, 
        llm_analysis: str, 
        schema_dict: Dict[str, List[Dict]]
    ) -> Dict:
        """Parse LLM's analysis (same as before)"""
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
        layer_details: Dict,
        validated_elements: Dict,
        critical_fks: List[Dict],
        pruned_schema: Dict
    ) -> str:
        """Generate comprehensive reasoning showing all three layers"""
        reasoning = "STEP 1: ENHANCED SCHEMA LINKING (THREE-LAYER APPROACH)\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Question: {question}\n\n"
        
        reasoning += "Operations: " + ", ".join(entities['operations']) + "\n\n"
        
        # Layer 1 Results
        reasoning += "LAYER 1: String Matching Pre-filter\n"
        reasoning += "-" * 50 + "\n"
        
        layer1 = layer_details['layer1']
        reasoning += f"Pre-filtered Tables ({len(layer1['tables'])}):\n"
        for table in sorted(layer1['tables']):
            reasoning += f"  • {table}\n"
        
        reasoning += f"\nPre-filtered Columns ({sum(len(c) for c in layer1['columns'].values())}):\n"
        for table, cols in sorted(layer1['columns'].items()):
            if cols:
                reasoning += f"  • {table}: {', '.join(sorted(cols))}\n"
        
        reasoning += "\n"
        
        # Layer 2 Results
        reasoning += "LAYER 2: LLM Analysis with Hints\n"
        reasoning += "-" * 50 + "\n"
        
        layer2 = layer_details['layer2']
        llm_elements = layer2['parsed_elements']
        
        reasoning += f"LLM Identified Tables ({len(llm_elements['tables'])}):\n"
        for table in sorted(llm_elements['tables']):
            in_layer1 = "✓" if table in layer1['tables'] else "+"
            reasoning += f"  {in_layer1} {table}\n"
        
        reasoning += "\n"
        reasoning += "✓ = Confirmed from Layer 1, + = Added by LLM\n\n"
        
        # Layer 3 Results
        reasoning += "LAYER 3: Post-Validation\n"
        reasoning += "-" * 50 + "\n"
        
        validation_log = validated_elements.get('validation_log', [])
        
        corrected = [v for v in validation_log if v['status'] == 'corrected']
        if corrected:
            reasoning += "Corrections Made:\n"
            for correction in corrected:
                reasoning += f"  • {correction['element']} → {correction['corrected_to']}\n"
            reasoning += "\n"
        
        reasoning += f"Final Validated Tables ({len(validated_elements['tables'])}):\n"
        for table in sorted(validated_elements['tables']):
            reasoning += f"  • {table}\n"
        
        reasoning += f"\nFinal Validated Columns ({sum(len(c) for c in validated_elements['columns'].values())}):\n"
        for table, cols in sorted(validated_elements['columns'].items()):
            reasoning += f"  • {table}: {', '.join(sorted(cols))}\n"
        
        if critical_fks:
            reasoning += f"\nForeign Keys ({len(critical_fks)}):\n"
            for fk in critical_fks:
                reasoning += f"  • {fk['from_table']}.{fk['from_column']} → {fk['to_table']}.{fk['to_column']}\n"
        
        reasoning += "\n" + "=" * 50 + "\n"
        reasoning += "SUMMARY\n"
        reasoning += "=" * 50 + "\n"
        reasoning += f"Layer 1 Pre-filter: {len(layer1['tables'])} tables\n"
        reasoning += f"Layer 2 LLM Analysis: {len(llm_elements['tables'])} tables\n"
        reasoning += f"Layer 3 Post-validation: {len(validated_elements['tables'])} tables\n"
        reasoning += f"Final Pruned Schema: {len(pruned_schema)} tables, {sum(len(c) for c in pruned_schema.values())} columns\n"
        
        return reasoning