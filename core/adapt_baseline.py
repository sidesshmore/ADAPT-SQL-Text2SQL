"""
ADAPT-SQL Baseline - Complete Pipeline (Steps 1-11)
Schema Linking + Complexity + Preliminary SQL + Example Selection + Routing +
Generation + Validation + Retry + Execute + Evaluate
"""
import os
import ollama as _ollama_module
from typing import Dict, List, Optional

# Patch ollama module-level functions to read OLLAMA_HOST at call time.
# This is needed because ollama binds its default client at import time,
# so setting OLLAMA_HOST after import has no effect without this patch.
# Clients are cached per host to avoid creating a new SSL context on every call.
# A 90-second timeout prevents infinite hangs when the model stalls mid-generation.
_ollama_client_cache: dict = {}

def _get_patched_client():
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if host not in _ollama_client_cache:
        _ollama_client_cache[host] = _ollama_module.Client(host=host, timeout=90)
    return _ollama_client_cache[host]

def _voyager_chat(**kwargs):
    """Route ollama.chat() calls to ASU Voyager OpenAI-compatible API."""
    from openai import OpenAI
    api_key = os.environ.get("VOYAGER_API_KEY", "")
    base_url = os.environ.get("VOYAGER_BASE_URL", "https://openai.rc.asu.edu/v1")
    model = os.environ.get("VOYAGER_MODEL", "llama4-scout-17b")
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = kwargs.get("messages", [])
    openai_resp = client.chat.completions.create(model=model, messages=messages, timeout=90)
    content = openai_resp.choices[0].message.content or ""
    # Return an object that matches the ollama response shape all pipeline modules expect:
    # response['message']['content']
    return {"message": {"content": content, "role": "assistant"}}

def _patched_chat(**kwargs):
    if os.environ.get("VOYAGER_API_KEY"):
        return _voyager_chat(**kwargs)
    return _get_patched_client().chat(**kwargs)

def _patched_embeddings(**kwargs):
    return _get_patched_client().embeddings(**kwargs)

_ollama_module.chat = _patched_chat
_ollama_module.embeddings = _patched_embeddings

import ollama
from pipeline.schema_linking import EnhancedSchemaLinker
from pipeline.query_complexity import QueryComplexityClassifier, ComplexityClass
from pipeline.prel_sql_prediction import PreliminaryPredictor
from pipeline.vector_search import DualSimilaritySelector
from utils.vector_store import SQLVectorStore
from pipeline.routing_strategy import RoutingStrategy, GenerationStrategy
from pipeline.few_shot import FewShotGenerator
from pipeline.intermediate_repr import IntermediateRepresentationGenerator
from pipeline.decomposed_generation import DecomposedGenerator
from pipeline.validate_sql import SQLValidator
from pipeline.validation_feedback_retry import ValidationFeedbackRetry
from pipeline.execute_compare import DatabaseManager
from pipeline.evaluation import Text2SQLEvaluator
from pipeline.sql_normalizer import normalize_sql_post_generation
from utils.structural_similarity import enhance_example_selection
from pipeline.checker_chain import CheckerChain
from pipeline.python_pivot import PythonPivot
from pipeline.set_op_detector import SetOpDetector
from pipeline.candidate_selector import CandidateSelector
from utils.value_retrieval import retrieve_db_values


class ADAPTBaseline:
    def __init__(
        self,
        model: str = "qwen3-coder",
        vector_store_path: str = None,
        max_retries: int = 2,
        execution_timeout: int = 30,
        enable_sql_normalization: bool = True,
        enable_structural_reranking: bool = True,
        ollama_host: str = None
    ):
        """
        Initialize ADAPT-SQL with Ollama model and optional vector store

        Args:
            model: Ollama model name (e.g., "llama3.2", "codellama", "mistral")
            vector_store_path: Path to pre-built FAISS vector store
            max_retries: Maximum validation retry attempts (default: 2)
            execution_timeout: SQL execution timeout in seconds (default: 30)
            ollama_host: Ollama server URL (e.g., "http://127.0.0.1:11437")
        """
        host = ollama_host or os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        os.environ["OLLAMA_HOST"] = host

        self.model = model
        self.max_retries = max_retries
        self.execution_timeout = execution_timeout
        self.enable_sql_normalization = enable_sql_normalization  
        self.enable_structural_reranking = enable_structural_reranking  
        
        # Initialize all pipeline components
        self.schema_linker = EnhancedSchemaLinker(model=model)
        self.complexity_classifier = QueryComplexityClassifier(model=model)
        self.preliminary_predictor = PreliminaryPredictor(model=model)
        self.routing_strategy = RoutingStrategy(model=model)
        self.sql_validator = SQLValidator()
        self.retry_engine = ValidationFeedbackRetry(model=model, max_retries=max_retries)
        self.db_manager = DatabaseManager(timeout=execution_timeout)
        self.evaluator = Text2SQLEvaluator()
        
        # Initialize all three generation strategies
        self.few_shot_generator = FewShotGenerator(model=model)
        self.intermediate_generator = IntermediateRepresentationGenerator(model=model)
        self.decomposed_generator = DecomposedGenerator(model=model)
        self.python_pivot = PythonPivot(model=model)
        self.set_op_detector = SetOpDetector()
        self.candidate_selector = CandidateSelector(db_manager=self.db_manager)
        
        # Load vector store if path provided
        self.vector_store = None
        self.example_selector = None
        
        if vector_store_path:
            self.vector_store = SQLVectorStore()
            if self.vector_store.load_index(vector_store_path):
                self.example_selector = DualSimilaritySelector(self.vector_store)
                print("✅ Vector store loaded successfully")
            else:
                print("⚠️ Vector store loading failed")
    
    def run_step1_schema_linking(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> Dict:
        """STEP 1: Enhanced Schema Linking"""
        return self.schema_linker.link_schema(
            natural_query, 
            schema_dict, 
            foreign_keys
        )
    
    def run_step2_complexity_classification(
        self,
        natural_query: str,
        step1_result: Dict
    ) -> Dict:
        """STEP 2: Query Complexity Classification"""
        return self.complexity_classifier.classify_query(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links']
        )
    
    def run_step3_preliminary_sql(
        self,
        natural_query: str,
        step1_result: Dict
    ) -> Dict:
        """STEP 3: Preliminary SQL Prediction"""
        return self.preliminary_predictor.predict_sql_skeleton(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links']
        )
    
    def run_step4_similarity_search(
        self,
        natural_query: str,
        k: int = 10,
        preliminary_sql: str = None
    ) -> Dict:
        """STEP 4: Similarity Search with DAIL-SQL Structural Reranking"""
        if self.example_selector is None:
            return {
                'similar_examples': [],
                'reasoning': 'Vector store not loaded',
                'query': natural_query,
                'total_found': 0
            }
        
        # Get similar examples from vector store.
        # Pass preliminary_sql as sql_hint so the FAISS query embeds
        # question + SQL skeleton — symmetric with how the index was built.
        result = self.example_selector.search_similar_examples(
            natural_query,
            k=k,
            sql_hint=preliminary_sql,
        )
        
        # Apply DAIL-SQL structural + style reranking if enabled
        if self.enable_structural_reranking and preliminary_sql:
            print("   Applying DAIL-SQL structural + style reranking...")
            
            from utils.structural_similarity import enhance_example_selection
            
            # Weights: 0.45 semantic + 0.25 structural + 0.2 style + 0.1 reasoning path
            reranked_examples = enhance_example_selection(
                examples=result['similar_examples'],
                preliminary_sql=preliminary_sql,
                semantic_weight=0.45,
                structural_weight=0.25,
                style_weight=0.2,
                reasoning_path_weight=0.1
            )

            result['similar_examples'] = reranked_examples
            result['reranking_applied'] = True
            result['reranking_method'] = 'DAIL-SQL + reasoning path (semantic + structural + style + clause-set)'

            print(f"   ✓ Reranked {len(reranked_examples)} examples")
            print(f"   Weights: 45% semantic + 25% structural + 20% style + 10% reasoning path")
        else:
            result['reranking_applied'] = False
            result['reranking_method'] = 'None'
        
        return result
    
    def run_step5_routing(
        self,
        complexity_class: ComplexityClass
    ) -> Dict:
        """STEP 5: Route to Appropriate Generation Strategy"""
        return self.routing_strategy.route_to_strategy(complexity_class.value)
    
    def run_step6a_few_shot_generation(
        self,
        natural_query: str,
        step1_result: Dict,
        step4_result: Dict,
        set_op_hint: str = ''
    ) -> Dict:
        """STEP 6a: Simple Few-Shot Generation (for EASY queries)"""
        return self.few_shot_generator.generate_sql_easy(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step4_result['similar_examples'],
            set_op_hint=set_op_hint
        )

    def run_step6b_intermediate_generation(
        self,
        natural_query: str,
        step1_result: Dict,
        step4_result: Dict,
        set_op_hint: str = ''
    ) -> Dict:
        """STEP 6b: Intermediate Representation Generation (for NON_NESTED_COMPLEX)"""
        return self.intermediate_generator.generate_sql_with_intermediate(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step4_result['similar_examples'],
            set_op_hint=set_op_hint
        )
    
    def run_step6c_decomposed_generation(
        self,
        natural_query: str,
        step1_result: Dict,
        step2_result: Dict,
        step4_result: Dict,
        python_hint: str = ''
    ) -> Dict:
        """STEP 6c: Decomposed Generation with Subquery Handling (for NESTED_COMPLEX)"""
        return self.decomposed_generator.generate_sql_decomposed(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step2_result['sub_questions'],
            step4_result['similar_examples'],
            few_shot_generator=self.few_shot_generator,
            intermediate_generator=self.intermediate_generator,
            python_hint=python_hint
        )
    
    @staticmethod
    def _make_distinct_variant(sql: str) -> str:
        """Return a DISTINCT version of the SQL (for SELECT or COUNT).
        Returns empty string if the SQL already has DISTINCT or if transformation is unclear."""
        import re as _re
        sql_stripped = sql.strip().rstrip(';')
        sql_upper = sql_stripped.upper()
        if 'DISTINCT' in sql_upper:
            return ''  # already has DISTINCT
        # COUNT(col) → COUNT(DISTINCT col)  (not COUNT(*))
        count_match = _re.search(r'COUNT\s*\(\s*(?!\*)(\w+)\s*\)', sql_stripped, _re.IGNORECASE)
        if count_match:
            return _re.sub(
                r'COUNT\s*\(\s*(?!\*)(\w+)\s*\)',
                lambda m: f'COUNT(DISTINCT {m.group(1)})',
                sql_stripped, count=1, flags=_re.IGNORECASE
            ) + ';'
        # SELECT col FROM ... → SELECT DISTINCT col FROM ...
        # but skip if there's an aggregation (COUNT/SUM/AVG/MAX/MIN) — DISTINCT on agg is wrong
        has_agg = bool(_re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_upper))
        if not has_agg:
            select_match = _re.match(r'^(SELECT\s+)', sql_stripped, _re.IGNORECASE)
            if select_match:
                return select_match.group(1) + 'DISTINCT ' + sql_stripped[len(select_match.group(1)):] + ';'
        return ''

    @staticmethod
    def _result_shape_issue(sql: str, exec_result: dict) -> str:
        """Return a description of the result shape issue if the result is implausible, else ''."""
        if not exec_result or not exec_result.get('success'):
            return ''
        rows = exec_result.get('result_rows', [])
        row_count = len(rows)
        sql_upper = sql.upper()
        has_agg = any(f in sql_upper for f in ['COUNT(', 'SUM(', 'AVG(', 'MAX(', 'MIN('])
        has_group = 'GROUP BY' in sql_upper
        if has_agg and not has_group and row_count > 1:
            return f"aggregation without GROUP BY returned {row_count} rows (expected 1)"
        if 'LIMIT 1' in sql_upper and row_count > 1:
            return f"LIMIT 1 returned {row_count} rows"
        return ''

    @staticmethod
    def _is_negation_query(question: str) -> bool:
        """Return True if the question likely expects 0 rows (negation/exclusion intent)."""
        q = ' ' + question.lower() + ' '
        patterns = [
            " never ", " no ", " not ", " without ", " none ", " zero ",
            "n't ", " except ", " exclud", " neither ", " nor ",
        ]
        return any(p in q for p in patterns)

    def _repair_candidate(
        self,
        sql: str,
        directive: str,
        generation_query: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
    ) -> str:
        """ExCoT / LitE-SQL: 1 LLM call to fix a candidate using a CheckerChain directive.
        Called only for candidates that failed a deterministic checker — the directive is
        specific and actionable (e.g. 'column X not found, use Y')."""
        schema_str = '\n'.join(
            f"  {tbl}({', '.join(c['column_name'] for c in cols[:8])})"
            for tbl, cols in list(pruned_schema.items())[:6]
        )
        prompt = (
            f"Fix the SQL according to the error below. Output ONLY the corrected SQL.\n\n"
            f"Question: {generation_query}\n"
            f"Schema:\n{schema_str}\n\n"
            f"SQL to fix:\n{sql}\n\n"
            f"Error: {directive}\n\n"
            f"Corrected SQL:"
        )
        try:
            import re as _re_repair
            raw = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )['message']['content'].strip()
            raw = _re_repair.sub(r'```(?:sql)?', '', raw, flags=_re_repair.IGNORECASE).strip().rstrip('`').strip()
            return raw if raw.upper().startswith('SELECT') else ''
        except Exception:
            return ''

    def _llm_judge_candidates(
        self,
        question: str,
        candidates: List[str],
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
    ) -> str:
        """
        LLM-as-Judge candidate selection (CHASE-SQL approach).

        Called when execution majority vote has no clear winner (all candidates
        return different result sets). The LLM evaluates candidates on semantic
        correctness — catches cases where a minority-voted candidate is actually
        correct but execution results diverge due to wrong but consistent errors.

        Returns the judge's preferred SQL, or '' on failure.
        """
        if not candidates:
            return ''

        schema_lines = []
        for tbl, cols in sorted(pruned_schema.items()):
            col_names = ', '.join(c['column_name'] for c in cols)
            schema_lines.append(f"  {tbl}({col_names})")
        schema_str = '\n'.join(schema_lines)

        fks = schema_links.get('foreign_keys', [])
        fk_str = '; '.join(
            f"{fk.get('from_table','')}.{fk.get('from_column','')} = "
            f"{fk.get('to_table','')}.{fk.get('to_column','')}"
            for fk in fks[:6]
            if fk.get('from_table') and fk.get('to_table')
        )

        candidates_block = ''
        for i, sql in enumerate(candidates[:5], 1):
            candidates_block += f"Candidate {i}:\n{sql}\n\n"

        prompt = (
            f"You are a SQL expert. Select the SQL candidate that best answers the question.\n\n"
            f"Question: {question}\n\n"
            f"Schema:\n{schema_str}\n\n"
            f"{'Foreign keys: ' + fk_str + chr(10) + chr(10) if fk_str else ''}"
            f"Candidates:\n{candidates_block}"
            "Rules:\n"
            "1. Select the candidate that most accurately answers the question.\n"
            "2. Prefer candidates with correct SELECT columns, proper JOINs, and right conditions.\n"
            "3. Check: does the SQL return the right type of data (count, name, value)?\n"
            "4. Output ONLY the number of the best candidate (1, 2, 3, 4, or 5).\n\n"
            "Best candidate number:"
        )

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are a SQL evaluation expert. Output only a single digit.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.0}
            )
            raw = response['message']['content'].strip()
            # Extract first digit
            import re as _re
            m = _re.search(r'\b([1-5])\b', raw)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass
        return ''

    def _generate_alternative_candidate(
        self,
        question: str,
        pruned_schema: Dict[str, List[Dict]],
        schema_links: Dict,
        strategy_value: str,
        selected_examples: List[Dict] = None
    ) -> str:
        """
        Generate an alternative SQL using Graph-of-Thought (GoT) reasoning.
        Explicitly builds a table relationship graph, traverses it to identify
        the join path, then generates SQL — structurally different from the
        primary NatSQL-based candidate (AP-SQL GoT approach).
        """
        # Build table relationship graph description
        fks = schema_links.get('foreign_keys', [])
        graph_nodes = sorted(pruned_schema.keys())
        graph_edges = []
        for fk in fks:
            ft = fk.get('from_table') or fk.get('source_table', '')
            fc = fk.get('from_column') or fk.get('source_column', '')
            tt = fk.get('to_table') or fk.get('target_table', '')
            tc = fk.get('to_column') or fk.get('target_column', '')
            if ft and tt:
                graph_edges.append(f"  {ft}.{fc} ── {tt}.{tc}")

        graph_str = "Nodes: " + ", ".join(graph_nodes) + "\n"
        graph_str += "Edges (FK links):\n"
        graph_str += ("\n".join(graph_edges) if graph_edges else "  (no foreign keys)")

        schema_str = ""
        for table_name, cols in sorted(pruned_schema.items()):
            col_desc = ", ".join(
                f"{c['column_name']}({c.get('data_type', '?')})" for c in cols
            )
            schema_str += f"  {table_name}: {col_desc}\n"

        examples_str = ""
        if selected_examples:
            top_examples = sorted(
                selected_examples,
                key=lambda x: x.get('combined_score', x.get('similarity_score', 0)),
                reverse=True
            )[:3]
            for ex in top_examples:
                q = ex.get('question', '')
                sql = ex.get('query', '')
                if q and sql:
                    examples_str += f"Q: {q}\nSQL: {sql}\n\n"

        few_shot_section = ""
        if examples_str:
            few_shot_section = f"\nSimilar examples:\n{examples_str}"

        prompt = f"""Generate a SQLite query for the following question using Graph-of-Thought reasoning.

Question: {question}

Schema:
{schema_str}
Table Relationship Graph:
{graph_str}
{few_shot_section}
Reasoning steps:
STEP 0 — Traverse the graph: Which tables are needed? Which FK edges connect them?
STEP 1 — Output: What columns appear in SELECT? Is aggregation (COUNT/SUM/AVG/MAX/MIN) needed?
STEP 2 — Filters: What goes in WHERE or HAVING? (Aggregate conditions like COUNT(*) > N go in HAVING, not WHERE.)
STEP 3 — Write the SQL using the join path identified in STEP 0.

IMPORTANT: Prefer ⭐ columns from the schema — they are confirmed relevant.
Output ONLY the final SQL query (no explanation, no markdown):"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': 'You are an expert SQLite query writer. Follow the Graph-of-Thought steps, then output ONLY the SQL query.'},
                    {'role': 'user', 'content': prompt}
                ],
                options={'temperature': 0.35}
            )
            raw = response['message']['content'].strip()
            import re as _re
            raw = _re.sub(r'```sql\s*', '', raw, flags=_re.IGNORECASE)
            raw = _re.sub(r'```\s*', '', raw)
            lines = raw.split('\n')
            sql_lines = []
            in_sql = False
            for line in lines:
                lu = line.strip().upper()
                if any(lu.startswith(kw) for kw in ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE']):
                    in_sql = True
                if in_sql:
                    sql_lines.append(line)
                    if line.strip().endswith(';'):
                        break
            if sql_lines:
                raw = '\n'.join(sql_lines)
            raw = raw.strip()
            if not raw.endswith(';'):
                raw += ';'
            return raw
        except Exception as e:
            print(f"   ⚠️  Alternative candidate generation failed: {e}")
            return ""

    @staticmethod
    def _is_sql_truncated(sql: str) -> bool:
        """Return True if the SQL appears to have been cut off mid-generation."""
        sql = sql.strip().rstrip(';').strip()
        if not sql:
            return False
        # Unbalanced parentheses — strongest signal
        if sql.count('(') > sql.count(')'):
            return True
        # Last character is a dangling connector (table.  or list,  or open ()
        if sql[-1] in {'.', ',', '('}:
            return True
        # Last meaningful token expects continuation
        last = sql.upper().split()[-1] if sql.split() else ''
        return last in {'SELECT', 'FROM', 'WHERE', 'JOIN', 'AND', 'OR', 'ON', 'BY', 'AS', 'IN', 'NOT', 'HAVING'}

    def run_step7_validation(
        self,
        generated_sql: str,
        step1_result: Dict
    ) -> Dict:
        """STEP 7: SQL Validation"""
        return self.sql_validator.validate_sql_enhanced(
            generated_sql,
            step1_result['pruned_schema'],
            step1_result['schema_links']
        )
    
    def run_step8_retry(
        self,
        natural_query: str,
        step1_result: Dict,
        generated_sql: str,
        step7_result: Dict,
        generation_strategy: str,
        step4_result: Dict = None
    ) -> Dict:
        """STEP 8: Validation-Feedback Retry"""
        original_examples = step4_result['similar_examples'] if step4_result else None
        
        return self.retry_engine.retry_with_feedback(
            question=natural_query,
            pruned_schema=step1_result['pruned_schema'],
            schema_links=step1_result['schema_links'],
            generated_sql=generated_sql,
            validation_result=step7_result,
            generation_strategy=generation_strategy,
            original_examples=original_examples
        )
    
    def run_step10_execute(
        self,
        sql: str,
        db_path: str
    ) -> Dict:
        """STEP 10: Execute SQL Query"""
        return self.db_manager.execute_query(sql, db_path)
    
    def run_step10_execute_both(
        self,
        generated_sql: str,
        gold_sql: str,
        db_path: str
    ) -> Dict:
        """STEP 10: Execute Both Generated and Gold SQL"""
        return self.db_manager.execute_both_queries(
            generated_sql,
            gold_sql,
            db_path
        )
    
    def run_step11_evaluate(
        self,
        question: str,
        generated_sql: str,
        gold_sql: str,
        generated_execution: Dict,
        gold_execution: Dict
    ) -> Dict:
        """STEP 11: Evaluate Generated SQL"""
        return self.evaluator.evaluate_example(
            question,
            generated_sql,
            gold_sql,
            generated_execution,
            gold_execution
        )
    
    def run_steps_1_to_4(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        k_examples: int = 10
    ) -> Dict:
        """Run Steps 1 through 4 of the ADAPT-SQL pipeline"""
        # Step 1: Schema Linking
        step1_result = self.run_step1_schema_linking(
            natural_query, schema_dict, foreign_keys
        )
        
        # Step 2: Complexity Classification
        step2_result = self.run_step2_complexity_classification(
            natural_query, step1_result
        )
        
        # Step 3: Preliminary SQL Prediction
        step3_result = self.run_step3_preliminary_sql(
            natural_query, step1_result
        )
        
        # Step 4: Similarity Search
        step4_result = self.run_step4_similarity_search(
            natural_query,
            k=k_examples,
            preliminary_sql=step3_result.get('predicted_sql') if self.enable_structural_reranking else None
        )
        
        return {
            'step1': step1_result,
            'step2': step2_result,
            'step3': step3_result,
            'step4': step4_result
        }
    
    def run_full_pipeline(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        k_examples: int = 10,
        enable_retry: bool = True,
        db_path: str = None,
        gold_sql: str = None,
        enable_execution: bool = False,
        enable_evaluation: bool = False,
        enable_execution_retry: bool = True,
        enable_multi_candidate: bool = False,
        generation_strategy_override: Optional[str] = None,
    ) -> Dict:
        """
        Run complete ADAPT-SQL pipeline (Steps 1-11)
        
        This is the main entry point for end-to-end SQL generation with validation, 
        retry, execution, and evaluation.
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples to retrieve
            enable_retry: Enable validation-feedback retry (Step 8)
            db_path: Path to database (required for execution)
            gold_sql: Ground truth SQL (required for evaluation)
                enable_execution: Enable SQL execution (Step 10)
            enable_evaluation: Enable evaluation (Step 11)
            enable_execution_retry: Retry when generated SQL returns 0 rows (default True)
            enable_multi_candidate: Generate a 2nd SQL candidate and pick by execution (default False)
            
        Returns:
            Complete results dictionary with all steps
        """
        print("\n" + "="*70)
        print("RUNNING COMPLETE ADAPT-SQL PIPELINE (STEPS 1-11)")
        print("="*70)
        
        # Steps 1-4: Analysis and Example Retrieval
        results = self.run_steps_1_to_4(
            natural_query, schema_dict, foreign_keys, k_examples
        )
        
        # Step 5: Routing based on complexity
        step5_result = self.run_step5_routing(results['step2']['complexity_class'])
        results['step5'] = step5_result
        
        # Step 6: SQL Generation based on selected strategy
        results['step6a'] = None
        results['step6b'] = None
        results['step6c'] = None
        generated_sql = None

        # B: Set operation detection — inject hint into generation prompt
        set_op_hint = self.set_op_detector.make_hint(natural_query)
        if set_op_hint:
            print(f"\n   [B] Set-op detected — injecting hint into generation prompt")
        results['set_op_detected'] = bool(set_op_hint)

        # Value retrieval: look up actual DB values for WHERE clause anchoring
        value_matches = {}
        generation_query = natural_query
        if db_path and enable_execution:
            try:
                value_matches = retrieve_db_values(
                    natural_query, results['step1']['pruned_schema'], db_path
                )
                if value_matches:
                    value_hint_str = "Note — verified DB values: " + "; ".join(
                        f"{k}='{v}'" for k, v in value_matches.items()
                    )
                    generation_query = f"{natural_query}\n[{value_hint_str}]"
                    print(f"   [Val] DB value hints injected: {list(value_matches.keys())}")
            except Exception as e:
                print(f"   [Val] Value retrieval skipped: {e}")
        results['value_matches'] = value_matches

        strategy = step5_result['strategy']
        natsql_candidate_for_pool = None  # set by INTERMEDIATE_REPRESENTATION branch if template-adapted promoted to primary

        # Cross-strategy retry override (XiYan-SQL / ExCoT): on retry attempts, use a
        # genuinely different generation path so the LLM can't produce the same wrong SQL.
        # Attempt 2 → skeleton_first (structure-first, bypasses NatSQL)
        # Attempt 3 → direct_sql (no NatSQL round-trip, different prompt structure)
        _override_handled = False
        if generation_strategy_override:
            _override_sql = None
            if (generation_strategy_override == "skeleton_first"
                    and strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION):
                try:
                    _override_sql = self.intermediate_generator.generate_skeleton_first(
                        question=generation_query,
                        pruned_schema=results['step1']['pruned_schema'],
                        schema_links=results['step1']['schema_links'],
                        selected_examples=results['step4'].get('similar_examples', []),
                        set_op_hint=set_op_hint,
                    )
                except Exception as _oe:
                    print(f"   ⚠️  skeleton_first override failed: {_oe}")
            elif generation_strategy_override == "direct_sql" and strategy in (
                GenerationStrategy.INTERMEDIATE_REPRESENTATION,
                GenerationStrategy.DECOMPOSED_GENERATION,
            ):
                try:
                    _override_sql = self.intermediate_generator.generate_direct_sql(
                        question=generation_query,
                        pruned_schema=results['step1']['pruned_schema'],
                        schema_links=results['step1']['schema_links'],
                        selected_examples=results['step4'].get('similar_examples', []),
                        set_op_hint=set_op_hint,
                    )
                except Exception as _oe:
                    print(f"   ⚠️  direct_sql override failed: {_oe}")

            if _override_sql and _override_sql.upper().strip().startswith('SELECT'):
                print(f"   [RETRY-OVERRIDE] {generation_strategy_override} primary")
                key = 'step6b' if strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION else 'step6c'
                results[key] = {'generated_sql': _override_sql, 'strategy_override': generation_strategy_override}
                generated_sql = _override_sql
                _override_handled = True

        if not _override_handled:
            # Normal routing (no cross-strategy override active)
            pass

        if _override_handled:
            pass  # generated_sql already set by override above
        elif strategy == GenerationStrategy.SIMPLE_FEW_SHOT:
            step6a_result = self.run_step6a_few_shot_generation(
                generation_query,
                results['step1'],
                results['step4'],
                set_op_hint=set_op_hint
            )
            results['step6a'] = step6a_result
            generated_sql = step6a_result['generated_sql']

        elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
            step6b_result = self.run_step6b_intermediate_generation(
                generation_query,
                results['step1'],
                results['step4'],
                set_op_hint=set_op_hint
            )
            results['step6b'] = step6b_result
            generated_sql = step6b_result['generated_sql']

            # Template-adapted primary promotion (CHASE-SQL inspired):
            # When the top retrieved example has similarity ≥ 0.7, generate a template-
            # adapted candidate and make it the PRIMARY (candidate 0). The NatSQL result
            # is saved for the diversity pool — it still participates in majority vote but
            # can no longer form a correlated majority against the correct answer.
            natsql_candidate_for_pool = None
            if enable_multi_candidate:
                similar_examples = results['step4'].get('similar_examples', [])
                if similar_examples:
                    top_ex = max(similar_examples,
                        key=lambda x: x.get('combined_score', x.get('similarity_score', 0)))
                    top_sim = top_ex.get('combined_score', top_ex.get('similarity_score', 0))
                    if top_sim >= 0.70:
                        try:
                            tmpl_primary = self.intermediate_generator.generate_template_adapted_sql(
                                question=generation_query,
                                pruned_schema=results['step1']['pruned_schema'],
                                schema_links=results['step1']['schema_links'],
                                selected_examples=similar_examples,
                                set_op_hint=set_op_hint,
                            )
                            if (tmpl_primary
                                    and tmpl_primary.upper().strip().startswith('SELECT')
                                    and not tmpl_primary.strip().startswith('--')):
                                print(f"   [TMPL-PRIMARY] Similarity={top_sim:.3f} → promoting template-adapted to primary")
                                natsql_candidate_for_pool = generated_sql
                                generated_sql = tmpl_primary
                        except Exception as _tp_err:
                            print(f"   ⚠️  Template-primary promotion failed: {_tp_err}")

        elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
            # G': Python pivot — get oracle result shape hint before generation
            python_hint = ''
            if db_path and enable_execution:
                print("\n   [G'] Python pivot: generating oracle hint...")
                python_hint = self.python_pivot.get_hint(
                    question=natural_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    foreign_keys=results['step1']['schema_links'].get('foreign_keys', []),
                    db_path=db_path,
                    db_manager=self.db_manager
                )
                if python_hint:
                    print(f"   [G'] Oracle hint: {python_hint}")
                else:
                    print("   [G'] Oracle hint unavailable — continuing without hint")
            step6c_result = self.run_step6c_decomposed_generation(
                generation_query,
                results['step1'],
                results['step2'],
                results['step4'],
                python_hint=python_hint
            )
            results['step6c'] = step6c_result
            generated_sql = step6c_result['generated_sql']

        else:
            print(f"\n⚠️ Unknown strategy: {strategy.value}")

        # Truncation check — retry once if the SQL was cut off mid-statement
        if generated_sql and self._is_sql_truncated(generated_sql):
            print("   ⚠️  Generated SQL appears truncated — retrying generation once")
            if strategy == GenerationStrategy.SIMPLE_FEW_SHOT:
                r6 = self.run_step6a_few_shot_generation(generation_query, results['step1'], results['step4'], set_op_hint=set_op_hint)
                generated_sql = r6['generated_sql']
                results['step6a'] = r6
            elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
                r6 = self.run_step6b_intermediate_generation(generation_query, results['step1'], results['step4'], set_op_hint=set_op_hint)
                generated_sql = r6['generated_sql']
                results['step6b'] = r6
            elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
                r6 = self.run_step6c_decomposed_generation(generation_query, results['step1'], results['step2'], results['step4'], python_hint=python_hint)
                generated_sql = r6['generated_sql']
                results['step6c'] = r6

        # Multi-candidate selection (XiYan-SQL / DeepEye-SQL approach):
        # Generate temperature-diverse candidates + GoT alt, then use CandidateSelector
        # to pick the winner by execution-based majority voting.
        results['multi_candidate'] = None
        if enable_multi_candidate and generated_sql and db_path:
            print("\n" + "="*70)
            print("STEP 6 (ALT): MULTI-CANDIDATE SELECTION")
            print("="*70 + "\n")

            all_candidates = [generated_sql]

            # NatSQL result demoted from primary (saved during template-primary promotion)
            if natsql_candidate_for_pool and natsql_candidate_for_pool != generated_sql:
                all_candidates.append(natsql_candidate_for_pool)

            # Preliminary SQL from step 3 — free additional candidate (zero LLM calls).
            # Uses a simpler generation path than step 6 generators; provides diversity
            # when the primary candidate has systematic NatSQL errors.
            prel_sql = results.get('step3', {}).get('predicted_sql', '').strip()
            if prel_sql and prel_sql.upper().startswith('SELECT') and prel_sql != generated_sql:
                all_candidates.append(prel_sql)

            # Temperature-diverse candidate at temp=0.4 (same strategy, different seed)
            try:
                extra = []
                if strategy == GenerationStrategy.SIMPLE_FEW_SHOT:
                    extra = self.few_shot_generator.generate_candidates(
                        question=generation_query,
                        pruned_schema=results['step1']['pruned_schema'],
                        schema_links=results['step1']['schema_links'],
                        selected_examples=results['step4'].get('similar_examples', []),
                        n=1,
                        set_op_hint=set_op_hint,
                    )
                elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
                    extra = self.intermediate_generator.generate_candidates(
                        question=generation_query,
                        pruned_schema=results['step1']['pruned_schema'],
                        schema_links=results['step1']['schema_links'],
                        selected_examples=results['step4'].get('similar_examples', []),
                        n=1,  # 1 NatSQL diversity candidate — template-adapted is now primary so less NatSQL needed
                        set_op_hint=set_op_hint,
                    )
                    # Skeleton-first (RESDSQL-inspired): structure first, entities second
                    # Directly targets SELECT clause errors (69% of hard failures)
                    try:
                        skel_sql = self.intermediate_generator.generate_skeleton_first(
                            question=generation_query,
                            pruned_schema=results['step1']['pruned_schema'],
                            schema_links=results['step1']['schema_links'],
                            selected_examples=results['step4'].get('similar_examples', []),
                            set_op_hint=set_op_hint,
                        )
                        if skel_sql:
                            extra.append(skel_sql)
                    except Exception as _e_sk:
                        pass
                    # Direct SQL (bypass NatSQL) — cross-strategy diversity
                    # When NatSQL intermediate consistently fails on a pattern,
                    # this candidate provides a genuinely different generation path.
                    try:
                        direct_sql = self.intermediate_generator.generate_direct_sql(
                            question=generation_query,
                            pruned_schema=results['step1']['pruned_schema'],
                            schema_links=results['step1']['schema_links'],
                            selected_examples=results['step4'].get('similar_examples', []),
                            set_op_hint=set_op_hint,
                        )
                        if direct_sql:
                            extra.append(direct_sql)
                    except Exception as _e_direct:
                        pass
                elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
                    extra = self.decomposed_generator.generate_candidates(
                        question=generation_query,
                        pruned_schema=results['step1']['pruned_schema'],
                        schema_links=results['step1']['schema_links'],
                        sub_questions=results['step2'].get('sub_questions', []),
                        selected_examples=results['step4'].get('similar_examples', []),
                        few_shot_generator=self.few_shot_generator,
                        intermediate_generator=self.intermediate_generator,
                        n=1,
                    )
                    # Direct SQL candidate for NESTED as well — subquery decomposition
                    # can fail systematically; direct generation gives a simpler fallback.
                    try:
                        direct_sql_nested = self.intermediate_generator.generate_direct_sql(
                            question=generation_query,
                            pruned_schema=results['step1']['pruned_schema'],
                            schema_links=results['step1']['schema_links'],
                            selected_examples=results['step4'].get('similar_examples', []),
                            set_op_hint=set_op_hint,
                        )
                        if direct_sql_nested:
                            extra.append(direct_sql_nested)
                    except Exception:
                        pass
                all_candidates.extend(c for c in extra if c)
            except Exception as e:
                print(f"   ⚠️  Extra candidate generation failed: {e}")

            # Template-adapted candidate (CHASE-SQL inspired):
            # Directly adapts the highest-similarity example's SQL for the new question.
            # Bypasses the NatSQL round-trip — a structurally different generation path
            # that directly targets SELECT clause errors (69% of hard failures).
            try:
                tmpl_sql = self.intermediate_generator.generate_template_adapted_sql(
                    question=generation_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    schema_links=results['step1']['schema_links'],
                    selected_examples=results['step4'].get('similar_examples', []),
                    set_op_hint=set_op_hint,
                )
                if tmpl_sql:
                    all_candidates.append(tmpl_sql)
            except Exception as e:
                print(f"   ⚠️  Template-adapted candidate failed: {e}")

            # GoT alt candidate (AP-SQL Graph-of-Thought reasoning)
            # Uses generation_query so value retrieval hints are also passed.
            try:
                got_sql = self._generate_alternative_candidate(
                    question=generation_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    schema_links=results['step1']['schema_links'],
                    strategy_value=strategy.value,
                    selected_examples=results['step4'].get('similar_examples', []),
                )
                if got_sql:
                    all_candidates.append(got_sql)
            except Exception as e:
                print(f"   ⚠️  GoT candidate generation failed: {e}")

            # DISTINCT execution-based detection: add a DISTINCT variant when question
            # signals it may be needed. CandidateSelector picks correct one by execution.
            # (Replaces keyword-injection F' which caused regressions — this is safe
            # because it only adds a candidate, never removes the primary.)
            _DISTINCT_Q = ['different', 'unique', 'distinct', 'how many types',
                           'how many kinds', 'how many different', 'kind of', 'type of']
            if any(sig in natural_query.lower() for sig in _DISTINCT_Q) and generated_sql:
                distinct_sql = self._make_distinct_variant(generated_sql)
                if distinct_sql and distinct_sql.upper() != generated_sql.upper():
                    all_candidates.append(distinct_sql)

            print(f"   Candidates generated: {len(all_candidates)}")

            # Execute all candidates and select winner by majority result-set agreement
            if len(all_candidates) > 1:
                sel = self.candidate_selector.select(all_candidates, db_path)
                winner_sql = sel['winner_sql']
                winner_idx = sel['winner_index']
                print(f"   Winner: idx={winner_idx} ({sel['winner_reason']})")

                # LLM-as-Judge fallback (CHASE-SQL): when execution majority finds no
                # clear winner (every candidate returns a different result set), ask the
                # LLM to pick the semantically best candidate.  Adds 1 LLM call only in
                # this edge case (~15-20% of queries where candidates disagree).
                n_groups = len(sel.get('group_counts', {}))
                n_successful = sum(1 for r in sel.get('exec_results', []) if r.get('success'))
                no_majority = n_groups >= n_successful and n_successful > 1
                if no_majority:
                    try:
                        judge_sql = self._llm_judge_candidates(
                            question=natural_query,
                            candidates=all_candidates,
                            pruned_schema=results['step1']['pruned_schema'],
                            schema_links=results['step1']['schema_links'],
                        )
                        if judge_sql:
                            print(f"   LLM judge override (no majority, {n_groups} groups)")
                            winner_sql = judge_sql
                    except Exception as _je:
                        print(f"   ⚠️  LLM judge failed: {_je}")

                if winner_sql:
                    generated_sql = winner_sql
                results['multi_candidate'] = {
                    **sel,
                    'n_candidates': len(all_candidates),
                    'primary_sql': all_candidates[0],
                }
            else:
                print("   ⚠️  Only 1 candidate — multi-candidate skipped")

        # Step 6.5: SQL Normalization (if enabled)
        if self.enable_sql_normalization and generated_sql:
            print("\n" + "="*70)
            print("STEP 6.5: SQL NORMALIZATION")
            print("="*70 + "\n")
            
            normalization_result = normalize_sql_post_generation(
                generated_sql=generated_sql,
                ground_truth_sql=gold_sql,
                enable_normalization=self.enable_sql_normalization
            )
            
            results['step6_5_normalization'] = normalization_result
            
            # Update generated_sql to normalized version
            if normalization_result['normalized_sql']:
                print(f"âœ… Normalized SQL: {len(normalization_result['changes_made'])} changes")
                generated_sql = normalization_result['normalized_sql']
            
            print("\n" + "="*70)
            print("STEP 6.5 COMPLETED")
            print("="*70 + "\n")
        else:
            results['step6_5_normalization'] = None
        
        # Step 7: Validation
        if generated_sql:
            step7_result = self.run_step7_validation(
                generated_sql,
                results['step1']
            )
            results['step7'] = step7_result
        else:
            results['step7'] = {
                'is_valid': False,
                'errors': [{'type': 'GENERATION_ERROR', 'message': 'No SQL generated', 'severity': 'CRITICAL'}],
                'warnings': [],
                'suggestions': [],
                'validation_score': 0.0,
                'reasoning': 'No SQL was generated in Step 6'
            }
        
        # E': Deterministic Checker Chain (DeepEye-SQL)
        results['checker_chain'] = None
        if generated_sql and enable_retry:
            checker = CheckerChain(schema_links=results['step1']['schema_links'])
            check_result = checker.run(
                generated_sql,
                db_path=db_path if enable_execution else None,
                db_manager=self.db_manager if enable_execution else None,
                question=natural_query
            )
            results['checker_chain'] = check_result
            if not check_result['passed']:
                print(f"⚠️  Checker '{check_result['checker']}' failed — injecting directive into retry")
                results['step7']['errors'].append({
                    'type': 'CHECKER_CHAIN',
                    'message': check_result['directive'],
                    'severity': 'HIGH'
                })
                results['step7']['is_valid'] = False

        # Step 8: Validation-Feedback Retry (if enabled and needed)
        final_sql = generated_sql
        final_is_valid = results['step7']['is_valid']

        if enable_retry and generated_sql:
            step8_result = self.run_step8_retry(
                natural_query,
                results['step1'],
                generated_sql,
                results['step7'],
                strategy.value,
                results['step4']
            )
            results['step8'] = step8_result
            
            final_sql = step8_result['final_sql']
            final_is_valid = step8_result['is_valid']
        else:
            results['step8'] = None
        
        # Add final results
        results['final_sql'] = final_sql
        results['final_is_valid'] = final_is_valid
        
        # Step 10: Execute SQL (if enabled and db_path provided)
        results['step10_generated'] = None
        results['step10_gold'] = None
        results['exec_retry'] = None

        if enable_execution and db_path and final_sql:
            # Execute generated SQL
            results['step10_generated'] = self.run_step10_execute(final_sql, db_path)

            # Execution error retry: SQL completely failed to execute (syntax/schema error)
            results['exec_error_retry'] = None
            exec_result = results['step10_generated']
            if (enable_execution_retry and
                    exec_result and
                    not exec_result.get('success')):
                exec_error_retry = self.retry_engine.retry_with_exec_error_feedback(
                    question=natural_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    schema_links=results['step1']['schema_links'],
                    current_sql=final_sql,
                    error_msg=exec_result.get('error', ''),
                    generation_strategy=strategy.value,
                    db_path=db_path,
                    db_manager=self.db_manager
                )
                results['exec_error_retry'] = exec_error_retry
                if exec_error_retry.get('sql_changed'):
                    final_sql = exec_error_retry['final_sql']
                    results['final_sql'] = final_sql
                    results['step10_generated'] = self.run_step10_execute(final_sql, db_path)

            # Execution-driven retry: fix queries that return 0 rows on a positive question
            exec_result = results['step10_generated']
            if (enable_execution_retry and
                    exec_result and
                    exec_result.get('success') and
                    len(exec_result.get('result_rows', [])) == 0 and
                    not self._is_negation_query(natural_query)):
                exec_retry = self.retry_engine.retry_with_execution_feedback(
                    question=natural_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    schema_links=results['step1']['schema_links'],
                    current_sql=final_sql,
                    generation_strategy=strategy.value,
                    db_path=db_path,
                    db_manager=self.db_manager
                )
                results['exec_retry'] = exec_retry
                if exec_retry.get('sql_changed'):
                    final_sql = exec_retry['final_sql']
                    results['final_sql'] = final_sql
                    results['step10_generated'] = self.run_step10_execute(final_sql, db_path)

            # Plausibility-triggered retry: fix scalar-agg queries returning wrong row count
            results['plausibility_retry'] = None
            exec_after_0row = results['step10_generated']
            plausibility = exec_after_0row.get('plausibility_check', {}) if exec_after_0row else {}
            if (enable_execution_retry and
                    exec_after_0row and exec_after_0row.get('success') and
                    not plausibility.get('plausible', True) and
                    len(exec_after_0row.get('result_rows', [])) > 0):
                plausibility_retry = self.retry_engine.retry_with_plausibility_feedback(
                    question=natural_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    schema_links=results['step1']['schema_links'],
                    current_sql=final_sql,
                    generation_strategy=strategy.value,
                    plausibility_issue=plausibility.get('issue', ''),
                    db_path=db_path,
                    db_manager=self.db_manager
                )
                results['plausibility_retry'] = plausibility_retry
                if plausibility_retry.get('sql_changed'):
                    final_sql = plausibility_retry['final_sql']
                    results['final_sql'] = final_sql
                    results['step10_generated'] = self.run_step10_execute(final_sql, db_path)

            # Result shape retry: fix queries where SQL structure contradicts result shape
            # (e.g. aggregation without GROUP BY returning multiple rows, LIMIT 1 returning many)
            results['shape_retry'] = None
            exec_for_shape = results['step10_generated']
            shape_issue = self._result_shape_issue(final_sql, exec_for_shape) if exec_for_shape else ''
            if enable_execution_retry and shape_issue:
                print(f"   [Shape] Issue detected: {shape_issue} — retrying")
                shape_retry = self.retry_engine.retry_with_plausibility_feedback(
                    question=natural_query,
                    pruned_schema=results['step1']['pruned_schema'],
                    schema_links=results['step1']['schema_links'],
                    current_sql=final_sql,
                    generation_strategy=strategy.value,
                    plausibility_issue=shape_issue,
                    db_path=db_path,
                    db_manager=self.db_manager
                )
                results['shape_retry'] = shape_retry
                if shape_retry.get('sql_changed'):
                    final_sql = shape_retry['final_sql']
                    results['final_sql'] = final_sql
                    results['step10_generated'] = self.run_step10_execute(final_sql, db_path)

            # Execute gold SQL if provided
            if gold_sql:
                results['step10_gold'] = self.run_step10_execute(gold_sql, db_path)
        
        # Step 11: Evaluate (if enabled and gold_sql provided)
        results['step11'] = None
        
        if enable_evaluation and gold_sql and results['step10_generated'] and results['step10_gold']:
            results['step11'] = self.run_step11_evaluate(
                natural_query,
                final_sql,
                gold_sql,
                results['step10_generated'],
                results['step10_gold']
            )
        
        print("\n" + "="*70)
        status = "✅" if final_is_valid else "⚠️"
        print(f"PIPELINE COMPLETED - ALL STEPS {status}")
        print("="*70 + "\n")
        
        return results
    
    def generate_sql(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        k_examples: int = 10,
        enable_retry: bool = True
    ) -> str:
        """
        Convenience method: Run full pipeline and return only the final SQL
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples to retrieve
            enable_retry: Enable validation-feedback retry
            
        Returns:
            Final SQL query as string (after validation and potential retry)
        """
        result = self.run_full_pipeline(
            natural_query, schema_dict, foreign_keys, k_examples, enable_retry
        )
        
        return result.get('final_sql', '-- No SQL generated')