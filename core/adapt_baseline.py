"""
ADAPT-SQL Baseline - Complete Pipeline (Steps 1-11)
Schema Linking + Complexity + Preliminary SQL + Example Selection + Routing +
Generation + Validation + Retry + Execute + Evaluate
"""
import os
import ollama as _ollama_module
from typing import Dict, List

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

def _patched_chat(**kwargs):
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
        
        # Get similar examples from vector store
        result = self.example_selector.search_similar_examples(
            natural_query,
            k=k
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
        step4_result: Dict
    ) -> Dict:
        """STEP 6a: Simple Few-Shot Generation (for EASY queries)"""
        return self.few_shot_generator.generate_sql_easy(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step4_result['similar_examples']
        )
    
    def run_step6b_intermediate_generation(
        self,
        natural_query: str,
        step1_result: Dict,
        step4_result: Dict
    ) -> Dict:
        """STEP 6b: Intermediate Representation Generation (for NON_NESTED_COMPLEX)"""
        return self.intermediate_generator.generate_sql_with_intermediate(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step4_result['similar_examples']
        )
    
    def run_step6c_decomposed_generation(
        self,
        natural_query: str,
        step1_result: Dict,
        step2_result: Dict,
        step4_result: Dict
    ) -> Dict:
        """STEP 6c: Decomposed Generation with Subquery Handling (for NESTED_COMPLEX)"""
        return self.decomposed_generator.generate_sql_decomposed(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step2_result['sub_questions'],
            step4_result['similar_examples'],
            few_shot_generator=self.few_shot_generator,
            intermediate_generator=self.intermediate_generator
        )
    
    @staticmethod
    def _is_negation_query(question: str) -> bool:
        """Return True if the question likely expects 0 rows (negation/exclusion intent)."""
        q = ' ' + question.lower() + ' '
        patterns = [
            " never ", " no ", " not ", " without ", " none ", " zero ",
            "n't ", " except ", " exclud", " neither ", " nor ",
        ]
        return any(p in q for p in patterns)

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
STEP 2 — Filters: What goes in WHERE or HAVING?
STEP 3 — Write the SQL using the join path identified in STEP 0.

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
        enable_multi_candidate: bool = False
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
        
        strategy = step5_result['strategy']
        
        if strategy == GenerationStrategy.SIMPLE_FEW_SHOT:
            step6a_result = self.run_step6a_few_shot_generation(
                natural_query,
                results['step1'],
                results['step4']
            )
            results['step6a'] = step6a_result
            generated_sql = step6a_result['generated_sql']
            
        elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
            step6b_result = self.run_step6b_intermediate_generation(
                natural_query,
                results['step1'],
                results['step4']
            )
            results['step6b'] = step6b_result
            generated_sql = step6b_result['generated_sql']
            
        elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
            step6c_result = self.run_step6c_decomposed_generation(
                natural_query,
                results['step1'],
                results['step2'],
                results['step4']
            )
            results['step6c'] = step6c_result
            generated_sql = step6c_result['generated_sql']
            
        else:
            print(f"\n⚠️ Unknown strategy: {strategy.value}")

        # Truncation check — retry once if the SQL was cut off mid-statement
        if generated_sql and self._is_sql_truncated(generated_sql):
            print("   ⚠️  Generated SQL appears truncated — retrying generation once")
            if strategy == GenerationStrategy.SIMPLE_FEW_SHOT:
                r6 = self.run_step6a_few_shot_generation(natural_query, results['step1'], results['step4'])
                generated_sql = r6['generated_sql']
                results['step6a'] = r6
            elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
                r6 = self.run_step6b_intermediate_generation(natural_query, results['step1'], results['step4'])
                generated_sql = r6['generated_sql']
                results['step6b'] = r6
            elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
                r6 = self.run_step6c_decomposed_generation(natural_query, results['step1'], results['step2'], results['step4'])
                generated_sql = r6['generated_sql']
                results['step6c'] = r6

        # Multi-candidate selection: generate a 2nd SQL and pick by execution outcome
        # Only runs when db_path is available (needed to execute candidates for comparison)
        results['multi_candidate'] = None
        if enable_multi_candidate and generated_sql and db_path:
            print("\n" + "="*70)
            print("STEP 6 (ALT): MULTI-CANDIDATE GENERATION")
            print("="*70 + "\n")

            alt_sql = self._generate_alternative_candidate(
                question=natural_query,
                pruned_schema=results['step1']['pruned_schema'],
                schema_links=results['step1']['schema_links'],
                strategy_value=strategy.value,
                selected_examples=results['step4'].get('similar_examples', [])
            )

            if alt_sql:
                primary_exec = self.db_manager.execute_query(generated_sql, db_path)
                alt_exec = self.db_manager.execute_query(alt_sql, db_path)

                primary_rows = len(primary_exec.get('result_rows', [])) if primary_exec.get('success') else -1
                alt_rows = len(alt_exec.get('result_rows', [])) if alt_exec.get('success') else -1

                primary_plaus = primary_exec.get('plausibility_check') or {}
                alt_plaus = alt_exec.get('plausibility_check') or {}
                primary_plausible = primary_plaus.get('plausible', True)
                alt_plausible = alt_plaus.get('plausible', True)

                # Alternative wins when primary returns nothing OR primary fails plausibility while alt passes
                use_alt = False
                reason = ""
                if primary_rows == 0 and alt_rows > 0:
                    use_alt = True
                    reason = f"{alt_rows} rows vs 0 for primary"
                elif not primary_plausible and alt_plausible and alt_rows > 0:
                    use_alt = True
                    reason = f"primary failed plausibility ({primary_plaus.get('issue', '?')}), alt passed"

                if use_alt:
                    print(f"   ✅ Alternative candidate selected ({reason})")
                    generated_sql = alt_sql
                else:
                    print(f"   Primary candidate kept ({primary_rows} rows vs {alt_rows} for alternative)")

                results['multi_candidate'] = {
                    'primary_sql': results.get('final_sql', generated_sql),
                    'alt_sql': alt_sql,
                    'primary_rows': primary_rows,
                    'alt_rows': alt_rows,
                    'winner': 'alt' if use_alt else 'primary'
                }
            else:
                print("   ⚠️  Alternative candidate generation skipped (empty output)")

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
        # Run 6 rule-based checkers and inject any directive into step7 errors
        # so that the retry engine receives an explicit correction instruction.
        results['checker_chain'] = None
        if generated_sql and enable_retry:
            checker = CheckerChain(schema_links=results['step1']['schema_links'])
            check_result = checker.run(
                generated_sql,
                db_path=db_path if enable_execution else None,
                db_manager=self.db_manager if enable_execution else None
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