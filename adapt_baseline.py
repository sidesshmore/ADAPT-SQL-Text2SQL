"""
ADAPT-SQL Baseline - Complete Pipeline (Steps 1-8)
Schema Linking + Complexity + Preliminary SQL + Example Selection + Routing + Generation + Validation + Retry
"""
from typing import Dict, List
from schema_linking import EnhancedSchemaLinker
from query_complexity import QueryComplexityClassifier, ComplexityClass
from prel_sql_prediction import PreliminaryPredictor
from vector_search import DualSimilaritySelector
from vector_store import SQLVectorStore
from routing_strategy import RoutingStrategy, GenerationStrategy
from few_shot import FewShotGenerator
from intermediate_repr import IntermediateRepresentationGenerator
from decomposed_generation import DecomposedGenerator
from validate_sql import SQLValidator
from validation_feedback_retry import ValidationFeedbackRetry


class ADAPTBaseline:
    def __init__(
        self, 
        model: str = "llama3.2",
        vector_store_path: str = None,
        max_retries: int = 2
    ):
        """
        Initialize ADAPT-SQL with Ollama model and optional vector store
        
        Args:
            model: Ollama model name (e.g., "llama3.2", "codellama", "mistral")
            vector_store_path: Path to pre-built FAISS vector store
            max_retries: Maximum validation retry attempts (default: 2)
        """
        self.model = model
        self.max_retries = max_retries
        
        # Initialize all pipeline components
        self.schema_linker = EnhancedSchemaLinker(model=model)
        self.complexity_classifier = QueryComplexityClassifier(model=model)
        self.preliminary_predictor = PreliminaryPredictor(model=model)
        self.routing_strategy = RoutingStrategy(model=model)
        self.sql_validator = SQLValidator()
        self.retry_engine = ValidationFeedbackRetry(model=model, max_retries=max_retries)
        
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
        """
        STEP 1: Enhanced Schema Linking
        
        Identifies relevant tables, columns, and foreign keys for the query.
        """
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
        """
        STEP 2: Query Complexity Classification
        
        Classifies query as EASY, NON_NESTED_COMPLEX, or NESTED_COMPLEX.
        """
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
        """
        STEP 3: Preliminary SQL Prediction
        
        Generates rough SQL skeleton for example matching.
        """
        return self.preliminary_predictor.predict_sql_skeleton(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links']
        )
    
    def run_step4_similarity_search(
        self,
        natural_query: str,
        k: int = 10
    ) -> Dict:
        """
        STEP 4: Similarity Search in Vector Database
        
        Retrieves similar examples from the vector store.
        """
        if self.example_selector is None:
            return {
                'similar_examples': [],
                'reasoning': 'Vector store not loaded',
                'query': natural_query,
                'total_found': 0
            }
        
        return self.example_selector.search_similar_examples(
            natural_query,
            k=k
        )
    
    def run_step5_routing(
        self,
        complexity_class: ComplexityClass
    ) -> Dict:
        """
        STEP 5: Route to Appropriate Generation Strategy
        
        Determines which SQL generation strategy to use based on complexity.
        """
        return self.routing_strategy.route_to_strategy(complexity_class.value)
    
    def run_step6a_few_shot_generation(
        self,
        natural_query: str,
        step1_result: Dict,
        step4_result: Dict
    ) -> Dict:
        """
        STEP 6a: Simple Few-Shot Generation (for EASY queries)
        
        Direct SQL generation using similar examples.
        """
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
        """
        STEP 6b: Intermediate Representation Generation (for NON_NESTED_COMPLEX)
        
        Two-stage generation: NatSQL intermediate → Final SQL.
        """
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
        """
        STEP 6c: Decomposed Generation with Subquery Handling (for NESTED_COMPLEX)
        
        Three-stage generation:
        1. Generate SQL for each sub-question
        2. Create intermediate representation combining sub-queries
        3. Convert to final nested SQL
        """
        return self.decomposed_generator.generate_sql_decomposed(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links'],
            step2_result['sub_questions'],
            step4_result['similar_examples'],
            few_shot_generator=self.few_shot_generator,
            intermediate_generator=self.intermediate_generator
        )
    
    def run_step7_validation(
        self,
        generated_sql: str,
        step1_result: Dict
    ) -> Dict:
        """
        STEP 7: SQL Validation
        
        Validates generated SQL for syntax, schema compliance, and logical correctness.
        """
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
        """
        STEP 8: Validation-Feedback Retry
        
        Regenerates SQL based on validation errors (if any).
        """
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
    
    def run_steps_1_to_4(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        k_examples: int = 10
    ) -> Dict:
        """
        Run Steps 1 through 4 of the ADAPT-SQL pipeline
        """
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
            natural_query, k=k_examples
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
        enable_retry: bool = True
    ) -> Dict:
        """
        Run complete ADAPT-SQL pipeline (Steps 1-8)
        
        This is the main entry point for end-to-end SQL generation with validation and retry.
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples to retrieve
            enable_retry: Enable validation-feedback retry (Step 8)
            
        Returns:
            {
                'step1': {...},
                'step2': {...},
                'step3': {...},
                'step4': {...},
                'step5': {...},
                'step6a': {...} (if EASY),
                'step6b': {...} (if NON_NESTED_COMPLEX),
                'step6c': {...} (if NESTED_COMPLEX),
                'step7': {...},
                'step8': {...} (if retry enabled),
                'final_sql': str,
                'final_is_valid': bool
            }
        """
        print("\n" + "="*70)
        print("RUNNING COMPLETE ADAPT-SQL PIPELINE (STEPS 1-8)")
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
            # EASY queries: Direct few-shot generation
            step6a_result = self.run_step6a_few_shot_generation(
                natural_query,
                results['step1'],
                results['step4']
            )
            results['step6a'] = step6a_result
            generated_sql = step6a_result['generated_sql']
            
        elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
            # NON_NESTED_COMPLEX queries: Two-stage with intermediate representation
            step6b_result = self.run_step6b_intermediate_generation(
                natural_query,
                results['step1'],
                results['step4']
            )
            results['step6b'] = step6b_result
            generated_sql = step6b_result['generated_sql']
            
        elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
            # NESTED_COMPLEX queries: Three-stage decomposed generation
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
            
            # Update final SQL with retry result
            final_sql = step8_result['final_sql']
            final_is_valid = step8_result['is_valid']
        else:
            results['step8'] = None
        
        # Add final results
        results['final_sql'] = final_sql
        results['final_is_valid'] = final_is_valid
        
        print("\n" + "="*70)
        status = "âœ…" if final_is_valid else "⚠️ "
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