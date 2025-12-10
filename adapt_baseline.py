"""
ADAPT-SQL Baseline - Complete Pipeline (Steps 1-7)
Schema Linking + Complexity + Preliminary SQL + Example Selection + Routing + Generation + Validation
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


class ADAPTBaseline:
    def __init__(
        self, 
        model: str = "llama3.2",
        vector_store_path: str = None
    ):
        """
        Initialize ADAPT-SQL with Ollama model and optional vector store
        
        Args:
            model: Ollama model name (e.g., "llama3.2", "codellama", "mistral")
            vector_store_path: Path to pre-built FAISS vector store
        """
        self.model = model
        
        # Initialize all pipeline components
        self.schema_linker = EnhancedSchemaLinker(model=model)
        self.complexity_classifier = QueryComplexityClassifier(model=model)
        self.preliminary_predictor = PreliminaryPredictor(model=model)
        self.routing_strategy = RoutingStrategy(model=model)
        self.sql_validator = SQLValidator()  # NEW: Validator doesn't need model
        
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
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            
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
        
        Args:
            natural_query: Natural language question
            step1_result: Output from Step 1
            
        Returns:
            {
                'complexity_class': ComplexityClass,
                'required_tables': Set[str],
                'sub_questions': List[str],
                'needs_joins': bool,
                'needs_subqueries': bool,
                'aggregations': List[str],
                'has_grouping': bool,
                'has_ordering': bool,
                'reasoning': str
            }
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
        
        Args:
            natural_query: Natural language question
            step1_result: Output from Step 1
            
        Returns:
            {
                'predicted_sql': str,
                'sql_skeleton': str,
                'sql_keywords': List[str],
                'sql_structure': Dict,
                'reasoning': str
            }
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
        
        Args:
            natural_query: Natural language question
            k: Number of similar examples to retrieve
            
        Returns:
            {
                'similar_examples': List[Dict],
                'reasoning': str,
                'query': str,
                'total_found': int
            }
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
        
        Args:
            complexity_class: Query complexity from Step 2
            
        Returns:
            {
                'strategy': GenerationStrategy,
                'reasoning': str,
                'description': str
            }
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
        
        Args:
            natural_query: Natural language question
            step1_result: Output from Step 1
            step4_result: Output from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
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
        
        Args:
            natural_query: Natural language question
            step1_result: Output from Step 1
            step4_result: Output from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'natsql_intermediate': str,
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
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
        
        Args:
            natural_query: Natural language question
            step1_result: Output from Step 1
            step2_result: Output from Step 2 (includes sub_questions)
            step4_result: Output from Step 4
            
        Returns:
            {
                'generated_sql': str,
                'sub_sql_list': List[Dict],
                'natsql_intermediate': str,
                'confidence': float,
                'reasoning': str,
                'examples_used': int
            }
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
        
        Args:
            generated_sql: SQL query from Step 6
            step1_result: Output from Step 1 (for schema reference)
            
        Returns:
            {
                'is_valid': bool,
                'errors': List[Dict],
                'warnings': List[Dict],
                'suggestions': List[str],
                'validation_score': float,
                'reasoning': str
            }
        """
        return self.sql_validator.validate_sql_enhanced(
            generated_sql,
            step1_result['pruned_schema'],
            step1_result['schema_links']
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
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples to retrieve
            
        Returns:
            {
                'step1': {...},
                'step2': {...},
                'step3': {...},
                'step4': {...}
            }
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
        k_examples: int = 10
    ) -> Dict:
        """
        Run complete ADAPT-SQL pipeline (Steps 1-7)
        
        This is the main entry point for end-to-end SQL generation and validation.
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples to retrieve
            
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
                'step7': {...}  (validation results)
            }
        """
        print("\n" + "="*70)
        print("RUNNING COMPLETE ADAPT-SQL PIPELINE (STEPS 1-7)")
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
        
        # Step 7: Validation (NEW)
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
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED - ALL 7 STEPS ✓")
        print("="*70 + "\n")
        
        return results
    
    def generate_sql(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        k_examples: int = 10
    ) -> str:
        """
        Convenience method: Run full pipeline and return only the generated SQL
        
        Args:
            natural_query: Natural language question
            schema_dict: Full database schema
            foreign_keys: Foreign key relationships
            k_examples: Number of similar examples to retrieve
            
        Returns:
            Generated SQL query as string
        """
        result = self.run_full_pipeline(
            natural_query, schema_dict, foreign_keys, k_examples
        )
        
        # Extract generated SQL from whichever step produced it
        if result.get('step6a'):
            return result['step6a']['generated_sql']
        elif result.get('step6b'):
            return result['step6b']['generated_sql']
        elif result.get('step6c'):
            return result['step6c']['generated_sql']
        else:
            return "-- No SQL generated"