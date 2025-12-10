"""
ADAPT-SQL Baseline - Complete Pipeline (Steps 1-6b)
Schema Linking + Complexity + Preliminary SQL + Example Selection + Routing + Generation
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


class ADAPTBaseline:
    def __init__(
        self, 
        model: str = "llama3.2",
        vector_store_path: str = None
    ):
        """Initialize ADAPT-SQL with Ollama model and optional vector store"""
        self.model = model
        self.schema_linker = EnhancedSchemaLinker(model=model)
        self.complexity_classifier = QueryComplexityClassifier(model=model)
        self.preliminary_predictor = PreliminaryPredictor(model=model)
        self.routing_strategy = RoutingStrategy(model=model)
        self.few_shot_generator = FewShotGenerator(model=model)
        self.intermediate_generator = IntermediateRepresentationGenerator(model=model)
        
        # Load vector store if path provided
        self.vector_store = None
        self.example_selector = None
        
        if vector_store_path:
            self.vector_store = SQLVectorStore()
            if self.vector_store.load_index(vector_store_path):
                self.example_selector = DualSimilaritySelector(self.vector_store)
                print("✅ Vector store loaded successfully")
            else:
                print("⚠️  Vector store loading failed")
    
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
        k: int = 10
    ) -> Dict:
        """STEP 4: Similarity Search in Vector Database"""
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
    
    def run_steps_1_to_4(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict],
        k_examples: int = 10
    ) -> Dict:
        """
        Run Steps 1 through 4
        
        Returns:
            {
                'step1': {...},
                'step2': {...},
                'step3': {...},
                'step4': {...}
            }
        """
        # Step 1
        step1_result = self.run_step1_schema_linking(
            natural_query, schema_dict, foreign_keys
        )
        
        # Step 2
        step2_result = self.run_step2_complexity_classification(
            natural_query, step1_result
        )
        
        # Step 3
        step3_result = self.run_step3_preliminary_sql(
            natural_query, step1_result
        )
        
        # Step 4 - Similarity Search
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
        Run complete ADAPT-SQL pipeline (Steps 1-6b)
        
        Returns:
            {
                'step1': {...},
                'step2': {...},
                'step3': {...},
                'step4': {...},
                'step5': {...},
                'step6a': {...} (if EASY),
                'step6b': {...} (if NON_NESTED_COMPLEX)
            }
        """
        print("\n" + "="*70)
        print("RUNNING COMPLETE ADAPT-SQL PIPELINE")
        print("="*70)
        
        # Steps 1-4
        results = self.run_steps_1_to_4(
            natural_query, schema_dict, foreign_keys, k_examples
        )
        
        # Step 5: Routing
        step5_result = self.run_step5_routing(results['step2']['complexity_class'])
        results['step5'] = step5_result
        
        # Step 6: Generation based on strategy
        results['step6a'] = None
        results['step6b'] = None
        
        if step5_result['strategy'] == GenerationStrategy.SIMPLE_FEW_SHOT:
            step6a_result = self.run_step6a_few_shot_generation(
                natural_query,
                results['step1'],
                results['step4']
            )
            results['step6a'] = step6a_result
            
        elif step5_result['strategy'] == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
            step6b_result = self.run_step6b_intermediate_generation(
                natural_query,
                results['step1'],
                results['step4']
            )
            results['step6b'] = step6b_result
            
        else:
            print(f"\n⚠️  Strategy {step5_result['strategy'].value} not yet implemented")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED")
        print("="*70 + "\n")
        
        return results