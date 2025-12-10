"""
ADAPT-SQL Baseline - Steps 1 & 2
Schema Linking + Complexity Classification
"""
from typing import Dict, List
from schema_linking import EnhancedSchemaLinker
from query_complexity import QueryComplexityClassifier


class ADAPTBaseline:
    def __init__(self, model: str = "llama3.2"):
        """Initialize ADAPT-SQL with Ollama model"""
        self.model = model
        self.schema_linker = EnhancedSchemaLinker(model=model)
        self.complexity_classifier = QueryComplexityClassifier(model=model)
    
    def run_step1_schema_linking(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> Dict:
        """
        Run STEP 1: Enhanced Schema Linking
        
        Returns:
            {
                'pruned_schema': Dict,
                'schema_links': Dict,
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
        Run STEP 2: Query Complexity Classification
        
        Args:
            natural_query: Natural language question
            step1_result: Output from Step 1
            
        Returns:
            {
                'complexity_class': ComplexityClass,
                'required_tables': Set[str],
                'sub_questions': List[str],
                'reasoning': str,
                ...
            }
        """
        return self.complexity_classifier.classify_query(
            natural_query,
            step1_result['pruned_schema'],
            step1_result['schema_links']
        )
    
    def run_steps_1_and_2(
        self,
        natural_query: str,
        schema_dict: Dict[str, List[Dict]],
        foreign_keys: List[Dict]
    ) -> Dict:
        """
        Run both Step 1 and Step 2
        
        Returns:
            {
                'step1': {...},
                'step2': {...}
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
        
        return {
            'step1': step1_result,
            'step2': step2_result
        }