"""
STEP 5: Routing Strategy
Routes to appropriate generation strategy based on complexity classification
"""
from typing import Dict, List
from enum import Enum


class GenerationStrategy(Enum):
    """Generation strategies for different complexity classes"""
    SIMPLE_FEW_SHOT = "SIMPLE_FEW_SHOT"
    INTERMEDIATE_REPRESENTATION = "INTERMEDIATE_REPRESENTATION"
    DECOMPOSED_GENERATION = "DECOMPOSED_GENERATION"


class RoutingStrategy:
    def __init__(self, model: str = "llama3.2"):
        """Initialize routing strategy"""
        self.model = model
    
    def route_to_strategy(
        self,
        complexity_class: str
    ) -> Dict:
        """
        STEP 5: Route to Appropriate Generation Strategy
        
        Args:
            complexity_class: Complexity classification from Step 2
            
        Returns:
            {
                'strategy': GenerationStrategy,
                'reasoning': str,
                'description': str
            }
        """
        print(f"\n{'='*60}")
        print("STEP 5: ROUTING TO GENERATION STRATEGY")
        print(f"{'='*60}\n")
        
        print(f"Input Complexity: {complexity_class}")
        
        # Route based on complexity
        if complexity_class == "EASY":
            strategy = GenerationStrategy.SIMPLE_FEW_SHOT
            description = "Single table or simple JOIN - Use few-shot examples directly"
            
        elif complexity_class == "NON_NESTED_COMPLEX":
            strategy = GenerationStrategy.INTERMEDIATE_REPRESENTATION
            description = "Multiple JOINs, aggregations - Use intermediate representation"
            
        elif complexity_class == "NESTED_COMPLEX":
            strategy = GenerationStrategy.DECOMPOSED_GENERATION
            description = "Subqueries required - Decompose into sub-problems"
            
        else:
            # Default fallback
            strategy = GenerationStrategy.SIMPLE_FEW_SHOT
            description = "Unknown complexity - Default to simple few-shot"
        
        print(f"Selected Strategy: {strategy.value}")
        print(f"Description: {description}")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            complexity_class, strategy, description
        )
        
        print(f"\n{'='*60}")
        print("STEP 5 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'strategy': strategy,
            'reasoning': reasoning,
            'description': description
        }
    
    def _generate_reasoning(
        self,
        complexity_class: str,
        strategy: GenerationStrategy,
        description: str
    ) -> str:
        """Generate reasoning for routing decision"""
        reasoning = "STEP 5: ROUTING STRATEGY\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Input Complexity Class: {complexity_class}\n"
        reasoning += f"Selected Strategy: {strategy.value}\n\n"
        
        reasoning += "Strategy Description:\n"
        reasoning += f"  {description}\n\n"
        
        reasoning += "Strategy Details:\n"
        
        if strategy == GenerationStrategy.SIMPLE_FEW_SHOT:
            reasoning += "  • Use 3-5 similar examples from vector store\n"
            reasoning += "  • Single LLM call with few-shot prompt\n"
            reasoning += "  • Direct SQL generation\n"
            reasoning += "  • Best for: Single table or simple JOIN queries\n"
            
        elif strategy == GenerationStrategy.INTERMEDIATE_REPRESENTATION:
            reasoning += "  • Generate intermediate representation first\n"
            reasoning += "  • Break down into logical steps\n"
            reasoning += "  • Then translate to SQL\n"
            reasoning += "  • Best for: Multiple JOINs, complex aggregations\n"
            
        elif strategy == GenerationStrategy.DECOMPOSED_GENERATION:
            reasoning += "  • Identify sub-questions\n"
            reasoning += "  • Generate SQL for each sub-question\n"
            reasoning += "  • Compose final SQL from sub-queries\n"
            reasoning += "  • Best for: Nested queries, comparisons with aggregates\n"
        
        return reasoning