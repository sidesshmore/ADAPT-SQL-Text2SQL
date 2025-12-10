"""
STEP 4: Similarity Search
Returns similar examples with their questions, SQL queries, and similarity scores
"""
from typing import List, Dict
from vector_store import SQLVectorStore


class DualSimilaritySelector:
    def __init__(self, vector_store: SQLVectorStore):
        """Initialize with loaded vector store"""
        self.vector_store = vector_store
    
    def search_similar_examples(
        self,
        question: str,
        k: int = 10
    ) -> Dict:
        """
        STEP 4: Similarity Search in Vector Database
        
        Args:
            question: Natural language question
            k: Number of similar examples to return
            
        Returns:
            {
                'similar_examples': List[Dict],
                'reasoning': str,
                'query': str,
                'total_found': int
            }
        """
        print(f"\n{'='*60}")
        print("STEP 4: SIMILARITY SEARCH")
        print(f"{'='*60}\n")
        
        print(f"Query: {question}")
        print(f"Searching for top {k} similar examples...")
        
        # Get similar examples from vector store
        similar_examples = self.vector_store.search(question, k=k)
        
        print(f"✓ Found {len(similar_examples)} similar examples")
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            question, similar_examples, k
        )
        
        print(f"\n{'='*60}")
        print("STEP 4 COMPLETED ✓")
        print(f"{'='*60}\n")
        
        return {
            'similar_examples': similar_examples,
            'reasoning': reasoning,
            'query': question,
            'total_found': len(similar_examples)
        }
    
    def _generate_reasoning(
        self,
        question: str,
        similar_examples: List[Dict],
        k: int
    ) -> str:
        """Generate reasoning for Step 4"""
        reasoning = "STEP 4: SIMILARITY SEARCH\n"
        reasoning += "=" * 50 + "\n\n"
        
        reasoning += f"Query: {question}\n"
        reasoning += f"Requested: {k} similar examples\n"
        reasoning += f"Found: {len(similar_examples)} examples\n\n"
        
        reasoning += "Similar Examples (Ranked by Similarity):\n"
        reasoning += "-" * 50 + "\n"
        
        for i, example in enumerate(similar_examples, 1):
            reasoning += f"\n{i}. Similarity Score: {example.get('similarity_score', 0):.4f}\n"
            reasoning += f"   Database: {example.get('db_id', 'unknown')}\n"
            reasoning += f"   Question: {example.get('question', 'N/A')}\n"
            
            # Show SQL query (truncated if too long)
            sql = example.get('query', 'N/A')
            if len(sql) > 150:
                reasoning += f"   SQL: {sql[:150]}...\n"
            else:
                reasoning += f"   SQL: {sql}\n"
        
        return reasoning