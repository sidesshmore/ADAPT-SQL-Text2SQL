"""
STEP 4: Similarity Search
Returns similar examples with their questions, SQL queries, and similarity scores
"""
import re as _re
from typing import List, Dict
from utils.vector_store import SQLVectorStore


class DualSimilaritySelector:
    def __init__(self, vector_store: SQLVectorStore):
        """Initialize with loaded vector store"""
        self.vector_store = vector_store
    
    def search_similar_examples(
        self,
        question: str,
        k: int = 10,
        sql_hint: str = None,
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
        similar_examples = self.vector_store.search(question, k=k, sql_hint=sql_hint)

        print(f"✓ Found {len(similar_examples)} similar examples")

        # SAFE-SQL: Reasoning path matching — boost examples whose SQL complexity
        # signature matches the preliminary SQL (sql_hint from Step 3).
        # Prevents homogeneous example sets when the question needs GROUP BY/HAVING/etc.
        if sql_hint and similar_examples:
            target_sig = self._complexity_signature(sql_hint)
            if target_sig:
                for ex in similar_examples:
                    ex_sig = self._complexity_signature(ex.get('query', ex.get('sql', '')))
                    overlap = len(target_sig & ex_sig) / len(target_sig)
                    ex['combined_score'] = (
                        ex.get('combined_score', ex.get('similarity_score', 0)) + 0.05 * overlap
                    )
                similar_examples.sort(key=lambda x: x.get('combined_score', 0), reverse=True)

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
    
    @staticmethod
    def _complexity_signature(sql: str) -> set:
        """Extract SQL complexity features for SAFE-SQL reasoning path matching."""
        sql_up = sql.upper()
        sig = set()
        if _re.search(r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql_up): sig.add('AGG')
        if 'GROUP BY' in sql_up: sig.add('GROUP')
        if 'HAVING' in sql_up: sig.add('HAVING')
        if sql_up.count('SELECT') > 1: sig.add('SUBQUERY')
        if _re.search(r'\b(EXCEPT|UNION|INTERSECT)\b', sql_up): sig.add('SETOP')
        if 'DISTINCT' in sql_up: sig.add('DISTINCT')
        if len(_re.findall(r'\bJOIN\b', sql_up)) >= 2: sig.add('MULTI_JOIN')
        return sig

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
        
        # Check reranking method
        has_structural = any('structural_similarity' in ex for ex in similar_examples)
        has_style = any('style_similarity' in ex for ex in similar_examples)
        
        if has_structural and has_style:
            reasoning += "Ranking Method: DAIL-SQL (Semantic + Structural + Style)\n"
            reasoning += "  - Semantic Weight: 50%\n"
            reasoning += "  - Structural Weight: 30%\n"
            reasoning += "  - Style Weight: 20%\n\n"
        elif has_structural:
            reasoning += "Ranking Method: Combined (Semantic + Structural)\n"
            reasoning += "  - Semantic Weight: 70%\n"
            reasoning += "  - Structural Weight: 30%\n\n"
        else:
            reasoning += "Ranking Method: Semantic Similarity Only\n\n"
        
        reasoning += "Similar Examples (Ranked by Similarity):\n"
        reasoning += "-" * 50 + "\n"
        
        for i, example in enumerate(similar_examples, 1):
            reasoning += f"\n{i}. "
            
            if has_structural and has_style:
                sem_score = example.get('similarity_score', 0)
                struct_score = example.get('structural_similarity', 0)
                style_score = example.get('style_similarity', 0)
                combined_score = example.get('combined_score', 0)
                
                reasoning += f"Combined: {combined_score:.4f}\n"
                reasoning += f"   (Sem: {sem_score:.4f}, Struct: {struct_score:.4f}, Style: {style_score:.4f})\n"
            elif has_structural:
                sem_score = example.get('similarity_score', 0)
                struct_score = example.get('structural_similarity', 0)
                combined_score = example.get('combined_score', 0)
                
                reasoning += f"Combined: {combined_score:.4f} "
                reasoning += f"(Sem: {sem_score:.4f}, Struct: {struct_score:.4f})\n"
            else:
                reasoning += f"Similarity Score: {example.get('similarity_score', 0):.4f}\n"
            
            reasoning += f"   Database: {example.get('db_id', 'unknown')}\n"
            reasoning += f"   Question: {example.get('question', 'N/A')}\n"
            
            sql = example.get('query', 'N/A')
            if len(sql) > 150:
                reasoning += f"   SQL: {sql[:150]}...\n"
            else:
                reasoning += f"   SQL: {sql}\n"
        
        return reasoning