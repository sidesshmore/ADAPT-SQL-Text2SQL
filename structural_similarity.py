"""
Structural Similarity for Enhanced Example Selection
Combines semantic similarity with SQL structural similarity

Improves both EM and EX by selecting examples with similar:
1. SQL structure (SELECT patterns, JOIN patterns, aggregations)
2. Query complexity (number of tables, subqueries, etc.)
"""
import re
from typing import Dict, List, Tuple
from collections import Counter


class SQLStructuralAnalyzer:
    def __init__(self):
        """Initialize structural analyzer"""
        pass
    
    def analyze_structure(self, sql: str) -> Dict:
        """
        Analyze SQL structure for similarity comparison
        
        Returns:
            {
                'select_pattern': str,
                'join_pattern': str,
                'num_tables': int,
                'num_joins': int,
                'has_subquery': bool,
                'aggregations': List[str],
                'has_group_by': bool,
                'has_having': bool,
                'has_order_by': bool,
                'has_limit': bool,
                'where_complexity': int,
                'structure_vector': List[float]
            }
        """
        sql_upper = sql.upper()
        
        # Basic structure
        structure = {
            'select_pattern': self._extract_select_pattern(sql),
            'join_pattern': self._extract_join_pattern(sql),
            'num_tables': self._count_tables(sql),
            'num_joins': sql_upper.count('JOIN'),
            'has_subquery': sql_upper.count('SELECT') > 1,
            'aggregations': self._extract_aggregations(sql),
            'has_group_by': 'GROUP BY' in sql_upper,
            'has_having': 'HAVING' in sql_upper,
            'has_order_by': 'ORDER BY' in sql_upper,
            'has_limit': 'LIMIT' in sql_upper,
            'where_complexity': self._calculate_where_complexity(sql)
        }
        
        # Create structure vector for similarity calculation
        structure['structure_vector'] = self._create_structure_vector(structure)
        
        return structure
    
    def _extract_select_pattern(self, sql: str) -> str:
        """
        Extract SELECT pattern
        Examples:
            - "SINGLE" (single column)
            - "MULTI" (multiple columns)
            - "AGG" (aggregation only)
            - "MIXED" (columns + aggregations)
        """
        sql_upper = sql.upper()
        
        # Extract SELECT clause
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper, re.DOTALL)
        if not select_match:
            return "UNKNOWN"
        
        select_clause = select_match.group(1)
        
        # Check for aggregations
        has_agg = any(agg in select_clause for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN'])
        
        # Count columns (rough estimate)
        num_cols = select_clause.count(',') + 1
        
        if has_agg and num_cols == 1:
            return "AGG"
        elif has_agg and num_cols > 1:
            return "MIXED"
        elif num_cols == 1:
            return "SINGLE"
        else:
            return "MULTI"
    
    def _extract_join_pattern(self, sql: str) -> str:
        """
        Extract JOIN pattern
        Examples:
            - "NONE" (no joins)
            - "SIMPLE" (1 join)
            - "MULTI" (2+ joins)
            - "COMPLEX" (nested or multiple types)
        """
        sql_upper = sql.upper()
        
        num_joins = sql_upper.count('JOIN')
        
        if num_joins == 0:
            return "NONE"
        elif num_joins == 1:
            return "SIMPLE"
        elif num_joins >= 2:
            # Check for different join types
            has_left = 'LEFT JOIN' in sql_upper
            has_right = 'RIGHT JOIN' in sql_upper
            has_inner = 'INNER JOIN' in sql_upper
            
            join_types = sum([has_left, has_right, has_inner])
            
            if join_types > 1:
                return "COMPLEX"
            else:
                return "MULTI"
        
        return "UNKNOWN"
    
    def _count_tables(self, sql: str) -> int:
        """Count number of tables in query"""
        sql_upper = sql.upper()
        
        # Count FROM tables
        from_matches = len(re.findall(r'FROM\s+(\w+)', sql_upper))
        
        # Count JOIN tables
        join_matches = len(re.findall(r'JOIN\s+(\w+)', sql_upper))
        
        return from_matches + join_matches
    
    def _extract_aggregations(self, sql: str) -> List[str]:
        """Extract aggregation functions used"""
        sql_upper = sql.upper()
        
        aggregations = []
        for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN']:
            if agg in sql_upper:
                aggregations.append(agg)
        
        return aggregations
    
    def _calculate_where_complexity(self, sql: str) -> int:
        """
        Calculate WHERE clause complexity
        Higher = more complex
        """
        sql_upper = sql.upper()
        
        if 'WHERE' not in sql_upper:
            return 0
        
        complexity = 0
        
        # Count AND/OR
        complexity += sql_upper.count(' AND ')
        complexity += sql_upper.count(' OR ')
        
        # Count comparisons
        complexity += sql_upper.count('=')
        complexity += sql_upper.count('>')
        complexity += sql_upper.count('<')
        complexity += sql_upper.count('LIKE')
        complexity += sql_upper.count('IN')
        
        # Count subqueries in WHERE
        where_match = re.search(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|$)', sql_upper, re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            complexity += where_clause.count('SELECT') * 3  # Subqueries add more complexity
        
        return complexity
    
    def _create_structure_vector(self, structure: Dict) -> List[float]:
        """
        Create numerical vector for similarity calculation
        """
        vector = []
        
        # Normalize values to [0, 1] range
        
        # Number of tables (normalize by max 10)
        vector.append(min(structure['num_tables'] / 10.0, 1.0))
        
        # Number of joins (normalize by max 5)
        vector.append(min(structure['num_joins'] / 5.0, 1.0))
        
        # Has subquery (binary)
        vector.append(1.0 if structure['has_subquery'] else 0.0)
        
        # Number of aggregations (normalize by max 5)
        vector.append(min(len(structure['aggregations']) / 5.0, 1.0))
        
        # Has GROUP BY (binary)
        vector.append(1.0 if structure['has_group_by'] else 0.0)
        
        # Has HAVING (binary)
        vector.append(1.0 if structure['has_having'] else 0.0)
        
        # Has ORDER BY (binary)
        vector.append(1.0 if structure['has_order_by'] else 0.0)
        
        # WHERE complexity (normalize by max 20)
        vector.append(min(structure['where_complexity'] / 20.0, 1.0))
        
        # SELECT pattern (one-hot encoding)
        select_patterns = ['SINGLE', 'MULTI', 'AGG', 'MIXED']
        for pattern in select_patterns:
            vector.append(1.0 if structure['select_pattern'] == pattern else 0.0)
        
        # JOIN pattern (one-hot encoding)
        join_patterns = ['NONE', 'SIMPLE', 'MULTI', 'COMPLEX']
        for pattern in join_patterns:
            vector.append(1.0 if structure['join_pattern'] == pattern else 0.0)
        
        return vector
    
    def calculate_similarity(
        self,
        structure1: Dict,
        structure2: Dict
    ) -> float:
        """
        Calculate structural similarity between two SQL structures
        Returns score in [0, 1] where 1 = identical structure
        """
        vec1 = structure1['structure_vector']
        vec2 = structure2['structure_vector']
        
        if len(vec1) != len(vec2):
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        
        return max(0.0, min(1.0, similarity))


class EnhancedExampleSelector:
    def __init__(self, semantic_weight: float = 0.7, structural_weight: float = 0.3):
        """
        Initialize enhanced selector
        
        Args:
            semantic_weight: Weight for semantic similarity (default: 0.7)
            structural_weight: Weight for structural similarity (default: 0.3)
        """
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight
        self.analyzer = SQLStructuralAnalyzer()
    
    def rerank_examples(
        self,
        examples: List[Dict],
        preliminary_sql: str
    ) -> List[Dict]:
        """
        Rerank examples using combined semantic + structural similarity
        
        Args:
            examples: Examples with 'similarity_score' and 'query' fields
            preliminary_sql: Preliminary SQL from Step 3
            
        Returns:
            Reranked examples with updated 'combined_score' field
        """
        # Analyze preliminary SQL structure
        target_structure = self.analyzer.analyze_structure(preliminary_sql)
        
        # Calculate structural similarity for each example
        for example in examples:
            example_sql = example.get('query', '')
            
            if not example_sql:
                example['structural_similarity'] = 0.0
                example['combined_score'] = example.get('similarity_score', 0.0)
                continue
            
            # Analyze example structure
            example_structure = self.analyzer.analyze_structure(example_sql)
            
            # Calculate structural similarity
            structural_sim = self.analyzer.calculate_similarity(
                target_structure, example_structure
            )
            
            example['structural_similarity'] = structural_sim
            
            # Calculate combined score
            semantic_score = example.get('similarity_score', 0.0)
            combined_score = (
                self.semantic_weight * semantic_score +
                self.structural_weight * structural_sim
            )
            
            example['combined_score'] = combined_score
        
        # Sort by combined score (descending)
        examples.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
        
        return examples


# ============================================================================
# Integration Helpers
# ============================================================================

def enhance_example_selection(
    examples: List[Dict],
    preliminary_sql: str,
    semantic_weight: float = 0.7,
    structural_weight: float = 0.3
) -> List[Dict]:
    """
    Convenience function for enhanced example selection
    
    Args:
        examples: Examples from vector search (Step 4)
        preliminary_sql: Preliminary SQL from Step 3
        semantic_weight: Weight for semantic similarity
        structural_weight: Weight for structural similarity
        
    Returns:
        Reranked examples with combined scores
    """
    selector = EnhancedExampleSelector(semantic_weight, structural_weight)
    return selector.rerank_examples(examples, preliminary_sql)


