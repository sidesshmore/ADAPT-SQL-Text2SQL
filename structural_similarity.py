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

    def analyze_ground_truth_style(self, sql: str) -> Dict:
        """
        Analyze stylistic patterns common in ground truth SQL
        These patterns improve EM matching
        
        Returns:
            {
                'alias_style': str,  # 'explicit' vs 'implicit' vs 'none'
                'join_order': str,  # 'alphabetical' vs 'dependency' vs 'random'
                'clause_spacing': str,  # 'compact' vs 'spaced'
                'keyword_case': str,  # 'upper' vs 'lower' vs 'mixed'
                'column_selection': str,  # 'explicit' vs 'wildcard'
                'aggregation_naming': str,  # 'function_style' vs 'descriptive'
                'style_vector': List[float]
            }
        """
        sql_upper = sql.upper()
        
        style = {}
        
        # 1. Alias style (AVG(age) vs AVG(age) AS avg_age)
        has_as = ' AS ' in sql_upper
        num_as = sql_upper.count(' AS ')
        num_agg = sum(sql_upper.count(agg) for agg in ['COUNT', 'SUM', 'AVG', 'MAX', 'MIN'])
        
        if num_agg > 0 and num_as >= num_agg:
            style['alias_style'] = 'explicit'
        elif num_as > 0:
            style['alias_style'] = 'implicit'
        else:
            style['alias_style'] = 'none'
        
        # 2. JOIN order pattern
        joins = re.findall(r'JOIN\s+(\w+)', sql_upper)
        if len(joins) > 1:
            alphabetical = all(joins[i] <= joins[i+1] for i in range(len(joins)-1))
            style['join_order'] = 'alphabetical' if alphabetical else 'dependency'
        else:
            style['join_order'] = 'single' if joins else 'none'
        
        # 3. Keyword case
        upper_keywords = sum(1 for kw in ['SELECT', 'FROM', 'WHERE', 'JOIN'] if kw in sql)
        lower_keywords = sum(1 for kw in ['select', 'from', 'where', 'join'] if kw in sql)
        
        if upper_keywords > lower_keywords:
            style['keyword_case'] = 'upper'
        elif lower_keywords > upper_keywords:
            style['keyword_case'] = 'lower'
        else:
            style['keyword_case'] = 'mixed'
        
        # 4. Column selection style
        if 'SELECT *' in sql_upper or 'SELECT  *' in sql_upper:
            style['column_selection'] = 'wildcard'
        else:
            style['column_selection'] = 'explicit'
        
        # 5. Aggregation naming convention
        # Check if aggregations use function-style names (e.g., avg(age)) or descriptive (e.g., average_age)
        agg_matches = re.findall(r'(COUNT|SUM|AVG|MAX|MIN)\s*\([^)]+\)(?:\s+AS\s+(\w+))?', sql_upper)
        descriptive_aliases = sum(1 for _, alias in agg_matches if alias and len(alias) > 8)
        
        if descriptive_aliases > 0:
            style['aggregation_naming'] = 'descriptive'
        else:
            style['aggregation_naming'] = 'function_style'
        
        # 6. Clause spacing (compact vs spaced)
        avg_line_length = len(sql) / (sql.count('\n') + 1)
        style['clause_spacing'] = 'compact' if avg_line_length > 80 else 'spaced'
        
        # Create style vector for similarity comparison
        style['style_vector'] = self._create_style_vector(style)
        
        return style

    def _create_style_vector(self, style: Dict) -> List[float]:
        """Convert style features to numerical vector"""
        vector = []
        
        # Alias style (one-hot)
        for val in ['explicit', 'implicit', 'none']:
            vector.append(1.0 if style['alias_style'] == val else 0.0)
        
        # JOIN order (one-hot)
        for val in ['alphabetical', 'dependency', 'single', 'none']:
            vector.append(1.0 if style['join_order'] == val else 0.0)
        
        # Keyword case (one-hot)
        for val in ['upper', 'lower', 'mixed']:
            vector.append(1.0 if style['keyword_case'] == val else 0.0)
        
        # Column selection (binary)
        vector.append(1.0 if style['column_selection'] == 'explicit' else 0.0)
        
        # Aggregation naming (binary)
        vector.append(1.0 if style['aggregation_naming'] == 'function_style' else 0.0)
        
        # Clause spacing (binary)
        vector.append(1.0 if style['clause_spacing'] == 'compact' else 0.0)
        
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

    def calculate_style_similarity(
        self,
        style1: Dict,
        style2: Dict
    ) -> float:
        """
        Calculate stylistic similarity between two SQL queries
        Returns score in [0, 1] where 1 = identical style
        """
        vec1 = style1['style_vector']
        vec2 = style2['style_vector']
        
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
        preliminary_sql: str,
        semantic_weight: float = 0.5,
        structural_weight: float = 0.3,
        style_weight: float = 0.2
    ) -> List[Dict]:
        """
        Rerank examples using DAIL-SQL approach with style similarity
        
        Args:
            examples: Examples with 'similarity_score' and 'query' fields
            preliminary_sql: Preliminary SQL from Step 3
            semantic_weight: Weight for semantic similarity (default: 0.5)
            structural_weight: Weight for structural similarity (default: 0.3)
            style_weight: Weight for ground-truth style similarity (default: 0.2)
            
        Returns:
            Reranked examples with updated scores
        """
        # Analyze preliminary SQL
        target_structure = self.analyzer.analyze_structure(preliminary_sql)
        target_style = self.analyzer.analyze_ground_truth_style(preliminary_sql)
        
        # Calculate similarities for each example
        for example in examples:
            example_sql = example.get('query', '')
            
            if not example_sql:
                example['structural_similarity'] = 0.0
                example['style_similarity'] = 0.0
                example['combined_score'] = example.get('similarity_score', 0.0)
                continue
            
            # Analyze example
            example_structure = self.analyzer.analyze_structure(example_sql)
            example_style = self.analyzer.analyze_ground_truth_style(example_sql)
            
            # Calculate structural similarity
            structural_sim = self.analyzer.calculate_similarity(
                target_structure, example_structure
            )
            
            # Calculate style similarity
            style_sim = self.analyzer.calculate_style_similarity(
                target_style, example_style
            )
            
            example['structural_similarity'] = structural_sim
            example['style_similarity'] = style_sim
            
            # Calculate combined score (DAIL-SQL approach)
            semantic_score = example.get('similarity_score', 0.0)
            combined_score = (
                semantic_weight * semantic_score +
                structural_weight * structural_sim +
                style_weight * style_sim
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
        semantic_weight: float = 0.5,
        structural_weight: float = 0.3,
        style_weight: float = 0.2
    ) -> List[Dict]:
        """
        Convenience function for enhanced example selection (DAIL-SQL approach)
        
        Args:
            examples: Examples from vector search (Step 4)
            preliminary_sql: Preliminary SQL from Step 3
            semantic_weight: Weight for semantic similarity (default: 0.5)
            structural_weight: Weight for structural similarity (default: 0.3)
            style_weight: Weight for ground-truth style similarity (default: 0.2)
            
        Returns:
            Reranked examples with combined scores
        """
        selector = EnhancedExampleSelector(semantic_weight, structural_weight)
        # Pass style_weight to rerank_examples
        return selector.rerank_examples(examples, preliminary_sql, 
                                        semantic_weight, structural_weight, style_weight)


