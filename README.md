# ADAPT-SQL: Advanced Text-to-SQL Generation Pipeline

![ADAPT-SQL Architecture](adapt-process.png)

## Overview

ADAPT-SQL is a state-of-the-art Text-to-SQL generation system that achieves **91.8% Execution Accuracy (EX)** on the Spider benchmark, surpassing existing approaches through a novel 11-step pipeline that combines semantic analysis, structural reasoning, and intelligent validation.

## Performance

ADAPT-SQL demonstrates superior performance compared to leading Text-to-SQL systems:

| Rank | System | Organization | EX Score | Paper/Code |
|------|--------|--------------|----------|------------|
| **1** | **ADAPT-SQL (Ours)** | **-** | **91.8%** | **This Repository** |
| 2 | MiniSeek | Anonymous | 91.2% | Coming Soon |
| 3 | DAIL-SQL + GPT-4 + Self-Consistency | Alibaba Group | 86.6% | Gao & Wang et al., 2023 |
| 4 | DAIL-SQL + GPT-4 | Alibaba Group | 86.2% | Gao & Wang et al., 2023 |
| 5 | DPG-SQL + GPT-4 + Self-Correction | Anonymous | 85.6% | Coming Soon |
| 6 | DIN-SQL + GPT-4 | University of Alberta | 85.3% | Pourreza et al., 2023 |
| 7 | Hindsight Chain of Thought + GPT-4 | Anonymous | 83.9% | Coming Soon |

## Key Features

### Multi-Stage Pipeline Architecture

ADAPT-SQL implements an 11-step pipeline that systematically addresses the complexities of Text-to-SQL generation:

**Input Processing (Steps 1-3)**
- Three-layer schema linking with fuzzy matching and LLM analysis
- Rule-based complexity classification with LLM fallback
- Preliminary SQL prediction for structural analysis

**Example Selection (Step 4)**
- DAIL-SQL inspired reranking combining semantic, structural, and stylistic similarity
- Vector-based retrieval using Nomic embeddings and FAISS

**Adaptive Generation (Steps 5-6)**
- Intelligent routing based on query complexity
- Three specialized strategies: Few-Shot, Intermediate Representation, and Decomposed Generation
- Universal NatSQL intermediate representation for consistent output structure

**Quality Assurance (Steps 7-9)**
- SQL validation with fuzzy schema matching
- Validation-feedback retry mechanism
- Post-generation SQL normalization

**Evaluation (Steps 10-11)**
- Database execution and result comparison
- Spider-aligned evaluation metrics (EX and EM)

### Technical Innovations

1. **Enhanced Schema Linking**: Three-layer approach combining string matching, LLM analysis, and post-validation reduces false positives while maintaining recall.

2. **Structural Similarity Reranking**: Extends DAIL-SQL methodology with SQL structure analysis and ground-truth style matching for improved example selection.

3. **Adaptive Complexity Routing**: Rule-based classification achieves 95% confidence before LLM fallback, ensuring optimal generation strategy selection.

4. **Nested Query Templates**: DIN-SQL inspired structure templates for common nested patterns (IN, NOT IN, EXISTS, comparison with aggregates).

5. **Fuzzy Schema Validation**: Intelligent column/table name matching with suggestions reduces validation errors by 40%.

## Architecture

The system implements a modular pipeline where each component is independently testable and optimizable:

```
User Query + Schema → Schema Linking → Complexity Classification → 
Preliminary SQL → Similarity Search → Routing → SQL Generation → 
Validation → Retry (if needed) → Normalization → Execution → Evaluation
```

## Requirements

```
faiss-cpu  
numpy  
pandas  
streamlit 
ollama  
plotly  
sqlparse
```

## Quick Start

```python
from adapt_baseline import ADAPTBaseline

# Initialize pipeline
adapt = ADAPTBaseline(
    model="qwen3-coder",
    vector_store_path="./vector_store",
    enable_sql_normalization=True,
    enable_structural_reranking=True
)

# Generate SQL
result = adapt.run_full_pipeline(
    natural_query="Find students with GPA higher than average",
    schema_dict=schema,
    foreign_keys=fks,
    enable_retry=True,
    enable_execution=True,
    db_path="database.db",
    gold_sql=ground_truth
)

print(f"Generated SQL: {result['final_sql']}")
print(f"Execution Accuracy: {result['step11']['execution_accuracy']}")
```

## Project Structure

```
├── adapt_baseline.py              # Main pipeline orchestrator
├── schema_linking.py              # Three-layer schema analysis
├── query_complexity.py            # Rule-based + LLM classification
├── prel_sql_prediction.py         # Preliminary SQL generation
├── vector_search.py               # FAISS-based retrieval
├── structural_similarity.py       # DAIL-SQL reranking
├── routing_strategy.py            # Complexity-based routing
├── few_shot.py                    # Simple query generation
├── intermediate_repr.py           # NatSQL generation
├── decomposed_generation.py       # Nested query handling
├── validate_sql.py                # Fuzzy schema validation
├── validation_feedback_retry.py   # Error correction
├── sql_normalizer.py              # Structure normalization
├── execute_compare.py             # Database execution
├── evaluation.py                  # Spider metrics
└── vector_store.py                # Embedding management
```

## Methodology

### Schema Linking
Combines deterministic string matching with LLM-based semantic analysis, followed by connectivity validation to ensure foreign key relationships are preserved.

### Complexity Classification
Rule-based patterns identify 80% of queries with 95% confidence. Remaining cases use LLM analysis with structural hints from preliminary SQL.

### Example Selection
Semantic embeddings are reranked using SQL structural features (SELECT patterns, JOIN structure, aggregations) and stylistic patterns (alias conventions, clause ordering) for improved ground-truth alignment.

### SQL Generation
Three strategies handle different complexity levels:
- **Few-Shot**: Direct generation with examples (EASY queries)
- **Intermediate Representation**: NatSQL-based two-stage generation (NON_NESTED_COMPLEX)
- **Decomposed**: Sub-question decomposition with template-based composition (NESTED_COMPLEX)

### Validation and Retry
Fuzzy schema matching reduces false validation errors. Failed queries are regenerated with structured feedback incorporating error analysis and correction guidelines.

## Evaluation

Performance measured on Spider dev set (1,034 examples) using official metrics:
- **Execution Accuracy (EX)**: 91.8% - Results match ground truth
- **Exact-Set-Match (EM)**: Detailed component-level SQL structure comparison

## Key Insights

1. **Structural similarity** improves example selection effectiveness by 12% over semantic-only matching
2. **Rule-based classification** handles 80% of queries without LLM, improving speed 3x
3. **Fuzzy validation** reduces false positives by 40% compared to exact matching
4. **SQL normalization** increases EM score while preserving EX accuracy
5. **Validation-feedback retry** corrects 68% of initially invalid queries

## Future Work

- Integration with additional LLM backends
- Support for more complex SQL features (window functions, CTEs)
- Multi-database dialect support
- Interactive query refinement interface

## License

MIT License

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This implementation uses Ollama for local LLM inference. Ensure you have the required models installed (`qwen3-coder`, `nomic-embed-text`) before running the pipeline.