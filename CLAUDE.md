# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ADAPT-SQL is a Text-to-SQL generation system achieving 91.8% Execution Accuracy on the Spider benchmark. It implements an 11-step pipeline that transforms natural language queries into SQL statements through schema linking, complexity classification, adaptive generation strategies, and validation with retry mechanisms.

**Core Architecture**: Natural language → Schema Linking → Complexity Classification → Preliminary SQL → Example Selection → Routing → SQL Generation (3 strategies) → Validation → Retry → Normalization → Execution → Evaluation

## Development Environment

### Setup
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Required Ollama models (must be installed separately)
ollama pull qwen3-coder       # Primary LLM for generation
ollama pull nomic-embed-text  # Embeddings for vector search
```

### Running the Application
```bash
# Interactive Streamlit UI (main interface)
streamlit run ui/app.py

# Batch processing UI (for dataset evaluation)
streamlit run ui/pages/batch_processing.py

# Multi-model comparison UI
streamlit run ui/pages/multimodel.py
```

### Building Vector Store
Before first use, build the FAISS index from Spider training data:
```bash
python utils/vector_store.py
```
This creates `vector_store/` with embeddings for example selection (Step 4).

## Project Structure

```
adapt-sql/
├── core/                      # Core pipeline orchestrator
│   └── adapt_baseline.py      # Main pipeline coordinator (Steps 1-11)
├── pipeline/                  # Individual pipeline steps
│   ├── schema_linking.py      # Step 1: Schema linking
│   ├── query_complexity.py    # Step 2: Complexity classification
│   ├── prel_sql_prediction.py # Step 3: Preliminary SQL
│   ├── vector_search.py       # Step 4: Example retrieval
│   ├── routing_strategy.py    # Step 5: Route to generation strategy
│   ├── few_shot.py            # Step 6a: Simple queries
│   ├── intermediate_repr.py   # Step 6b: NatSQL generation
│   ├── decomposed_generation.py # Step 6c: Nested queries
│   ├── validate_sql.py        # Step 7: SQL validation
│   ├── validation_feedback_retry.py # Step 8: Error retry
│   ├── sql_normalizer.py      # Step 9: Normalization
│   ├── execute_compare.py     # Step 10: Execution
│   └── evaluation.py          # Step 11: Metrics
├── utils/                     # Utility modules
│   ├── vector_store.py        # FAISS index management
│   ├── structural_similarity.py # SQL structure analysis
│   ├── fuzzy_schema_validator.py # Fuzzy name matching
│   └── rule_based_complexity.py  # Rule-based classification
├── ui/                        # Streamlit interfaces
│   ├── app.py                 # Main UI
│   ├── display_utils.py       # UI display functions
│   ├── batch_utils.py         # Batch processing utilities
│   ├── enhanced_retry_engine.py # Advanced retry engine
│   └── pages/
│       ├── batch_processing.py # Dataset evaluation UI
│       └── multimodel.py       # Multi-model comparison UI
├── data/                      # Spider dataset
├── vector_store/              # FAISS index files
├── RESULTS/                   # Evaluation results
└── venv/                      # Virtual environment
```

## Pipeline Architecture

The system follows a strict 11-step sequential pipeline orchestrated by `core/adapt_baseline.py`:

### Steps 1-3: Input Analysis
- **Step 1** (`pipeline/schema_linking.py`): Three-layer schema linking (string matching → LLM analysis → connectivity validation)
- **Step 2** (`pipeline/query_complexity.py`): Classify as EASY/NON_NESTED_COMPLEX/NESTED_COMPLEX using rule-based patterns + LLM fallback
- **Step 3** (`pipeline/prel_sql_prediction.py`): Generate preliminary SQL for structural analysis

### Step 4: Example Selection
- **Step 4** (`pipeline/vector_search.py` + `utils/structural_similarity.py`):
  - Semantic search via FAISS (Nomic embeddings)
  - Structural reranking based on SQL patterns (DAIL-SQL inspired)

### Steps 5-6: Adaptive Generation
- **Step 5** (`pipeline/routing_strategy.py`): Route to strategy based on complexity
- **Step 6**: Generate SQL using one of three strategies:
  - `pipeline/few_shot.py`: Direct generation for EASY queries
  - `pipeline/intermediate_repr.py`: NatSQL intermediate representation for NON_NESTED_COMPLEX
  - `pipeline/decomposed_generation.py`: Sub-question decomposition for NESTED_COMPLEX

### Steps 7-9: Quality Assurance
- **Step 7** (`pipeline/validate_sql.py`): Fuzzy schema validation with suggestions
- **Step 8** (`pipeline/validation_feedback_retry.py`): Regenerate with structured error feedback
- **Step 9** (`pipeline/sql_normalizer.py`): Post-generation normalization (formatting, alias consistency)

### Steps 10-11: Evaluation
- **Step 10** (`pipeline/execute_compare.py`): Execute SQL against SQLite databases
- **Step 11** (`pipeline/evaluation.py`): Calculate Spider metrics (EX, EM)

## Key Data Structures

### Schema Dictionary Format
```python
{
    "table_name": [
        {"column_name": "id", "type": "INTEGER"},
        {"column_name": "name", "type": "TEXT"}
    ]
}
```

### Foreign Keys Format
```python
[
    {
        "source_table": "orders",
        "source_column": "customer_id",
        "target_table": "customers",
        "target_column": "id"
    }
]
```

### Pipeline Result Structure
All steps return dictionaries that accumulate through the pipeline. Final result contains:
```python
{
    "step1": {...},  # Schema linking results
    "step2": {...},  # Complexity classification
    # ... through step11
    "final_sql": "SELECT ...",
    "retry_count": int,
    "retry_history": [...]
}
```

## Dataset Structure

### Spider Dataset Location
- **Dev set**: `data/spider/dev.json` (1,034 examples for evaluation)
- **Train set**: `data/spider/spider_data/train_spider.json`
- **Databases**: `data/spider/spider_data/database/{db_name}/{db_name}.sqlite`

Each example in Spider JSON:
```python
{
    "db_id": "database_name",
    "question": "natural language query",
    "query": "ground truth SQL",
    "question_toks": [...],
    "query_toks": [...]
}
```

## Important Implementation Details

### LLM Integration
All modules use Ollama for local inference. The pattern is:
```python
response = ollama.chat(
    model=self.model,
    messages=[{"role": "user", "content": prompt}]
)
result = response['message']['content']
```

### Complexity Classification Logic
Rule-based patterns identify 80% of queries with 95% confidence before LLM fallback:
- **EASY**: Single table, simple WHERE, basic JOINs
- **NON_NESTED_COMPLEX**: Multiple tables, aggregations, GROUP BY, but no subqueries
- **NESTED_COMPLEX**: Subqueries in WHERE/SELECT/FROM, correlated subqueries, EXISTS/IN

### Validation with Fuzzy Matching
`utils/fuzzy_schema_validator.py` provides intelligent name matching:
- Handles common variations (snake_case vs camelCase, plurals)
- Generates suggestions for misspelled columns/tables
- Reduces false positives by 40% vs exact matching

### Vector Store Architecture
- Nomic embeddings (768-dimensional)
- FAISS IndexFlatL2 for exact nearest neighbor search
- Metadata stored in parallel `examples.json`
- Reranking applied after retrieval using structural similarity

## Testing and Evaluation

### Running Evaluations
Use the batch processing interface to evaluate on datasets:
1. Load `data/spider/dev.json` in Streamlit UI
2. Configure pipeline options (enable/disable normalization, structural reranking)
3. Results saved to `RESULTS/` as CSV and PDF

### Key Metrics
- **Execution Accuracy (EX)**: Whether generated SQL produces same results as ground truth
- **Exact-Set-Match (EM)**: Whether generated SQL exactly matches ground truth (after normalization)

Current benchmarks:
- Dev set (1,034 examples): 91.8% EX, 35.0% EM
- Test set (1,000 examples): Results in `RESULTS/` directory

## Module Dependencies

**Core orchestrator**: `core/adapt_baseline.py` → All pipeline modules

**Pipeline flow**:
- `pipeline/schema_linking.py` → `pipeline/query_complexity.py` → `pipeline/prel_sql_prediction.py`
- `utils/vector_store.py` → `pipeline/vector_search.py` → `utils/structural_similarity.py`
- `pipeline/routing_strategy.py` → {`pipeline/few_shot.py`, `pipeline/intermediate_repr.py`, `pipeline/decomposed_generation.py`}
- `pipeline/validate_sql.py` (uses `utils/fuzzy_schema_validator.py`) → `pipeline/validation_feedback_retry.py`
- `pipeline/sql_normalizer.py` → `pipeline/execute_compare.py` → `pipeline/evaluation.py`

**UI components**:
- `ui/app.py` + `ui/display_utils.py`: Main interactive interface
- `ui/pages/batch_processing.py` + `ui/batch_utils.py`: Dataset evaluation
- `ui/pages/multimodel.py`: Multi-model comparison
- `ui/enhanced_retry_engine.py`: Extended retry with multiple strategies

## Common Patterns

### Adding a New Generation Strategy
1. Create module in `pipeline/` (e.g., `pipeline/new_strategy.py`)
2. Implement class with `generate(natural_query, schema, foreign_keys, examples)` method
3. Add to `GenerationStrategy` enum in `pipeline/routing_strategy.py`
4. Update routing logic in `RoutingStrategy.route_to_strategy()`
5. Import and instantiate in `core/adapt_baseline.py.__init__()`
6. Add case in `ADAPTBaseline.run_step6_sql_generation()`

### Modifying Validation Logic
Primary validation in `pipeline/validate_sql.py` (schema compliance)
Fuzzy matching in `utils/fuzzy_schema_validator.py` (name similarity, suggestions)
Retry generation in `pipeline/validation_feedback_retry.py` (error-driven regeneration)

### Changing LLM Model
Set via `ADAPTBaseline(model="your-ollama-model")`. Must be available in Ollama. Different models may require prompt adjustments in individual modules.

## Performance Considerations

- Vector search: Pre-build FAISS index to avoid runtime embedding overhead
- Database execution: 30s timeout per query (configurable)
- Batch processing: Checkpoints saved every N examples (see `ui/batch_utils.py`)
- LLM inference: Local Ollama = no API costs but requires compute
- Structural reranking: ~20% slower than semantic-only but +12% accuracy

## Import Pattern

All modules use absolute imports from the project root. Examples:
```python
# In pipeline modules
from utils.vector_store import SQLVectorStore
from utils.fuzzy_schema_validator import FuzzySchemaValidator

# In UI modules (add parent to path first)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.adapt_baseline import ADAPTBaseline
from ui.batch_utils import save_checkpoint
```

## ASU SOL (HPC) Deployment

The project runs on ASU's SOL cluster (Rubik). Key details:

### Paths on SOL
- Project: `/scratch/smore123/ADAPT-SQL-Text2SQL/`
- Ollama binary: `/scratch/smore123/ollama_install/bin/ollama`
- Ollama models: `/scratch/smore123/ollama_models/`
- Spider data: `/scratch/smore123/ADAPT-SQL-Text2SQL/data/spider/`

### SOL Setup (run `setup_sol.sh` on a GPU compute node)
```bash
# Request GPU compute node first
srun --pty --partition=public --gres=gpu:a100:1 --mem=32G --cpus-per-task=4 --time=4:00:00 /bin/bash

# Then run the setup script
bash /scratch/smore123/setup_sol.sh
```

The setup script handles: HOME env, CUDA module, Ollama server startup, model verification.

### Ollama on SOL
- System Ollama version: 0.13.2 (too old, can't run Gemma 4)
- Custom Ollama: v0.22.0 installed at `/scratch/smore123/ollama_install/`
- Must use the install directory binary (not a copy) so CUDA libs are found
- Server port: **11437** (11434 = system old server, 11435/11436 may be taken)
- Always start with: `OLLAMA_MODELS=/scratch/smore123/ollama_models OLLAMA_HOST=127.0.0.1:11437 ollama serve &`
- Models pulled: `gemma4` (9.6GB), `nomic-embed-text`

### Running the UI on SOL
```bash
# On SOL compute node
export OLLAMA_HOST=http://127.0.0.1:11437
streamlit run ui/app.py --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false

# Access via OOD at: https://ood06.sol.rc.asu.edu
# Or SSH tunnel from Mac: ssh -L 8501:localhost:8501 smore123@sol.rc.asu.edu
```

In the Streamlit sidebar, set **Ollama Host** to `http://127.0.0.1:11437`.

### Critical: Ollama Host Configuration
`ollama.chat()` and `ollama.embeddings()` are patched at module level in `core/adapt_baseline.py` to read `OLLAMA_HOST` dynamically at call time. This is necessary because the ollama Python library binds its default client at import time — setting `OLLAMA_HOST` after import has no effect without this patch. Do not remove this patch.

### Known Issues Fixed
- `from structural_similarity import` → must be `from utils.structural_similarity import`
- `from validate_sql import` → must be `from pipeline.validate_sql import`
- UI data paths default to `/scratch/smore123/ADAPT-SQL-Text2SQL/data/spider/`

## License

MIT License - See LICENSE file
