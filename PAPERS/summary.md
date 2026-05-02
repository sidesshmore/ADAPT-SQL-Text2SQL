# Text-to-SQL Research: Comprehensive Literature Summary
### 15 Papers × ADAPT-SQL Cross-Analysis

> **Reading time saved**: ~120 hours of paper reading distilled here.  
> **Scope**: 15 papers covering schema linking, multi-agent systems, retrieval, RL training, and robustness.  
> **Your system**: ADAPT-SQL — 11-step pipeline, local Ollama (qwen3-coder), actual **81.5% EX** on Spider dev (1,034 queries) after Priority-1 fixes.

---

## Table of Contents

1. [Quick Reference — All 15 Papers](#1-quick-reference--all-15-papers)
2. [Results Leaderboards](#2-results-leaderboards)
3. [ADAPT-SQL vs The Field](#3-adapt-sql-vs-the-field)
4. [Per-Paper Deep Dives](#4-per-paper-deep-dives)
   - [A. Schema Linking & Structure](#a-schema-linking--structure)
   - [B. Multi-Agent & Multi-Generator](#b-multi-agent--multi-generator)
   - [C. Retrieval & In-Context Learning](#c-retrieval--in-context-learning)
   - [D. Training & Reinforcement Learning](#d-training--reinforcement-learning)
   - [E. Robustness Evaluation](#e-robustness-evaluation)
5. [Technique Comparison Matrix](#5-technique-comparison-matrix)
6. [Pipeline Step Comparison](#6-pipeline-step-comparison)
7. [Gap Analysis & Improvement Roadmap](#7-gap-analysis--improvement-roadmap)
8. [Robustness Notes (Dr.Spider)](#8-robustness-notes-drspider)

---

## 1. Quick Reference — All 15 Papers

| # | Paper | Core Idea (one line) | Best Spider EX | Best BIRD EX | Adoptable for ADAPT-SQL? |
|---|-------|----------------------|---------------|--------------|--------------------------|
| 1 | **RESDSQL** | Rank schema items, then generate SQL skeleton before filling entities | 84.1% dev | — | Yes — schema ranking |
| 2 | **LitE-SQL** | Lightweight vector schema retrieval + execution-error-driven retry | 88.45% test | 72.1% dev | **Yes — retry redesign** |
| 3 | **LinkAlign** | Multi-round semantic retrieval for massive multi-database settings | 33.09%* | — | Partial — for larger deploys |
| 4 | **View-Oriented** | Abstract multi-table joins into SQL views → simpler skeleton generation | +2.1% (Spider-EM) | — | Yes — join simplification |
| 5 | **DTS-SQL** | Two-stage fine-tuning: schema linking first, SQL second (7B models) | 85.5% dev | 60.3% test | Partial — fine-tuning required |
| 6 | **MAC-SQL** | Three specialized agents: Selector → Decomposer → Refiner | 86.75% dev | 59.6% test | Yes — refiner agent pattern |
| 7 | **XiYan-SQL** | Generate N SQL candidates from multiple strategies, rerank by execution | 89.65% test | 75.6% dev | **Yes — multi-gen ensemble** |
| 8 | **DeepEye-SQL** | SDLC-inspired: N-version SQL + unit tests + confidence-aware selection | 89.8% test | 73.5% dev | **Yes — N-version + testing** |
| 9 | **SAFE-SQL** | Filter ICL examples by semantic + structural + reasoning path quality | ~87.9% easy | — | **Yes — example re-scoring** |
| 10 | **AP-SQL** | Schema filtering (3B model) + RAG + CoT/GoT routing by complexity | 89.7% (GPT-4o) | — | Yes — schema pre-filter |
| 11 | **YORO** | Bake DB knowledge into model weights; no schema at inference time | 78.5% dev | 34.0% dev | No — needs per-DB fine-tuning |
| 12 | **Pi-SQL** | Use Python as intermediate language to reason before generating SQL | +3.2% BIRD | — | Yes — code-as-CoT |
| 13 | **SQL-R1** | RL with 4-component reward (format + execution + result + length) | 88.7% test | 67.1% test | No — training infrastructure |
| 14 | **ExCoT** | Execution-guided DPO + iterative chain-of-thought preference alignment | 86.59% test | 68.5% dev | Partial — CoT pattern reusable |
| 15 | **Dr.Spider** | Benchmark of 17 perturbation types; measures true model robustness | N/A (eval only) | N/A | Yes — test your own robustness |

*Spider 2.0-Lite score, not comparable to standard Spider

---

## 2. Results Leaderboards

### 2a. Spider Dev — Execution Accuracy

| System | Spider Dev EX | Model Used | Open Source? |
|--------|--------------|------------|--------------|
| DeepEye-SQL | ~89%+ | GPT-4 class | Partial |
| AP-SQL | 89.7% | GPT-4o | No |
| MAC-SQL | 86.75% | GPT-4 + SQL-Llama 7B | Partial |
| ExCoT (best) | ~87%+ | Qwen-2.5-Coder 32B | Yes |
| DTS-SQL | 85.5% | DeepSeek 7B | Yes |
| RESDSQL | 84.1% | T5-3B / BERT | Yes |
| **ADAPT-SQL (after-fix-1)** | **81.5%** | **qwen3-coder (local Ollama)** | **Yes** |
| YORO (Mistral-7B) | 78.5% | Mistral 7B | Yes |
| SAFE-SQL (Llama-70B) | ~45% | Llama-3.3-70B | Yes |

> **Note**: ADAPT-SQL documentation lists 91.8% EX as a target. Baseline full-dev-set run (2026-05-01) measured **80.6%**. After Priority-1 fixes (changes A+B+C, 2026-05-02) measured **81.5%**. Numbers in this document use the latest verified run.

### 2b. Spider Test — Execution Accuracy

| System | Spider Test EX | Model |
|--------|---------------|-------|
| DeepEye-SQL | **89.8%** | GPT-4 class |
| XiYan-SQL | 89.65% | Multi-model ensemble |
| AP-SQL | 89.7% | GPT-4o |
| LitE-SQL | 88.45% | — |
| SQL-R1 (7B) | 88.7% | Qwen2.5-Coder-7B |
| ExCoT | 86.59% | LLaMA-3.1-70B |
| RESDSQL | 79.9% | T5-3B |
| DTS-SQL | — | 7B fine-tuned |

### 2c. BIRD Benchmark

| System | BIRD Dev EX | BIRD Test EX | Notes |
|--------|------------|-------------|-------|
| DeepEye-SQL | 73.5% | **75.07%** | Best open test result |
| XiYan-SQL | **75.63%** | — | Best dev result here |
| LitE-SQL | 72.1% | — | |
| ExCoT | 68.51% | — | LLaMA-3.1-70B |
| SQL-R1 (14B) | 66.6% | 67.1% | Qwen2.5-Coder-14B |
| MAC-SQL | — | 59.59% | GPT-4 + SQL-Llama |
| DTS-SQL | — | 60.31% | DeepSeek 7B |
| YORO | 34.0% | — | Mistral 7B, schema-free |

> ADAPT-SQL is not evaluated on BIRD — Spider-only for now.

### 2d. Efficiency Comparison

| System | LLM Calls / Query | Schema Tokens (avg) | Can Run Locally? |
|--------|------------------|---------------------|-----------------|
| YORO | 1 | **41–47** | Yes (7B) |
| AP-SQL | ~3 | ~300 (filtered) | Partial |
| LitE-SQL | 2–4 | ~400 (vector-filtered) | Yes |
| ADAPT-SQL | **5–10** (up to 4 retries) | Full schema | Yes (Ollama) |
| MAC-SQL | 3–5 per agent | Full schema | Partial |
| DeepEye-SQL | N×3 + selection | Full schema | No (GPT-4) |
| XiYan-SQL | N generators | Full schema | Partial |

ADAPT-SQL uses the most LLM calls per query — a meaningful target for optimization.

---

## 3. ADAPT-SQL vs The Field

### Actual Performance — Run History

| Metric | Baseline (2026-05-01) | After-Fix-1 (2026-05-02) | Δ |
|--------|-----------------------|--------------------------|---|
| Total queries | 1,034 | 1,034 | — |
| Valid SQL | 1,034 (100%) | 1,034 (100%) | — |
| Execution success | 1,018 (98.5%) | 1,010 (97.7%) | **−0.8%** |
| **EX** | **833 / 1,034 = 80.6%** | **843 / 1,034 = 81.5%** | **+0.9%** |
| EM | 182 / 1,034 = 17.6% | 183 / 1,034 = 17.7% | +0.1% |
| Avg composite score | 68.0% | 68.8% | +0.8% |

> Fix-1 applied: (A) multi-candidate generation + execution-based winner selection, (B) execution-driven retry for 0-row results, (C) schema column pre-filtering.  
> **Expected gain: +4–8% EX. Actual gain: +0.9% EX.** See revised roadmap in Section 7.

### By Complexity (After-Fix-1)

| Complexity | Count | % | Notes vs Baseline |
|------------|-------|---|-------------------|
| EASY | 269 | 26.0% | +25 vs baseline (244) — schema filtering made queries appear simpler |
| NON_NESTED_COMPLEX | 652 | 63.1% | −25 vs baseline (677) — reclassified due to reduced schema context |
| NESTED_COMPLEX | 113 | 10.9% | Unchanged |

### By Generation Strategy

| Strategy | Applied To | EX Accuracy |
|----------|-----------|-------------|
| SIMPLE_FEW_SHOT | EASY queries | **86.1%** |
| INTERMEDIATE_REPRESENTATION (NatSQL) | NON_NESTED | **79.3%** |
| DECOMPOSED_GENERATION | NESTED | **76.1%** |

### Retry Behavior

| Retry Attempts | Query Count | % |
|---------------|-------------|---|
| 1st attempt succeeds | 676 | 65.4% |
| 2 attempts | 120 | 11.6% |
| 3 attempts | 25 | 2.4% |
| Hit max (4 attempts) | **213** | **20.6%** |

> **Key issue**: 1 in 5 queries exhaust retries. These are likely hard failures — improving the retry signal (execution-error feedback vs. just validation feedback) is the highest-ROI fix.

### What ADAPT-SQL Does That Papers Don't

- **Fully local**: No API costs, runs on Ollama — uniquely deployable on HPC/offline environments
- **11-step pipeline**: More stages than most (schema linking → complexity → preliminary SQL → retrieval → routing → 3 strategies → validate → retry → normalize → execute → evaluate)
- **Adaptive routing**: Explicitly maps complexity class to generation strategy — most papers use a single strategy
- **Fuzzy schema validation**: 40% false positive reduction vs. exact matching — not commonly described in papers
- **Structural reranking**: DAIL-SQL-inspired SQL pattern reranking on top of FAISS semantic search

### Where ADAPT-SQL Lags

- **~9% behind SOTA** (80.6% vs. 89.8% DeepEye-SQL on Spider)
- **No multi-candidate generation**: Only one SQL generated per attempt
- **Retry is validation-driven**, not execution-driven (can't learn from wrong-results, only syntax/schema errors)
- **No CoT reasoning trace** before SQL generation
- **FAISS retrieval is semantic-only**: No structural or reasoning path quality filter on retrieved examples

---

## 4. Per-Paper Deep Dives

---

### A. Schema Linking & Structure

---

#### RESDSQL — Decoupling Schema Linking and Skeleton Parsing

**Problem**: Standard seq2seq models try to simultaneously learn schema linking AND SQL generation, creating interference. Schema errors in generation are hard to separate from reasoning errors.

**How it works**:
1. **Ranking-Enhanced Encoder**: A cross-encoder scores every (query, table/column) pair for relevance. Irrelevant schema items are filtered before generation.
2. **Skeleton-Aware Decoder**: Two-phase decoding — first generates SQL structure/skeleton (SELECT, FROM, WHERE, GROUP BY keywords without entity names), then fills in the actual table/column names.
3. Uses T5-3B fine-tuned on Spider training set.

**Key numbers**:
| Dataset | EX |
|---------|-----|
| Spider Dev | 84.1% |
| Spider Test | 79.9% |

**Ablation insight**: Removing the skeleton-aware decoder drops ~3%, removing the cross-encoder ranking drops ~2%. Both components matter independently.

**Limitations**: Cross-encoder adds inference overhead. Skeleton generation fails on complex nesting. Still below 90% because schema filtering can remove needed columns.

**TL;DR for ADAPT-SQL**: Your schema linking (Step 1) does string matching + LLM analysis + connectivity. RESDSQL's cross-encoder ranking is more principled — it scores each schema item for relevance. Adding a relevance pre-filter that reduces schema from "all columns" to "top-K ranked columns" before prompt construction could cut hallucination noise significantly.

---

#### LitE-SQL — Lightweight Execution-Guided Self-Correction

**Problem**: Large encoder models are expensive. Schema linking can be done with lightweight vector retrieval. And most retry mechanisms re-prompt blindly — they should use the actual execution error as feedback.

**How it works**:
1. **Vector-based schema retriever**: Encodes query and each schema column as vectors. Retrieves top-K most relevant columns (reduces context by 60-70%).
2. **Stage 1 - Generate**: Initial SQL given filtered schema.
3. **Stage 2 - Execute & Correct**: Runs SQL against DB. If it fails or returns wrong structure, feeds the execution error message directly back to the LLM for targeted correction. Iterates up to 3 rounds.

**Key numbers**:
| Dataset | EX |
|---------|-----|
| Spider Test | 88.45% |
| BIRD Dev | 72.1% |

**Ablation insight**: Execution-guided self-correction alone adds **+3–5% EX** over static generation. Schema pre-filtering reduces context size 60-70% with less than 2% accuracy loss.

**Limitations**: If execution error messages are ambiguous (e.g., SQLite errors don't always say which column is wrong), correction quality degrades. Fails when all correction attempts still produce syntax errors.

**TL;DR for ADAPT-SQL**: ADAPT-SQL's Step 8 retries when validation fails. LitE-SQL retries when *execution results* are wrong. These are different: your retry fires on schema violations, LitE-SQL's fires on wrong output. Adding execution-result checking (does the executed result make sense?) as a retry trigger would catch a class of errors your current system misses entirely. **This is the single highest-impact change you can make.**

---

#### LinkAlign — Scalable Schema Linking for Multi-Database Environments

**Problem**: Enterprise SQL deployments have thousands of tables across hundreds of databases. Current methods assume you know which database to query. LinkAlign solves database retrieval + schema item grounding at massive scale.

**How it works**:
1. **Multi-round semantic retrieval**: Iteratively rewrites the user query using LLM reflection, then retrieves candidate databases via embedding similarity. Exponential decay for diversity.
2. **Response filtering**: Prunes irrelevant schema items from retrieved databases using embedding comparison.
3. **Multi-agent schema parsing**: A "Data Analyst" agent + "Database Expert" agent debate which schema items are relevant. A "Schema Auditor" infers missing elements.

**Key numbers** (Spider 2.0-Lite benchmark — harder, multi-DB):
| System | Score |
|--------|-------|
| LinkAlign + DeepSeek-R1 | **33.09%** (SOTA on Spider 2.0-Lite) |
| Spider-Agent + o3-mini | 23.40% |
| DIN-SQL + GPT-4o | 1.46% |

**Ablation insight**: Multi-agent debate improves schema grounding by ~4% over single-agent. Multi-round retrieval is essential — single-round misses the correct database 23.6% of the time.

**Limitations**: Computational cost scales with database pool size. Schema parsing with debate increases latency significantly in pipeline mode.

**TL;DR for ADAPT-SQL**: Not immediately relevant for Spider (single DB per query). But if you move to BIRD or enterprise settings, LinkAlign's multi-round query rewriting + schema pruning pattern is directly adoptable. The "Schema Auditor" idea — inferring missing schema links — is interesting for your Step 1 connectivity validation.

---

#### View-Oriented Skeleton Generation

**Problem**: Multi-table JOIN inference is the hardest part of SQL generation. Models frequently generate wrong join conditions. Views (pre-defined SQL abstractions combining tables) reduce this to single-table lookups.

**How it works**:
1. **View Classifier**: Determines if the query maps to a predefined view pattern (two-table equi-join, non-equi-join, etc.)
2. **Path A — View path (Text2SQLSkeleton)**: T5 generates SQL skeleton for that view type. A rule template then completes table/FROM clauses automatically.
3. **Path B — Non-view path (Text2SQL)**: BERT with multi-head attention for queries not matching any view.

**Key numbers**: +1.55% on Spider-syn, +2.13% on Spider-EM compared to baseline. Modest but consistent.

**Ablation insight**: View classification accuracy is the bottleneck — incorrect view type assignment propagates errors. Two-path design prevents the view approach from degrading performance on non-view queries.

**Limitations**: Requires pre-defining view patterns. Only covers ~40% of multi-table queries. Improvements are modest.

**TL;DR for ADAPT-SQL**: Your `decomposed_generation.py` already handles complex queries. The view idea is a lighter version — for NON_NESTED queries with standard join patterns, you could template the FROM clause skeleton and only let the LLM fill in SELECT/WHERE. Small gain but low implementation cost.

---

### B. Multi-Agent & Multi-Generator

---

#### DTS-SQL — Decomposed Two-Stage Fine-Tuning

**Problem**: Asking a 7B model to do schema linking AND SQL generation in one pass is too hard. Decomposing into two separate fine-tuning objectives, each specialized, dramatically improves small model performance.

**How it works**:
1. **Stage 1 model (schema linker)**: Fine-tuned to identify which tables and columns from the schema are relevant to the query. Outputs a filtered schema.
2. **Stage 2 model (SQL generator)**: Fine-tuned to generate SQL given only the filtered schema (not the full DB schema). Reduced context, focused task.
3. Both models based on DeepSeek 7B / Mistral 7B.

**Key numbers**:
| Dataset | EX |
|---------|-----|
| Spider Dev | 85.5% |
| BIRD Test | 60.31% |

**Ablation insight**: Two-stage decomposition outperforms single-stage end-to-end fine-tuning significantly. Errors in Stage 1 compound — schema linking accuracy is the critical bottleneck.

**Limitations**: Performance still below larger/proprietary models. Stage 1 errors are unrecoverable. Requires fine-tuning two separate models (training compute).

**TL;DR for ADAPT-SQL**: You already have a conceptual two-stage design (schema linking in Step 1, generation in Step 6). The DTS-SQL insight is that *specializing* the model on each sub-task via fine-tuning outperforms a single general LLM doing everything. If you ever fine-tune qwen3-coder, separate the schema-linking LoRA from the SQL-generation LoRA.

---

#### MAC-SQL — Multi-Agent Collaborative Framework

**Problem**: A single LLM juggling schema linking, query decomposition, and SQL generation makes compound errors. Specialized agents that each do one thing well and check each other reduce error cascades.

**How it works**:
1. **Selector Agent**: Reads the natural language query, filters the schema to only relevant tables/columns, produces a focused schema representation.
2. **Decomposer Agent**: Takes the focused schema + query and decomposes it into logical sub-questions (especially for nested/complex queries). Breaks "find employees with salary higher than average" into sub-queries.
3. **Refiner Agent**: Takes the generated SQL, executes it, and if it fails, provides structured error feedback to regenerate. Catches 5-8% of errors post-generation.
4. **SQL-Llama (7B)**: Base model fine-tuned on Spider for the Refiner agent specifically.

**Key numbers**:
| Dataset | EX |
|---------|-----|
| Spider Dev | 86.75% |
| BIRD Test | 59.59% |

**Ablation insight**: Removing any single agent causes measurable drops. Refiner agent alone catches ~6% of queries the Decomposer would have gotten wrong.

**Limitations**: Multi-agent coordination adds significant latency. Selector errors cascade — if wrong columns are selected, neither Decomposer nor Refiner can recover. GPT-4 dependency for main agents.

**TL;DR for ADAPT-SQL**: Your pipeline already has schema linking (Step 1) + decomposition (Step 6c for NESTED) + retry (Step 8). The key MAC-SQL insight you're missing: **a dedicated refinement agent with structured error feedback**. Your retry (Step 8) currently uses validation errors — MAC-SQL's Refiner uses *execution* errors. Try replacing validation-retry with execution-retry.

---

#### XiYan-SQL — Multi-Generator Ensemble Framework

**Problem**: Any single SQL generation strategy has blind spots. Different prompting strategies (schema-first, example-first, CoT step-by-step) produce different correct/incorrect patterns. Ensembling leverages their complementary strengths.

**How it works**:
1. **Schema Filter Module**: Pre-generation, removes irrelevant tables/columns from schema context. Reduces LLM distraction.
2. **N Generators**: Multiple SQL generation models (different fine-tuning, different prompting strategies) each produce a candidate SQL independently.
3. **Execution Reranking**: Execute all candidates against the DB. Candidates that execute successfully and return non-empty results are preferred. Among successful candidates, use semantic similarity/confidence scoring to select best.
4. Final selection combines execution success + result plausibility + generator confidence.

**Key numbers**:
| Dataset | EX |
|---------|-----|
| Spider Test | **89.65%** |
| BIRD Dev | 75.63% |

**Ablation insight**: Single best generator underperforms the ensemble by 2-4%. Execution-based reranking adds ~2% over simple confidence-based selection. Schema filtering adds ~1.5%.

**Limitations**: Requires running N generators (N × inference cost). Execution reranking fails when all candidates produce errors. Schema filter can prune needed columns.

**TL;DR for ADAPT-SQL**: This is the most directly impactful change you can make without any training. Add a second generation pass: after your current Step 6 generates SQL₁, generate SQL₂ with a different prompt template (e.g., CoT-style instead of few-shot). Execute both, pick the one that returns a result. If only one executes, take it. This alone is likely worth **+4–6% EX** based on XiYan-SQL's ablations.

---

#### DeepEye-SQL — Software-Engineering-Inspired SDLC Workflow

**Problem**: SQL generation is like writing code — it needs requirements analysis, implementation variants, and testing. Most systems treat it as a single inference step. DeepEye-SQL applies the full software development lifecycle.

**How it works**:
1. **Semantic Value Retrieval**: Extracts semantic information from query + DB documentation to understand intent precisely.
2. **Robust Schema Linking**: Multi-step schema disambiguation, especially for column name conflicts across tables.
3. **N-Version Programming**: Generates N diverse SQL candidates using different prompting strategies simultaneously:
   - Schema-first (analyze tables, then write SQL)
   - Example-first (find similar examples, then adapt)
   - Step-by-step CoT reasoning
4. **Unit Testing**: Executes each candidate. Validates: syntax, row count plausibility, data type consistency.
5. **Confidence-Aware Selection**: Combines generation confidence + execution success + result plausibility scores. Selects the highest-confidence candidate.

**Key numbers**:
| Dataset | EX |
|---------|-----|
| Spider Test | **89.8%** |
| BIRD Dev | 73.5% |
| BIRD Test | 75.07% |

**Ablation insight**: N-version programming alone adds ~3% over single-strategy. Unit testing (execution validation) catches an additional 8-12% of initially wrong candidates before selection.

**Limitations**: N × generation cost + execution cost per query. Semantic value retrieval requires good DB documentation. Confidence scoring can be miscalibrated for domain-specific queries.

**TL;DR for ADAPT-SQL**: DeepEye-SQL and XiYan-SQL tell the same story: **generate multiple SQL candidates, test them, pick the best**. The unit testing angle (not just "does it execute" but "does the result make sense — right row count, right data type?") is especially practical. You could implement this in `validate_sql.py` or as a new Step 7b.

---

### C. Retrieval & In-Context Learning

---

#### SAFE-SQL — Self-Augmented In-Context Learning with Fine-Grained Example Selection

**Problem**: Finding truly relevant in-context examples from training data is hard for rare query types. Self-generating examples risks noise. SAFE-SQL generates synthetic examples then filters them rigorously before using them.

**How it works**:
1. **Schema Linking**: Identify relevant tables/columns from the test query first.
2. **Example Generation**: LLM generates 10 synthetic NL-SQL pairs that are structurally similar to the test question.
3. **Fine-Grained Scoring** (0–10 scale, three components):
   - **Semantic Similarity S**: Cosine similarity on question embeddings — same topic/domain?
   - **Structural Alignment A**: Do the key DB elements match (same JOINs, aggregations, filters)?
   - **Reasoning Path Quality R**: Do the logical derivation steps align? (SELECT/WHERE/GROUP BY patterns)
   - Combined: `Rel = α·S + β·A + γ·R`
4. **Threshold Filtering**: Only keep examples scoring ≥ 8/10. ~65.7% of generated examples pass.
5. **Inference**: Final SQL generation using only the filtered high-quality examples.

**Key numbers**:
| System | Spider Easy | Spider Medium | Spider Hard | Spider Extra | Overall |
|--------|-----------|--------------|------------|-------------|---------|
| SAFE-SQL (GPT-4o) | 87.9% | 77.8% | 63.5% | 38.9% | **59.1%** |
| Few-shot baseline | 79.8% | 68.3% | 54.1% | 22.8% | 49.6% |

> Note: SAFE-SQL overall (59.1%) appears low because this is GPT-4 few-shot baseline comparison, not a fully-optimized system.

**Ablation insight** (drop from 63.5% BIRD baseline):

| Component Removed | EX Drop |
|------------------|---------|
| Without Reasoning Path | -3.5% |
| Without Relevance Filtering | -5.8% |
| Without Schema Linking | -7.5% |
| Without Similar Examples | -10.3% |

Schema linking + example filtering together account for ~18% of performance.

**Limitations**: Dependent on LLM quality for example generation. Framework degrades with weaker models. Threshold tuning required per domain.

**TL;DR for ADAPT-SQL**: Your Step 4 (vector search) uses FAISS semantic similarity for example retrieval. SAFE-SQL adds two missing filters: **structural alignment** (do the retrieved examples have the same SQL clause structure as what you'll generate?) and **reasoning path quality** (do the JOINs/aggregations in the example match what the query needs?). Your `structural_similarity.py` already handles the structural part — you just need to integrate it as a filter on retrieved examples, not just a reranker.

---

#### Auto Prompt SQL (AP-SQL) — Resource-Efficient Architecture

**Problem**: Resource-constrained environments (edge devices, no big models) need text-to-SQL. AP-SQL achieves competitive performance with small models by being smarter about schema filtering and prompt design.

**How it works**:
1. **Schema Filtering**: A fine-tuned Qwen-3B model selects only the top 3 tables and top 3 columns per table. Dramatically reduces prompt length.
2. **RAG Examples**: Retrieves top-K NL-SQL pairs from training set (K=3 optimal).
3. **Schema Linking**: Two-stage (table selection → column selection) with 1-10 relevance scoring, threshold 6.
4. **SQL Generation**: Routes based on complexity:
   - Simple (single-table): Chain-of-Thought (CoT) prompting
   - Complex (multi-table, nested): Graph-of-Thought (GoT) prompting — builds a graph of table relationships, then reasons over it

**Key numbers** (Spider):
| Model | EX | Test Suite Acc |
|-------|-----|---------------|
| Qwen-7B | 68.3% | 60.8% |
| Llama-8B | 72.4% | 64.1% |
| GPT-4o-mini | 83.2% | 75.8% |
| GPT-4o | **89.7%** | 82.6% |

**Ablation insight**: Schema filtering (3 tables, 3 columns each) alone reduces errors significantly. GoT vs. CoT: GoT adds ~3% for complex queries, no benefit for simple ones.

**Limitations**: Schema filter is a fine-tuned model (training required). K=3 retrieval may be too conservative for very complex queries.

**TL;DR for ADAPT-SQL**: AP-SQL's **GoT (Graph-of-Thought)** prompting for complex queries is directly actionable. For NON_NESTED_COMPLEX queries in your Step 6b, instead of just intermediate representation (NatSQL), try building an explicit relationship graph of the needed tables first, then generating SQL over it. Also: their schema filtering to top-3-tables is aggressive but effective — your schema linking may pass too many irrelevant columns to the generator.

---

#### YORO — You Only Read Once (Knowledge Internalization)

**Problem**: Every text-to-SQL inference re-reads the full database schema (1979 tokens for CodeS, 979 for PICARD). This is slow, expensive, and wastes context on static information. YORO bakes database knowledge into model weights during training.

**How it works**:
1. **SQL Skeleton Extraction**: Extract all table names, column names, aliases, values from training SQLs.
2. **Synthetic SQL Generation**: Generate diverse SQL queries over the database using extracted skeletons.
3. **NLQ Generation**: Generate natural language questions for each synthetic SQL.
4. **Expert Model Training**: Train one LoRA-fine-tuned model per database. Each "expert" specializes in one DB and knows its schema without being told.
5. **Inference**: Input is just the NLQ (41–47 tokens average vs. 979–1979 for baselines). The model produces SQL from internalized knowledge.

**Key numbers**:
| Dataset | YORO (Mistral-7B) | CodeS (comparison) |
|---------|------------------|-------------------|
| Spider Dev | 78.5% | 80.2% |
| KaggleDBA | 39.0% | 43.4% |
| BIRD Dev | 34.0% | 35.7% |

**Ablation insight**: Synthetic data is the most critical component — removing it drops Spider performance by 63.3%. Domain expert models account for 11% of performance vs. a single monolithic model.

**Limitations**: Requires retraining when DB schema changes. Struggles with column disambiguation and value abbreviations. Doesn't scale well to thousands of tables. Performance is actually slightly below CodeS on most benchmarks.

**TL;DR for ADAPT-SQL**: YORO's approach is too infrastructure-heavy for your current setup (requires per-DB fine-tuning). However, the synthetic data generation technique — using your existing SQLs to create more training pairs — is directly applicable if you ever fine-tune. The most interesting finding: **a 7B model with internalized schema performs comparably to much larger models with full schema context**, suggesting schema compression/distillation is worth exploring.

---

#### Pi-SQL — Python as Pivot Language

**Problem**: There's a semantic gap between natural language and SQL (a low-resource, specialized language). Python is high-resource, well-understood by LLMs, and can express the same logic. Use Python as a stepping stone.

**How it works**:
1. **Python Generation (3 strategies)**:
   - *Merge-First*: Load relevant columns, merge dataframes, then analyze
   - *Filter-First*: Filter rows first by condition, then process
   - *Vanilla Direct*: Generate Python analysis code without constraints
2. **Self-Consistency Verification**: Execute all 3 Python programs, determine consensus on the correct reference result.
3. **SQL Generation with Guidance**: Generate SQL guided by the Python program + its execution result. If SQL output matches Python output → keep it. If not → retry with the discrepancy highlighted.

**Key numbers** (BIRD Dev):
- +3.2% EX improvement over best non-pivot baseline (TA-SQL)
- Works across EASY/NON_NESTED/NESTED difficulty levels

**Ablation insight**: Python guidance as a "double-check" mechanism is the primary value. The three diverse strategies ensure coverage of different query interpretations.

**Limitations**: Python generation overhead (3× generation). Primarily evaluated on BIRD (complex queries) not Spider. Computational cost not fully analyzed.

**TL;DR for ADAPT-SQL**: The core idea is: **generate executable code in a language the model knows better (Python), verify the result, then use that result to constrain SQL generation**. For ADAPT-SQL, this could mean: for NESTED_COMPLEX queries, first generate a Python pandas equivalent, execute it to get the expected result shape, then use that as a hint in your SQL generation prompt. Low-overhead version: just ask the LLM to "think in Python first" as a CoT step before writing SQL.

---

### D. Training & Reinforcement Learning

---

#### SQL-R1 — Reinforcement Learning for Text-to-SQL

**Problem**: Supervised fine-tuning (SFT) creates models that memorize patterns but don't reason through new complex cases. RL with execution feedback trains models to actually solve the problem, not imitate solutions.

**How it works**:
1. **Stage 1 — SFT Cold Start**: Fine-tune on SynSQL-200K (synthetic NL-SQL pairs). Gives the model a baseline before RL.
2. **Stage 2 — RL with GRPO**: Group Relative Policy Optimization — generates multiple SQL candidates per question, rewards correct ones, penalizes wrong ones.
3. **4-Component Reward Function**:
   - **Format Reward (Sf)**: Is the output properly formatted SQL? (0 or 1)
   - **Execution Reward (Se)**: Does it execute without errors? (+2 if yes, -2 if not)
   - **Result Reward (Sr)**: Does execution return the correct result? (+3 if yes, -3 if no)
   - **Length Reward (Sl)**: Penalize unnecessarily long queries
4. **Self-Consistency Voting**: Among RL-generated candidates, select by majority vote on execution result.

**Key numbers**:
| Dataset | EX | Model |
|---------|-----|-------|
| Spider Dev | 87.6% | Qwen2.5-Coder-7B |
| Spider Test | **88.7%** | Qwen2.5-Coder-7B |
| BIRD Dev | 66.6% | Qwen2.5-Coder-7B |
| BIRD Test | 67.1% | Qwen2.5-Coder-14B |

**Ablation insight** (removing each reward component from 87.6% baseline):
| Removed | EX Drop |
|---------|---------|
| Format Reward | -3.0% |
| Length Reward | -1.1% |
| Execution Reward | -0.7% |
| Result Reward | -0.7% |

Format correctness is surprisingly the most impactful reward — the model learns SQL structure before learning correctness.

**Limitations**: Requires large synthetic training dataset (SynSQL-2.5M). Limited to SQLite dialect. Expensive training compute.

**TL;DR for ADAPT-SQL**: Not directly adoptable without training infrastructure (GPU cluster). But the reward design is educational: execution correctness matters more than exact-match correctness as a training signal. The most practical takeaway: **the format of your retry prompt matters a lot** — SQL-R1's biggest gain came from format reward, suggesting that structuring your error feedback messages (format issues vs. logic issues vs. schema issues) could significantly improve retry success.

---

#### ExCoT — Execution-Guided Chain-of-Thought via DPO

**Problem**: Adding CoT to text-to-SQL doesn't help without training — models add reasoning steps that don't lead to better SQL. DPO (Direct Preference Optimization) with execution feedback trains models to produce reasoning that actually leads to correct SQL.

**How it works**:
1. **Stage 1 — CoT SFT**: GPT-4o generates diverse CoT variants (no-CoT, simple-CoT, complex-CoT) for 5,600 verified examples. Fine-tune the model on verified CoT outputs.
2. **Stage 2 — Off-Policy DPO**: Generate up to 32 SQL candidates per question. Execute all. Form preference pairs: (correct SQL with CoT) vs. (wrong SQL with CoT). Train with maximum edit distance strategy — hardest negative examples.
3. **Stage 3 — On-Policy Iterative DPO**: Use the model itself to generate new preference pairs, then retrain. Repeat 2-3 rounds.

**Key numbers**:
| Stage | BIRD Dev EX | Spider Test EX |
|-------|------------|---------------|
| LLaMA-3.1-70B Base | 57.37% | 78.81% |
| + CoT SFT | 62.03% | 83.00% |
| + Off-Policy DPO | 66.30% | 82.49% |
| + On-Policy DPO | **68.51%** | **86.59%** |
| CoT token length (evolution) | 560 → 910 tokens | — |

**Ablation insight**: Off-policy DPO with *furthest* (hardest) negative examples: 66.30%. With random negatives: only 64.08%. Harder negatives matter. Iterative on-policy rounds improve by another ~2%.

**Limitations**: 3-stage training complexity. Intermediate reasoning can introduce partial truths. Requires 70B+ model to get these results.

**TL;DR for ADAPT-SQL**: Without training infrastructure, you can still borrow the **CoT pattern**: before generating SQL, add a reasoning step to your prompt: "Step 1: Identify which tables are needed. Step 2: Determine the WHERE conditions. Step 3: Decide if aggregation is needed. Step 4: Write the SQL." Even without training, structured CoT prompting often improves LLM SQL generation quality, especially for NESTED_COMPLEX queries. Try adding this to `decomposed_generation.py`.

---

### E. Robustness Evaluation

---

#### Dr.Spider — Diagnostic Evaluation Benchmark for Robustness

**Problem**: All text-to-SQL benchmarks measure accuracy on clean, well-formed questions. Real users rephrase things, use synonyms, abbreviate column names, and change table content. How robust are SOTA systems really?

**How it works**:
Introduces 17 perturbation types across 3 dimensions applied to the Spider dev set:

**DB Perturbations** (3 types):
- `Schema-synonym`: "country" → "nation" in column names
- `Schema-abbreviation`: "rank_points" → "rank_pts"
- `DBcontent-equivalence`: Split text fields, convert booleans, change number representations

**NLQ Perturbations** (9 types):
- `Keyword-synonym`: "highest" → "largest", "most" → "greatest"
- `Column-carrier`: "teachers not born in Little Lever" instead of "teachers whose hometown is not Little Lever"
- `Value-synonym`: "France" → referenced differently
- `Column-value`, `Column-attribute`, `Multitype`, etc.

**SQL Perturbations** (5 types):
- `Comparison`: Flip `>` to `>=`, `<` to `<=`
- `Sort-order`: ASC ↔ DESC
- `DB-text`, `DB-number`: Change specific values in WHERE clauses

**Key robustness results** (pre → post perturbation):
| Model | Pre-Perturb EX | Post-Perturb EX | Drop |
|-------|---------------|----------------|------|
| RATSQL | 73.2% | 25.2% | **-48.0%** |
| GrAPPA | 69.1% | 25.2% | **-43.9%** |
| SmbOP | 65.7% | 25.2% | -40.5% |
| T5-3B | 67.4% | 54.3% | -13.1% |
| **PICARD** | 70.9% | **51.2%** | **-19.7%** |

T5-family models are dramatically more robust than GNN/attention-based models. PICARD is the most robust overall.

**Most damaging perturbations**:
| Perturbation | EX Drop |
|-------------|---------|
| DBcontent-equivalence | -50.7% |
| Value-synonym | -26.9% |
| Column-value | -21.6% |
| Comparison operator | -19.8% |

**TL;DR for ADAPT-SQL**: You should run Dr.Spider's perturbations against your system. Based on your architecture, you're most vulnerable to: (1) `Schema-synonym` — your fuzzy matcher helps but won't catch all synonyms; (2) `Column-carrier` — implicit column references without explicit naming; (3) `DBcontent-equivalence` — split fields and boolean conversions. Your strongest defense is the fuzzy schema validator (Step 7) — the LLM-based approaches above have no such layer.

---

## 5. Technique Comparison Matrix

| Technique | Papers Using It | ADAPT-SQL Status | Estimated EX Gain |
|-----------|----------------|-----------------|-------------------|
| Schema pre-filtering (reduce tokens) | RESDSQL, LitE-SQL, AP-SQL, XiYan-SQL | **Partial** (Step 1 links schema but passes all to generator) | +1–3% |
| SQL skeleton generation | RESDSQL, View-Oriented, MAC-SQL | **No** | +1–2% |
| NatSQL / intermediate representation | ADAPT-SQL, Pi-SQL | **Yes** (Step 6b) | Already implemented |
| Multi-generator ensemble | XiYan-SQL, DeepEye-SQL | **No** | **+4–6%** |
| Execution-driven retry | LitE-SQL, DeepEye-SQL, MAC-SQL | **No** (validation-driven only) | **+3–5%** |
| Execution-based candidate selection | XiYan-SQL, DeepEye-SQL, SQL-R1 | **No** | +2–4% |
| Structured CoT before SQL | ExCoT, AP-SQL (GoT), Pi-SQL | **No** | +2–3% |
| Fine-grained example scoring | SAFE-SQL | **Partial** (structural reranking exists, no reasoning path filter) | +1–2% |
| Fuzzy schema validation | ADAPT-SQL | **Yes** (Step 7) | Already implemented |
| Structural example reranking | ADAPT-SQL (DAIL-SQL inspired) | **Yes** (Step 4) | Already implemented |
| Complexity routing | ADAPT-SQL, AP-SQL | **Yes** (Step 5) | Already implemented |
| Validation + feedback retry | ADAPT-SQL, MAC-SQL | **Yes** (Step 8) | Already implemented |
| RL/DPO training | SQL-R1, ExCoT | **No** | +5–8% (training required) |
| Knowledge internalization | YORO | **No** | +2% (infra required) |
| Multi-database retrieval | LinkAlign | **No** | Not applicable for Spider |
| Robustness testing | Dr.Spider | **No** | Diagnostic only |

---

## 6. Pipeline Step Comparison

For each of ADAPT-SQL's 11 steps, what do the papers do and how does it compare?

| ADAPT-SQL Step | What You Do | Best Paper Equivalent | Gap |
|---------------|-------------|----------------------|-----|
| **Step 1: Schema Linking** | 3-layer: string match → LLM analysis → connectivity | RESDSQL cross-encoder ranking, LinkAlign multi-round retrieval | You pass full matched schema to generator; papers filter to top-K before generation |
| **Step 2: Complexity Classification** | Rule-based (80%) + LLM fallback | AP-SQL routes CoT/GoT by complexity | Similar concept; AP-SQL's GoT for complex is worth adding |
| **Step 3: Preliminary SQL** | Generate preliminary SQL for structural analysis | RESDSQL skeleton generation | Your preliminary SQL informs example retrieval; RESDSQL's skeleton directly guides decoding |
| **Step 4: Example Retrieval** | FAISS semantic + structural reranking | SAFE-SQL (adds reasoning path filter), DAIL-SQL | Missing: reasoning path quality filter on retrieved examples |
| **Step 5: Routing** | Map complexity → strategy | AP-SQL, MAC-SQL (implicit routing) | Similar; consider adding GoT routing for complex |
| **Step 6a: Few-Shot (EASY)** | Direct few-shot generation | AP-SQL CoT, ExCoT | Add structured CoT step before SQL |
| **Step 6b: NatSQL (NON_NESTED)** | Intermediate representation | Pi-SQL Python pivot | Add Python/CoT hint for harder non-nested queries |
| **Step 6c: Decomposed (NESTED)** | Sub-question decomposition | MAC-SQL Decomposer, DTS-SQL | MAC-SQL uses dedicated fine-tuned model; yours uses prompting only |
| **Step 7: Validation** | Fuzzy schema validation | RESDSQL, DeepEye-SQL (unit testing) | Add execution-based validation: does result have expected shape? |
| **Step 8: Retry** | Regenerate with validation error feedback | LitE-SQL (execution feedback), MAC-SQL Refiner | Switch from validation-driven to execution-driven retry |
| **Step 9: Normalization** | Format, alias consistency | SQL-R1 (format reward) | Good; normalization covers format issues |
| **Steps 10-11: Execute + Evaluate** | SQLite execution + Spider metrics | All papers (baseline) | Consider adding result plausibility check to Step 10 |
| **Missing: Multi-candidate** | — | XiYan-SQL, DeepEye-SQL | Generate 2–3 candidates, select by execution |

---

## 7. Gap Analysis & Improvement Roadmap

### Post-Fix-1 Reality Check (Updated 2026-05-02)

Changes A, B, C were implemented and a full 1,034-query dev run completed. Results:

| Change | Expected EX Gain | Realized Contribution | Status |
|--------|-----------------|----------------------|--------|
| A — Multi-candidate + execution selection | +4–6% | Marginal | ⚠️ Candidates not diverse enough |
| B — Execution retry for 0-row results | +3–5% | Marginal | ⚠️ Fires too rarely; most failures aren't 0-row |
| C — Schema column pre-filtering | +1–3% | **Negative side effect** | ❌ Execution success −0.8%; over-filters needed columns |
| **Combined A+B+C** | **+4–8%** | **+0.9% EX** | Under-delivered |

**Root cause analysis:**

- **Change A**: Both generated candidates use similar prompts. When the underlying LLM produces two outputs from nearly identical contexts, they fail in the same way. Gain only materializes when candidates are *structurally diverse* (e.g., one uses JOIN, one uses subquery).
- **Change B**: Most wrong SQL returns a *non-empty but incorrect* result set. The 0-row trigger is too narrow — it only catches the subset that returns nothing.
- **Change C**: The pre-filter is too aggressive. The 25-query complexity reclassification (NON_NESTED → EASY) and the −8 execution successes show that columns being pruned were sometimes load-bearing. Schema filtering only works safely if the relevance scorer is high-precision.

---

### Revised Priority 1 — Fix the Regressions First

#### A′. Make Multi-Candidate Genuinely Diverse
**Problem**: SQL₁ and SQL₂ from the same prompt template fail identically.  
**Fix**: Use *structurally different* prompt strategies for each candidate — one uses the NatSQL intermediate representation, the other uses direct few-shot with chain-of-thought. Divergent reasoning paths produce divergent outputs.  
**Files**: `pipeline/few_shot.py`, `pipeline/intermediate_repr.py` — generate SQL₂ using the *other* strategy (EASY → try IR style; NON_NESTED → try few-shot style)  
**Realistic EX gain**: +2–4% (vs. the +4–6% estimate; adjusted for single-model diversity ceiling)

#### B′. Extend Execution Retry Beyond 0-Row
**Problem**: Change B only retries when result is empty. Wrong non-empty results (e.g., returns 5 rows instead of 1, or different column values) slip through.  
**Fix**: After execution, compare result *shape* to query intent: if query asks for one value (MAX, COUNT, MIN) but returns multiple rows → retry. If query has no aggregation but returns a single scalar → retry. Keep 0-row trigger as-is and add these shape checks.  
**Files**: `pipeline/execute_compare.py`, `pipeline/validation_feedback_retry.py`  
**Realistic EX gain**: +1–3%

#### C′. Tune Schema Filtering — Soft Filter, Not Hard Cut
**Problem**: Hard top-K filtering removes needed columns.  
**Fix**: Instead of dropping columns, *reorder* them — put high-relevance columns first in the prompt. The LLM attends more to earlier context. Never remove a column that appears verbatim in the query or is a foreign key. This gives context reduction without information loss.  
**Files**: `pipeline/schema_linking.py`  
**Goal**: Recover the lost −0.8% execution success and convert it to a gain  
**Realistic EX gain**: +0.5–1.5% (net, after recovering regression)

---

### Priority 2 — Untried, Low Effort, Real Evidence

#### D. Structured CoT in Generation Prompts ← **Try This First**
**What**: Add a reasoning preamble to each generation prompt: "Before writing SQL, identify: (1) which tables are needed and why, (2) the JOIN conditions, (3) whether aggregation/subquery is required. Then write the SQL."  
**Files**: Prompt templates in `pipeline/few_shot.py`, `pipeline/intermediate_repr.py`, `pipeline/decomposed_generation.py` — prompt string edits only, no logic changes  
**Evidence**: ExCoT CoT SFT adds +5% to BIRD, +4% Spider Test; Pi-SQL adds +3.2% on BIRD  
**Effort**: <1 hour — pure prompt edits  
**Realistic EX gain**: +2–3% (without training, gains are lower than paper numbers)

#### E. Plausibility Check on Execution Results
**What**: Rule-based sanity checks after execution — if the query contains MAX/MIN/COUNT with no GROUP BY, result should be exactly 1 row; if query asks "how many X" result should be a single integer; etc. Flag implausible results → trigger retry.  
**Files**: `pipeline/execute_compare.py`  
**Evidence**: DeepEye-SQL unit testing catches 8–12% of initially wrong candidates  
**Effort**: 1–2 hours  
**Realistic EX gain**: +1–2%

#### F. Reasoning Path Filter on Retrieved Examples
**What**: Extend Step 4 structural reranking to also score by SQL clause alignment — retrieved examples that require the same clause set (GROUP BY, HAVING, subquery in WHERE) as the current query get priority over structurally similar but clause-divergent ones.  
**Files**: `utils/structural_similarity.py`, `pipeline/vector_search.py`  
**Evidence**: SAFE-SQL: removing reasoning path filter costs −3.5%  
**Effort**: 3–4 hours  
**Realistic EX gain**: +1–2%

---

### Priority 3 — Longer-Term

#### G. RL Fine-Tuning (SQL-R1 approach)
Fine-tune qwen3-coder with GRPO + execution-based reward on SOL GPU nodes. Evidence: SQL-R1 reaches 88.7% Spider test with 7B model. High effort — requires training pipeline + compute allocation.

#### H. Multi-Agent Architecture (MAC-SQL pattern)
Separate schema selection, SQL generation, and execution-based refinement into dedicated agent calls. Evidence: MAC-SQL 86.75% Spider dev. High effort — significant pipeline refactor.

---

### Updated Quick Win Summary

| Change | Status | Files | Effort | Realistic EX Gain |
|--------|--------|-------|--------|-------------------|
| **D — CoT in prompts** | Not tried | 3 prompt templates | <1 hr | **+2–3%** |
| **E — Result plausibility check** | Not tried | `execute_compare.py` | 1–2 hr | **+1–2%** |
| **C′ — Soft schema reordering** | Replaces C | `schema_linking.py` | 1–2 hr | +0.5–1.5% |
| **B′ — Extended execution retry** | Extends B | `execute_compare.py` | 2 hr | +1–3% |
| **A′ — Cross-strategy candidates** | Fixes A | 2 generation modules | 3 hr | +2–4% |
| **F — Reasoning path example filter** | Not tried | `structural_similarity.py` | 4 hr | +1–2% |

**Realistic target**: Implementing D + E + C′ could push **81.5% → 84–86% EX** without any training. Adding A′ and B′ could push toward **87–88%**.

---

## 8. Robustness Notes (Dr.Spider)

Dr.Spider reveals that even strong models lose 20–48% accuracy under perturbations. Based on ADAPT-SQL's architecture, here's where you're most and least vulnerable:

### Your Vulnerability Assessment

| Perturbation Type | Your Risk | Why | Your Defense |
|------------------|-----------|-----|--------------|
| `Schema-synonym` (column renamed) | **Medium** | Fuzzy matcher helps but won't catch all semantic synonyms | `fuzzy_schema_validator.py` |
| `Schema-abbreviation` | **Low** | Fuzzy matching handles most abbreviations | `fuzzy_schema_validator.py` |
| `DBcontent-equivalence` (split fields) | **High** | No content-awareness in your pipeline | None currently |
| `Column-carrier` (implicit column refs) | **High** | Your schema linking relies on explicit keywords | String matching in Step 1 |
| `Value-synonym` (France → République) | **High** | No value normalization or synonym mapping | None currently |
| `Keyword-synonym` (highest → largest) | **Medium** | LLM-based schema linking handles some paraphrasing | LLM fallback in Step 1 |
| `Comparison` (> vs >=) | **Low** | Normalization step handles some of these | `sql_normalizer.py` |
| `Sort-order` (ASC vs DESC) | **Low** | Execution-based evaluation catches this | EX metric |

### How to Test Your Own Robustness

Quick test: Take 50 dev questions and manually apply 3 perturbations each (keyword synonym, column synonym, schema abbreviation). Run through your pipeline. Any drop > 10% is a real robustness gap.

### Quick Fixes for Robustness
1. **Value synonym handling**: When schema linking extracts column values from the question, also expand with a synonym set (e.g., "largest" → ["biggest", "highest", "maximum"]). Low effort, high impact on `value-synonym` perturbations.
2. **Column-carrier handling**: Add a step to your schema linking that asks the LLM: "What implicit column references appear in this question?" (e.g., "hometown" → `city` column). This aligns with RESDSQL's ranking approach.
3. **Content-equivalence awareness**: For queries that mention split fields (first name + last name), your schema linking should recognize this pattern and link both columns.

---

*Summary generated from 15 papers: RESDSQL, LitE-SQL, LinkAlign, View-Oriented Skeleton, DTS-SQL, MAC-SQL, XiYan-SQL, DeepEye-SQL, SAFE-SQL, AP-SQL, YORO, Pi-SQL, SQL-R1, ExCoT, Dr.Spider.*  
*ADAPT-SQL results from actual run: `RESULTS/adapt_sql_results_2026-05-01_23-00-35.csv`, 1,034 Spider dev queries.*
