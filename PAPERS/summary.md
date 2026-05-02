# Text-to-SQL Research Papers — Comprehensive Summary (Spider 1.0)

> 9 papers evaluated on Spider 1.0, covering schema linking, multi-generator ensembles, RL-based training, candidate selection, and efficiency. Everything you need to know without reading each paper.

---

## 1. All 9 Papers at a Glance

| # | Paper | Core Idea in One Line | Spider Dev EX | Spider Test EX | Fine-Tuned? |
|---|-------|----------------------|--------------|----------------|-------------|
| 1 | **DTS-SQL** | Two-stage fine-tuning: schema linking first, SQL generation second | 85.5% | — | Yes (7B) |
| 2 | **DeepEye-SQL** | SDLC-inspired: N-version programming + deterministic SQL unit tests | — | **89.8%** | No |
| 3 | **ExCoT** | Iterative on-policy DPO using SQLite execution as reward signal | — | 86.59% | Yes (DPO) |
| 4 | **LitE-SQL** | Vector-DB schema linking + two-stage SFT→DPO training | — | 88.45% | Yes (7B) |
| 5 | **MAC-SQL** | Three-agent pipeline: Selector → Decomposer → Refiner | — | ~89% | Optional |
| 6 | **SAFE-SQL** | Self-generated examples + composite relevance scoring, zero training | 87.9% | — | No |
| 7 | **SQL-R1** | Layered RL rewards (format + execution + result + length) on 7B model | 78.1% | — | Yes (GRPO) |
| 8 | **XiYan-SQL** | Multi-generator ensemble + consistency-based candidate selection | — | 89.65% | Yes |
| 9 | **YORO** | Internalize DB schema into model weights — 47-token prompts at inference | 78.5% | — | Yes/per-DB |

---

## 2. Papers Grouped by Theme

---

### Theme A — Schema Linking

> Getting the right tables and columns before generation is the single highest-leverage step. DTS-SQL proves this: perfect schema linking → 90.3% BIRD vs. actual 85.5%.

---

#### DTS-SQL — Decomposed Two-Stage Fine-Tuning on Small (7B) Models

**The problem:** Small LLMs struggle when schema linking and SQL generation are mixed into one task. Both tasks compete for the model's capacity.

**Solution:** Split into two separately fine-tuned stages:
1. **Schema Linker:** Input = question + all tables (ranked by embedding similarity). Output = relevant tables/columns only.
2. **SQL Generator:** Input = question + filtered schema only. Output = SQL.

Each stage has its own fine-tuning loss. The schema linker uses vector similarity to rank tables before feeding to LLM, reducing the context the SQL generator sees.

**Upper bound analysis:** Perfect schema linking → 90.3% BIRD. Actual achieved: 85.5%. The 4.8pp gap is entirely recoverable with better schema linking — no generation improvements needed.

**Results:**
| Model | Spider Dev EX | BIRD Hold-out |
|-------|--------------|---------------|
| DTS-SQL + DeepSeek 7B | **85.5%** | **60.3%** |
| DIN-SQL + GPT-4 (baseline) | 84.4% | — |
| LLaMA2 7B prompting | 66.7% | — |

**Borrow for ADAPT-SQL:** The upper-bound framing is directly useful — run your pipeline with gold schema links to measure how much EX you're losing to schema linking failures vs. generation failures. Your pipeline already separates these stages; the fine-tuning aspect requires model weight access.

---

### Theme B — Multi-Generator / Candidate Generation

> Generate multiple SQL candidates using diverse strategies, then pick the best. Diversity in generation prevents systematic failures.

---

#### XiYan-SQL — Multi-Generator Ensemble with Consistency-Based Selection

**Three components:**

1. **Schema Filter:** Multi-path retrieval (tables + columns + values via cosine similarity). PFKeyIdentifier algorithm handles partial and composite foreign keys correctly.

2. **Multiple SQL Generators:** Several fine-tuned generators with intentionally distinct reasoning styles — some prefer CTEs, others subqueries, others flat JOINs. Diversity is the goal.

3. **SQL Selection:** Candidates reorganized by execution consistency (same result → same cluster). Best cluster representative chosen by a trained selection model using listwise comparison.

**Training diversity trick:** Multi-format SQL enhancement — same query trained in structural variants (CTEs vs. subqueries), stylistic variants (formatting differences), and refined variants. Forces generators to diverge.

**Results:**
| Dataset | EX% |
|---------|-----|
| BIRD Dev | 75.63% (SOTA at time) |
| Spider Test | **89.65%** |

**Borrow for ADAPT-SQL:** Execution-consistency clustering for candidate selection is immediately applicable without training. Generate 3–5 candidates, execute all, group by execution result, pick the most-agreed-upon answer.

---

### Theme C — Candidate Selection & Verification

> Once you have SQL candidates, how do you pick the right one? And how do you verify quality before execution?

---

#### DeepEye-SQL — Software Engineering Principles Applied to SQL (No Fine-Tuning)

**Inspiration:** Software Development Life Cycle (SDLC) — requirements → design → implementation → testing → QA.

**Four phases:**

1. **Intent Scoping — Semantic Value Retrieval (SVR):** Offline index of all cell values in all databases. Online retrieval via embeddings + edit distance. Grounds SQL WHERE conditions in actual values that exist in the data.

2. **N-Version Programming:** Multiple independent generators with different reasoning strategies run in parallel under a fixed compute budget. One uses CTEs, another subqueries, another flat JOINs.

3. **SQL Unit Testing — Deterministic checkers (not probabilistic):**
   - Syntax Checker: parses SQL AST for structural validity
   - Logic Checker: validates JOIN conditions, subquery scope, alias references
   - Data Quality Checker: checks NULL handling, type mismatches, ambiguous columns

4. **Confidence-Aware Selection:** Execution-based clustering + consensus frequency scoring.

**Key advantage:** No fine-tuning required — uses ~30B open-source LLMs.

**Results:**
| Dataset | EX% |
|---------|-----|
| BIRD Dev | 73.5% |
| BIRD Test | 75.07% |
| Spider Test | **89.8%** |

**Borrow for ADAPT-SQL:** The deterministic SQL unit testing checkers (Syntax + Logic + Data Quality) are a direct upgrade to your `validate_sql.py`. Your current validator catches schema errors but not JOIN logic errors or NULL mismatches. SVR value retrieval is essential for BIRD but also helps Spider WHERE conditions.

---

#### LitE-SQL — Lightweight Vector-Based Schema Linking + Two-Stage Training

**Schema Linker:**
- Pre-compute embeddings for every column (name + description + table + value description) **offline once**
- At inference: cosine similarity lookup only — no LLM forward pass for schema linking
- **Hard-Negative Supervised Contrastive Loss (HN-SupCon):** Uses 0.1 margin threshold to filter semantically similar but irrelevant columns. Distinguishes `customer_id` in Orders vs. `customer_id` in Returns.

**Two-stage training:**
1. **SFT:** Supervised fine-tuning on schema + question → SQL (quality baseline)
2. **RFT (DPO):** Execution-correct SQLs = preferred, execution-failed = dispreferred. Trains model to prefer executable SQL over syntactically plausible but wrong SQL.

**Self-correction:** Generate multiple candidates, use actual error messages as feedback for iterative refinement. No external model sampling needed.

**Results:**
| Dataset | EX% |
|---------|-----|
| BIRD Dev | 72.1% |
| Spider Test | **88.45%** |
| CHASE-SQL (reference) | 87.60% Spider Test |

**Borrow for ADAPT-SQL:** Pre-computing schema embeddings and doing inference-time cosine retrieval (instead of LLM-based schema linking) would dramatically speed up Step 1 and reduce 2–3 LLM calls per query. HN-SupCon is a fine-tuning technique for later; the vector-DB schema approach works without training.

---

### Theme D — RL & Training-Based Improvement

> Use execution results as a reward signal to train the model to generate better SQL. The database itself acts as the oracle — no human labels needed.

---

#### SQL-R1 — Layered Reward RL with Cold-Start SFT

**The insight:** A single binary reward (correct/not) is too sparse for RL training. Layered rewards provide dense, shaped feedback at multiple levels.

**Four reward components:**
```
R_format  = 1 if SQL format is correct (parseable), else 0
R_exec    = +2 (executable), 0 (not executable), -2 (parse error)
R_result  = +3 (results match ground truth), -3 (results wrong)
R_length  = penalty for unnecessarily long/complex queries
Total R   = R_format + R_exec + R_result + R_length
```

**Training pipeline:**
1. **SFT cold-start** on SynSQL-200k (easy examples only) — teaches the model the expected reasoning format before RL
2. **GRPO RL** on SynSQL-Complex-5k (hard queries only) — uses execution feedback to improve on challenging cases

The cold-start phase prevents reward hacking: without it, the model learns to produce valid-looking SQL that isn't semantically correct.

**Dataset:** SynSQL-2.5M synthetic samples across 16,583 databases (generated by OmniSQL pipeline).

**Results:**
| Dataset | Dev EX | Test EX |
|---------|--------|---------|
| Spider Dev | 78.1% | — |
| BIRD Dev | 87.6% | 88.7% |

(Qwen2.5-Coder-7B)

**Borrow for ADAPT-SQL:** If you fine-tune any model on SOL's A100, use this exact reward structure. The layered reward is the clearest RL recipe in the literature. Cold-start SFT before RL is mandatory to avoid training instability.

---

#### ExCoT — Chain-of-Thought + Iterative On-Policy DPO

**The insight:** Reasoning quality matters as much as SQL quality. Train models to reason correctly about schema and query intent, not just produce syntactically valid SQL.

**Three CoT variants tested:**
- **No-CoT:** Direct SQL generation
- **Simple-CoT:** Schema identification step → SQL
- **Complex-CoT:** Sub-question decomposition → intermediate SQL → final SQL

**Data creation:** Few-shot prompting GPT-4o generates 5.6k verified CoT examples from BIRD/Spider training data.

**Off-policy DPO:** Preference pairs — preferred = correct SQL with CoT, dispreferred = incorrect SQL with CoT. Maximum edit distance selects the most informative dispreferred examples.

**On-policy iterative DPO (3 rounds — 48 GPU hours total):**
1. Generate SQL candidates with current model
2. Execute each in SQLite → correctness signal
3. Form (correct, incorrect) preference pairs
4. DPO train → updated model
5. Repeat with updated model (on-policy = new pairs each round)

**Results:**
| Model | BIRD Dev | Spider Test |
|-------|----------|-------------|
| LLaMA-3.1 70B (ExCoT) | 68.51% | **86.59%** |
| Qwen-2.5-Coder 32B | 68.25% | 85.14% |
| vs. zero-shot CoT | +10.14% | — |

**Borrow for ADAPT-SQL:** The iterative DPO loop using SQLite execution as oracle is the most practical fine-tuning approach available — no human labels needed. Your existing batch runs already produce (correct, incorrect) pairs. Collect them, run DPO on SOL A100.

---

### Theme E — Efficiency & Novel Paradigms

> Papers that challenge the standard prompting+retrieval approach with fundamentally different ideas.

---

#### SAFE-SQL — Self-Augmented ICL with Composite Example Quality Scoring (No Training)

**The problem:** FAISS retrieves examples that are semantically similar but structurally wrong for the target query — similar question wording, but different SQL patterns needed.

**Solution:** Generate 10 synthetic examples per query at inference time, then filter by a composite quality score.

**Composite Relevance Score:**
```
Rel = α × S(Qt, Qe) + β × A(Qt, Qe) + γ × R

S = Semantic Similarity    — LLM-based intent alignment between target and example
A = Structural Alignment   — key entity/relationship correspondence
R = Reasoning Path Quality — logical derivation step validation
```

**Threshold:** Score ≥ 8/10 → include as few-shot example. Critical because 65.71% of self-generated examples score 10/10 (LLMs are biased toward their own output) — filtering removes noise.

**No fine-tuning. No external training data.** Works directly with any LLM.

**Results:**
| Dataset | EX% | EM% |
|---------|-----|-----|
| Spider Dev (GPT-4o) | **87.9%** | 78.3% |
| Zero-shot baseline | 49.2% | — |
| Standard few-shot | 51.2–57.0% | — |

**Borrow for ADAPT-SQL:** The composite relevance scoring is a direct drop-in upgrade for Step 4 (`vector_search.py`). Instead of pure cosine similarity, score each retrieved example on `semantic + structural + reasoning_quality` and filter by threshold before including in the prompt.

---

#### YORO — Internalize Database Knowledge into Model Weights

**The radical idea:** Instead of encoding the database schema at every inference call (expensive, context-heavy), bake the schema knowledge into model weights during a training phase. At inference, only the question is needed.

**How it works:**
1. For each target database, automatically generate synthetic NLQ-SQL pairs: SQL skeleton extraction → SQL generation → NLQ synthesis
2. Fine-tune an expert model per database on these pairs
3. At inference: input = question only (no schema, no values, no retrieval)

**The savings:**
| Method | Avg Input Tokens (BIRD) | Avg Input Tokens (Spider) |
|--------|------------------------|--------------------------|
| CodeS (standard) | 1,979 | 713 |
| YORO | **47** | **31** |
| Reduction | **97.6%** | **95.7%** |

**Results:**
| Dataset | EX% |
|---------|-----|
| Spider Dev | **78.5%** |
| KaggleDDQA | 39.0% |
| BIRD Dev | 34.0% (schema alone insufficient for BIRD's external knowledge) |

**Borrow for ADAPT-SQL:** For production deployments with fixed databases (enterprise use cases), YORO's per-DB expert approach eliminates all retrieval overhead. Not suitable for Spider/BIRD evaluation (you'd need to train per-DB), but the synthetic NLQ-SQL generation approach is useful for augmenting the vector store.

---

#### MAC-SQL — Three-Agent Collaborative Framework

**Three specialized agents with distinct roles:**

1. **Selector Agent** — Decomposes the full database into a minimal relevant sub-database. Prunes irrelevant tables before any generation. Reduces schema noise.

2. **Decomposer Agent** — Breaks complex questions into sub-questions with CoT reasoning. Generates SQL for each sub-question, combines into final query.

3. **Refiner Agent** — Detects and corrects SQL errors using external execution feedback. Checks: syntax errors, execution feasibility, non-empty result verification, semantic correctness.

**SQL-Llama:** Fine-tuned 7B model trained on agent instruction data from BIRD/Spider. Allows running the full pipeline without GPT-4.

**Ablation — contribution of each agent:**
| Component | EX% Contribution |
|-----------|-----------------|
| Selector | +8.45% |
| Decomposer | +14.04% |
| Refiner | +10.15% |

**Results:**
| Setting | EX% |
|---------|-----|
| GPT-4 backbone | 59.59% BIRD |
| Qwen2.5 backbone | **~89% Spider** |
| SQL-Llama 7B | 46.35% BIRD |

**Borrow for ADAPT-SQL:** Your pipeline already implements the Decomposer (Steps 2–6) and Refiner (Steps 7–8) patterns. The Selector agent — minimal sub-database extraction before any LLM generation — is the missing piece and the highest single-component gain (+8.45%). Particularly valuable for BIRD and larger schemas.

---

## 3. Key Techniques Explained

### GRPO — Group Relative Policy Optimization
Used by: SQL-R1

Standard RL (PPO) requires a critic/value network, making training unstable and expensive. GRPO replaces it with group-relative scoring: generate a group of N outputs, score each one, normalize rewards within the group, use each output's deviation from the group mean as its advantage signal. Much more stable for LLM fine-tuning. Reward is typically layered (see SQL-R1) rather than binary.

### DPO — Direct Preference Optimization
Used by: ExCoT, LitE-SQL

Instead of explicit RL, DPO frames alignment as preference learning. Collect (preferred SQL, rejected SQL) pairs — preferred = execution-correct, rejected = execution-incorrect. Train the model to increase the log-probability ratio of preferred over rejected outputs. More stable than PPO, works on smaller datasets (~5k pairs). Iterative on-policy DPO (ExCoT) regenerates new pairs with the updated model each round, maintaining relevance.

### Execution-as-Reward Signal
Used by: ExCoT, LitE-SQL, SQL-R1

Execute the generated SQL against a SQLite database. If the result matches the gold result → reward = 1, else = 0. This is the key insight enabling RL/DPO for Text-to-SQL without human labelers — the database itself is the oracle. Works because SQL execution is deterministic. Your existing batch runs already produce these (correct, incorrect) labels — you have training data already.

### N-Version Programming
Used by: DeepEye-SQL, XiYan-SQL

Borrowed from safety-critical software engineering: generate N independent SQL implementations using different reasoning strategies, then vote/select. One generator uses CTEs, another subqueries, another flat JOINs. Diversity prevents systematic failure — if all generators share the same bug, voting can't help; truly independent generators have uncorrelated errors.

### Self-Consistency / Candidate Selection
Used by: XiYan-SQL, DeepEye-SQL

Generate N SQL candidates via temperature sampling or diverse prompts, execute all, group by execution result. Most common result = winner (majority vote). Self-consistency consistently adds 5–15pp over greedy decoding with N=10–20 candidates. No training required — just run your pipeline N times per query.

### Semantic Value Retrieval (SVR)
Used by: DeepEye-SQL

Offline: index all cell values across all database tables using embeddings. Online: for each query, retrieve matching cell values via cosine similarity + edit distance. Ground SQL WHERE conditions in values that actually exist in the data (e.g., `WHERE country = 'USA'` vs `WHERE country = 'United States'`). Critical for BIRD; also helps Spider queries with string matching.

### Composite Example Relevance Scoring
Used by: SAFE-SQL

Score retrieved few-shot examples on three dimensions: semantic similarity (does the question ask the same thing?), structural alignment (does it need the same SQL pattern?), and reasoning path quality (is the derivation logic transferable?). Weighted combination with threshold filtering removes low-quality examples that pure cosine similarity would include.

---

## 4. Benchmark Comparison Tables

### Table 1 — Spider Results

| Paper | Spider Dev EX | Spider Test EX | Model | Fine-Tuned? |
|-------|--------------|----------------|-------|-------------|
| **DeepEye-SQL** | — | **89.8%** | ~30B OSS | No |
| **XiYan-SQL** | — | 89.65% | Fine-tuned | Yes |
| **MAC-SQL** | — | ~89% | Qwen2.5 | Optional |
| **LitE-SQL** | — | 88.45% | 7B | Yes |
| **SAFE-SQL** | 87.9% | — | GPT-4o | No |
| **ExCoT** | — | 86.59% | LLaMA-3.1 70B | Yes |
| **DTS-SQL** | 85.5% | — | DeepSeek 7B | Yes |
| **YORO** | 78.5% | — | Mistral 7B | Yes/per-DB |
| **SQL-R1** | 78.1% | — | Qwen2.5-Coder 7B | Yes |
| **ADAPT-SQL (ours)** | **~92–93%*** | — | qwen3-coder | No |

*After fixing data leakage (vector store now built from train_spider.json). Previous number of 95.1% reflected dev-set retrieval leakage.

### Table 2 — Training Requirements

| Paper | Requires Weights | Training Data | GPU Budget | Algorithm |
|-------|-----------------|---------------|-----------|-----------|
| **SQL-R1** | Yes | SynSQL-200k SFT + 5k RL | High (A100) | GRPO + layered reward |
| **ExCoT** | Yes | 5.6k CoT pairs + 3 iterative rounds | 48 GPU-hours | On-policy DPO |
| **LitE-SQL** | Yes | BIRD/Spider + execution runs | Moderate | SFT → DPO |
| **DTS-SQL** | Yes | Spider/BIRD schema annotations | Moderate | Two-stage SFT |
| **YORO** | Yes (per DB) | Synthetic NLQ-SQL per database | Per-DB | Expert SFT |
| **XiYan-SQL** | Yes | Multi-format SQL variants | Moderate-High | Ensemble SFT |
| **MAC-SQL** | Optional | Agent instruction data | Optional | Multi-agent prompting |
| **DeepEye-SQL** | **No** | None | None | N-version + unit tests |
| **SAFE-SQL** | **No** | None (self-generated) | None | Self-augmented ICL |

### Table 3 — Efficiency

| Paper | Key Efficiency Feature | Metric | Detail |
|-------|----------------------|--------|--------|
| **YORO** | DB knowledge in weights | 97.6% input reduction | 1,979 → 47 tokens (BIRD) |
| **LitE-SQL** | Pre-computed schema embeddings | No LLM per schema link | O(1) cosine lookup at inference |
| **DeepEye-SQL** | Deterministic checkers vs. LLM validation | No retry LLM calls | Rule-based AST parsing |
| **SAFE-SQL** | No retrieval infrastructure needed | Zero setup | Self-generates examples per query |
| **DTS-SQL** | Smaller context for SQL generator | ~40% context reduction | Filtered schema only |

### Table 4 — Technique Overlap (which papers share ideas)

| Technique | Papers Using It |
|-----------|----------------|
| Execution as reward/feedback | SQL-R1, ExCoT, LitE-SQL |
| Multi-candidate generation + selection | XiYan-SQL, DeepEye-SQL |
| Separate schema linking stage | DTS-SQL, LitE-SQL, DeepEye-SQL |
| Agent-based pipeline | MAC-SQL |
| No fine-tuning required | DeepEye-SQL, SAFE-SQL |
| Iterative self-correction loop | MAC-SQL (Refiner), ExCoT, LitE-SQL |

---

## 5. What ADAPT-SQL Can Borrow — Prioritized Integration Table

Current state: ~92–93% EX Spider Dev (clean baseline), 11-step pipeline, Ollama local inference, no fine-tuning.

| # | Technique | From Paper | Effort | Expected EX Gain | How to Integrate |
|---|-----------|-----------|--------|-----------------|-----------------|
| 1 | **Composite example relevance scoring** | SAFE-SQL | Low | +1–2pp | In `pipeline/vector_search.py`: add structural + reasoning quality scoring on top of current cosine similarity |
| 2 | **Execution-consistency candidate selection** | XiYan-SQL | Low-Med | +1–3pp | Run pipeline 3–5× per query, execute all, group by result, pick consensus — no training needed |
| 3 | **Deterministic SQL unit test checkers** | DeepEye-SQL | Medium | +0.5–1pp | Extend `pipeline/validate_sql.py` with Logic Checker (JOIN scope) and Data Quality Checker (NULL/type) |
| 4 | **Selector agent (sub-database extraction)** | MAC-SQL | Medium | +1–2pp | Add a pre-generation step that prunes `schema_dict` to only the most relevant tables before Steps 3–6 |
| 5 | **Divide-and-Conquer CoT for NESTED** | DTS-SQL / MAC-SQL | Medium | +1–2pp | Extend `pipeline/decomposed_generation.py` with a D&C CoT path that generates sub-SQLs before combining |
| 6 | **Vector-DB schema linking** | LitE-SQL | Medium | +0.5–1pp (speed) | Pre-embed all column names offline, replace LLM schema linking with cosine lookup at inference |
| 7 | **Iterative DPO with execution oracle** | ExCoT / LitE-SQL | High | +3–5pp | Collect (correct, incorrect) pairs from existing checkpoint runs → DPO fine-tune on SOL A100 |
| 8 | **GRPO with layered rewards** | SQL-R1 | High | +4–6pp | Full RL fine-tuning: R_exec + R_result + R_format + R_length. Needs model weights + A100 |

**Immediate wins (no training, no infrastructure):**
- Items 1 and 2 can be implemented and tested in a day against existing checkpoints
- Item 3 extends existing `validate_sql.py` with 2–3 new rule-based checks
- Item 4 adds one LLM call before generation, with no other changes

---

## 6. Quick Reference Cards

---

**DTS-SQL** · Schema linking and SQL generation as two separate fine-tuned tasks
> Upper bound proof: perfect schema linking = 90.3% — the gap is diagnosable and fixable
> Top innovations: two-stage fine-tuning, embedding-based table ranking, upper-bound analysis
> Best result: 85.5% Spider Dev (7B model, competitive with GPT-4)

---

**DeepEye-SQL** · Software SDLC applied to SQL — no fine-tuning needed
> N-version programming + deterministic AST-based unit tests replace probabilistic self-correction
> Top innovations: Semantic Value Retrieval, SQL unit testing tool-chain (Syntax + Logic + Data Quality)
> Best result: **89.8% Spider Test** (best no-fine-tuning result)

---

**ExCoT** · Iterative on-policy DPO using SQLite as the ground truth oracle
> SQLite execution replaces human annotation for preference labels; 3-round iterative refinement
> Top innovations: on-policy iterative DPO, complex CoT variants, edit-distance pair selection
> Best result: 86.59% Spider Test (+10.14% over zero-shot CoT)

---

**LitE-SQL** · Lightweight inference via pre-computed schema embeddings + DPO alignment
> Vector DB stores schema embeddings offline; O(1) cosine lookup replaces LLM schema encoding
> Top innovations: HN-SupCon hard-negative contrastive loss, two-stage SFT→DPO
> Best result: 88.45% Spider Test, 72.1% BIRD Dev (7B model)

---

**MAC-SQL** · Three specialized agents collaborate: Selector → Decomposer → Refiner
> Each agent handles one concern; ablation shows Decomposer contributes +14pp, Refiner +10pp
> Top innovations: Selector agent for schema pruning, open-source SQL-Llama 7B
> Best result: ~89% Spider (Qwen2.5 backbone), 59.59% BIRD (GPT-4)

---

**SAFE-SQL** · Self-generated few-shot examples with composite quality filtering — zero training
> Generate 10 synthetic examples per query, score on semantic + structural + reasoning, filter at ≥8/10
> Top innovations: composite relevance score, threshold filtering, no retrieval infrastructure needed
> Best result: **87.9% Spider Dev** with GPT-4o (no training, no vector store)

---

**SQL-R1** · Layered reward RL on 7B model: the clearest fine-tuning recipe in the literature
> Four reward components (format + execution + result + length) with cold-start SFT before GRPO
> Top innovations: layered reward design, SynSQL-2.5M dataset, cold-start prevents reward hacking
> Best result: 78.1% Spider Dev, 87.6% BIRD Dev (Qwen2.5-Coder-7B)

---

**XiYan-SQL** · Multi-generator ensemble with consistency-based candidate selection
> Multiple generators with intentionally diverse reasoning styles; cluster by execution result, pick best
> Top innovations: PFKeyIdentifier for composite FKs, multi-format training, consistency clustering
> Best result: **89.65% Spider Test**, 75.63% BIRD Dev SOTA

---

**YORO** · Internalize schema knowledge into model weights — 47 tokens at inference
> Per-database expert fine-tuning eliminates all runtime schema encoding (97.6% input reduction)
> Top innovations: synthetic per-DB NLQ-SQL generation, knowledge internalization, expert routing
> Best result: 78.5% Spider Dev with average 31 input tokens

---

*9 papers — Spider 1.0 only. Last updated May 2026.*
