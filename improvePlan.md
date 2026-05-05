# ADAPT-SQL: Beat SOTA (89.8% EX) — Implementation Plan

## Context

ADAPT-SQL currently achieves 83.8% EX on Spider dev (1,034 queries). SOTA is DeepEye-SQL at 89.8%. The gap is ~6% absolute. This plan describes the minimum set of prompt-only, no-fine-tuning changes needed to plausibly cross 89%, using Qwen2.5-Coder-32B as the LLM backbone.

**Empirical constraint**: In this project, paper-reported gains transfer at ~20% (5× discount) due to local Ollama inference vs GPU-served fine-tuned models. The 32B model is assumed to reduce this discount to ~30-40% because it's the strongest instruction-tuned code model available without fine-tuning.

---

## Hard Failure Taxonomy (Current Baseline)

From CSV analysis of 167 queries wrong in both fix-3 and fix-4:
- SELECT wrong: 68.9% (wrong columns, missing DISTINCT, wrong aggregation)
- WHERE wrong: 55.1% (missing predicates, wrong filters)
- FROM/JOIN wrong: 41.3% (wrong tables, cartesian products)
- Set operations: ~3% of Spider dev; system **never** generates EXCEPT/INTERSECT/UNION — always replaced with WHERE filters
- Schema naming mismatches: ~8% of errors (pettype vs pet_type style)

---

## SOTA Reference

| System | EX | Core Technique |
|--------|-----|----------------|
| DeepEye-SQL | 89.8% | Checker chain + execution selection |
| AP-SQL | 89.7% | Schema simplification + multi-step prompting |
| XiYan-SQL | 89.65% | N diverse generators + majority voting |
| DAIL-SQL | ~86% | Structural few-shot (we already use this) |
| Pi-SQL | ~85% | Python intermediate repr |
| RESDSQL | ~84% | Schema linking + skeleton parsing |

---

## Phase A — Model Upgrade to Qwen2.5-Coder-32B

**Expected gain**: +1.5–2.5% EX  
**Implementation risk**: Low  
**Effort**: 1 parameter change + `ollama pull qwen2.5-coder:32b`

### Why
32B parameters vs ~7B for qwen3-coder. The 32B instruction-tuned model handles complex SQL patterns (nested subqueries, EXCEPT, correlated WHERE) substantially better in benchmarks. At 32B we expect the 5× discount to shrink — call it 3× — because the model itself is closer to what paper authors used.

### Files
- `core/adapt_baseline.py` line ~45: `model: str = "qwen3-coder"` → `"qwen2.5-coder:32b"`
- No other changes needed — all modules inherit model from the orchestrator

### Verify
Run 50 NESTED queries only: `python -c "from core.adapt_baseline import ADAPTBaseline; ..."`. Compare EX before/after on the 167 hard failures.

---

## Phase B — Set Operation Detection + Forced Generation

**Expected gain**: +0.8–1.5% EX  
**Implementation risk**: Low-Medium  
**Effort**: ~150 lines across 2–3 files

### Why
~30 Spider dev queries require EXCEPT/INTERSECT/UNION. Currently 0% accuracy on these. Even 50% hit rate = +1.5% EX. Signal detection is precise enough to avoid false positives.

### New File: `pipeline/set_op_detector.py`

```python
class SetOpDetector:
    EXCEPT_SIGNALS = ['but not', 'except', 'excluding', 'not in', 'who are not', 'that are not', 'never']
    INTERSECT_SIGNALS = ['both', 'who are also', 'that are also', 'in common', 'shared']
    UNION_SIGNALS = ['either', 'or both', 'combined with', 'as well as']

    def detect(self, question: str) -> dict:
        # Returns {'op': 'EXCEPT'|'INTERSECT'|'UNION'|None, 'confidence': float}
        q = question.lower()
        for sig in self.EXCEPT_SIGNALS:
            if sig in q:
                return {'op': 'EXCEPT', 'confidence': 0.8}
        for sig in self.INTERSECT_SIGNALS:
            if sig in q:
                return {'op': 'INTERSECT', 'confidence': 0.7}
        for sig in self.UNION_SIGNALS:
            if sig in q:
                return {'op': 'UNION', 'confidence': 0.6}
        return {'op': None, 'confidence': 0.0}
```

### Modify `pipeline/few_shot.py` and `pipeline/intermediate_repr.py`

Add `set_op_hint: str = ''` parameter to `generate()` methods. If non-empty, inject before the question in the prompt:

```
IMPORTANT: This question requires a {op} set operation.
Structure your answer as: SELECT ... FROM ... {op} SELECT ... FROM ...
Do NOT use WHERE filters to approximate set operations.
```

### Modify `core/adapt_baseline.py`

In `run_full_pipeline()`, before step 6:
```python
from pipeline.set_op_detector import SetOpDetector
set_op = SetOpDetector().detect(natural_query)
set_op_hint = f"Use {set_op['op']}" if set_op['op'] else ''
```
Pass `set_op_hint` to the generation step.

---

## Phase C — Multi-Candidate Diversity + Execution-Based Selection

**Expected gain**: +1.0–1.5% EX  
**Implementation risk**: Medium  
**Effort**: ~350 lines across 4 files + 1 new file

### Why
XiYan-SQL's main advantage. Generate N=3 candidates using different temperatures / prompt styles, execute all against SQLite, pick winner by majority result agreement. This catches cases where the primary generation fails but a secondary attempt succeeds.

### New File: `pipeline/candidate_selector.py`

```python
class CandidateSelector:
    def select(self, candidates: list[str], db_path: str, db_manager) -> str:
        """
        candidates: list of SQL strings (3-5)
        Returns: best SQL string

        Algorithm:
        1. Execute each candidate. Collect (sql, result_rows, success).
        2. Group successful candidates by result_set_key = frozenset of sorted row tuples.
        3. Majority group wins (most candidates agree on same result).
        4. Tie: prefer lowest temperature (index 0).
        5. No successful candidates: return candidates[0] (primary, let retry handle it).
        """
```

### Modify Generation Modules

In `pipeline/few_shot.py`:
```python
def generate_candidates(self, question, schema, fk, examples, n=3) -> list[str]:
    temps = [0.0, 0.3, 0.6]
    return [self._generate_at_temp(question, schema, fk, examples, t) for t in temps]
```

Same pattern in `pipeline/intermediate_repr.py` and `pipeline/decomposed_generation.py`.

### Modify `core/adapt_baseline.py`

After step 6 generates primary SQL, generate 2 more candidates and run selector:
```python
candidates = [primary_sql] + step6_module.generate_candidates(...)
best_sql = self.candidate_selector.select(candidates, db_path, db_manager)
results['step6']['final_sql'] = best_sql
results['step6']['candidates'] = candidates
```

### Execution overhead
Each query: 3 LLM calls + 3 SQLite executions. Adds ~60-90s per query. For batch processing, this is acceptable on SOL GPU node.

---

## Phase D — Schema Naming Normalization

**Expected gain**: +0.4–0.7% EX  
**Implementation risk**: Low  
**Effort**: ~60 lines in 2 files

### Why
~8% of failures involve schema naming mismatches (pettype vs pet_type, StudentInfo vs student_info). The fuzzy validator catches some but not all. Pre-normalizing tokens before schema linking improves match rate.

### Modify `pipeline/schema_linking.py`

Add normalization step before string matching:
```python
def _normalize_token(self, s: str) -> str:
    # camelCase → snake_case, remove underscores for comparison
    import re
    s = re.sub(r'([A-Z])', r'_\1', s).lower().lstrip('_')
    return re.sub(r'_+', '_', s)
```

Apply to both question tokens and schema column/table names during the string matching phase only (not in the final SQL generation).

### Modify `pipeline/validate_sql.py`

When schema validation fails, try normalized form before flagging as error:
```python
normalized_col = normalize_token(col)
if any(normalize_token(c) == normalized_col for c in schema_cols):
    # match found via normalization — pass
```

---

## Phase E — Skeleton-First NESTED Generation (RESDSQL-inspired)

**Expected gain**: +0.3–0.6% EX  
**Implementation risk**: Medium  
**Effort**: ~120 lines in `pipeline/decomposed_generation.py`

### Why
NESTED queries fail at 24.8% (75.2% EX). Generating a structural skeleton first (SELECT ___ FROM ___ WHERE EXISTS (SELECT ___ FROM ___ WHERE ___)) constrains the search space. This becomes candidate[2] for NESTED queries when multi-candidate is active.

### Modify `pipeline/decomposed_generation.py`

Add `generate_skeleton_first()`:
```python
def generate_skeleton_first(self, question, schema, fk, examples) -> str:
    # Step 1: generate skeleton
    skeleton_prompt = f"Generate only the SQL SKELETON for: {question}\n"
                      "Use ___ for all column names, conditions, and values.\n"
                      "Show the structure: SELECT, FROM, JOIN, WHERE, subquery pattern only."
    skeleton = self._call_llm(skeleton_prompt, temp=0.1)
    
    # Step 2: fill in skeleton
    fill_prompt = f"Fill in this SQL skeleton with real column names and values:\n{skeleton}\n"
                  f"Schema: {schema_str}\nQuestion: {question}"
    return self._call_llm(fill_prompt, temp=0.1)
```

Used as 3rd candidate for NESTED in the multi-candidate selector.

---

## Phase F — Complexity-Aware Few-Shot Selection

**Expected gain**: +0.2–0.4% EX  
**Implementation risk**: Low  
**Effort**: ~30 lines in `pipeline/vector_search.py`

### Why
Currently examples are retrieved by semantic similarity regardless of complexity. NESTED examples given to EASY queries add noise. Filter by same complexity tier first.

### Modify `pipeline/vector_search.py`

After FAISS retrieval, filter results where `example['complexity'] == query_complexity`. If fewer than `k` remain, relax the filter. This requires complexity labels in `vector_store/examples.json` — check if these exist; if not, derive them from SQL patterns at build time.

---

## Expected EX Trajectory

| Phase | Change | Central Gain | Cumulative EX |
|-------|--------|-------------|---------------|
| After-Fix-4 | Baseline | — | 83.8% |
| A | Qwen2.5-Coder-32B | +2.0% | 85.8% |
| B | Set operation detection | +1.2% | 87.0% |
| C | Multi-candidate + voting | +1.2% | 88.2% |
| D | Schema normalization | +0.5% | 88.7% |
| E | Skeleton-first NESTED | +0.4% | 89.1% |
| F | Complexity-aware FSS | +0.3% | 89.4% |

**Optimistic case** (upper bounds): 91.0% — beats SOTA  
**Central case** (midpoints): 89.4% — beats SOTA  
**Conservative case** (lower bounds): 87.1% — does not beat SOTA

**Honest assessment**: Beating 89.8% requires Phase A to outperform the 5× discount AND both B and C hit their upper estimates. The central case barely beats SOTA at 89.4%. If after Phase A+B+C the EX is below 87%, the gap cannot be closed with the remaining phases alone — that would indicate the model underperforms expectations and different prompting strategies for schema linking (AP-SQL style) should be explored.

---

## Implementation Order (EX gain / effort ratio)

1. **Phase A** — 1 line, +2% expected. Do this first, re-run 50 hard NESTED queries to calibrate actual gain.
2. **Phase B** — ~150 lines, very targeted, low false-positive risk. Do before multi-candidate to know baseline gain on set op queries.
3. **Phase C** — ~350 lines, requires new file + wiring. Biggest structural change.
4. **Phase D** — ~60 lines, low risk, incremental.
5. **Phase E** — ~120 lines, adds 3rd NESTED candidate.
6. **Phase F** — ~30 lines, last resort gain.

---

## Files to Modify / Create

| File | Action | Phase |
|------|--------|-------|
| `core/adapt_baseline.py` | Change default model; wire set_op_hint, multi-candidate, selector | A, B, C |
| `pipeline/set_op_detector.py` | CREATE — signal detection + op classification | B |
| `pipeline/few_shot.py` | Add `generate_candidates(n=3)`, accept `set_op_hint` | B, C |
| `pipeline/intermediate_repr.py` | Add `generate_candidates(n=3)`, accept `set_op_hint` | B, C |
| `pipeline/decomposed_generation.py` | Add `generate_candidates(n=3)`, add `generate_skeleton_first()` | C, E |
| `pipeline/candidate_selector.py` | CREATE — majority voting on result sets | C |
| `pipeline/schema_linking.py` | Add `_normalize_token()`, apply to string matching | D |
| `pipeline/validate_sql.py` | Try normalized form before flagging schema error | D |
| `pipeline/vector_search.py` | Filter by complexity tier after FAISS retrieval | F |
| `ui/batch_utils.py` | Add `Set_Op_Detected`, `Candidates_Generated`, `Winner_Index` CSV columns | C, B |

---

## Verification Plan

1. **After Phase A**: Run 100 NESTED queries. Expect EX to rise from 75.2% → 78%+. If not, 32B model is underperforming — investigate prompt format compatibility.
2. **After Phase B**: Manually inspect all queries where set_op_detector fires. Confirm SQL uses EXCEPT/INTERSECT/UNION. Check false positive rate on non-set-op queries.
3. **After Phase C**: Full dev set run (1,034 queries). Check `Winner_Index` distribution — if 0 wins 90%+, diversity is insufficient (raise temperatures).
4. **Full run**: Compare results CSV against `RESULTS/after-fix-4/` baseline. Verify per-complexity EX improvements match expectations.

---

## Risk Factors

- **Qwen2.5-Coder-32B VRAM**: ~20GB in fp16. ASU SOL A100 (40GB) handles this comfortably. Request with `--gres=gpu:a100:1`.
- **Multi-candidate latency**: 3× LLM calls per query. Batch processing will take ~3× longer. Use checkpoint saving every 10 queries (already implemented in batch_utils.py).
- **Set op false positives**: Phrases like "except" appear in non-set-op contexts ("except for ties"). Mitigated by: (1) still running checker chain post-generation, (2) multi-candidate selector will prefer the correct SQL if the set op version fails execution.
- **Ollama host compatibility**: `ollama.chat()` is patched in `core/adapt_baseline.py` to read OLLAMA_HOST at call time. Do not move this patch. All new modules must call `ollama.chat()` (not direct client instantiation) to respect the patch.
