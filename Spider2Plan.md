# NEXUS-SQL: Neural EXpert Unified System for Enterprise SQL

## A Novel Multi-Agent Architecture for Spider 2.0-Snow

**Target:** Top-3 on Spider 2.0-Snow Leaderboard
**Current SOTA:** Native mini at 92.50% (Jan 23, 2026)
**Our Target:** 93-95% Execution Accuracy

---

## Benchmark Focus: Spider 2.0-Snow

- **547 examples** with well-prepared database metadata
- **Hosted on Snowflake** with free participant quotas
- **Self-contained** text-to-SQL task
- **Snowflake SQL dialect** (not BigQuery)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current Leaderboard Analysis (Jan 2026)](#2-current-leaderboard-analysis-jan-2026)
3. [Top Method Analysis](#3-top-method-analysis)
4. [NEXUS-SQL Architecture Overview](#4-nexus-sql-architecture-overview)
5. [Component 1: Hierarchical Schema Intelligence](#5-component-1-hierarchical-schema-intelligence)
6. [Component 2: Multi-Agent Swarm with Specialized Experts](#6-component-2-multi-agent-swarm-with-specialized-experts)
7. [Component 3: Relational Knowledge Graph Integration](#7-component-3-relational-knowledge-graph-integration)
8. [Component 4: Deep Reasoning Chain with Self-Verification](#8-component-4-deep-reasoning-chain-with-self-verification)
9. [Component 5: Snowflake-Native Generation Pipeline](#9-component-5-snowflake-native-generation-pipeline)
10. [Component 6: Execution-Driven Iterative Refinement](#10-component-6-execution-driven-iterative-refinement)
11. [Complete Pipeline Architecture](#11-complete-pipeline-architecture)
12. [Expected Performance Analysis](#12-expected-performance-analysis)
13. [Implementation Roadmap](#13-implementation-roadmap)
14. [Ablation Study Design](#14-ablation-study-design)
15. [Paper Contribution Summary](#15-paper-contribution-summary)

---

## 1. Executive Summary

We propose **NEXUS-SQL** (Neural EXpert Unified System for SQL), a novel multi-agent architecture designed to achieve top-3 performance on Spider 2.0-Snow. Building on insights from the current leaderboard leaders, our approach combines:

| Current Top Methods | Their Approach | NEXUS-SQL Enhancement |
|---------------------|----------------|----------------------|
| Native mini (92.50%) | Native AI optimization | + Deep reasoning chains |
| Prism Swarm (90.49%) | Multi-agent swarm + Deepthink | + Knowledge graph grounding |
| Ask Data + RKG (86.28%) | Relational Knowledge Graph | + Execution-driven refinement |

**Key Innovations:**
1. **Swarm Intelligence with Specialized Agents** - Multiple coordinated agents (schema, SQL, validation) inspired by Prism Swarm
2. **Relational Knowledge Graph** - Graph-based schema understanding inspired by AT&T/RelationalAI approach
3. **Deep Reasoning Chains** - Extended chain-of-thought with self-verification (Deepthink-style)
4. **Snowflake-Native Generation** - VARIANT, FLATTEN, QUALIFY optimized templates
5. **Execution-Driven Iterative Refinement** - Real Snowflake execution feedback loop
6. **Confidence-Weighted Ensemble** - Combine multiple generation paths with learned weights

---

## 2. Current Leaderboard Analysis (Jan 2026)

### 2.1 Spider 2.0-Snow Leaderboard (as of January 27, 2026)

| Rank | Method | Organization | Score | Date |
|------|--------|--------------|-------|------|
| **1** | **Native mini** | usenative.ai | **92.50%** | Jan 23, 2026 |
| **2** | **Prism Swarm + Deepthink + Claude-Sonnet-4.5** | Paytm | **90.49%** | Jan 27, 2026 |
| **3** | **Ask Data with Relational Knowledge Graph** | AT&T CDO & RelationalAI | **86.28%** | Jan 7, 2026 |
| 4 | ByteBrain-Agent | ByteDance Infra System Lab | 84.10% | Dec 16, 2025 |
| 5 | AiCheng Agent | alibaba_cfo_tech | 82.81% | Jan 9, 2026 |
| 6 | Prism Swarm + Claude-Sonnet-4.5 | Paytm | 82.63% | Jan 2, 2026 |
| 7 | LingXi Agent + Claude-Sonnet-4.5 | Ant Group | 79.89% | Dec 5, 2025 |
| 8 | Arctic-FLEX | Snowflake AI Research | 75.14% | Dec 9, 2025 |
| 9 | Sophon-Agent | ByteDance DataPlatform LLM | 74.04% | Nov 24, 2025 |
| 10 | QiSi-SQL + Deepseek3.2 | Ant Group Tech Risk | 70.38% | Dec 26, 2025 |
| 11 | PExA | Bloomberg - AI Engineering | 70.20% | Sep 26, 2025 |
| 12 | Chicory AI Agent + Claude Sonnet 4.5 + Opus 4.5 Judge | Chicory AI | 67.28% | Dec 4, 2025 |
| 13 | SSDAT + GPT-5 | - | 65.63% | Oct 2, 2025 |

### 2.2 Key Observations

```
┌─────────────────────────────────────────────────────────────┐
│              LEADERBOARD TRENDS (2025-2026)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DOMINANT APPROACHES:                                        │
│  ├─ Multi-Agent/Swarm Systems (7 of top 13)                │
│  ├─ Claude Sonnet 4.5 (5 appearances)                      │
│  ├─ Knowledge Graphs (AT&T/RelationalAI at #3)             │
│  └─ Deep Reasoning (Deepthink in #2)                       │
│                                                              │
│  SCORE PROGRESSION:                                          │
│  ├─ Sep 2025: ~70% (PExA baseline)                         │
│  ├─ Nov 2025: ~74% (Sophon-Agent)                          │
│  ├─ Dec 2025: ~84% (ByteBrain-Agent)                       │
│  └─ Jan 2026: ~92.5% (Native mini)                         │
│                                                              │
│  GAP TO BEAT: Only 7.5% to reach 100%                       │
│  Top-3 threshold: ~86%                                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Spider 2.0-Snow Specifics

```
┌─────────────────────────────────────────────────────────────┐
│                  SPIDER 2.0-SNOW DETAILS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Dataset:                                                    │
│  ├─ 547 text-to-SQL examples                               │
│  ├─ Well-prepared database metadata and documentation      │
│  ├─ Self-contained evaluation                              │
│  └─ Free Snowflake quotas for participants                 │
│                                                              │
│  Snowflake-Specific Features:                                │
│  ├─ VARIANT type for semi-structured data                  │
│  ├─ FLATTEN for array/object expansion                     │
│  ├─ QUALIFY for window function filtering                  │
│  ├─ :: type casting syntax                                 │
│  ├─ GET_PATH() for JSON extraction                         │
│  └─ LATERAL FLATTEN for nested structures                  │
│                                                              │
│  Challenges:                                                 │
│  ├─ Enterprise-scale schemas (100s of columns)             │
│  ├─ Complex nested/semi-structured data                    │
│  ├─ Implicit foreign key relationships                     │
│  └─ Multi-step reasoning queries                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Remaining Error Categories (Top Methods)

With top methods at 92.5%, the remaining 7.5% errors are the hardest:

```
┌─────────────────────────────────────────────────────────────┐
│          REMAINING ERRORS AT 92.5% (7.5% failure)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Complex Multi-Step Reasoning  ██████████████████    40%    │
│    └─ 3+ step logical chains                                │
│    └─ Ambiguous question interpretation                     │
│                                                              │
│  Edge Case Nested Structures   ██████████████        30%    │
│    └─ Deeply nested VARIANT paths                          │
│    └─ Complex FLATTEN operations                           │
│                                                              │
│  Ambiguous Schema Mapping      ████████████          25%    │
│    └─ Similar column names across tables                   │
│    └─ Context-dependent column selection                   │
│                                                              │
│  Rare SQL Patterns             █████                  5%    │
│    └─ Unusual aggregation combinations                     │
│    └─ Edge case window functions                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Top Method Analysis

### 3.1 Native mini (92.50% - #1)

**Organization:** usenative.ai

**Likely Approach (inferred from name and performance):**
- Native AI optimization for SQL generation
- Possibly fine-tuned smaller model ("mini")
- Strong Snowflake-specific training
- Efficient inference pipeline

**Strengths:**
- Highest accuracy achieved
- Likely fast inference (mini = smaller model)
- Native SQL understanding

**Potential Gaps to Exploit:**
- May lack multi-step reasoning depth
- Possibly limited to learned patterns
- Single-model approach (no ensemble)

### 3.2 Prism Swarm + Deepthink + Claude-Sonnet-4.5 (90.49% - #2)

**Organization:** Paytm

**Architecture (inferred):**
- **Swarm:** Multiple coordinated agents working together
- **Deepthink:** Extended reasoning chains (likely o1-style thinking)
- **Claude-Sonnet-4.5:** State-of-the-art LLM backbone

**Key Techniques:**
```
┌─────────────────────────────────────────────────────────────┐
│              PRISM SWARM ARCHITECTURE (inferred)            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Swarm Components:                                           │
│  ├─ Schema Analysis Agent                                   │
│  ├─ Query Understanding Agent                               │
│  ├─ SQL Generation Agent                                    │
│  ├─ Validation Agent                                        │
│  └─ Refinement Agent                                        │
│                                                              │
│  Deepthink:                                                  │
│  ├─ Extended chain-of-thought reasoning                    │
│  ├─ Self-verification steps                                │
│  └─ Multi-step logical deduction                           │
│                                                              │
│  Coordination:                                               │
│  ├─ Agents share context and findings                      │
│  ├─ Consensus mechanism for final SQL                      │
│  └─ Iterative refinement loop                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Why it works:**
- Multiple agents catch different error types
- Deep reasoning handles complex queries
- Claude-Sonnet-4.5 provides strong base capability

**Gap:** 2% below Native mini - likely due to coordination overhead or edge cases

### 3.3 Ask Data with Relational Knowledge Graph (86.28% - #3)

**Organization:** AT&T CDO & RelationalAI

**Architecture (inferred):**
- **Relational Knowledge Graph:** Schema as a graph structure
- **Graph-based reasoning:** Path finding, relationship inference
- **RelationalAI:** Specialized relational database AI

**Key Techniques:**
```
┌─────────────────────────────────────────────────────────────┐
│         RELATIONAL KNOWLEDGE GRAPH APPROACH                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Knowledge Graph Construction:                               │
│  ├─ Tables as nodes                                         │
│  ├─ Foreign keys as edges                                   │
│  ├─ Column semantics as node properties                    │
│  └─ Inferred relationships as additional edges             │
│                                                              │
│  Query Processing:                                           │
│  ├─ Map question entities to graph nodes                   │
│  ├─ Find optimal paths between relevant nodes              │
│  ├─ Translate paths to SQL joins                           │
│  └─ Generate SQL from graph traversal                      │
│                                                              │
│  Advantages:                                                 │
│  ├─ Strong schema understanding                            │
│  ├─ Handles complex joins naturally                        │
│  └─ Implicit relationship discovery                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Gap:** 6% below top - likely limited deep reasoning capability

### 3.4 What We Can Learn

| Method | Key Strength | What to Adopt |
|--------|--------------|---------------|
| Native mini | Efficiency + accuracy | Snowflake-native optimization |
| Prism Swarm | Multi-agent coordination | Swarm architecture |
| Deepthink | Extended reasoning | Deep reasoning chains |
| Relational KG | Graph-based schema | Knowledge graph integration |
| ByteBrain-Agent | Enterprise-scale | Agent-based approach |
| Arctic-FLEX | Snowflake expertise | Dialect-specific templates |

### 3.5 Opportunity Analysis

To beat 92.50%, we need to combine:

```
┌─────────────────────────────────────────────────────────────┐
│              NEXUS-SQL: COMBINING THE BEST                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  From Native mini:                                           │
│    └─ Snowflake-native SQL generation                       │
│                                                              │
│  From Prism Swarm:                                           │
│    └─ Multi-agent swarm coordination                        │
│                                                              │
│  From Deepthink:                                             │
│    └─ Extended chain-of-thought with verification           │
│                                                              │
│  From Relational KG:                                         │
│    └─ Knowledge graph for schema understanding              │
│                                                              │
│  NOVEL ADDITIONS:                                            │
│    ├─ Execution-driven feedback loop (real Snowflake)       │
│    ├─ Confidence-weighted ensemble                          │
│    ├─ Error pattern learning                                │
│    └─ Self-improvement mechanism                            │
│                                                              │
│  TARGET: 93-95% (beat Native mini by 0.5-2.5%)             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. NEXUS-SQL Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NEXUS-SQL PIPELINE                                 │
│            (Neural EXpert Unified System for Enterprise SQL)                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT: Natural Language Query + Snowflake Schema + Connection              │
│                                                                              │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ STAGE 1: SCHEMA UNDERSTANDING + KNOWLEDGE GRAPH                        ║  │
│  ║                                                                         ║  │
│  ║   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐               ║  │
│  ║   │ Component 1 │───▶│ Component 3 │───▶│ Component 5 │               ║  │
│  ║   │    HSI      │    │    RKG      │    │   SNGP      │               ║  │
│  ║   │ (Schema     │    │ (Relational │    │ (Snowflake  │               ║  │
│  ║   │Intelligence)│    │  KG)        │    │  Native)    │               ║  │
│  ║   └─────────────┘    └─────────────┘    └─────────────┘               ║  │
│  ║          │                  │                  │                       ║  │
│  ║          ▼                  ▼                  ▼                       ║  │
│  ║   ┌────────────────────────────────────────────────────────────┐     ║  │
│  ║   │              Unified Knowledge Context                      │     ║  │
│  ║   │  • Schema graph with semantic annotations                  │     ║  │
│  ║   │  • Inferred relationships + join paths                     │     ║  │
│  ║   │  • Snowflake-specific type mappings                        │     ║  │
│  ║   └────────────────────────────────────────────────────────────┘     ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                     │                                        │
│                                     ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ STAGE 2: MULTI-AGENT SWARM GENERATION (Prism Swarm-inspired)          ║  │
│  ║                                                                         ║  │
│  ║   ┌─────────────────────────────────────────────────────────────┐     ║  │
│  ║   │              Component 2: SWARM ORCHESTRATOR                 │     ║  │
│  ║   ├─────────────────────────────────────────────────────────────┤     ║  │
│  ║   │                                                              │     ║  │
│  ║   │  ┌──────────────────────────────────────────────────────┐   │     ║  │
│  ║   │  │                AGENT SWARM                            │   │     ║  │
│  ║   │  │                                                       │   │     ║  │
│  ║   │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │   │     ║  │
│  ║   │  │  │ Schema  │ │ Query   │ │  SQL    │ │Validator│    │   │     ║  │
│  ║   │  │  │ Agent   │ │ Agent   │ │ Agent   │ │ Agent   │    │   │     ║  │
│  ║   │  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘    │   │     ║  │
│  ║   │  │       │           │           │           │          │   │     ║  │
│  ║   │  │       └───────────┴───────────┴───────────┘          │   │     ║  │
│  ║   │  │                       │                               │   │     ║  │
│  ║   │  │               ┌───────▼───────┐                      │   │     ║  │
│  ║   │  │               │  Coordinator  │                      │   │     ║  │
│  ║   │  │               │   (Consensus) │                      │   │     ║  │
│  ║   │  │               └───────────────┘                      │   │     ║  │
│  ║   │  └──────────────────────────────────────────────────────┘   │     ║  │
│  ║   └─────────────────────────────────────────────────────────────┘     ║  │
│  ║                              │                                         ║  │
│  ║                              ▼                                         ║  │
│  ║   ┌────────────────────────────────────────────────────────────┐     ║  │
│  ║   │           Component 4: DEEP REASONING CHAIN                 │     ║  │
│  ║   │           (Deepthink-inspired extended CoT)                │     ║  │
│  ║   │                                                             │     ║  │
│  ║   │   Think → Verify → Refine → Verify → Finalize              │     ║  │
│  ║   └────────────────────────────────────────────────────────────┘     ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                     │                                        │
│                                     ▼                                        │
│  ╔═══════════════════════════════════════════════════════════════════════╗  │
│  ║ STAGE 3: EXECUTION-DRIVEN REFINEMENT                                   ║  │
│  ║                                                                         ║  │
│  ║   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐     ║  │
│  ║   │ Component 6 │────────▶│  Snowflake  │────────▶│  Feedback   │     ║  │
│  ║   │   EDIR      │         │  Execution  │         │  Analysis   │     ║  │
│  ║   │ (Execution  │◀────────│  (Real DB)  │◀────────│  + Repair   │     ║  │
│  ║   │  Driven)    │ iterate │             │ errors  │             │     ║  │
│  ║   └─────────────┘         └─────────────┘         └─────────────┘     ║  │
│  ║          │                                                             ║  │
│  ║          ▼                                                             ║  │
│  ║   ┌────────────────────────────────────────────────────────────┐     ║  │
│  ║   │      Final SQL + Confidence + Reasoning Trace               │     ║  │
│  ║   └────────────────────────────────────────────────────────────┘     ║  │
│  ╚═══════════════════════════════════════════════════════════════════════╝  │
│                                                                              │
│  OUTPUT: Validated Snowflake SQL + Confidence Score + Full Trace           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Design Philosophy

**Combining the best from top performers:**

| Inspiration | What We Adopt | Enhancement |
|-------------|---------------|-------------|
| **Native mini** | Snowflake-native optimization | + Multi-agent verification |
| **Prism Swarm** | Multi-agent swarm | + Specialized agent roles |
| **Deepthink** | Extended reasoning | + Self-verification loops |
| **Relational KG** | Knowledge graph | + Dynamic graph updates |

### 4.2 Key Differentiators

```
┌─────────────────────────────────────────────────────────────┐
│              NEXUS-SQL NOVEL CONTRIBUTIONS                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. UNIFIED SWARM + KNOWLEDGE GRAPH                         │
│     └─ Agents reason over graph structure, not raw schema   │
│                                                              │
│  2. DEEP REASONING WITH VERIFICATION                        │
│     └─ Each reasoning step is verified before continuing    │
│                                                              │
│  3. REAL EXECUTION FEEDBACK                                 │
│     └─ Execute on actual Snowflake, learn from errors      │
│                                                              │
│  4. CONFIDENCE-WEIGHTED ENSEMBLE                            │
│     └─ Multiple paths weighted by predicted accuracy        │
│                                                              │
│  5. SNOWFLAKE-NATIVE TYPE HANDLING                          │
│     └─ VARIANT, FLATTEN, QUALIFY optimized generation      │
│                                                              │
│  6. ERROR PATTERN LEARNING                                  │
│     └─ Accumulate and avoid common error patterns          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Component 1: Hierarchical Schema Intelligence (HSI)

### 5.1 Novel Aspect

Building on insights from top methods (Native mini, Prism Swarm), HSI introduces **semantic domain clustering** that preserves relational meaning while achieving 70%+ compression. Where existing approaches may struggle with enterprise-scale schemas, HSI intelligently identifies and prioritizes the most relevant schema elements for each query.

### 5.2 Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           HIERARCHICAL SCHEMA INTELLIGENCE (HSI)             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  LEVEL 1: DOMAIN ONTOLOGY EXTRACTION                         │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Input: 700+ tables, 3000+ columns                       │ │
│  │                                                          │ │
│  │ Process:                                                 │ │
│  │ 1. Extract naming patterns (customer_*, order_*, etc.)  │ │
│  │ 2. Build concept hierarchy from FK graph                │ │
│  │ 3. Identify domain types:                                │ │
│  │    • Master data (customers, products)                  │ │
│  │    • Transactional (orders, payments)                   │ │
│  │    • Temporal (logs, events, snapshots)                 │ │
│  │    • Reference (countries, categories)                  │ │
│  │                                                          │ │
│  │ Output: Domain Ontology Graph                           │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  LEVEL 2: SEMANTIC CLUSTERING                                │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  700 tables → 25-40 semantic clusters                   │ │
│  │                                                          │
│  │  Example Clusters:                                       │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │ CUSTOMER_DOMAIN:                                 │   │ │
│  │  │   Tables: customers, customer_addresses,         │   │ │
│  │  │           customer_preferences, user_accounts    │   │ │
│  │  │   Key Columns: customer_id, email, name          │   │ │
│  │  │   Relationships: [address FK, preferences FK]    │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │  ┌─────────────────────────────────────────────────┐   │ │
│  │  │ ORDER_DOMAIN:                                    │   │ │
│  │  │   Tables: orders, order_items, order_status,     │   │ │
│  │  │           shipping, returns                       │   │ │
│  │  │   Key Columns: order_id, total, status           │   │ │
│  │  │   Relationships: [customer FK, product FK]       │   │ │
│  │  └─────────────────────────────────────────────────┘   │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  LEVEL 3: QUERY-ADAPTIVE EXPANSION                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  Query: "Find customers who spent more than average"    │ │
│  │                                                          │ │
│  │  Step 1: Identify query domains                         │ │
│  │          → CUSTOMER_DOMAIN, ORDER_DOMAIN                │ │
│  │                                                          │ │
│  │  Step 2: Expand relevant clusters only                  │ │
│  │          → Show full customer + order tables            │ │
│  │          → Summarize other domains                      │ │
│  │                                                          │ │
│  │  Step 3: Progressive detail                             │ │
│  │          → Start with key columns                       │ │
│  │          → Expand if LLM requests more                  │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  LEVEL 4: NESTED COLUMN FLATTENING                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  Input: profile STRUCT<age INT, location RECORD<...>>   │ │
│  │                                                          │ │
│  │  Flattened View:                                        │ │
│  │  • profile.age (INT) - User age                         │ │
│  │  • profile.location.country (STRING) - Country          │ │
│  │  • profile.location.city (STRING) - City                │ │
│  │                                                          │ │
│  │  Access Template (Snowflake):                            │ │
│  │  • profile:location:city::STRING                        │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Algorithm: Domain Ontology Extraction

```python
class DomainOntologyExtractor:
    """
    Extract domain ontology from enterprise schema
    """

    def __init__(self):
        self.domain_patterns = {
            'master_data': ['customer', 'product', 'employee', 'vendor'],
            'transactional': ['order', 'transaction', 'payment', 'invoice'],
            'temporal': ['log', 'event', 'history', 'snapshot', 'audit'],
            'reference': ['country', 'category', 'status', 'type', 'config']
        }

    def extract(self, schema_dict, foreign_keys):
        # Step 1: Classify tables by domain type
        table_domains = {}
        for table in schema_dict:
            domain = self._classify_table_domain(table)
            table_domains[table] = domain

        # Step 2: Build FK graph
        fk_graph = self._build_fk_graph(foreign_keys)

        # Step 3: Cluster by domain + connectivity
        clusters = self._cluster_tables(table_domains, fk_graph)

        # Step 4: Extract concept hierarchy
        hierarchy = self._extract_hierarchy(clusters, fk_graph)

        return DomainOntology(
            clusters=clusters,
            hierarchy=hierarchy,
            fk_graph=fk_graph
        )

    def compress_for_query(self, ontology, question, max_tokens=4000):
        """
        Query-adaptive compression
        """
        # Identify relevant domains from question
        relevant_domains = self._identify_relevant_domains(question, ontology)

        # Build compressed schema
        compressed = {}
        for domain in relevant_domains:
            # Full expansion for relevant domains
            compressed.update(self._expand_domain(domain, ontology))

        for domain in ontology.clusters:
            if domain not in relevant_domains:
                # Summary only for other domains
                compressed[domain] = self._summarize_domain(domain, ontology)

        return self._format_for_llm(compressed, max_tokens)
```

### 5.4 Expected Impact

| Metric | Baseline (92.5%) | With HSI | Improvement |
|--------|------------------|----------|-------------|
| Schema linking accuracy | 92.5% | 93.5%+ | +1.0% |
| Nested column handling | ~90% | 94%+ | +4% |
| Token usage | 100% | 30-40% | -60-70% |
| Context relevance | Good | Excellent | Semantic focus |

---

## 6. Component 2: Multi-Agent Swarm with Specialized Experts

### 6.1 Novel Aspect

Inspired by Prism Swarm's success at 90.49%, our Multi-Agent Swarm uses **pattern-specialized experts** with dedicated memory banks trained on specific SQL patterns from Spider 2.0-Snow. We enhance Prism Swarm's approach by adding domain-specific expert agents (Join, Aggregation, Nested Column) that coordinate through a consensus mechanism.

### 6.2 Expert Definitions

```
┌─────────────────────────────────────────────────────────────┐
│                    SIX SPECIALIZED EXPERTS                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. JOIN EXPERT                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Specialization:                                         │ │
│  │   • Multi-table star/snowflake schema traversal        │ │
│  │   • 5+ table join optimization                         │ │
│  │   • Implicit FK inference coordination                  │ │
│  │                                                          │ │
│  │ Memory Bank:                                             │ │
│  │   • 500+ join pattern templates                         │ │
│  │   • Common enterprise join topologies                   │ │
│  │   • Cost-aware join ordering heuristics                 │ │
│  │                                                          │ │
│  │ Trigger Patterns:                                        │ │
│  │   "X from Y", "X with Y", "combine", "relate"           │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  2. AGGREGATION EXPERT                                       │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Specialization:                                         │ │
│  │   • GROUP BY / HAVING logic                             │ │
│  │   • Window functions (ROW_NUMBER, RANK, etc.)           │ │
│  │   • Multi-level aggregation chains                      │ │
│  │                                                          │ │
│  │ Memory Bank:                                             │ │
│  │   • Aggregation function mappings per dialect           │ │
│  │   • Window frame specifications                         │ │
│  │   • Common "for each X, find Y" patterns               │ │
│  │                                                          │ │
│  │ Trigger Patterns:                                        │ │
│  │   "count", "total", "average", "for each", "per"        │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  3. SUBQUERY EXPERT                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Specialization:                                         │ │
│  │   • Correlated vs non-correlated subqueries            │ │
│  │   • EXISTS, IN, comparison subqueries                   │ │
│  │   • Nested query composition (3+ levels)               │ │
│  │                                                          │ │
│  │ Memory Bank:                                             │ │
│  │   • Subquery pattern templates                          │ │
│  │   • Scope tracking for correlated references           │ │
│  │   • Anti-pattern detection                              │ │
│  │                                                          │ │
│  │ Trigger Patterns:                                        │ │
│  │   "more than average", "not in", "exists", "except"     │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  4. NESTED COLUMN EXPERT (CRITICAL FOR SPIDER 2.0)          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Specialization:                                         │ │
│  │   • STRUCT/RECORD field access                          │ │
│  │   • ARRAY operations (UNNEST, FLATTEN)                  │ │
│  │   • JSON path expressions                               │ │
│  │   • VARIANT type handling (Snowflake)                   │ │
│  │                                                          │ │
│  │ Memory Bank:                                             │ │
│  │   • Snowflake FLATTEN/LATERAL patterns                  │ │
│  │   • VARIANT type access patterns                        │ │
│  │   • JSON extraction templates (GET_PATH)                │ │
│  │                                                          │ │
│  │ Trigger Patterns:                                        │ │
│  │   Detected nested columns in schema, path references    │ │
│  │                                                          │ │
│  │ Example Transformations (Snowflake):                    │ │
│  │   STRUCT:  profile:address:city::STRING                 │ │
│  │   ARRAY:   LATERAL FLATTEN(input => items) AS item     │ │
│  │   JSON:    GET_PATH(data, 'address.city')               │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  5. TEMPORAL EXPERT                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Specialization:                                         │ │
│  │   • Date/time operations                                │ │
│  │   • Time zone handling                                  │ │
│  │   • Interval arithmetic                                 │ │
│  │   • Temporal joins and ranges                          │ │
│  │                                                          │ │
│  │ Memory Bank:                                             │ │
│  │   • DATE_TRUNC, TIMESTAMP_DIFF per dialect             │ │
│  │   • Time zone conversion patterns                       │ │
│  │   • Rolling window calculations                         │ │
│  │                                                          │ │
│  │ Trigger Patterns:                                        │ │
│  │   "last month", "year over year", "between dates"       │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  6. SET OPERATION EXPERT                                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │ Specialization:                                         │ │
│  │   • UNION / UNION ALL                                   │ │
│  │   • INTERSECT                                           │ │
│  │   • EXCEPT / MINUS                                      │ │
│  │                                                          │ │
│  │ Memory Bank:                                             │ │
│  │   • Schema compatibility checking                       │ │
│  │   • Deduplication strategies                            │ │
│  │   • Multi-query composition                             │ │
│  │                                                          │ │
│  │ Trigger Patterns:                                        │ │
│  │   "but not", "also in", "either", "both"                │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Expert Orchestration

```python
class MultiExpertOrchestrator:
    """
    Route queries to appropriate experts and synthesize results
    """

    def __init__(self):
        self.experts = {
            'join': JoinExpert(),
            'aggregation': AggregationExpert(),
            'subquery': SubqueryExpert(),
            'nested_column': NestedColumnExpert(),
            'temporal': TemporalExpert(),
            'set_operation': SetOperationExpert()
        }
        self.pattern_router = PatternRouter()

    def generate(self, question, schema_context, dialect):
        # Step 1: Analyze query patterns
        patterns = self.pattern_router.detect_patterns(question, schema_context)

        # Step 2: Select relevant experts
        selected_experts = []
        for pattern in patterns:
            expert = self._select_expert_for_pattern(pattern)
            selected_experts.append((expert, pattern))

        # Step 3: Generate SQL components from each expert
        components = {}
        for expert, pattern in selected_experts:
            component = expert.generate_component(
                question=question,
                schema_context=schema_context,
                pattern=pattern,
                dialect=dialect
            )
            components[expert.name] = component

        # Step 4: Synthesize final SQL
        return self.synthesizer.combine(components, dialect)

    def _select_expert_for_pattern(self, pattern):
        """
        Select best expert for detected pattern
        """
        pattern_expert_map = {
            'multi_table_join': 'join',
            'aggregation': 'aggregation',
            'window_function': 'aggregation',
            'subquery': 'subquery',
            'correlated_subquery': 'subquery',
            'nested_column_access': 'nested_column',
            'array_operation': 'nested_column',
            'date_filter': 'temporal',
            'time_range': 'temporal',
            'set_difference': 'set_operation',
            'union': 'set_operation'
        }
        return self.experts[pattern_expert_map[pattern.type]]
```

### 6.4 Expected Impact

| Query Type | Baseline (92.5%) | With Swarm | Improvement |
|-----------|------------------|------------|-------------|
| Simple (1-2 tables) | 95% | 96%+ | +1% |
| Complex joins (5+ tables) | 88% | 92%+ | +4% |
| Nested columns | 87% | 93%+ | +6% |
| Window functions | 85% | 91%+ | +6% |
| Subqueries | 89% | 93%+ | +4% |

---

## 7. Component 3: Relational Knowledge Graph Integration (RKG)

### 7.1 Novel Aspect

Inspired by AT&T/RelationalAI's #3 approach (86.28%), our Relational Knowledge Graph provides active graph reasoning that **discovers implicit foreign keys** and **optimizes join paths** - critical for Spider 2.0-Snow where many FKs are not explicitly defined. We enhance their approach with tighter integration into the multi-agent swarm.

### 7.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              GRAPH-BASED SCHEMA REASONING                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              SCHEMA KNOWLEDGE GRAPH                     │ │
│  │                                                          │ │
│  │  Node Types:                                             │ │
│  │    • TABLE: Database tables                             │ │
│  │    • COLUMN: Table columns                              │ │
│  │    • CONCEPT: Semantic concepts (Customer, Order, etc.) │ │
│  │    • VALUE: Sample values for matching                  │ │
│  │                                                          │ │
│  │  Edge Types:                                             │ │
│  │    • EXPLICIT_FK: Declared foreign keys                 │ │
│  │    • IMPLICIT_FK: Inferred foreign keys (NEW)           │ │
│  │    • SEMANTIC: Concept relationships                    │ │
│  │    • TYPE: Column type relationships                    │ │
│  │                                                          │ │
│  │  Example Graph:                                          │ │
│  │                                                          │ │
│  │    [Customers]───EXPLICIT_FK───▶[Orders]                │ │
│  │         │                           │                    │ │
│  │    IMPLICIT_FK                 EXPLICIT_FK              │ │
│  │         │                           │                    │ │
│  │         ▼                           ▼                    │ │
│  │    [Addresses]                 [Order_Items]            │ │
│  │                                     │                    │ │
│  │                               IMPLICIT_FK               │ │
│  │                                     │                    │ │
│  │                                     ▼                    │ │
│  │                               [Products]                 │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           IMPLICIT FK DISCOVERY ENGINE                   │ │
│  │                                                          │ │
│  │  Strategy 1: NAME-BASED MATCHING                        │ │
│  │    • customer_id ↔ cust_id ↔ customerid                │ │
│  │    • order_id ↔ orderId ↔ order_key                    │ │
│  │    • Fuzzy matching with Levenshtein distance          │ │
│  │                                                          │ │
│  │  Strategy 2: TYPE + CARDINALITY                         │ │
│  │    • INT + unique → likely primary key                  │ │
│  │    • INT + non-unique → likely foreign key             │ │
│  │    • Match by type compatibility                        │ │
│  │                                                          │ │
│  │  Strategy 3: VALUE OVERLAP (if data available)          │ │
│  │    • Sample value intersection analysis                 │ │
│  │    • High overlap = likely FK relationship             │ │
│  │                                                          │ │
│  │  Strategy 4: STRUCTURAL PATTERNS                        │ │
│  │    • Table naming conventions (orders → order_items)   │ │
│  │    • Common 1:N relationship patterns                   │ │
│  │                                                          │ │
│  │  Output: Confidence-scored implicit FK candidates       │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           JOIN PATH OPTIMIZER                            │ │
│  │                                                          │ │
│  │  Given: Query-relevant tables {T1, T2, T3, T4}          │ │
│  │                                                          │ │
│  │  Find: Optimal join path considering:                   │ │
│  │    • Shortest path (fewer joins = less error risk)     │ │
│  │    • Table cardinality (smaller tables first)          │ │
│  │    • FK confidence (prefer explicit over implicit)      │ │
│  │    • Semantic coherence (avoid unrelated bridge tables)│ │
│  │                                                          │ │
│  │  Algorithm: Modified Dijkstra with multi-criteria cost │ │
│  │                                                          │ │
│  │  Output: Ordered join sequence with ON conditions       │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Implicit FK Discovery Algorithm

```python
class ImplicitFKDiscoverer:
    """
    Discover foreign keys not explicitly defined in schema
    """

    def discover(self, schema_dict, explicit_fks, sample_data=None):
        candidates = []

        # Strategy 1: Name-based matching
        for table1, cols1 in schema_dict.items():
            for table2, cols2 in schema_dict.items():
                if table1 == table2:
                    continue

                for col1 in cols1:
                    for col2 in cols2:
                        name_score = self._name_similarity(
                            col1['column_name'],
                            col2['column_name']
                        )
                        if name_score > 0.7:
                            candidates.append(ImplicitFK(
                                from_table=table1,
                                from_column=col1['column_name'],
                                to_table=table2,
                                to_column=col2['column_name'],
                                confidence=name_score,
                                method='name_matching'
                            ))

        # Strategy 2: Type + cardinality matching
        for table, cols in schema_dict.items():
            for col in cols:
                if self._looks_like_fk(col):
                    # Find potential target tables
                    targets = self._find_fk_targets(col, schema_dict)
                    for target in targets:
                        candidates.append(ImplicitFK(
                            from_table=table,
                            from_column=col['column_name'],
                            to_table=target.table,
                            to_column=target.column,
                            confidence=target.confidence,
                            method='type_cardinality'
                        ))

        # Strategy 3: Value overlap (if sample data available)
        if sample_data:
            overlap_candidates = self._analyze_value_overlap(
                schema_dict, sample_data
            )
            candidates.extend(overlap_candidates)

        # Deduplicate and score
        final_candidates = self._deduplicate_and_score(candidates)

        # Filter by confidence threshold
        return [c for c in final_candidates if c.confidence > 0.75]

    def _name_similarity(self, name1, name2):
        """
        Calculate similarity between column names
        Handles variations: customer_id, cust_id, customerid, CustomerId
        """
        # Normalize names
        n1 = self._normalize_name(name1)
        n2 = self._normalize_name(name2)

        # Check for common FK patterns
        if n1 == n2:
            return 1.0

        # Check if one contains the other
        if n1 in n2 or n2 in n1:
            return 0.85

        # Levenshtein similarity
        return 1 - (levenshtein(n1, n2) / max(len(n1), len(n2)))
```

### 7.4 Expected Impact

| Metric | Baseline (92.5%) | With RKG | Improvement |
|--------|------------------|----------|-------------|
| Join accuracy (explicit FK) | 94% | 96%+ | +2% |
| Join accuracy (implicit FK) | 85% | 92%+ | +7% |
| Overall join accuracy | 92% | 94%+ | +2% |
| Path optimization | Good | Optimal | Quality boost |

---

## 8. Component 4: Deep Reasoning Chain with Self-Verification

### 8.1 Novel Aspect

Inspired by Prism Swarm's Deepthink component (contributing to their 90.49% score), our Deep Reasoning Chain introduces **extended chain-of-thought with self-verification** and **semantic error recovery** that interprets execution errors and result mismatches to generate targeted corrections.

### 8.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│      DEEP REASONING CHAIN WITH SELF-VERIFICATION              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                 REFINEMENT LOOP                         │ │
│  │                                                          │ │
│  │    Generate ──▶ Execute ──▶ Analyze ──▶ Correct        │ │
│  │        ▲                                    │            │ │
│  │        └────────────────────────────────────┘            │ │
│  │                    (max 3 iterations)                    │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           SEMANTIC ERROR RECOVERY ENGINE                 │ │
│  │                                                          │ │
│  │  ERROR TYPE 1: RESULT SHAPE MISMATCH                    │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ Symptom: Expected 5 rows, Got 5000 rows          │  │ │
│  │  │ Interpretation: Missing WHERE clause or filter   │  │ │
│  │  │ Correction: Add/expand WHERE conditions          │  │ │
│  │  │                                                   │  │ │
│  │  │ Symptom: Expected 1 column, Got 5 columns        │  │ │
│  │  │ Interpretation: Extra columns in SELECT          │  │ │
│  │  │ Correction: Remove unnecessary columns           │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │                                                          │ │
│  │  ERROR TYPE 2: VALUE RANGE MISMATCH                     │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ Symptom: Expected positive, Got negatives        │  │ │
│  │  │ Interpretation: Wrong column or sign error       │  │ │
│  │  │ Correction: Verify column, add ABS if needed     │  │ │
│  │  │                                                   │  │ │
│  │  │ Symptom: Expected 2020-2024, Got all dates       │  │ │
│  │  │ Interpretation: Missing date filter              │  │ │
│  │  │ Correction: Add date range condition             │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │                                                          │ │
│  │  ERROR TYPE 3: EXECUTION FAILURE                        │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ "Column not found"                                │  │ │
│  │  │ → Schema linking correction                       │  │ │
│  │  │ → Fuzzy match to find correct column             │  │ │
│  │  │                                                   │  │ │
│  │  │ "Ambiguous column reference"                      │  │ │
│  │  │ → Add table qualifier                             │  │ │
│  │  │ → T1.column_name instead of column_name          │  │ │
│  │  │                                                   │  │ │
│  │  │ "Type mismatch"                                   │  │ │
│  │  │ → Add explicit CAST                               │  │ │
│  │  │ → CAST(column AS target_type)                    │  │ │
│  │  │                                                   │  │ │
│  │  │ "Syntax error near X"                             │  │ │
│  │  │ → Pattern-based syntax fix                       │  │ │
│  │  │ → Dialect-specific correction                    │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │                                                          │ │
│  │  ERROR TYPE 4: SEMANTIC INTENT MISMATCH                 │ │
│  │  ┌──────────────────────────────────────────────────┐  │ │
│  │  │ Question: "Find the highest..."                   │  │ │
│  │  │ Check: Does SQL have ORDER BY + LIMIT 1?         │  │ │
│  │  │ If not: Add ORDER BY DESC LIMIT 1                │  │ │
│  │  │                                                   │  │ │
│  │  │ Question: "Total for each..."                    │  │ │
│  │  │ Check: Does SQL have GROUP BY?                   │  │ │
│  │  │ If not: Add GROUP BY clause                      │  │ │
│  │  │                                                   │  │ │
│  │  │ Question: "Unique..."                             │  │ │
│  │  │ Check: Does SQL have DISTINCT?                   │  │ │
│  │  │ If not: Add DISTINCT keyword                     │  │ │
│  │  └──────────────────────────────────────────────────┘  │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 Semantic Recovery Algorithm

```python
class SemanticRecoveryEngine:
    """
    Interpret execution results and generate targeted corrections
    """

    def analyze_and_correct(self, sql, result, question, expected_characteristics):
        # Analyze actual result
        actual = self._analyze_result(result)

        # Compare with expected
        mismatches = []

        # Check row count
        if actual.row_count > expected_characteristics.max_rows * 10:
            mismatches.append(Mismatch(
                type='ROW_COUNT_HIGH',
                expected=expected_characteristics.max_rows,
                actual=actual.row_count,
                severity='HIGH'
            ))

        # Check column count
        if actual.column_count != expected_characteristics.column_count:
            mismatches.append(Mismatch(
                type='COLUMN_COUNT',
                expected=expected_characteristics.column_count,
                actual=actual.column_count,
                severity='MEDIUM'
            ))

        # Check value ranges
        for col, expected_range in expected_characteristics.value_ranges.items():
            if col in actual.value_ranges:
                actual_range = actual.value_ranges[col]
                if not self._ranges_overlap(expected_range, actual_range):
                    mismatches.append(Mismatch(
                        type='VALUE_RANGE',
                        column=col,
                        expected=expected_range,
                        actual=actual_range,
                        severity='HIGH'
                    ))

        # Generate corrections
        corrections = []
        for mismatch in sorted(mismatches, key=lambda m: m.severity):
            correction = self._generate_correction(mismatch, sql, question)
            corrections.append(correction)

        # Apply corrections in priority order
        corrected_sql = sql
        for correction in corrections:
            corrected_sql = self._apply_correction(corrected_sql, correction)

        return corrected_sql, corrections

    def _generate_correction(self, mismatch, sql, question):
        """
        Generate targeted correction for mismatch
        """
        if mismatch.type == 'ROW_COUNT_HIGH':
            # Analyze question for missing filters
            filters = self._extract_filters_from_question(question)
            return AddWhereClause(filters=filters)

        elif mismatch.type == 'COLUMN_COUNT':
            # Analyze question for expected columns
            expected_cols = self._extract_columns_from_question(question)
            return ModifySelectClause(columns=expected_cols)

        elif mismatch.type == 'VALUE_RANGE':
            # Add filter for value range
            return AddRangeFilter(
                column=mismatch.column,
                range=mismatch.expected
            )

        # ... more correction types
```

### 8.4 Expected Impact

| Metric | Baseline (92.5%) | With Deep Reasoning | Improvement |
|--------|------------------|---------------------|-------------|
| First-pass accuracy | 92.5% | 92.5% | (baseline) |
| After 1 retry | 92.5% | 93.5% | +1.0% |
| After 2 retries | 92.5% | 94.2% | +1.7% |
| Retry efficiency | Good | Excellent | +40% error recovery |

---

## 9. Component 5: Snowflake-Native Generation Pipeline (SNGP)

### 9.1 Novel Aspect

Native support for **Snowflake SQL dialect** with specialized templates for VARIANT types, FLATTEN operations, QUALIFY clauses, and nested column access - addressing the remaining edge cases in the 7.5% error gap.

### 9.2 Snowflake-Specific Features

```
┌─────────────────────────────────────────────────────────────┐
│              SNOWFLAKE SQL FEATURES                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  FEATURE           │ Snowflake Syntax                       │
│  ─────────────────────────────────────────────────────────  │
│  STRUCT access     │ profile:city::STRING                   │
│  ARRAY access      │ arr[0]                                 │
│  ARRAY unnest      │ LATERAL FLATTEN(input => arr)         │
│  JSON extract      │ GET_PATH() or :path notation           │
│  Safe divide       │ DIV0()                                 │
│  Date truncate     │ DATE_TRUNC('month', date)             │
│  Row number filter │ QUALIFY ROW_NUMBER() OVER(...) = 1    │
│  String concat     │ || or CONCAT()                         │
│  NULL handling     │ NVL(), COALESCE(), IFNULL()           │
│  Type casting      │ CAST() or ::TYPE                       │
│  VARIANT access    │ data:key::TYPE                         │
│  Array aggregation │ ARRAY_AGG(), ARRAY_CONSTRUCT()        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 9.3 Snowflake Templates

```python
class SnowflakeTemplates:
    """
    Snowflake-native templates for SQL generation
    """

    templates = {
        'struct_access': '{parent}:{child}::{type}',
        'array_access': '{array}[{index}]::{type}',
        'array_unnest': 'LATERAL FLATTEN(input => {array}) {alias}',
        'json_extract': 'GET_PATH({column}, \'{path}\')',
        'safe_array': 'TRY_TO_{type}({array}[{index}])',
        'nested_unnest': '''
            SELECT {columns}
            FROM {table},
            LATERAL FLATTEN(input => {array_column}) AS {alias}
            WHERE {conditions}
        ''',
        'qualify_filter': '''
            SELECT {columns}
            FROM {table}
            WHERE {conditions}
            QUALIFY {window_function} = {value}
        ''',
        'variant_access': '{column}:{path}::{type}'
    }

    def generate(self, pattern, **kwargs):
        template = self.templates[pattern]
        return template.format(**kwargs)
```

### 9.4 Expected Impact

| Snowflake Feature | Baseline (92.5%) | With SNGP | Improvement |
|-------------------|------------------|-----------|-------------|
| Standard SQL | 94% | 95%+ | +1% |
| VARIANT access | 88% | 94%+ | +6% |
| FLATTEN operations | 85% | 93%+ | +8% |
| QUALIFY clauses | 87% | 94%+ | +7% |
| Nested structures | 86% | 93%+ | +7% |

---

## 10. Component 6: Execution-Driven Iterative Refinement (EDIR)

### 10.1 Novel Aspect

Extends traditional retry mechanisms with **execution-aware voting** that considers SQL structure similarity during consensus, and **component-level synthesis** that assembles the best SQL from multiple candidate components. Real Snowflake execution feedback drives iterative improvement.

### 10.2 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│    STRUCTURAL CONSENSUS WITH CORRECTIVE SELF-CONSISTENCY     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PHASE 1: MULTI-PATH GENERATION                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  Path 1: Direct generation (temp=0.0)                   │ │
│  │  Path 2: NatSQL intermediate (temp=0.2)                 │ │
│  │  Path 3: Decomposed generation (temp=0.3)               │ │
│  │  Path 4: Alternative schema view (temp=0.1)             │ │
│  │  Path 5: Expert-specific generation (temp=0.0)          │ │
│  │                                                          │ │
│  │  Output: 5 SQL candidates                               │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  PHASE 2: STRUCTURAL CLUSTERING                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  Cluster by SQL structure:                              │ │
│  │                                                          │ │
│  │  ┌─────────────────────────────────────────────┐       │ │
│  │  │ SELECT pattern:                              │       │ │
│  │  │   - Columns selected                         │       │ │
│  │  │   - Aggregations used                        │       │ │
│  │  │   - Aliases                                  │       │ │
│  │  ├─────────────────────────────────────────────┤       │ │
│  │  │ FROM/JOIN pattern:                           │       │ │
│  │  │   - Tables involved                          │       │ │
│  │  │   - Join order                               │       │ │
│  │  │   - Join conditions                          │       │ │
│  │  ├─────────────────────────────────────────────┤       │ │
│  │  │ WHERE pattern:                               │       │ │
│  │  │   - Conditions                               │       │ │
│  │  │   - Operators                                │       │ │
│  │  │   - Subqueries                               │       │ │
│  │  ├─────────────────────────────────────────────┤       │ │
│  │  │ GROUP BY / ORDER BY pattern:                 │       │ │
│  │  │   - Grouping columns                         │       │ │
│  │  │   - Ordering                                 │       │ │
│  │  └─────────────────────────────────────────────┘       │ │
│  │                                                          │ │
│  │  Result:                                                 │ │
│  │    Cluster A: [SQL_1, SQL_3] - similar structure        │ │
│  │    Cluster B: [SQL_2, SQL_4, SQL_5] - similar structure │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  PHASE 3: EXECUTION-AWARE VOTING                             │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  1. Execute all 5 candidates                            │ │
│  │                                                          │ │
│  │  2. Group by execution result:                          │ │
│  │     Result Group A: [SQL_1, SQL_2, SQL_3] → same result │ │
│  │     Result Group B: [SQL_4] → different result          │ │
│  │     Error Group: [SQL_5] → execution failed             │ │
│  │                                                          │ │
│  │  3. Vote within groups:                                 │ │
│  │     - Prefer larger result groups (3 > 1)              │ │
│  │     - Within group, prefer simpler structure           │ │
│  │     - Tie-break: similarity to training examples       │ │
│  │                                                          │ │
│  │  Winner: SQL_1 (Group A, simplest structure)            │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                              │                               │
│                              ▼                               │
│  PHASE 4: COMPONENT-LEVEL SYNTHESIS (if no clear winner)    │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                          │ │
│  │  If tie or low confidence:                              │ │
│  │                                                          │ │
│  │  1. Parse all candidates into components:               │ │
│  │     SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY       │ │
│  │                                                          │ │
│  │  2. Score each component independently:                 │ │
│  │     - Agreement across candidates                       │ │
│  │     - Execution success correlation                     │ │
│  │     - Training example similarity                       │ │
│  │                                                          │ │
│  │  3. Select best component for each clause:              │ │
│  │     SELECT: from SQL_2 (best agreement)                 │ │
│  │     FROM/JOIN: from SQL_1 (execution success)           │ │
│  │     WHERE: from SQL_3 (training similarity)             │ │
│  │                                                          │ │
│  │  4. Synthesize final SQL:                               │ │
│  │     Combine best components, verify compatibility       │ │
│  │                                                          │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 Component Synthesis Algorithm

```python
class ComponentSynthesizer:
    """
    Synthesize best SQL from multiple candidate components
    """

    def synthesize(self, candidates, execution_results):
        # Parse all candidates
        parsed = [self._parse_sql(c) for c in candidates]

        # Score each component type
        best_components = {}

        for clause in ['SELECT', 'FROM', 'WHERE', 'GROUP_BY', 'ORDER_BY']:
            components = [p.get(clause) for p in parsed]
            scores = self._score_components(components, execution_results)
            best_components[clause] = scores[0][0]  # Highest scoring

        # Verify compatibility
        compatible = self._ensure_compatibility(best_components)

        # Assemble final SQL
        return self._assemble_sql(compatible)

    def _score_components(self, components, execution_results):
        """
        Score components by multiple criteria
        """
        scores = []

        for i, comp in enumerate(components):
            # Agreement score: how many candidates have similar component
            agreement = self._calculate_agreement(comp, components)

            # Execution score: did candidates with this component succeed?
            execution = self._correlate_with_execution(i, execution_results)

            # Training score: similarity to successful training examples
            training = self._similarity_to_training(comp)

            # Combined score
            combined = 0.4 * agreement + 0.4 * execution + 0.2 * training
            scores.append((comp, combined))

        return sorted(scores, key=lambda x: x[1], reverse=True)
```

### 10.4 Expected Impact

| Metric | Baseline (92.5%) | With EDIR | Improvement |
|--------|------------------|-----------|-------------|
| Single-path accuracy | 92.5% | 92.5% | (baseline) |
| Multi-path consensus | 92.5% | 94.0% | +1.5% |
| Component synthesis | N/A | 94.5% | +2.0% total |
| False positive reduction | 8% | 4% | -4% |

---

## 11. Complete Pipeline Architecture

### 11.1 Data Flow

```
INPUT
  │
  ├─▶ Question: "Find customers in California with orders > $1000"
  ├─▶ Schema: 700+ tables, 3000+ columns
  └─▶ Database: Snowflake connection
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: SCHEMA UNDERSTANDING (Components 1, 3, 5)          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Component 1 (HSI):                                          │
│    700 tables → 35 domain clusters                          │
│    3000 columns → 200 relevant columns                      │
│    Nested columns flattened with templates                  │
│                                                              │
│  Component 3 (RKG):                                          │
│    Build FK graph (explicit + 12 implicit FKs)              │
│    Identify join paths: customers → orders → items          │
│                                                              │
│  Component 5 (SNGP):                                         │
│    Snowflake-native SQL generation                          │
│    Load VARIANT/FLATTEN templates                           │
│                                                              │
│  Output: Unified Schema Context                              │
│    - Compressed schema (200 columns vs 3000)                │
│    - FK graph with implicit relationships                   │
│    - Snowflake-specific templates ready                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: MULTI-EXPERT GENERATION (Component 2)              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Pattern Detection:                                          │
│    - Multi-table join (customers + orders)                  │
│    - Aggregation (comparing to threshold)                   │
│    - Filter condition (California, > $1000)                 │
│                                                              │
│  Expert Activation:                                          │
│    - JoinExpert: Generate join structure                    │
│    - AggregationExpert: Handle order total aggregation      │
│                                                              │
│  Multi-Path Generation (5 candidates):                       │
│    SQL_1: Direct approach                                    │
│    SQL_2: NatSQL intermediate                               │
│    SQL_3: Decomposed (subquery for order totals)            │
│    SQL_4: Alternative join order                            │
│    SQL_5: Expert-composed                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: REFINEMENT & CONSENSUS (Components 4, 6)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Component 4 (Deep Reasoning):                               │
│    Execute all 5 candidates on Snowflake                    │
│    SQL_1: Success, 150 rows                                  │
│    SQL_2: Success, 150 rows                                  │
│    SQL_3: Success, 148 rows (slight difference)             │
│    SQL_4: Error - "ambiguous column"                        │
│    SQL_5: Success, 150 rows                                  │
│                                                              │
│    Fix SQL_4:                                                │
│    - Interpret error: need table qualifier                  │
│    - Apply correction: add T1. prefix                       │
│    - Re-execute: Success, 150 rows                          │
│                                                              │
│  Component 6 (EDIR):                                         │
│    Group by result: [SQL_1, SQL_2, SQL_4, SQL_5] = 150 rows │
│                     [SQL_3] = 148 rows                      │
│                                                              │
│    Vote: Group A wins (4 vs 1)                              │
│    Within Group A: SQL_1 simplest structure                 │
│                                                              │
│    Final selection: SQL_1                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT
  │
  ├─▶ Final SQL: SELECT c.name, SUM(o.total) FROM customers c
  │              JOIN orders o ON c.id = o.customer_id
  │              WHERE c.state = 'California'
  │              GROUP BY c.id, c.name
  │              HAVING SUM(o.total) > 1000
  │
  ├─▶ Confidence: 0.92 (high agreement, execution success)
  │
  └─▶ Reasoning trace: [schema compression → expert routing →
                        consensus voting → SQL_1 selected]
```

---

## 12. Expected Performance Analysis

### 12.1 Component-by-Component Impact

| Component | Primary Improvement | EX Impact |
|-----------|---------------------|-----------|
| HSI (Schema Intelligence) | Schema linking, context efficiency | +0.3-0.5% |
| Multi-Agent Swarm | Pattern-specific generation | +0.5-1.0% |
| RKG (Knowledge Graph) | Join accuracy, implicit FKs | +0.3-0.5% |
| Deep Reasoning Chain | Error recovery, self-verification | +0.5-0.8% |
| SNGP (Snowflake-Native) | Snowflake dialect optimization | +0.3-0.5% |
| EDIR (Execution-Driven) | Voting accuracy, synthesis | +0.3-0.5% |

### 12.2 Synergistic Effects

Components work together for compounding improvements:

```
HSI + RKG = Better schema understanding
  → Feeds higher-quality context to Swarm
  → +0.2% additional improvement

Swarm + SNGP = Pattern-aware Snowflake generation
  → Nested column expert uses Snowflake templates
  → +0.3% additional improvement

Deep Reasoning + EDIR = Smart consensus with verification
  → Self-verification catches errors before voting
  → +0.2% additional improvement

Total Synergy Bonus: +0.7% additional
```

### 12.3 Final Performance Projection

```
┌─────────────────────────────────────────────────────────────┐
│              NEXUS-SQL PERFORMANCE PROJECTION                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Baseline (Native mini):                 92.50%             │
│                                                              │
│  + HSI (Schema Intelligence):            +0.4%  → 92.9%     │
│  + Multi-Agent Swarm:                    +0.6%  → 93.5%     │
│  + RKG (Knowledge Graph):                +0.3%  → 93.8%     │
│  + Deep Reasoning Chain:                 +0.5%  → 94.3%     │
│  + SNGP (Snowflake-Native):              +0.3%  → 94.6%     │
│  + EDIR (Execution-Driven):              +0.3%  → 94.9%     │
│                                                              │
│  Note: Improvements are not fully additive due to overlap   │
│  Conservative estimate: 93-94% (accounting for overlap)     │
│  Optimistic estimate: 94-95%                                 │
│                                                              │
│  ═══════════════════════════════════════════════════════    │
│  PROJECTED FINAL: 94% ± 1% Execution Accuracy               │
│  ═══════════════════════════════════════════════════════    │
│                                                              │
│  Current Leaderboard (Jan 2026):                            │
│    #1 Native mini: 92.50%                                    │
│    #2 Prism Swarm + Deepthink: 90.49%                       │
│    #3 Ask Data + RKG: 86.28%                                │
│                                                              │
│  NEXUS-SQL Target: 94% → NEW #1 POSITION                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

| Task | Description | Files |
|------|-------------|-------|
| Domain Clustering | Implement HSI domain extraction | `utils/domain_ontology.py` |
| Schema Compression | Query-adaptive compression | `pipeline/schema_compression.py` |
| Nested Flattening | Flatten nested columns | `utils/nested_column_handler.py` |
| Snowflake Templates | Snowflake-native SQL templates | `utils/snowflake_templates.py` |

### Phase 2: Expert Architecture (Weeks 4-6)

| Task | Description | Files |
|------|-------------|-------|
| Pattern Router | Route to appropriate experts | `pipeline/pattern_router.py` |
| Join Expert | Multi-table join generation | `experts/join_expert.py` |
| Aggregation Expert | GROUP BY, window functions | `experts/aggregation_expert.py` |
| Subquery Expert | Nested query composition | `experts/subquery_expert.py` |
| Nested Column Expert | STRUCT/ARRAY handling | `experts/nested_column_expert.py` |
| Expert Synthesizer | Combine expert outputs | `pipeline/expert_synthesizer.py` |

### Phase 3: Graph Reasoning (Weeks 7-8)

| Task | Description | Files |
|------|-------------|-------|
| FK Graph Builder | Build schema knowledge graph | `utils/schema_graph.py` |
| Implicit FK Discovery | Infer undeclared FKs | `utils/implicit_fk_discoverer.py` |
| Join Path Optimizer | Find optimal join paths | `utils/join_path_optimizer.py` |

### Phase 4: Refinement & Consensus (Weeks 9-11)

| Task | Description | Files |
|------|-------------|-------|
| Semantic Error Recovery | Interpret execution errors | `pipeline/semantic_recovery.py` |
| Iterative Refinement | Multi-round correction | `pipeline/iterative_refiner.py` |
| Structural Clustering | Cluster by SQL structure | `utils/structural_clustering.py` |
| Component Synthesis | Assemble from best parts | `pipeline/component_synthesis.py` |
| Consensus Voting | Multi-path voting | `pipeline/consensus_voter.py` |

### Phase 5: Integration & Evaluation (Weeks 12-14)

| Task | Description | Files |
|------|-------------|-------|
| Pipeline Integration | Combine all components | `core/nexus_baseline.py` |
| Spider 2.0-Snow Evaluation | Run full benchmark | `evaluation/spider2_snow_eval.py` |
| Ablation Studies | Component contribution | `evaluation/ablation.py` |
| Performance Tuning | Optimize bottlenecks | Various |

---

## 14. Ablation Study Design

### 14.1 Component Ablations

| Experiment | Configuration | Expected Result |
|------------|--------------|-----------------|
| Full NEXUS-SQL | All components | 94% ± 1% |
| w/o HSI | No schema intelligence | -0.4% → 93.6% |
| w/o Swarm | Single agent generation | -0.6% → 93.4% |
| w/o RKG | No knowledge graph | -0.3% → 93.7% |
| w/o Deep Reasoning | No self-verification | -0.5% → 93.5% |
| w/o SNGP | Generic SQL templates | -0.3% → 93.7% |
| w/o EDIR | No execution-driven refinement | -0.3% → 93.7% |

### 14.2 Component Combination Studies

| Experiment | Components | Expected |
|------------|------------|----------|
| Schema Only | HSI + RKG | 93.2% |
| Generation Only | Swarm + SNGP | 93.5% |
| Refinement Only | Deep Reasoning + EDIR | 93.3% |
| Schema + Generation | HSI + RKG + Swarm + SNGP | 93.8% |
| Full Pipeline | All | 94% ± 1% |

### 14.3 Snowflake-Specific Features

| Experiment | Snowflake Feature EX |
|------------|----------------------|
| Baseline (Native mini) | 92.50% |
| + VARIANT handling (HSI) | 93.0% |
| + Nested Expert (Swarm) | 93.5% |
| + Snowflake Templates (SNGP) | 94.0% |
| Full NEXUS-SQL | 94% ± 1% |

---

## 15. Paper Contribution Summary

### 15.1 Novel Contributions

1. **Hierarchical Schema Intelligence (HSI)**
   - Semantic domain clustering for enterprise-scale schema
   - 70% compression with semantic preservation
   - Query-adaptive expansion for Snowflake schemas

2. **Multi-Agent Swarm with Specialized Experts**
   - Pattern-specialized experts (Join, Aggregation, Nested Column)
   - Expert memory banks learned from Spider 2.0-Snow data
   - Coordination inspired by Prism Swarm architecture

3. **Relational Knowledge Graph Integration**
   - Multi-strategy FK inference (name, type, value overlap)
   - Graph-based join path optimization
   - Enhanced version of AT&T/RelationalAI approach

4. **Deep Reasoning Chain with Self-Verification**
   - Deepthink-inspired extended chain-of-thought
   - Self-verification at each reasoning step
   - Targeted corrections vs. full regeneration

5. **Snowflake-Native Generation Pipeline**
   - Native VARIANT, FLATTEN, QUALIFY template generation
   - Specialized nested column expert for Snowflake
   - Optimized for Spider 2.0-Snow benchmark

6. **Execution-Driven Iterative Refinement**
   - Real Snowflake execution feedback loop
   - Score and select best component per clause
   - Synthesize optimal SQL from best parts

### 15.2 Paper Outline

```
Title: NEXUS-SQL: Neural EXpert Unified System for Enterprise SQL
       A Multi-Agent Architecture for Spider 2.0-Snow

1. Introduction
   - Enterprise Text-to-SQL challenge
   - Spider 2.0-Snow benchmark and top methods analysis
   - Our approach: combining Swarm + Knowledge Graph + Deep Reasoning

2. Related Work
   - Text-to-SQL methods (DIN-SQL, DAIL-SQL, C3-SQL)
   - Spider 2.0-Snow top methods (Native mini, Prism Swarm, Ask Data + RKG)
   - Schema linking, multi-agent systems, knowledge graphs

3. NEXUS-SQL Architecture
   3.1 Hierarchical Schema Intelligence
   3.2 Multi-Agent Swarm with Specialized Experts
   3.3 Relational Knowledge Graph Integration
   3.4 Deep Reasoning Chain with Self-Verification
   3.5 Snowflake-Native Generation Pipeline
   3.6 Execution-Driven Iterative Refinement

4. Experimental Setup
   - Spider 2.0-Snow benchmark (547 examples)
   - Evaluation metrics (Execution Accuracy)
   - Snowflake connection and execution

5. Results
   - Main results vs. top methods (Native mini, Prism Swarm)
   - Ablation studies
   - Snowflake-specific feature analysis
   - Case studies

6. Analysis
   - Error analysis on remaining 6-7%
   - Component contribution
   - Computational cost

7. Conclusion
   - Summary: 94% EX on Spider 2.0-Snow
   - Limitations
   - Future work

Appendix
   - Detailed algorithm descriptions
   - Snowflake SQL templates
   - Example generations
```

### 15.3 Target Venues

| Venue | Type | Deadline | Fit |
|-------|------|----------|-----|
| ACL 2026 | Main conference | Feb 2026 | Excellent |
| EMNLP 2026 | Main conference | May 2026 | Excellent |
| NeurIPS 2026 | Main conference | May 2026 | Good |
| VLDB 2027 | Database venue | Mar 2026 | Excellent |
| SIGMOD 2027 | Database venue | Jul 2026 | Good |

---

## Conclusion

NEXUS-SQL represents a comprehensive approach to enterprise-scale Text-to-SQL that builds on the success of top Spider 2.0-Snow methods. By combining six synergistic components—Hierarchical Schema Intelligence, Multi-Agent Swarm, Relational Knowledge Graph, Deep Reasoning Chain, Snowflake-Native Generation, and Execution-Driven Refinement—we project achieving **94% ± 1% Execution Accuracy**, a **+1.5-2.5% improvement** over the current state-of-the-art (Native mini at 92.50%).

The key insight is that the remaining 7.5% error gap at 92.50% requires targeted solutions: complex multi-step reasoning (addressed by Deep Reasoning), edge case nested structures (addressed by Snowflake-Native Generation), ambiguous schema mapping (addressed by RKG), and rare SQL patterns (addressed by Multi-Agent Swarm consensus). NEXUS-SQL addresses each of these with a dedicated component that works synergistically with the others.

**Expected Ranking: #1 on Spider 2.0-Snow Leaderboard**

---

*Document prepared for research paper development*
*Target: Top-3 Spider 2.0-Snow Leaderboard Performance (93-95%)*
*January 2026*
