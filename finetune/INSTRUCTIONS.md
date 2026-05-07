# Fine-Tuning Instructions — ADAPT-SQL on ASU SOL

## What this does
Trains a QLoRA fine-tune of Qwen2.5-Coder-32B on the Spider training set,
then exports it as an Ollama model. The resulting model replaces the LLM
component inside the existing ADAPT-SQL pipeline — all other pipeline
improvements (schema linking, multi-candidate, EX-only retry, etc.) remain.

**Best verified EX before fine-tuning: 83.8%** (After-Fix-4, 867/1034 queries,
git tag context: B'-fix + checker chain + GoT + structure-aware FAISS).
Source: PAPERS/summary.md, After-Fix-4 row.

Fix-5 (83.5%) was a regression — DISTINCT checker disabled in commit 059f209.
The SOTA improvements (qwen2.5-coder:32b + set-op detection + multi-candidate
majority vote, commit 5098ec7) are queued but unverified.

Fine-tuning relationship to pipeline:
  - train_sft.py / train_grpo.py train Qwen/Qwen2.5-Coder-32B-Instruct
    (the base HuggingFace weights) — NOT the pipeline code
  - The output model replaces the qwen2.5-coder:32b Ollama model
  - All pipeline improvements (schema linking, retry, normalization, etc.)
    stay active and stack on top of the fine-tuned LLM
  - Expected: 83.8% (pipeline) + fine-tuned LLM → ~86–88% EX

Expected improvement: +2–4% EX from domain-adapted weights.

---

## Prerequisites (one-time per account)

Run on a GPU compute node (not the login node):

```bash
srun --pty --partition=public --gres=gpu:a100:1 --mem=32G --cpus-per-task=4 --time=4:00:00 /bin/bash
bash /scratch/$USER/setup_sol.sh
```

Ensure the project is cloned and venv exists:
```bash
cd /scratch/$USER
git clone https://github.com/sidesshmore/ADAPT-SQL-Text2SQL.git ADAPT-SQL-Text2SQL
cd ADAPT-SQL-Text2SQL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running the Fine-Tune (sbatch — no interactive session needed)

```bash
cd /scratch/$USER/ADAPT-SQL-Text2SQL
git pull
sbatch finetune/finetune_sol.sh
```

SLURM settings in the script:
- **GPUs**: 4 × A100
- **Memory**: 160 GB
- **CPUs**: 16
- **Wall time**: 1 day 12 hours
- **Logs**: `/scratch/$USER/finetune_$JOBID.log`
- **Errors**: `/scratch/$USER/finetune_$JOBID.err`

Works for both `smore123` and `snande1` — auto-detects user.

### Monitor progress:
```bash
squeue -u $USER
tail -f /scratch/$USER/finetune_*.log
```

### Checkpoints saved to:
```
/scratch/$USER/finetune_checkpoints/sft/checkpoint-N/   ← every 100 steps
/scratch/$USER/finetune_checkpoints/sft/final/          ← end of Stage 1
/scratch/$USER/finetune_checkpoints/grpo/checkpoint-N/  ← every 50 steps
/scratch/$USER/finetune_checkpoints/grpo/final/         ← end of Stage 2
```

If the job dies, just resubmit — it resumes from the latest checkpoint automatically.

---

## After Training Completes — Merge + Export

```bash
bash /scratch/$USER/ADAPT-SQL-Text2SQL/finetune/merge_export.sh
```

This:
1. Merges the LoRA adapter into the full model (saved to `/scratch/$USER/finetune_merged/`)
2. Converts to GGUF via llama.cpp (saved to `/scratch/$USER/finetune_gguf/`)
3. Registers as `adapt-sql-coder` in Ollama

---

## Using the Fine-Tuned Model in ADAPT-SQL

After `merge_export.sh` completes, in the Streamlit batch UI:

- **Model dropdown** → select `adapt-sql-coder`

Or to add it to the dropdown, edit `ui/pages/batch_processing.py`:
```python
model = st.selectbox("🤖 Model", ["adapt-sql-coder", "qwen2.5-coder:32b", ...])
```

---

## Training Details

### Stage 1: SFT (`train_sft.py`)
- Base model: `Qwen/Qwen2.5-Coder-32B-Instruct` (from HuggingFace)
- QLoRA: 4-bit NF4, LoRA rank=64, alpha=128
- Data: Spider train set (~7,000 examples)
- Prompt format:
  ```
  ### Schema:
  CREATE TABLE ...

  ### Question:
  <natural language>

  ### SQL:
  <gold SQL>
  ```
- Loss computed only on the SQL completion (not schema/question)
- 3 epochs, lr=2e-5, cosine schedule, 4 GPUs

### Stage 2: GRPO (`train_grpo.py`)
- Loads SFT adapter, continues RL training
- Reward = format (0.1) + executability (0.3) + result match (0.5) + length penalty (-0.1)
- 1 epoch, lr=1e-5, group size=4 (4 candidates per prompt)

---

## Estimated Time

| Stage | Time (4×A100) |
|-------|--------------|
| SFT (3 epochs, ~7k examples) | ~6–8 hours |
| GRPO (1 epoch) | ~3–4 hours |
| Merge + export | ~30–60 min |
| **Total** | **~10–13 hours** |

The 12-hour wall time in the SLURM script should cover it. If SFT finishes but
GRPO doesn't, re-run just GRPO by editing `finetune_sol.sh` to skip the SFT block.

---

## Parallel Fine-Tune Option (Federated LoRA)

If you want to use 4 separate 4-hour OOD sessions instead of one 12-hour sbatch:

1. Split Spider train into 4 shards of ~1,750 examples each
2. Run `train_sft.py` on each shard independently (one per OOD session)
3. Average the 4 LoRA adapter weights:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load all 4 adapters and average their weights
adapters = [f"/scratch/$USER/finetune_checkpoints/shard_{i}/sft/final" for i in range(4)]
# ... weight averaging logic
```

This is more complex and usually gives slightly lower quality than full-data SFT.
Recommended to use sbatch with the full 12-hour run instead.
