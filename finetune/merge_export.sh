#!/bin/bash
# merge_export.sh — Merge LoRA adapter into base model, export to GGUF, register with Ollama
#
# Usage: bash finetune/merge_export.sh
# Run this AFTER finetune_sol.sh completes successfully.

USER=$(whoami)
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
CHECKPOINT_DIR=$SCRATCH/finetune_checkpoints
HF_CACHE=$SCRATCH/hf_cache
MERGED_DIR=$SCRATCH/finetune_merged
GGUF_DIR=$SCRATCH/finetune_gguf
OLLAMA_MODELS=$SCRATCH/ollama_models
OLLAMA_BIN=$SCRATCH/ollama_install/bin/ollama

echo "[$(date)] Starting merge + export for user: $USER"

export HOME=/home/$USER
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
module load cuda 2>/dev/null

source $PROJECT/venv/bin/activate

mkdir -p $MERGED_DIR $GGUF_DIR

# ── Step 1: Merge LoRA adapter into full-precision model ────────────────────
echo "[$(date)] Merging LoRA adapter..."

SCRATCH_DIR=$SCRATCH CHECKPOINT_DIR=$CHECKPOINT_DIR \
MERGED_DIR=$MERGED_DIR HF_CACHE=$HF_CACHE \
python3 - <<'PYEOF'
import os, sys, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

hf_cache = os.environ.get("HF_CACHE")
checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR"))
merged_dir = Path(os.environ.get("MERGED_DIR"))

# Prefer GRPO final, fall back to SFT final
grpo_final = checkpoint_dir / "grpo" / "final"
sft_final = checkpoint_dir / "sft" / "final"
adapter_path = grpo_final if grpo_final.exists() else sft_final

print(f"Using adapter: {adapter_path}")

tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="cpu",
    cache_dir=hf_cache,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, str(adapter_path))
model = model.merge_and_unload()

print(f"Saving merged model to {merged_dir}...")
model.save_pretrained(str(merged_dir), safe_serialization=True)
tokenizer.save_pretrained(str(merged_dir))
print("Merge complete.")
PYEOF

if [ $? -ne 0 ]; then
    echo "[✗] Merge failed"
    exit 1
fi
echo "[✓] Merged model saved to $MERGED_DIR"

# ── Step 2: Convert to GGUF with llama.cpp ──────────────────────────────────
echo "[$(date)] Converting to GGUF..."

LLAMACPP_DIR=$SCRATCH/llama.cpp
if [ ! -d "$LLAMACPP_DIR" ]; then
    echo "[~] Cloning llama.cpp..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git $LLAMACPP_DIR
fi

# Install llama.cpp python deps
pip install -q gguf sentencepiece

python3 $LLAMACPP_DIR/convert_hf_to_gguf.py \
    $MERGED_DIR \
    --outfile $GGUF_DIR/adapt-sql-coder.gguf \
    --outtype q8_0

if [ $? -ne 0 ]; then
    echo "[✗] GGUF conversion failed"
    exit 1
fi
echo "[✓] GGUF saved to $GGUF_DIR/adapt-sql-coder.gguf"

# ── Step 3: Register with Ollama ─────────────────────────────────────────────
echo "[$(date)] Registering with Ollama..."

cat > $GGUF_DIR/Modelfile <<EOF
FROM $GGUF_DIR/adapt-sql-coder.gguf

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "### Schema:"
PARAMETER stop "### Question:"

SYSTEM "You are an expert SQL generation assistant. Given a database schema and a natural language question, generate correct and efficient SQL."
EOF

OLLAMA_MODELS=$OLLAMA_MODELS OLLAMA_HOST=127.0.0.1:11437 \
    $OLLAMA_BIN create adapt-sql-coder -f $GGUF_DIR/Modelfile

if [ $? -eq 0 ]; then
    echo "[✓] Model registered as 'adapt-sql-coder' in Ollama"
    echo ""
    echo "To use it in ADAPT-SQL:"
    echo "  In Streamlit sidebar → Model → select 'adapt-sql-coder'"
    echo "  Or set model='adapt-sql-coder' in ADAPTBaseline()"
else
    echo "[✗] Ollama registration failed — GGUF file is still usable manually"
fi

echo "[$(date)] Export complete."
