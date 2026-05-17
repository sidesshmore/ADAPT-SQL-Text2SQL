#!/bin/bash
#SBATCH --job-name=adapt-eval-snande1
#SBATCH --account=class_cse57388551fall2025
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/snande1/eval_ft_%A_%a.log
#SBATCH --error=/scratch/snande1/eval_ft_%A_%a.err
#SBATCH --array=0-7

USER=snande1
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL

# 8 tasks × ~130 examples = 1034 total (mirrors eval_sota.sh pattern)
STARTS=(0 130 260 390 520 650 780 910)
NUMS=(130 130 130 130 130 130 130 124)

START=${STARTS[$SLURM_ARRAY_TASK_ID]}
NUM=${NUMS[$SLURM_ARRAY_TASK_ID]}
RANGE="${START}_$((START + NUM - 1))"

CHECKPOINT_DIR=$SCRATCH/eval_results_finetuned/range_$RANGE
OLLAMA_PORT=$((11437 + SLURM_ARRAY_TASK_ID))   # 11437–11444

echo "[$(date)] eval-ft job array_id=$SLURM_ARRAY_TASK_ID range=$START+$NUM port=$OLLAMA_PORT"

export HOME=/home/$USER
export PYTHONUNBUFFERED=1
module load cuda 2>/dev/null

source $PROJECT/venv/bin/activate

# ── Verify model exists ───────────────────────────────────────────────────────
GGUF_FILE=$SCRATCH/finetune_gguf/adapt-sql-coder.gguf
if [ ! -f "$GGUF_FILE" ]; then
    echo "[ERROR] GGUF not found at $GGUF_FILE — did merge_export.sh complete?"
    exit 1
fi

# ── Start Ollama ──────────────────────────────────────────────────────────────
OLLAMA_BIN=$SCRATCH/ollama_install/bin/ollama
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "[ERROR] Ollama not found at $OLLAMA_BIN"
    echo "Install it first: curl -L https://ollama.com/download/ollama-linux-amd64.tgz | tar -xz -C $SCRATCH/ollama_install"
    exit 1
fi

OLLAMA_MODELS=$SCRATCH/ollama_models \
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
    $OLLAMA_BIN serve &

OLLAMA_PID=$!
echo "[$(date)] Ollama PID=$OLLAMA_PID port=$OLLAMA_PORT"
sleep 20

# ── Register model if not already present ────────────────────────────────────
MODELFILE=$SCRATCH/finetune_gguf/Modelfile
if [ ! -f "$MODELFILE" ]; then
    cat > $MODELFILE <<EOF
FROM $GGUF_FILE

PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "### Schema:"
PARAMETER stop "### Question:"

SYSTEM "You are an expert SQL generation assistant. Given a database schema and a natural language question, generate correct and efficient SQL."
EOF
fi

OLLAMA_MODELS=$SCRATCH/ollama_models OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
    $OLLAMA_BIN create adapt-sql-coder -f $MODELFILE 2>/dev/null || true

echo "[$(date)] Model ready — starting eval..."

# ── Run eval ──────────────────────────────────────────────────────────────────
python $PROJECT/finetune/eval_batch.py \
    --start $START \
    --num $NUM \
    --model adapt-sql-coder \
    --ollama_host http://127.0.0.1:$OLLAMA_PORT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --project_dir $PROJECT

echo "[$(date)] eval-ft done for range $RANGE"
kill $OLLAMA_PID 2>/dev/null
