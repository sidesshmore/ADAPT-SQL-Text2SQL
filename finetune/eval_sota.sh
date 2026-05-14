#!/bin/bash
#SBATCH --job-name=adapt-eval-sota
#SBATCH --account=grp_hdavulcu
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-10:00:00
#SBATCH --output=/scratch/%u/eval_sota_%A_%a.log
#SBATCH --error=/scratch/%u/eval_sota_%A_%a.err
#SBATCH --array=0-7

USER=$(whoami)
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL

# 8 tasks × ~130 examples = 1034 total (full Spider dev set)
STARTS=(0 130 260 390 520 650 780 910)
NUMS=(130 130 130 130 130 130 130 104)

START=${STARTS[$SLURM_ARRAY_TASK_ID]}
NUM=${NUMS[$SLURM_ARRAY_TASK_ID]}
RANGE="${START}_$((START + NUM - 1))"

CHECKPOINT_DIR=$SCRATCH/eval_results_sota/range_$RANGE
OLLAMA_PORT=$((11437 + SLURM_ARRAY_TASK_ID))   # 11437–11444

echo "[$(date)] eval-sota job array_id=$SLURM_ARRAY_TASK_ID range=$START+$NUM port=$OLLAMA_PORT"

export HOME=/home/$USER
export PYTHONUNBUFFERED=1
module load cuda 2>/dev/null

source $PROJECT/venv/bin/activate

# Start Ollama
OLLAMA_MODELS=$SCRATCH/ollama_models \
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
    $SCRATCH/ollama_install/bin/ollama serve &

OLLAMA_PID=$!
echo "[$(date)] Ollama PID=$OLLAMA_PID port=$OLLAMA_PORT"
sleep 20  # wait for server to be ready

# Verify qwen2.5-coder:32b is available; pull if missing
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $SCRATCH/ollama_install/bin/ollama list | grep -q "qwen2.5-coder:32b" || {
    echo "[$(date)] qwen2.5-coder:32b not found — pulling now (this takes ~20 min first time)..."
    OLLAMA_MODELS=$SCRATCH/ollama_models \
    OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
        $SCRATCH/ollama_install/bin/ollama pull qwen2.5-coder:32b
}

# Run eval using the full SOTA pipeline (Phase A–E: 32B model, set-op detection,
# multi-candidate majority vote, schema normalization, skeleton-first NESTED)
python $PROJECT/finetune/eval_batch.py \
    --start $START \
    --num $NUM \
    --model qwen2.5-coder:32b \
    --ollama_host http://127.0.0.1:$OLLAMA_PORT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --project_dir $PROJECT

echo "[$(date)] eval-sota done for range $RANGE"
kill $OLLAMA_PID 2>/dev/null
