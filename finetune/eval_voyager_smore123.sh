#!/bin/bash
#SBATCH --job-name=adapt-voyager
#SBATCH --account=class_cse57388551fall2025
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/smore123/voyager_eval_%A_%a.log
#SBATCH --error=/scratch/smore123/voyager_eval_%A_%a.err
#SBATCH --array=0-7

USER=smore123
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL

# 8 tasks x ~130 examples = 1034 total
STARTS=(0 130 260 390 520 650 780 910)
NUMS=(130 130 130 130 130 130 130 124)

START=${STARTS[$SLURM_ARRAY_TASK_ID]}
NUM=${NUMS[$SLURM_ARRAY_TASK_ID]}
RANGE="${START}_$((START + NUM - 1))"
CHECKPOINT_DIR=$SCRATCH/eval_results_voyager/range_$RANGE
OLLAMA_PORT=$((11437 + SLURM_ARRAY_TASK_ID))

echo "[$(date)] voyager-eval array_id=$SLURM_ARRAY_TASK_ID range=$START+$NUM"

export HOME=/home/$USER
export PYTHONUNBUFFERED=1
export VOYAGER_API_KEY=$(grep VOYAGER_API_KEY $PROJECT/.env | cut -d= -f2)
export VOYAGER_BASE_URL=https://openai.rc.asu.edu/v1
export VOYAGER_MODEL=qwen3-235b-a22b-thinking-2507

if [ -z "$VOYAGER_API_KEY" ]; then
    echo "[ERROR] VOYAGER_API_KEY not found in $PROJECT/.env"
    exit 1
fi

source $PROJECT/venv/bin/activate

# Start Ollama for nomic-embed-text embeddings only (LLM calls go to Voyager)
OLLAMA_BIN=$SCRATCH/ollama_install/bin/ollama
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "[ERROR] Ollama not found at $OLLAMA_BIN"
    exit 1
fi

OLLAMA_MODELS=$SCRATCH/ollama_models \
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
    $OLLAMA_BIN serve &

OLLAMA_PID=$!
echo "[$(date)] Ollama PID=$OLLAMA_PID port=$OLLAMA_PORT (embeddings only)"
sleep 15

export OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT

echo "[$(date)] Starting eval range=$RANGE model=$VOYAGER_MODEL"

python $PROJECT/eval_voyager.py \
    --start $START \
    --num $NUM \
    --checkpoint_dir $CHECKPOINT_DIR \
    --checkpoint_every 25

EXIT_CODE=$?
echo "[$(date)] eval done exit_code=$EXIT_CODE"
kill $OLLAMA_PID 2>/dev/null
