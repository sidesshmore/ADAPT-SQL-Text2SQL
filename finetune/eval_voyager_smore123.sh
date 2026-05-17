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
export VOYAGER_MODEL=qwen3-coder-30b-a3b-instruct

if [ -z "$VOYAGER_API_KEY" ]; then
    echo "[ERROR] VOYAGER_API_KEY not found in $PROJECT/.env"
    exit 1
fi

source $PROJECT/venv/bin/activate

# Embeddings served from pre-computed cache (vector_store/dev_query_embeddings.pkl)
# No Ollama needed — cache lookup is instant
echo "[$(date)] Starting eval range=$RANGE model=$VOYAGER_MODEL (cached embeddings)"

python $PROJECT/eval_voyager.py \
    --start $START \
    --num $NUM \
    --checkpoint_dir $CHECKPOINT_DIR \
    --checkpoint_every 25

EXIT_CODE=$?
echo "[$(date)] eval done exit_code=$EXIT_CODE"
