#!/bin/bash
# sbatch array job — one model, one split, 150-query batches
#
# Usage (--array is passed on the command line, not hardcoded here):
#
#   dev  (1034 queries, 7 tasks):
#     VOYAGER_MODEL=qwen3-coder-30b-a3b-instruct \
#       sbatch --array=0-6 --export=ALL,SPLIT=dev run_sol_multimodel.sh
#
#   test (2147 queries, 15 tasks):
#     VOYAGER_MODEL=qwen3-235b-a22b-instruct \
#       sbatch --array=0-14 --export=ALL,SPLIT=test run_sol_multimodel.sh

#SBATCH --job-name=adapt-multimodel
#SBATCH --account=class_cse57388551fall2025
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-08:00:00
#SBATCH --output=/scratch/smore123/multimodel_%A_%a.log
#SBATCH --error=/scratch/smore123/multimodel_%A_%a.err

SPLIT="${SPLIT:-dev}"

USER=smore123
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL

# 150 queries per task; eval_voyager.py clips the last batch automatically
START=$(( SLURM_ARRAY_TASK_ID * 150 ))
NUM=150

MODEL="${VOYAGER_MODEL:-qwen3-coder-30b-a3b-instruct}"

echo "[$(date)] array=$SLURM_ARRAY_TASK_ID split=$SPLIT start=$START num=$NUM model=$MODEL"

export HOME=/home/$USER
export PYTHONUNBUFFERED=1
export VOYAGER_API_KEY=$(grep VOYAGER_API_KEY $PROJECT/.env | cut -d= -f2)
export VOYAGER_BASE_URL=https://openai.rc.asu.edu/v1
export VOYAGER_MODEL=$MODEL

if [ -z "$VOYAGER_API_KEY" ]; then
    echo "[ERROR] VOYAGER_API_KEY not found in $PROJECT/.env"
    exit 1
fi

source $PROJECT/venv/bin/activate

# Embeddings are pre-cached — no Ollama, no GPU needed
echo "[$(date)] Starting eval (cached embeddings)"

python $PROJECT/eval_voyager.py \
    --model "$MODEL" \
    --split "$SPLIT" \
    --start "$START" \
    --num "$NUM" \
    --checkpoint_every 25

EXIT_CODE=$?
echo "[$(date)] done exit_code=$EXIT_CODE"
exit $EXIT_CODE
