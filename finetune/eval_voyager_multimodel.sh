#!/bin/bash
#SBATCH --job-name=adapt-voyager
#SBATCH --account=class_cse57388551fall2025
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/smore123/voyager_logs/%x_%A_%a.log
#SBATCH --error=/scratch/smore123/voyager_logs/%x_%A_%a.err
#SBATCH --array=0-7

USER=smore123
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL

# Required env vars (pass via --export):
#   VOYAGER_MODEL  — model name on Voyager API
#   SPLIT          — "dev" or "test" (default: dev)
if [ -z "$VOYAGER_MODEL" ]; then
    echo "[ERROR] VOYAGER_MODEL not set."
    echo "Usage: sbatch --export=ALL,VOYAGER_MODEL=<model>,SPLIT=dev|test $0"
    exit 1
fi
SPLIT=${SPLIT:-dev}

# Partition sizes per split
if [ "$SPLIT" = "test" ]; then
    # 2147 queries across 8 tasks
    STARTS=(0 269 538 807 1076 1345 1614 1883)
    NUMS=(269 269 269 269 269 269 269 264)
    TOTAL=2147
else
    # 1034 queries across 8 tasks
    STARTS=(0 130 260 390 520 650 780 910)
    NUMS=(130 130 130 130 130 130 130 124)
    TOTAL=1034
fi

START=${STARTS[$SLURM_ARRAY_TASK_ID]}
NUM=${NUMS[$SLURM_ARRAY_TASK_ID]}
RANGE="${START}_$((START + NUM - 1))"

# Checkpoint dir: eval_results_voyager/<model_safe>/<split>/range_<range>/
MODEL_SAFE=$(echo "$VOYAGER_MODEL" | tr '/: ' '___')
CHECKPOINT_DIR=$SCRATCH/eval_results_voyager/${MODEL_SAFE}/${SPLIT}/range_${RANGE}

echo "[$(date)] voyager-eval model=$VOYAGER_MODEL split=$SPLIT array_id=$SLURM_ARRAY_TASK_ID range=$RANGE"

export HOME=/home/$USER
export PYTHONUNBUFFERED=1
export VOYAGER_API_KEY=$(grep VOYAGER_API_KEY $PROJECT/.env | cut -d= -f2)
export VOYAGER_BASE_URL=https://openai.rc.asu.edu/v1

if [ -z "$VOYAGER_API_KEY" ]; then
    echo "[ERROR] VOYAGER_API_KEY not found in $PROJECT/.env"
    exit 1
fi

mkdir -p $SCRATCH/voyager_logs
source $PROJECT/venv/bin/activate

echo "[$(date)] Starting eval model=$VOYAGER_MODEL split=$SPLIT range=$RANGE"

python $PROJECT/eval_voyager.py \
    --split $SPLIT \
    --start $START \
    --num $NUM \
    --checkpoint_dir $CHECKPOINT_DIR \
    --checkpoint_every 25

EXIT_CODE=$?
echo "[$(date)] eval done exit_code=$EXIT_CODE model=$VOYAGER_MODEL split=$SPLIT range=$RANGE"
