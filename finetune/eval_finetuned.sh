#!/bin/bash
#SBATCH --job-name=adapt-eval
#SBATCH --account=class_cse543spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --output=/scratch/%u/eval_%A_%a.log
#SBATCH --error=/scratch/%u/eval_%A_%a.err
#SBATCH --array=0-3

USER=$(whoami)
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL

# Range config: 4 jobs × ~259 examples = 1034 total
STARTS=(0 259 518 777)
NUMS=(259 259 259 257)

START=${STARTS[$SLURM_ARRAY_TASK_ID]}
NUM=${NUMS[$SLURM_ARRAY_TASK_ID]}
RANGE="${START}_$((START + NUM - 1))"

CHECKPOINT_DIR=$SCRATCH/eval_results/range_$RANGE
OLLAMA_PORT=$((11437 + SLURM_ARRAY_TASK_ID))   # 11437, 11438, 11439, 11440

echo "[$(date)] eval job array_id=$SLURM_ARRAY_TASK_ID range=$START+$NUM port=$OLLAMA_PORT"

export HOME=/home/$USER
module load cuda 2>/dev/null

source $PROJECT/venv/bin/activate

# Start Ollama
OLLAMA_MODELS=$SCRATCH/ollama_models \
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT \
    $SCRATCH/ollama_install/bin/ollama serve &

OLLAMA_PID=$!
echo "[$(date)] Ollama PID=$OLLAMA_PID port=$OLLAMA_PORT"
sleep 15  # wait for server to be ready

# Run eval
python $PROJECT/finetune/eval_batch.py \
    --start $START \
    --num $NUM \
    --model adapt-sql-coder \
    --ollama_host http://127.0.0.1:$OLLAMA_PORT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --project_dir $PROJECT

echo "[$(date)] eval done for range $RANGE"
kill $OLLAMA_PID 2>/dev/null
