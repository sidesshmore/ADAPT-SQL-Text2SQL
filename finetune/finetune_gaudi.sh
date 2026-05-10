#!/bin/bash
#SBATCH --job-name=adapt-sql-gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:4
#SBATCH --mem=80G
#SBATCH --cpus-per-task=32
#SBATCH --time=0-24:00:00
#SBATCH --output=/scratch/%u/finetune_gaudi_%j.log
#SBATCH --error=/scratch/%u/finetune_gaudi_%j.err

# ── Auto-detect user and project root ────────────────────────────────────────
USER=$(whoami)
SCRATCH=/scratch/$USER
# Resolve project dir from the location of this script (works regardless of
# whether the repo is at $SCRATCH/ADAPT-SQL-Text2SQL or $SCRATCH/sidessh/...)
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
CHECKPOINT_DIR=$SCRATCH/finetune_checkpoints_gaudi
HF_CACHE=$SCRATCH/hf_cache

echo "[$(date)] Starting Gaudi fine-tune job for user: $USER"
echo "[$(date)] Project: $PROJECT"
echo "[$(date)] Checkpoints: $CHECKPOINT_DIR"

# ── Environment ──────────────────────────────────────────────────────────────
export HOME=/home/$USER
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export PYTHONUNBUFFERED=1
mkdir -p $HF_CACHE $CHECKPOINT_DIR

# Gaudi-specific env
export PT_HPU_LAZY_MODE=0           # Eager mode (more stable for RL/GRPO)
export HABANA_LOGS=$SCRATCH/habana_logs
mkdir -p $HABANA_LOGS

# ── Venv + packages ──────────────────────────────────────────────────────────
source $PROJECT/venv/bin/activate

echo "[$(date)] Installing/verifying Gaudi fine-tune dependencies..."
pip install -q \
    "transformers>=4.45.0" \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0"
pip install -q --no-deps "optimum-habana" || true

echo "[$(date)] Dependencies ready."

# ── Stage 0: Build pipeline-format SFT data ──────────────────────────────────
SFT_DATA=$SCRATCH/pipeline_sft_data.json
if [ ! -f "$SFT_DATA" ]; then
    echo "[$(date)] Building pipeline-format SFT data..."
    python $PROJECT/finetune/build_pipeline_sft_data.py \
        --project_dir $PROJECT \
        --out $SFT_DATA
    echo "[$(date)] SFT data built: $SFT_DATA"
else
    echo "[$(date)] Using existing SFT data: $SFT_DATA"
fi

# ── Stage 1: SFT ─────────────────────────────────────────────────────────────
echo "[$(date)] Launching SFT training on 4 Gaudi 2 cards..."

python $PROJECT/finetune/gaudi_spawn.py \
    --nproc_per_node 4 \
    $PROJECT/finetune/train_sft_gaudi.py \
    --project_dir $PROJECT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --hf_cache $HF_CACHE \
    --data_file $SFT_DATA \
    --epochs 3 \
    --batch_size 2 \
    --grad_accum 4 \
    --lr 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128

SFT_EXIT=$?
echo "[$(date)] SFT finished with exit code $SFT_EXIT"

# ── Stage 2: GRPO (only if SFT succeeded) ────────────────────────────────────
if [ $SFT_EXIT -eq 0 ]; then
    echo "[$(date)] Launching GRPO training on Gaudi..."
    python $PROJECT/finetune/gaudi_spawn.py \
        --nproc_per_node 4 \
        $PROJECT/finetune/train_grpo_gaudi.py \
        --project_dir $PROJECT \
        --checkpoint_dir $CHECKPOINT_DIR \
        --hf_cache $HF_CACHE \
        --epochs 3 \
        --batch_size 1 \
        --grad_accum 4 \
        --lr 1e-5
    echo "[$(date)] GRPO finished with exit code $?"
else
    echo "[$(date)] Skipping GRPO — SFT did not complete cleanly."
fi

echo "[$(date)] All done. Checkpoints at: $CHECKPOINT_DIR"
echo "[$(date)] Run merge_export.sh to convert to Ollama model."
