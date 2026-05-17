#!/bin/bash
#SBATCH --job-name=adapt-grpo-snande1
#SBATCH --account=class_cse57388551fall2025
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --output=/scratch/snande1/finetune_grpo_%j.log
#SBATCH --error=/scratch/snande1/finetune_grpo_%j.err

USER=snande1
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
CHECKPOINT_DIR=$SCRATCH/finetune_checkpoints
HF_CACHE=$SCRATCH/hf_cache

echo "[$(date)] Starting GRPO+merge job: $USER"
echo "[$(date)] GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

export HOME=/home/$USER
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda 2>/dev/null
module load gcc 2>/dev/null

source $PROJECT/venv/bin/activate

pip install -q \
    "transformers>=4.45.0" \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.44.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0"

# ── Verify SFT adapter exists ─────────────────────────────────────────────────
SFT_FINAL=$CHECKPOINT_DIR/sft/final
if [ ! -f "$SFT_FINAL/adapter_model.safetensors" ]; then
    echo "[ERROR] SFT adapter not found at $SFT_FINAL — did SFT job complete?"
    exit 1
fi

echo "[$(date)] Launching GRPO training (single GPU — 4-bit QLoRA incompatible with DDP)..."
python $PROJECT/finetune/train_grpo.py \
    --project_dir $PROJECT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --hf_cache $HF_CACHE \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 16 \
    --lr 1e-5

GRPO_EXIT=$?
echo "[$(date)] GRPO finished with exit code $GRPO_EXIT"

if [ $GRPO_EXIT -eq 0 ]; then
    echo "[$(date)] Running merge + GGUF export..."
    bash $PROJECT/finetune/merge_export.sh
    echo "[$(date)] Merge/export finished with exit code $?"
else
    echo "[$(date)] Skipping merge — GRPO did not complete cleanly."
fi

echo "[$(date)] All done. Checkpoints at: $CHECKPOINT_DIR"
