#!/bin/bash
#SBATCH --job-name=adapt-sql-finetune
#SBATCH --account=class_cse543spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/%u/finetune_%j.log
#SBATCH --error=/scratch/%u/finetune_%j.err

# ── Auto-detect user ────────────────────────────────────────────────────────
USER=$(whoami)
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
CHECKPOINT_DIR=$SCRATCH/finetune_checkpoints
HF_CACHE=$SCRATCH/hf_cache

echo "[$(date)] Starting fine-tune job for user: $USER"
echo "[$(date)] Project: $PROJECT"
echo "[$(date)] Checkpoints: $CHECKPOINT_DIR"
echo "[$(date)] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

# ── Environment ─────────────────────────────────────────────────────────────
export HOME=/home/$USER
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
mkdir -p $HF_CACHE $CHECKPOINT_DIR

module load cuda 2>/dev/null
module load gcc 2>/dev/null

# ── Venv + packages ─────────────────────────────────────────────────────────
source $PROJECT/venv/bin/activate

echo "[$(date)] Installing/verifying fine-tune dependencies..."
pip install -q \
    "transformers>=4.45.0" \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.44.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0"

echo "[$(date)] Dependencies ready."

# ── Stage 0: Build pipeline-format SFT data ─────────────────────────────────
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

# ── Stage 1: SFT ────────────────────────────────────────────────────────────
echo "[$(date)] Launching SFT training on 4 GPUs..."

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    $PROJECT/finetune/train_sft.py \
    --project_dir $PROJECT \
    --checkpoint_dir $CHECKPOINT_DIR \
    --hf_cache $HF_CACHE \
    --data_file $SFT_DATA \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 2e-5 \
    --lora_rank 64 \
    --lora_alpha 128

SFT_EXIT=$?
echo "[$(date)] SFT finished with exit code $SFT_EXIT"

# ── Stage 2: GRPO (only if SFT succeeded) ───────────────────────────────────
if [ $SFT_EXIT -eq 0 ]; then
    echo "[$(date)] Launching GRPO training..."
    torchrun \
        --nproc_per_node=4 \
        --master_port=29501 \
        $PROJECT/finetune/train_grpo.py \
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
