#!/bin/bash
#SBATCH --job-name=adapt-sft-snande1
#SBATCH --account=class_cse543spring2026
#SBATCH --partition=public
#SBATCH --qos=class
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/snande1/finetune_sft_%j.log
#SBATCH --error=/scratch/snande1/finetune_sft_%j.err

USER=snande1
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
CHECKPOINT_DIR=$SCRATCH/finetune_checkpoints
HF_CACHE=$SCRATCH/hf_cache

echo "[$(date)] Starting SFT job: $USER"
echo "[$(date)] GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ', ')"

export HOME=/home/$USER
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export HF_DATASETS_CACHE=$HF_CACHE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $HF_CACHE $CHECKPOINT_DIR

module load cuda 2>/dev/null
module load gcc 2>/dev/null

source $PROJECT/venv/bin/activate

echo "[$(date)] Installing/verifying fine-tune dependencies..."
pip install -q \
    "transformers>=4.45.0" \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "bitsandbytes>=0.44.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0"

# ── Stage 0: Build SFT data ──────────────────────────────────────────────────
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
SFT_FINAL=$CHECKPOINT_DIR/sft/final
if [ -f "$SFT_FINAL/adapter_model.safetensors" ]; then
    echo "[$(date)] SFT already complete — nothing to do."
    exit 0
fi

echo "[$(date)] Launching SFT training on 4 A100s..."
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

echo "[$(date)] SFT finished with exit code $?"
