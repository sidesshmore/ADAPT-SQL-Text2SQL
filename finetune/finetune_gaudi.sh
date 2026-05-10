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
# SLURM_SUBMIT_DIR is set by sbatch to wherever "sbatch" was run from.
# Always run: cd /path/to/repo && sbatch finetune/finetune_gaudi.sh
PROJECT=${SLURM_SUBMIT_DIR:-$SCRATCH/sidessh/ADAPT-SQL-Text2SQL}
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

# Load Habana/Gaudi software stack
module load habana 2>/dev/null || \
    module load synapse 2>/dev/null || \
    module load intel-gaudi 2>/dev/null || \
    echo "[WARN] No Habana module found via module load"
[ -f /etc/profile.d/habanalabs.sh ] && source /etc/profile.d/habanalabs.sh || true
echo "[$(date)] hl-smi: $(hl-smi 2>/dev/null | head -1 || echo 'not found')"

# ── Use the system Gaudi Python env (has habana_frameworks pre-installed) ─────
GAUDI_PYTHON=/packages/envs/pytorch-2.9.0-gaudi/bin/python
GAUDI_PIP=/packages/envs/pytorch-2.9.0-gaudi/bin/pip
export PATH=/packages/envs/pytorch-2.9.0-gaudi/bin:$PATH

echo "[$(date)] Installing/verifying fine-tune dependencies into Gaudi env..."
$GAUDI_PIP install -q \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "datasets>=3.0.0" \
    --user 2>/dev/null || \
$GAUDI_PIP install -q \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "datasets>=3.0.0"

echo "[$(date)] Dependencies ready."

# ── Stage 0: Build pipeline-format SFT data ──────────────────────────────────
SFT_DATA=$SCRATCH/pipeline_sft_data.json
if [ ! -f "$SFT_DATA" ]; then
    echo "[$(date)] Building pipeline-format SFT data..."
    $GAUDI_PYTHON $PROJECT/finetune/build_pipeline_sft_data.py \
        --project_dir $PROJECT \
        --out $SFT_DATA
    echo "[$(date)] SFT data built: $SFT_DATA"
else
    echo "[$(date)] Using existing SFT data: $SFT_DATA"
fi

# ── Stage 1: SFT ─────────────────────────────────────────────────────────────
echo "[$(date)] Launching SFT training on 4 Gaudi 2 cards..."

$GAUDI_PYTHON $PROJECT/finetune/gaudi_spawn.py \
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
    $GAUDI_PYTHON $PROJECT/finetune/gaudi_spawn.py \
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
