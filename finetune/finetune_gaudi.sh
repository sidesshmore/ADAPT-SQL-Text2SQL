#!/bin/bash
#SBATCH --job-name=adapt-sql-gaudi
#SBATCH --account=class_cse59827694spring2026
#SBATCH --partition=gaudi
#SBATCH --qos=class_gaudi
#SBATCH --gres=gpu:hl225:4
#SBATCH --mem=400G
#SBATCH --cpus-per-task=64
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
# PT_HPU_LAZY_MODE=0 (eager) breaks _hpu_C.init() on this Synapse AI build — leave unset (defaults to lazy=1)
export HABANA_LOGS=$SCRATCH/habana_logs
mkdir -p $HABANA_LOGS

# Load Habana/Gaudi software stack
module load habana 2>/dev/null || \
    module load synapse 2>/dev/null || \
    module load intel-gaudi 2>/dev/null || \
    echo "[WARN] No Habana module found via module load"
[ -f /etc/profile.d/habanalabs.sh ] && source /etc/profile.d/habanalabs.sh || true
echo "[$(date)] hl-smi output:"
hl-smi 2>/dev/null || echo "hl-smi not found"

# ── HPU smoke test (verify device access before training) ────────────────────
echo "[$(date)] HPU smoke test..."
$GAUDI_BASE/bin/python -c "
import habana_frameworks.torch.core as htcore
import torch
count = torch.hpu.device_count() if hasattr(torch, 'hpu') else 0
print('HPU device_count:', count)
t = torch.zeros(1).to('hpu')
print('Smoke test passed:', t.device)
" && echo "[$(date)] HPU smoke test PASSED" || { echo "[$(date)] HPU smoke test FAILED — aborting"; exit 1; }

# ── Build a venv that inherits the Gaudi env (gets habana + torch for free) ───
GAUDI_BASE=/packages/envs/pytorch-2.9.0-gaudi
GAUDI_VENV=$SCRATCH/gaudi_venv
# Block ~/.local — a CUDA torch there conflicts with the Gaudi torch
export PYTHONNOUSERSITE=1

if [ ! -d "$GAUDI_VENV" ]; then
    echo "[$(date)] Creating Gaudi venv with system-site-packages..."
    $GAUDI_BASE/bin/python -m venv --system-site-packages $GAUDI_VENV
fi

export PATH=$GAUDI_VENV/bin:$GAUDI_BASE/bin:$PATH
GAUDI_PYTHON=$GAUDI_VENV/bin/python

echo "[$(date)] Installing trl/peft/datasets/optimum-habana into Gaudi venv..."
$GAUDI_VENV/bin/pip install -q \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "datasets>=3.0.0" \
    "optimum-habana"

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
SFT_FINAL=$CHECKPOINT_DIR/sft/final
if [ -f "$SFT_FINAL/adapter_model.safetensors" ]; then
    echo "[$(date)] SFT final model already exists at $SFT_FINAL — skipping SFT."
    SFT_EXIT=0
else
    echo "[$(date)] Launching SFT training on 4 Gaudi 2 cards..."
    $GAUDI_PYTHON $PROJECT/finetune/gaudi_spawn.py \
        --nproc_per_node 4 \
        $PROJECT/finetune/train_sft_gaudi.py \
        --project_dir $PROJECT \
        --checkpoint_dir $CHECKPOINT_DIR \
        --hf_cache $HF_CACHE \
        --data_file $SFT_DATA \
        --epochs 3 \
        --batch_size 4 \
        --grad_accum 4 \
        --lr 2e-5 \
        --lora_rank 64 \
        --lora_alpha 128
    SFT_EXIT=$?
    echo "[$(date)] SFT finished with exit code $SFT_EXIT"
fi

# ── Stage 2: GRPO (only if SFT succeeded) ────────────────────────────────────
GRPO_EXIT=1
if [ $SFT_EXIT -eq 0 ]; then
    echo "[$(date)] Launching GRPO training on Gaudi..."
    $GAUDI_PYTHON $PROJECT/finetune/train_grpo_gaudi.py \
        --project_dir $PROJECT \
        --checkpoint_dir $CHECKPOINT_DIR \
        --hf_cache $HF_CACHE \
        --epochs 3 \
        --batch_size 2 \
        --grad_accum 16 \
        --lr 1e-5
    GRPO_EXIT=$?
    echo "[$(date)] GRPO finished with exit code $GRPO_EXIT"
else
    echo "[$(date)] Skipping GRPO — SFT did not complete cleanly."
fi

echo "[$(date)] All done. Checkpoints at: $CHECKPOINT_DIR"
echo "[$(date)] Run merge_export.sh to convert to Ollama model."
exit $GRPO_EXIT
