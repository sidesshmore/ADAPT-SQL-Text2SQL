#!/bin/bash
# Run this on a Gaudi compute node after cloning the repo:
#   bash setup_gaudi.sh

set -e

USER=$(whoami)
SCRATCH=/scratch/$USER
PROJECT=$SCRATCH/sidessh/ADAPT-SQL-Text2SQL
HF_CACHE=$SCRATCH/hf_cache

export HOME=/home/$USER
export HF_HOME=$HF_CACHE
export TRANSFORMERS_CACHE=$HF_CACHE
export PYTHONUNBUFFERED=1

echo "[$(date)] Setting up ADAPT-SQL for $USER on Gaudi"
echo "[$(date)] Project: $PROJECT"

# ── Symlink Spider data from smore123 if not already present ─────────────────
if [ ! -d "$PROJECT/data" ]; then
    if [ -d "/scratch/smore123/ADAPT-SQL-Text2SQL/data" ]; then
        echo "[$(date)] Symlinking Spider data from smore123..."
        ln -s /scratch/smore123/ADAPT-SQL-Text2SQL/data $PROJECT/data
        echo "[$(date)] Done."
    else
        echo "[WARN] No data directory found. Copy or symlink Spider data manually:"
        echo "       ln -s /scratch/smore123/ADAPT-SQL-Text2SQL/data $PROJECT/data"
    fi
else
    echo "[$(date)] data/ already present, skipping."
fi

# ── Virtual environment ───────────────────────────────────────────────────────
mkdir -p $HF_CACHE

if [ ! -d "$PROJECT/venv" ]; then
    echo "[$(date)] Creating venv..."
    python3 -m venv $PROJECT/venv
fi

source $PROJECT/venv/bin/activate
echo "[$(date)] venv activated: $(which python)"

# ── Install dependencies ──────────────────────────────────────────────────────
echo "[$(date)] Installing dependencies..."
pip install --upgrade pip -q
pip install -r $PROJECT/requirements.txt -q
pip install -q \
    "transformers>=4.45.0" \
    "trl==1.3.0" \
    "peft>=0.13.0" \
    "accelerate>=0.34.0" \
    "datasets>=3.0.0" \
    "optimum-habana>=1.13.0"

echo "[$(date)] All packages installed."

# ── Verify Gaudi is accessible ────────────────────────────────────────────────
python - <<'EOF'
try:
    import habana_frameworks.torch.core as htcore
    import torch
    print(f"[OK] Gaudi detected — {torch.hpu.device_count()} HPU(s) available")
except ImportError:
    print("[WARN] habana_frameworks not found — will fall back to CPU/CUDA at runtime")
EOF

echo ""
echo "================================================================"
echo " Setup complete!"
echo " To submit the fine-tune job:"
echo "   cd $PROJECT"
echo "   sbatch finetune/finetune_gaudi.sh"
echo "================================================================"
