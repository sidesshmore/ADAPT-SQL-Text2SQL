#!/bin/bash
# setup_sol.sh — Run this on a GPU compute node on ASU SOL
# Usage: bash setup_sol.sh

set -e

SCRATCH=/scratch/smore123
OLLAMA_INSTALL=$SCRATCH/ollama_install
OLLAMA_BIN=$OLLAMA_INSTALL/bin/ollama
OLLAMA_MODELS_DIR=$SCRATCH/ollama_models
OLLAMA_PORT=11437
PROJECT_DIR=$SCRATCH/ADAPT-SQL-Text2SQL

echo "=== ADAPT-SQL Setup on ASU SOL ==="

# ── 1. Verify we're on a compute node ──────────────────────────────────────
if [[ "$SLURM_JOB_ID" == "" ]]; then
    echo ""
    echo "WARNING: No SLURM job detected. You may be on the login node."
    echo "GPU will not be available. For GPU, first run:"
    echo "  srun --pty --partition=public --gres=gpu:a100:1 --mem=32G --cpus-per-task=4 --time=4:00:00 /bin/bash"
    echo "Then re-run this script."
    echo ""
    read -p "Continue anyway? (y/n): " confirm
    [[ "$confirm" != "y" ]] && exit 1
fi

# ── 2. Set environment ──────────────────────────────────────────────────────
export HOME=/home/smore123
export PATH="$OLLAMA_INSTALL/bin:$PATH"
export OLLAMA_MODELS=$OLLAMA_MODELS_DIR
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

echo "[1/7] Environment set (HOME=$HOME, PORT=$OLLAMA_PORT)"

# ── 3. Load CUDA ────────────────────────────────────────────────────────────
module load cuda 2>/dev/null && echo "[2/7] CUDA module loaded" || echo "[2/7] CUDA module not found (continuing)"

# ── 4. Kill any existing Ollama servers ─────────────────────────────────────
echo "[3/7] Stopping any existing Ollama servers..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# ── 5. Start Ollama server with GPU ─────────────────────────────────────────
echo "[4/7] Starting Ollama server on port $OLLAMA_PORT..."
OLLAMA_MODELS=$OLLAMA_MODELS_DIR OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > $SCRATCH/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "      Server PID: $OLLAMA_PID (logs at $SCRATCH/ollama.log)"
sleep 6

# Check if GPU was detected
if grep -q "library=cuda" $SCRATCH/ollama.log; then
    echo "      GPU detected"
elif grep -q "library=cpu" $SCRATCH/ollama.log; then
    echo "      WARNING: Running on CPU only (no GPU detected)"
    echo "      Check $SCRATCH/ollama.log for details"
fi

# ── 6. Pull Gemma 4 if not already present ──────────────────────────────────
echo "[5/7] Checking for Gemma 4 model..."
if $OLLAMA_BIN list 2>/dev/null | grep -q "gemma4"; then
    echo "      Gemma 4 already downloaded, skipping pull"
else
    echo "      Pulling Gemma 4 (~9.6 GB, this will take a few minutes)..."
    $OLLAMA_BIN pull gemma4
fi

# ── 7. Test the model ────────────────────────────────────────────────────────
echo "[6/7] Testing Gemma 4..."
RESPONSE=$($OLLAMA_BIN run gemma4 "Reply with only: ready" 2>/dev/null)
echo "      Model response: $RESPONSE"

# ── 8. Activate venv and verify Python can reach Ollama ─────────────────────
echo "[7/7] Testing Python + Ollama integration..."
source $PROJECT_DIR/venv/bin/activate

python3 - <<'EOF'
import os
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11437"
try:
    import ollama
    r = ollama.chat(model="gemma4", messages=[{"role": "user", "content": "Reply with only: ready"}])
    print(f"      Python ollama OK: {r['message']['content'].strip()}")
except Exception as e:
    print(f"      Python ollama FAILED: {e}")
EOF

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Ollama is running on port $OLLAMA_PORT (PID $OLLAMA_PID)"
echo "To use Gemma 4 in the pipeline, set the model to 'gemma4' in the UI"
echo "or pass model='gemma4' to ADAPTBaseline()."
echo ""
echo "To run the UI (with port forwarding from your laptop):"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo "  streamlit run $PROJECT_DIR/ui/app.py --server.port 8501"
echo ""
echo "On your laptop:"
echo "  ssh -L 8501:localhost:8501 smore123@sol.asu.edu"
echo "  Then open http://localhost:8501"
echo ""
echo "Ollama logs: $SCRATCH/ollama.log"
