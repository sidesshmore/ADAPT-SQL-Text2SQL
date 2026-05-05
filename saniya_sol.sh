#!/bin/bash
# saniya_sol.sh — One-command startup for ADAPT-SQL on ASU SOL
# Usage: bash saniya_sol.sh

SCRATCH=/scratch/snande1
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
OLLAMA_BIN=$SCRATCH/ollama_install/bin/ollama
OLLAMA_MODELS=$SCRATCH/ollama_models
OLLAMA_PORT=11437
STREAMLIT_PORT=8501

# ── Environment ────────────────────────────────────────────────────────────
export HOME=/home/snande1
export PATH="$SCRATCH/ollama_install/bin:$PATH"
export OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$OLLAMA_MODELS

module load cuda 2>/dev/null

# ── Ollama server ──────────────────────────────────────────────────────────
if curl -s http://127.0.0.1:$OLLAMA_PORT/api/version &>/dev/null; then
    echo "[✓] Ollama already running on port $OLLAMA_PORT"
else
    echo "[~] Starting Ollama server..."
    pkill -f "ollama serve" 2>/dev/null; sleep 1
    OLLAMA_MODELS=$OLLAMA_MODELS OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > $SCRATCH/ollama.log 2>&1 &
    sleep 6
    if curl -s http://127.0.0.1:$OLLAMA_PORT/api/version &>/dev/null; then
        echo "[✓] Ollama started (logs: $SCRATCH/ollama.log)"
    else
        echo "[✗] Ollama failed to start — check $SCRATCH/ollama.log"
        exit 1
    fi
fi

# ── Git pull ───────────────────────────────────────────────────────────────
echo "[~] Pulling latest code..."
cd $PROJECT && git pull --quiet && echo "[✓] Code up to date"

# ── Venv ──────────────────────────────────────────────────────────────────
source $PROJECT/venv/bin/activate
echo "[✓] Virtual environment activated"

# ── Model check ────────────────────────────────────────────────────────────
MODEL=qwen2.5-coder:32b
if OLLAMA_MODELS=$OLLAMA_MODELS OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN list 2>/dev/null | grep -q "qwen2.5-coder:32b"; then
    echo "[✓] $MODEL already available"
else
    echo "[~] Pulling $MODEL (~20GB, this will take several minutes)..."
    OLLAMA_MODELS=$OLLAMA_MODELS OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN pull $MODEL
    echo "[✓] $MODEL downloaded"
fi

# ── Streamlit config (fixes OOD WebSocket origin rejection) ───────────────
mkdir -p $PROJECT/.streamlit
cat > $PROJECT/.streamlit/config.toml << 'EOF'
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
EOF
echo "[✓] Streamlit config written"

# ── Streamlit ─────────────────────────────────────────────────────────────
echo ""
echo "[✓] Starting Streamlit on port $STREAMLIT_PORT..."
echo "    Access via OOD browser, or SSH tunnel from your Mac:"
echo "    ssh -L $STREAMLIT_PORT:localhost:$STREAMLIT_PORT snande1@sol.rc.asu.edu"
echo "    Then open http://localhost:$STREAMLIT_PORT"
echo ""
echo "    In the sidebar: set Ollama Host = http://127.0.0.1:$OLLAMA_PORT"
echo ""

streamlit run $PROJECT/ui/pages/batch_processing.py \
    --server.port $STREAMLIT_PORT
