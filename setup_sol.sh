#!/bin/bash
# setup_sol.sh — Full one-time setup for ADAPT-SQL on ASU SOL
# Run on a GPU compute node:
#   srun --pty --partition=public --gres=gpu:a100:1 --mem=32G --cpus-per-task=4 --time=4:00:00 /bin/bash
#   bash /scratch/<username>/setup_sol.sh
#
# Usage:
#   bash setup_sol.sh             # full setup (includes vector store build ~1-2h)
#   bash setup_sol.sh --no-index  # skip vector store build (do it later manually)

set -e

# ── Dynamic paths (works for any SOL user) ─────────────────────────────────────
WHOAMI=$(whoami)
SCRATCH=/scratch/$WHOAMI
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
OLLAMA_INSTALL=$SCRATCH/ollama_install
OLLAMA_MODELS=$SCRATCH/ollama_models
OLLAMA_PORT=11437
VENV=$PROJECT/venv
GITHUB_REPO=https://github.com/sidesshmore/ADAPT-SQL-Text2SQL

export HOME=/home/$WHOAMI
export PATH="$OLLAMA_INSTALL/bin:$PATH"
export OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$OLLAMA_MODELS

BUILD_INDEX=true
if [[ "$1" == "--no-index" ]]; then
    BUILD_INDEX=false
fi

echo "========================================================"
echo "  ADAPT-SQL Setup"
echo "  User    : $WHOAMI"
echo "  Scratch : $SCRATCH"
echo "  Project : $PROJECT"
echo "  Index   : $BUILD_INDEX"
echo "========================================================"

# ── CUDA ───────────────────────────────────────────────────────────────────────
module load cuda 2>/dev/null && echo "[✓] CUDA loaded" || echo "[!] CUDA not found — continuing anyway"

# ── Directories ────────────────────────────────────────────────────────────────
mkdir -p "$OLLAMA_INSTALL" "$OLLAMA_MODELS" "$SCRATCH"
echo "[✓] Directories ready"

# ── Install Ollama ─────────────────────────────────────────────────────────────
if [ -f "$OLLAMA_INSTALL/bin/ollama" ]; then
    echo "[✓] Ollama already installed at $OLLAMA_INSTALL/bin/ollama"
else
    mkdir -p "$OLLAMA_INSTALL/bin"

    # Try 1: copy from a known working install on the same cluster
    KNOWN_INSTALL=""
    for candidate in /scratch/smore123/ollama_install /scratch/snande1/ollama_install; do
        if [ -f "$candidate/bin/ollama" ]; then
            KNOWN_INSTALL="$candidate"
            break
        fi
    done

    if [ -n "$KNOWN_INSTALL" ]; then
        echo "[~] Copying Ollama from existing install at $KNOWN_INSTALL..."
        cp -r "$KNOWN_INSTALL/." "$OLLAMA_INSTALL/"
        chmod +x "$OLLAMA_INSTALL/bin/ollama"
        echo "[✓] Ollama copied from $KNOWN_INSTALL"

    else
        # Try 2: download via multiple URL candidates (no sudo, no tgz)
        DOWNLOADED=false
        OLLAMA_URLS=(
            "https://ollama.com/download/ollama-linux-amd64"
            "https://github.com/ollama/ollama/releases/download/v0.3.14/ollama-linux-amd64"
            "https://github.com/ollama/ollama/releases/download/v0.2.8/ollama-linux-amd64"
        )
        for url in "${OLLAMA_URLS[@]}"; do
            echo "[~] Trying: $url"
            if curl -fSL --max-time 120 -o "$OLLAMA_INSTALL/bin/ollama" "$url" 2>/dev/null; then
                chmod +x "$OLLAMA_INSTALL/bin/ollama"
                echo "[✓] Ollama downloaded from $url"
                DOWNLOADED=true
                break
            else
                echo "[!] Failed: $url"
            fi
        done

        if [ "$DOWNLOADED" = false ]; then
            echo "[✗] All download attempts failed."
            echo "    Manual fix: ask smore123 to run:"
            echo "      chmod o+rx /scratch/smore123 /scratch/smore123/ollama_install /scratch/smore123/ollama_install/bin /scratch/smore123/ollama_install/bin/ollama"
            echo "    Then re-run this script."
            exit 1
        fi
    fi
fi

# ── Start Ollama server ────────────────────────────────────────────────────────
if curl -s "http://127.0.0.1:$OLLAMA_PORT/api/version" &>/dev/null; then
    echo "[✓] Ollama server already running on port $OLLAMA_PORT"
else
    echo "[~] Starting Ollama server on port $OLLAMA_PORT..."
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 1
    OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" \
        "$OLLAMA_INSTALL/bin/ollama" serve > "$SCRATCH/ollama.log" 2>&1 &
    sleep 8
    if curl -s "http://127.0.0.1:$OLLAMA_PORT/api/version" &>/dev/null; then
        echo "[✓] Ollama server started"
    else
        echo "[✗] Ollama failed to start — check $SCRATCH/ollama.log"
        exit 1
    fi
fi

# ── Pull models ────────────────────────────────────────────────────────────────
echo "[~] Pulling qwen3-coder (primary LLM — may take 15-30 min)..."
OLLAMA_HOST="http://127.0.0.1:$OLLAMA_PORT" "$OLLAMA_INSTALL/bin/ollama" pull qwen3-coder
echo "[✓] qwen3-coder ready"

echo "[~] Pulling nomic-embed-text (embeddings)..."
OLLAMA_HOST="http://127.0.0.1:$OLLAMA_PORT" "$OLLAMA_INSTALL/bin/ollama" pull nomic-embed-text
echo "[✓] nomic-embed-text ready"

# ── Clone / update repo ────────────────────────────────────────────────────────
if [ -d "$PROJECT/.git" ]; then
    echo "[~] Repo exists — pulling latest..."
    cd "$PROJECT" && git pull --quiet && echo "[✓] Repo up to date"
else
    echo "[~] Cloning $GITHUB_REPO..."
    git clone "$GITHUB_REPO" "$PROJECT"
    echo "[✓] Repo cloned"
fi

# ── Python virtual environment ─────────────────────────────────────────────────
if [ -d "$VENV" ]; then
    echo "[✓] venv already exists"
else
    echo "[~] Creating Python virtual environment..."
    python3 -m venv "$VENV"
    echo "[✓] venv created"
fi

source "$VENV/bin/activate"
echo "[✓] venv activated"

echo "[~] Installing Python dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r "$PROJECT/requirements.txt"
echo "[✓] Dependencies installed"

# ── Streamlit config ───────────────────────────────────────────────────────────
mkdir -p "$PROJECT/.streamlit"
cat > "$PROJECT/.streamlit/config.toml" << 'EOF'
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
EOF
echo "[✓] Streamlit config written"

# ── Build vector store ─────────────────────────────────────────────────────────
if [ "$BUILD_INDEX" = true ]; then
    echo "[~] Building FAISS vector store — embeds ~7,000 training examples"
    echo "    This takes 1-2 hours. Progress + ETA printed as it runs."
    cd "$PROJECT"
    OLLAMA_HOST="http://127.0.0.1:$OLLAMA_PORT" python utils/vector_store.py
    echo "[✓] Vector store built"
else
    echo "[!] Skipping vector store (--no-index). Build it later with:"
    echo "    cd $PROJECT && OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT python utils/vector_store.py"
fi

# ── Write reusable startup script ─────────────────────────────────────────────
cat > "$SCRATCH/start_adapt.sh" << 'STARTUP_SCRIPT'
#!/bin/bash
# start_adapt.sh — Run at the start of every SOL session (auto-generated)

WHOAMI=$(whoami)
SCRATCH=/scratch/$WHOAMI
PROJECT=$SCRATCH/ADAPT-SQL-Text2SQL
OLLAMA_INSTALL=$SCRATCH/ollama_install
OLLAMA_MODELS=$SCRATCH/ollama_models
OLLAMA_PORT=11437

export HOME=/home/$WHOAMI
export PATH="$OLLAMA_INSTALL/bin:$PATH"
export OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$OLLAMA_MODELS

module load cuda 2>/dev/null

# Pull latest code
cd "$PROJECT" && git pull --quiet && echo "[✓] Code up to date"

# Start Ollama if not already running
if curl -s "http://127.0.0.1:$OLLAMA_PORT/api/version" &>/dev/null; then
    echo "[✓] Ollama already running on port $OLLAMA_PORT"
else
    echo "[~] Starting Ollama..."
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 1
    OLLAMA_MODELS="$OLLAMA_MODELS" OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT" \
        "$OLLAMA_INSTALL/bin/ollama" serve > "$SCRATCH/ollama.log" 2>&1 &
    sleep 6
    curl -s "http://127.0.0.1:$OLLAMA_PORT/api/version" &>/dev/null \
        && echo "[✓] Ollama started" \
        || { echo "[✗] Ollama failed — check $SCRATCH/ollama.log"; exit 1; }
fi

source "$PROJECT/venv/bin/activate"
echo "[✓] venv activated"
echo ""
echo "Ready. To launch UI:"
echo "  cd $PROJECT"
echo "  streamlit run ui/app.py --server.port 8501 --server.enableCORS=false --server.enableXsrfProtection=false"
STARTUP_SCRIPT

chmod +x "$SCRATCH/start_adapt.sh"
echo "[✓] Startup script saved to $SCRATCH/start_adapt.sh"

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup complete for: $WHOAMI"
echo ""
echo "  Next sessions : bash $SCRATCH/start_adapt.sh"
echo "  Launch UI     : cd $PROJECT"
echo "                  streamlit run ui/app.py --server.port 8501 \\"
echo "                    --server.enableCORS=false \\"
echo "                    --server.enableXsrfProtection=false"
echo "  Ollama logs   : $SCRATCH/ollama.log"
echo "========================================================"
