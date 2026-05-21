#!/usr/bin/env bash
# Launch all 5 models × 2 splits × batches of 150 queries in tmux.
#
# Usage (on SOL compute node, inside tmux):
#   bash launch_all_models.sh dev    # launch only dev split
#   bash launch_all_models.sh test   # launch only test split
#   bash launch_all_models.sh both   # launch dev + test (warning: many windows)
#
# Prerequisites:
#   - tmux session already running
#   - venv activated in this shell (source venv/bin/activate)
#   - VOYAGER_API_KEY in .env
#
# Dev split:  1034 queries → 7 ranges of 150 (last range = 134)
# Test split: 2147 queries → 15 ranges of 150 (last range =  47)
#
# Each range runs in its own tmux window named: <model-slug>-<split>-<start>

set -euo pipefail

SPLIT="${1:-dev}"
PROJECT="/scratch/smore123/ADAPT-SQL-Text2SQL"
SESSION="${TMUX_PANE:+$(tmux display-message -p '#S')}"
SESSION="${SESSION:-adapt}"

# ── Verify/list available models first ──────────────────────────────────────
# Run this once manually to get exact API names:
#   curl -s https://openai.rc.asu.edu/v1/models \
#        -H "Authorization: Bearer $VOYAGER_API_KEY" | python3 -m json.tool | grep '"id"'

# EXACT Voyager API model names (verify with the curl command above)
MODELS=(
    "qwen3-coder-30b-a3b-instruct"   # Qwen3-Coder-30B   — confirmed working
    "qwen3-235b-a22b-instruct"        # Qwen3-235B-A22B   — verify name
    "devstral-2-123b"                 # Devstral2-123B    — verify name
    "gpt-oss-120b"                    # GPT-OSS-120B      — verify name
    "llama4-maverick-17b-128e-instruct"  # LLaMA4-Maverick — verify name
)

# Dev split ranges (150 per batch, last batch is remainder)
DEV_RANGES=(
    "0 150"
    "150 150"
    "300 150"
    "450 150"
    "600 150"
    "750 150"
    "900 150"   # actual end = 1034, script clips automatically
)

# Test split ranges (150 per batch, last batch is remainder)
TEST_RANGES=(
    "0 150"
    "150 150"
    "300 150"
    "450 150"
    "600 150"
    "750 150"
    "900 150"
    "1050 150"
    "1200 150"
    "1350 150"
    "1500 150"
    "1650 150"
    "1800 150"
    "1950 150"
    "2100 150"   # actual end = 2147, script clips automatically
)

launch_window() {
    local model="$1" split="$2" start="$3" num="$4"
    local slug
    slug=$(echo "$model" | sed 's|/|__|g' | cut -c1-20)
    local win="${slug}-${split}-${start}"
    local cmd="cd $PROJECT && source venv/bin/activate && \
python eval_voyager.py --model $model --split $split --start $start --num $num && \
echo '[DONE] $win' ; read"

    if [ -n "${TMUX:-}" ]; then
        tmux new-window -t "$SESSION" -n "$win" "bash -c '$cmd'"
        echo "[launched] $win"
    else
        echo "tmux new-window -n '$win' \"bash -c '$cmd'\""
    fi
}

echo "=== Launching model evals (split=$SPLIT) ==="
echo ""

for model in "${MODELS[@]}"; do
    slug=$(echo "$model" | sed 's|/|__|g' | cut -c1-20)

    if [[ "$SPLIT" == "dev" || "$SPLIT" == "both" ]]; then
        echo "--- $model [dev] ---"
        for range in "${DEV_RANGES[@]}"; do
            read -r start num <<< "$range"
            launch_window "$model" "dev" "$start" "$num"
            sleep 0.3  # avoid thundering herd on tmux
        done
    fi

    if [[ "$SPLIT" == "test" || "$SPLIT" == "both" ]]; then
        echo "--- $model [test] ---"
        for range in "${TEST_RANGES[@]}"; do
            read -r start num <<< "$range"
            launch_window "$model" "test" "$start" "$num"
            sleep 0.3
        done
    fi
done

echo ""
echo "=== All windows launched ==="
echo "Monitor: watch -n 60 'python $PROJECT/watch_voyager.py --sol'"
