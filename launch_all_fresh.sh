#!/bin/bash
# launch_all_fresh.sh — cancel everything, clear checkpoints, resubmit all models fresh
#
# Usage: bash launch_all_fresh.sh
#
# Time limits:
#   dev  (1034 queries, 7 tasks × 150):  12h each task
#   test (2147 queries, 15 tasks × 150): 24h each task

set -e

ACCOUNT=grp_hdavulcu
RESULTS_DIR=/scratch/smore123/ADAPT-SQL-Text2SQL/eval_results_voyager
PROJECT=/scratch/smore123/ADAPT-SQL-Text2SQL

# ── 1. Cancel all current jobs ────────────────────────────────────────────────
echo "[1/3] Cancelling all running jobs for smore123..."
scancel -u smore123 || true
echo "      Done."

# ── 2. Backup old results (preserve the data, start fresh) ───────────────────
echo ""
BACKUP="${RESULTS_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
if [ -d "$RESULTS_DIR" ]; then
    echo "[2/3] Backing up $RESULTS_DIR → $BACKUP"
    mv "$RESULTS_DIR" "$BACKUP"
    echo "      Done. (Old results preserved in backup)"
else
    echo "[2/3] No existing results dir found, skipping backup."
fi

# ── 3. Submit all models fresh ───────────────────────────────────────────────
echo ""
echo "[3/3] Submitting all models..."
echo ""

cd "$PROJECT"

MODELS=(
    "qwen3-coder-30b-a3b-instruct"
    "gemma4-31b-it"
    "devstral2-123b"
    "gpt-oss-120b"
    "llama4-maverick-17b"
    "qwen3-235b-a22b-instruct-2507"
)

printf "  %-45s  %s  %s\n" "Model" "Dev JobID" "Test JobID"
printf "  %-45s  %s  %s\n" "─────────────────────────────────────────────" "─────────" "──────────"

for MODEL in "${MODELS[@]}"; do
    DEV_JID=$(sbatch \
        --array=0-6 \
        --account=$ACCOUNT \
        --time=0-12:00:00 \
        --parsable \
        run_sol_multimodel.sh "$MODEL" dev)

    TEST_JID=$(sbatch \
        --array=0-14 \
        --account=$ACCOUNT \
        --time=1-00:00:00 \
        --parsable \
        run_sol_multimodel.sh "$MODEL" test)

    printf "  %-45s  %-9s  %-10s\n" "$MODEL" "$DEV_JID" "$TEST_JID"
done

echo ""
echo "All jobs submitted. Monitor with:"
echo "  watch -n 60 'python3 watch_voyager.py --sol 2>/dev/null'"
echo ""
echo "Dry-run fix after each model completes:"
echo "  source venv/bin/activate && python3 dry_run_fix.py --model MODEL --split dev"
